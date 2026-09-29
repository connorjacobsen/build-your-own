# 06 · Build a paged KV allocator

**Time:** 2–3 hours. **Prerequisite:** cache length versus capacity. **Deliverable:** a fixed-size block pool with ownership invariants. **Build:** `physical_slots` and an allocator variant.

A contiguous per-request cache has an awkward choice: allocate enough for the largest possible continuation, or repeatedly move/copy tensors as the sequence grows. Paging separates a sequence's logical positions from the physical blocks holding its cache.

For block size `B`, logical token position `p` has logical block `p // B` and offset `p % B`. A **block table** maps the logical block to a physical block ID. The slot is therefore `(table[p // B], p % B)`. The sequence stays logically contiguous even when physical block IDs are scattered.

Our pool allocates K and V tensors shaped `[layers,physical_blocks,block_size,kv_heads,head_dim]`. All layers use the same block ID mapping. A block is a unit of ownership across all layers, not a GPU thread block and not a Python memory page. Production attention backends may use different physical layouts.

```python
B = 4
table = [5, 1, 7]
positions = list(range(10))
slots = [(table[p // B], p % B) for p in positions]
for p, slot in zip(positions, slots):
    print(f"logical token {p:2} → physical block {slot[0]}, offset {slot[1]}")
assert slots[4] == (1, 0)
assert slots[9] == (7, 1)
```

## What paging saves, and what it does not

Under on-demand allocation, a nonempty sequence with `T` tokens needs `ceil(T/B)` blocks. Its last block wastes between zero and `B−1` slots. This is **internal fragmentation**. Paging also avoids requiring a single large contiguous region for each sequence.

The physical pool itself is preallocated and occupies its full tensor memory even when many blocks are free. Freeing a block makes capacity available to another request; it does not shrink the pool tensor or return that memory to the OS/device allocator. Distinguish resident bytes, assigned blocks, valid cached tokens, and unused slots.

Our integrated scheduler later reserves a request's maximum needed blocks at admission to keep the first implementation deadlock-free. That reservation adds unused capacity beyond last-block fragmentation. The formula below describes on-demand paging; it is not a claim that our scheduler always achieves that footprint.

```python
lengths = [1, 7, 9, 17, 31]
rows = []
for block_size in [1, 2, 4, 8, 16]:
    allocated = sum(math.ceil(n/block_size)*block_size for n in lengths)
    rows.append((block_size, sum(lengths), allocated, allocated-sum(lengths)))
print("block size, useful tokens, allocated slots, slack")
for row in rows:
    print(row)
plt.bar([str(r[0]) for r in rows], [r[3] for r in rows])
plt.xlabel("Block size"); plt.ylabel("Unused tail slots"); plt.show()
```

Smaller blocks reduce tail slack but increase metadata and can make kernel access less efficient. Larger blocks reduce table lengths but reduce reuse granularity. A block-size choice is a tradeoff, not a universal constant.

## Free lists and reference counts

A free-list entry means a physical block has no owner. Allocation removes IDs from the free queue and gives each one a reference count of one. Sharing adds owners; release removes an owner. A block returns to the free queue exactly when its count reaches zero.

Allocation failure must be atomic: either return all requested blocks or leave the pool unchanged. Partial allocation followed by an exception is a common leak. Double-free must fail loudly; otherwise the same physical block can appear twice in the free list and be assigned to unrelated requests.

```python
from nanovllm_course.cache import BlockPool
cfg = TinyConfig()
pool = BlockPool(cfg, num_blocks=4, block_size=4)
a = pool.allocate(3)
before = (pool.refs.copy(), list(pool.free))
try:
    pool.allocate(2)
except MemoryError:
    print("Allocation rejected without partial mutation")
assert before == (pool.refs, list(pool.free))
pool.retain([a[0]])
assert pool.refs[a[0]] == 2
pool.release(a)
assert pool.n_free == 3
pool.release([a[0]])
pool.check()
assert pool.n_free == 4
assert pool.storage_bytes == cfg.kv_bytes_per_token() * 16
```

## Recycled memory and valid length

A recycled block can contain stale data from its previous owner. Correctness must come from valid-length metadata and masking, not from hoping unused memory contains zeros. Our `read` gathers only the logical prefix requested by the caller; the model's causal mask limits each query's visible portion.

Zeroing every allocated block is expensive and unnecessary for mathematical correctness when valid lengths are correct. This course initializes the pool to zero for predictable debugging but does not clear it on reuse. Production isolation requirements are broader than this mathematical reference; for this single-user engine the key lesson is that stale slots must never be read as valid context.

```python
pool = BlockPool(cfg, num_blocks=3, block_size=4)
allocated = pool.allocate(3)
table = [allocated[2], allocated[0], allocated[1]]
k = torch.arange(7 * cfg.n_kv_heads * cfg.head_dim).float().view(7, cfg.n_kv_heads, cfg.head_dim)
v = -k
pool.write(0, table, 0, k, v)
read_k, read_v = pool.read(0, table, 7)
torch.testing.assert_close(read_k, k)
torch.testing.assert_close(read_v, v)
pool.release(table); pool.check()
```

## Build and check

Implement `physical_slots(table,block_size,start,count)` returning block IDs and offsets for a range. Reject ranges beyond the block table. Then build a small allocator using a `deque` and reference counts. Test exact-fit allocation, failure with one block too few, sharing, double-free, and reuse after completion.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("slots")
else:
    print("Slot-mapping exercise skipped.")
```

**Checkpoint:** for `B=4`, table `[6,2]`, and logical position 5, which physical flat token slot would a flattened pool use? **Answer:** `2*4+1=9`. The logical position remains 5 for RoPE; physical slot 9 is an address, not a model position.

Upstream bridge: inspect `get_new_blocks` and `free_blocks` in [vLLM's block pool](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py). Its cache-aware free queue is more sophisticated than ours. The [PagedAttention paper](https://arxiv.org/abs/2309.06180) explains the original systems motivation. We implement the memory indirection here, not its optimized CUDA kernel.

Next: [07 — Run attention through block tables](07_paged_attention.ipynb).
