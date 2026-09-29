> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 06 — Build and use physical pages

**Build:** `BlockPool` in `cache.py` and `TinyLM.forward_paged_batch` in `model.py`.  
**Prerequisite:** stage 05. **Estimated effort:** 4–6 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## From an expanding tensor to reusable capacity

The contiguous cache makes reuse easy to understand, but repeated concatenation copies historical data. A fixed per-request maximum buffer avoids copying but can strand memory in requests that finish early. Paging separates logical sequence order from physical storage so fixed-size blocks can be assigned and recycled independently.

Let B be block size, p a logical position, and T a request's block table. The logical block is `p // B`, the offset is `p % B`, and the physical block ID is `T[p // B]`. The model position is still p. Never apply RoPE using a physical address.

For B=4 and table `[5,1,7]`, logical position 6 lives in physical block 1, offset 2. Physical block IDs can be nonmonotonic. A test using only table `[0,1,2]` is insufficient because an implementation ignoring indirection can accidentally pass.

## BlockPool contract

`BlockPool(cfg,num_blocks=64,block_size=8,device="cpu",dtype=torch.float32)` owns K and V tensors named `k` and `v`, each shaped `[layers,num_blocks,block_size,Hkv,d]`. All layers use the same physical block IDs. Expose `num_blocks`, `block_size`, and `refs`, a list of integer ownership counts.

| API | Behavior |
|---|---|
| `n_free` | Number of physical blocks with zero owners |
| `storage_bytes` | Resident bytes in both tensor pools, including currently unassigned slots |
| `allocate(count)` | Return distinct available block IDs, with one owner each; fail atomically with `MemoryError` if insufficient capacity |
| `retain(blocks)` | Add an owner to each currently owned block |
| `release(blocks)` | Remove one owner; make a block reusable when count reaches zero; reject double-free with `ValueError` |
| `slots(table,start,count)` | Return per-token physical IDs and within-block offsets for that logical range, as tensor or list sequences |
| `write(layer,table,start,k,v)` | Store the new chunk; reject writes to blocks whose reference count is not exactly one |
| `read(layer,table,length)` | Return contiguous logical K/V for positions `[0,length)` |
| `check()` | Assert consistent free capacity and nonnegative ownership; return value is irrelevant |

Positive pool/block sizes are required. Negative allocation counts and logical ranges outside the table raise `ValueError`. Zero-length allocation returns no blocks. `read` and `write` operate on valid supplied shapes; you need not implement a general tensor schema validator.

Allocation order is your choice. Do not assume tests require FIFO. The `refs` list and named tensors are explicit inspection boundaries, not a requirement to use the instructor's internal free-list implementation. Copy on write and the `PrefixCache` class are stage 08 work.

## Ownership must be explicit

A request's table owns one reference to each block. Sharing adds another owner. Releasing one owner must not free a block still in use. A free block must not occur twice in the allocator's available pool; otherwise two future allocations can alias accidentally.

Atomic allocation means a failed request for three blocks cannot consume the two that were free. Check capacity or provide rollback before exposing a partial result. Double-free is an error, not an opportunity to silently add an ID to a free queue again.

The physical tensors keep their allocated size after blocks are released. `n_free` describes reusable capacity; it does not mean memory has returned to the OS or device allocator. With on-demand paging, tail slack for one sequence is less than B tokens. Our later beginner scheduler reserves future capacity too, so its reservation slack must be accounted for separately.

## Connect the model to the pool

Implement:

```text
forward_paged_batch(items,pool) -> list_of_logits
items = [(new_ids,block_table,num_computed_tokens), ...]
```

Each request's table already contains enough capacity. New IDs begin at its computed count. Preserve order, use absolute positions, and return `[new_tokens,vocab]` logits per item. An empty item list returns `[]`.

At each layer, project all new packed rows once, store their K/V through the table, and attend to each request's valid historical context. You may gather historical tensors and call your reference attention. This is real paged storage with a slow attention backend; it is not yet an optimized PagedAttention kernel.

The pool must be the source of truth. Do not maintain a separate contiguous cache that bypasses page reads, or reconstruct old states from saved token IDs. The grader edits physical V tensors between calls and checks the resulting logits. It also instruments packed projections to ensure only new positions are projected.

## Mask stale capacity, not just other requests

Recycled blocks may contain old data. Correctness comes from logical length and causal masks, not from zero-filled unused slots. The final physical block can be partially valid. Reading all B slots and masking only by the number of allocated blocks exposes stale values and changes softmax normalization.

Logical length limits the gathered history. For a multi-token new chunk, the offset mask additionally prevents an earlier new query from reading a later new key. These are two related but distinct boundaries.

## Acceptance and failure experiments

The gate tests shuffled physical blocks, unequal requests, chunks crossing page boundaries, atomic failure, ownership counts, write protection, memory byte counts, true page dependence, and new-row projection counts. Try prompt lengths B−1, B, and B+1 in your own scratch experiments; boundaries are where implicit assumptions become visible.

If numerical outputs fail only after an append, inspect which physical slots you wrote and read for the first new token. If two requests interfere, inspect table ownership and packed boundaries. If outputs are right but projection counts fail, you have preserved the function while missing the intended optimization.

**Reading:** [the PagedAttention paper](https://arxiv.org/abs/2309.06180) and [pinned block pool](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/block_pool.py#L257). This course makes no claim to reproduce the paper's throughput results.

**Next:** [Stage 07 — A living batch](07_scheduler.md).
