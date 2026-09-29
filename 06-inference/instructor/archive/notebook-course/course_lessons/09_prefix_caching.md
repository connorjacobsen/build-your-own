# 09 · Share prefixes without corrupting them

**Time:** 2–3 hours. **Prerequisite:** reference counts and computed-token frontiers. **Deliverable:** reusable full-prefix blocks with safe eviction, plus a copy-on-write experiment. **Build:** `reusable_prefix_tokens`.

Two requests beginning with the same token prefix under the same model can reuse the same computed K/V for that prefix. This saves prompt computation and may save physical capacity. It does not cache an answer: continuations are still generated independently with each request's sampling state.

A block's K/V depends on the whole context before it. Therefore, keying only by the tokens inside the block is wrong. The second block `[C,D]` after prefix `[A,B]` is not generally interchangeable with `[C,D]` after `[X,Y]`.

Our cache keys are exact tuples of all prompt tokens up to a full block boundary. This is easy to reason about and avoids lossy custom hashes, but storing many repeated prefixes costs metadata and tuple-building work. Production systems use chained hashes or other indexed structures. The cache here belongs to one engine with fixed weights and positional settings; never transfer it to another model or mutate the model while it is live.

## Why reuse only full blocks?

A full block's contents can remain immutable while requests append into later private blocks. Sharing a partial tail is harder because two continuations might write to the same physical slots. You need copy on write before either branch mutates that shared block.

The reference prefix cache shares only complete prompt blocks. At admission it reuses at most `floor((P−1)/B)*B` prompt positions. Leaving at least one input token to compute is necessary because KV entries do not contain the final prompt logits. At an exact block boundary this policy recomputes the final block. This matches the pinned source's block-aligned last-token limitation, not an unavoidable law of all cache designs.

```python
for length in [3,4,5,8,9]:
    reusable = ((length-1)//4)*4
    print(f"prompt={length}, block_size=4, maximum reusable tokens={reusable}")
assert ((8-1)//4)*4 == 4
```

## Observe real reuse and output preservation

Run the first request to completion before submitting the second. Requests admitted together before a prefix has been computed cannot reuse work that does not exist yet.

```python
model = make_model()
engine = Engine(model, block_size=4, num_blocks=12, prefill_chunk=4, token_budget=8)
prompt = [256,1,2,3,4,5,6,7,8]
engine.add_request("cold", prompt, 5)
engine.run()
cold_work = sum(step["tokens"] for step in engine.history)
mark = len(engine.history)
engine.add_request("warm", prompt, 5)
engine.run()
warm_work = sum(step["tokens"] for step in engine.history[mark:])
assert engine.requests["warm"].output == engine.requests["cold"].output
assert engine.requests["warm"].cached_tokens == 8
assert cold_work-warm_work == 8
print("Computed positions:", {"cold":cold_work, "warm":warm_work})
print("Retained prefix entries:", len(engine.prefix.entries))
```

This proves skipped computation. It does not guarantee a particular wall-clock speedup. Small Python workloads can be dominated by cache lookup and bookkeeping. Benchmark cold and warm runs separately and include prefix hit counts alongside time.

## Ownership and eviction

Our convention assigns one reference to each cache entry and one more to each active request using the block. An entry whose block has reference count one is retained only by the cache and can be evicted. Eviction deletes the lookup entry and releases that reference. A block still used by a request cannot be reclaimed.

This differs from vLLM's convention, where cached blocks can remain in an evictable free queue with zero active-request references. Both can be correct if all operations consistently use the same ownership model. Do not copy a reference-count check from one convention into the other.

```python
from nanovllm_course.cache import BlockPool, PrefixCache
pool = BlockPool(model.cfg, num_blocks=2, block_size=2)
cache = PrefixCache(pool)
owner = pool.allocate(2)
cache.publish([1,2,3,4], owner, computed=4)
shared = cache.acquire([1,2,3,4,5])
assert shared == owner
assert not cache.make_room(1)  # Active owners prevent reclamation.
pool.release(owner)
assert not cache.make_room(1)  # The second request still owns both blocks.
pool.release(shared)
assert cache.make_room(2)
assert pool.n_free == 2 and not cache.entries
pool.check()
```

Our LRU order is based on entry lookup/publishing. `make_room` evicts only unreferenced-by-request entries, skipping active ones. Capacity pressure may remove an early prefix entry while leaving a later entry; lookup stops at the first missing block, preserving correctness at the cost of possible reuse. A production design can improve the eviction/lookup data structures.

## Copy on write for a shared partial tail

This standalone experiment demonstrates a feature needed for branching generation. It is not used by our engine's full-block-only prefix sharing. To fork, copy the block table and retain its IDs. Before writing into a shared tail, allocate a private block, copy all layers' K/V, replace the child's mapping, and release the child's ownership of the old block.

```python
pool = BlockPool(model.cfg, num_blocks=3, block_size=4)
parent = pool.allocate(1)
pool.k[:,parent[0]].fill_(3)
child = parent.copy(); pool.retain(child)
old = parent[0]
new = pool.copy_on_write(child, 0)
assert new != old
assert torch.all(pool.k[:,new] == 3)
pool.k[:,new,2].fill_(99)
assert torch.all(pool.k[:,old,2] == 3)
pool.release(parent + child); pool.check()
assert pool.n_free == 3
```

If allocation fails, the original table and counts must remain valid. If the block is already privately owned, copy on write should do nothing. A shared prefix should never be mutated just because a request has permission to append later positions.

## Build and check

Implement `reusable_prefix_tokens(prompt_length,block_size)` with the last-logit rule. Then extend the cache exercise with a prefix that shares the same second-block tokens but a different first block. It must miss.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("prefix")
else:
    print("Prefix boundary exercise skipped.")
engine.clear_prefix_cache()
assert engine.pool.n_free == engine.pool.num_blocks
```

**Questions:** Can a cache hit be used after changing weights? Can a sampled token ID be published as a computed KV entry immediately? Why are hashes of local block tokens insufficient?

<details><summary>Answers</summary>

No: weights determine all cached states. No: sampling an ID does not compute that ID's K/V; it must first pass through the model. A block's representations depend on earlier context, so identity must include the prefix as well as local tokens and any model/adaptor settings.
</details>

Upstream bridge: read [the pinned prefix-caching design](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/prefix_caching.md) and `get_computed_blocks` in [KVCacheManager](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L154). Pay attention to full-block boundaries, identity, and the distinction between lookup and ownership acquisition.

Next: [10 — Local serving](10_serving.ipynb).
