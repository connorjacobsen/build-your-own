> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 08 — Share prefixes safely

**Build:** `PrefixCache`, `BlockPool.copy_on_write`, and engine prefix integration.  
**Prerequisite:** stage 07. **Estimated effort:** 3–5 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## Reuse computation, not answers

Requests with the same prefix under the same immutable model can reuse the prefix's K/V. Their continuations still use their own sampling states. A prefix cache is not a cache of generated text.

The K/V for a block depend on all earlier context, not only the token IDs inside that block. Block `[C,D]` after `[A,B]` generally differs from `[C,D]` after `[X,Y]`. Identity must include the prefix. For this engine you may use exact token tuples up to each block boundary. The cache belongs to one engine/model; swapping weights or positional settings requires invalidation. You are not implementing cross-model or multi-tenant sharing.

## PrefixCache contract

Construct with `PrefixCache(pool)`. The cache holds reusable full prompt blocks and exposes an `entries` mapping for inspection. The exact key representation is your choice, provided identity includes complete preceding context and cannot produce false hits.

| API | Required behavior |
|---|---|
| `publish(prompt,table,computed)` | Retain complete blocks covered by both prompt length and computed length; do not republish duplicate identities with extra ownership |
| `acquire(prompt)` | Return physical IDs for the longest available full prefix, acquire one request reference per returned block, and leave enough prompt work to produce final logits |
| `make_room(needed)` | Evict cache-only retained entries until `pool.n_free >= needed` if possible; return whether enough free blocks exist |
| `clear()` | Drop all cache ownership and lookup entries while preserving active request ownership |

Use one retained ownership reference per cache entry, plus one per active request using the block. A block with count one may be cache-only and evictable; higher counts mean an active owner remains. Use lookup/publish recency to prefer old unused entries when choosing victims. This convention is explicitly different from vLLM's cache-aware free-queue reference convention.

## Last-logit and block-boundary rules

KV state does not contain the final prompt logits. You must execute at least one prompt input position before sampling a continuation, unless you implement another explicitly stored output-state mechanism—which this stage does not.

With prompt length P and block size B, reuse at most `floor((P−1)/B)*B` positions. For P=8 and B=4, reuse four, then recompute the final four-token block. For P=9, reuse eight, then compute one. This block-aligned restriction makes shared pages immutable and the model interface straightforward.

Do not share a partially filled prompt tail in the engine. That tail can later receive generated-token state and must remain private. Standalone copy on write below explains how branching systems can safely generalize beyond this restriction.

## Integrate with admission and completion

When a waiting request is considered for admission, find reusable blocks and acquire their references before any eviction could reclaim them. Count them toward its reserved capacity. Reserve private blocks only for the remaining logical blocks. If admission fails, undo any acquired request references; the cache's own retention remains.

On a hit, initialize the request's computed frontier and `cached_tokens` from reused full blocks. Feed only the uncached suffix into model execution. Publish newly completed full prompt blocks only after their K/V have actually been written. A sampled output ID is not computed KV.

Completion releases request ownership but can leave cached retention. Clearing the cache after all requests finish returns every physical block to free capacity. Clearing while a request is active removes only cache ownership; that request must still be able to use its retained pages.

Two simultaneously arriving requests cannot reuse computation that has not yet happened. The grader therefore warms a prefix by completing one request before submitting the second. Cache metadata alone is not a valid hit.

## Copy on write contract

`pool.copy_on_write(table,logical_block)` returns the physical ID to use for future writes and updates that table entry if necessary. If the block has one owner, return it unchanged. If shared, allocate a private block, copy K/V for all layers, update only the supplied table, and release that branch's old reference.

If no physical block is available, raise `MemoryError` without changing the original table or ownership counts. Other branches must continue seeing the old content. This method is graded independently; the full-block engine policy does not require a branching-generation API.

The ordering matters. Updating the table before allocation succeeds can orphan the old block or expose an uninitialized new one. Dropping the old reference too early can allow reuse while another operation still expects its content.

## What the tests make observable

The grader tests false-prefix matches, exact block boundaries, partially computed blocks, active-owner eviction prevention, clear-with-live-owners, copy-on-write isolation, allocation failure, and capacity recycling under repeated prompts.

It also counts the real rows projected in cold and warm generation. A hit counter that claims eight reused tokens while the model still processes those eight positions fails. Output equivalence and skipped computation are separate requirements.

Your cache can retain useful entries even when all requests are finished, so `n_free == num_blocks` is not expected until retained cache state is cleared. Explain whether each reference in a debugging print belongs to a request or the cache. Anonymous reference increments are difficult to audit.

## Production bridge

Exact full-prefix tuples are pedagogically convenient but repeat metadata. Production designs often chain hashes or maintain richer indexes. They may include model/adaptor identity, multimodal inputs, or isolation salts. A more compact key must preserve the same identity semantics.

Read [the pinned prefix-caching design](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/prefix_caching.md) and [get_computed_blocks](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/kv_cache_manager.py#L154). Concentrate on ownership acquisition and last-token handling before adopting any data structure.

**Next:** [Stage 09 — Serving](09_serving.md).
