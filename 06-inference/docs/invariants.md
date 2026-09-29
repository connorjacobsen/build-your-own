# Invariants and debugging order

## Coordinates

There are three different coordinates. **Absolute model position** determines RoPE and causal visibility. **Packed row offset** identifies a new token's row in a combined execution tensor. **Physical cache slot** identifies storage through a block table. Confusing any two can preserve shapes while corrupting outputs.

For block size B, request table T, and logical position p:

```text
logical block = p // B
within-block offset = p % B
physical block = T[p // B]
flattened physical token slot = physical block * B + offset
```

The RoPE position is still p.

## Model and cache

- Each layer caches its own post-RoPE K and unrotated V.
- A query at absolute position p sees keys 0 through p, inclusive.
- Appending does not change previously computed states under fixed weights and model settings.
- Cache validity ends when weights or relevant execution semantics change.
- For GQA, query-head to KV-head mapping is consistent in every execution path.
- Dense, contiguous-cache, and paged logits should agree within stated floating-point tolerances.

## Request frontier

`tokens` includes the prompt and all sampled output IDs. `computed` counts token positions with valid KV entries. A sampled ID is not computed until it has passed through the model.

Before execution, pending work is `len(tokens)-computed`. A partial prompt chunk cannot sample. After a normal nonterminal emission, `len(tokens)-computed == 1`. After G outputs from a prompt of P tokens, the final computed count is P+G−1 (assuming G>0 and no unused precomputation). A zero-output request executes nothing.

## Capacity and progress

Admission reserves `ceil((P+G−1)/B)` logical blocks; prefix reuse replaces some with shared blocks. Private allocation is atomic. Requests larger than the total pool are rejected. A running request needs no additional pages and can therefore finish without cache-allocation deadlock. This proof depends on reservation; it does not transfer unchanged to an on-demand allocator.

FCFS admission can block small requests behind a large one. Round-robin execution prevents permanent execution starvation among admitted finite requests when the token budget is positive. No finite latency bound is claimed for arbitrarily growing incoming load.

## Ownership

Every free block has refcount zero and occurs once in the free queue. Every nonfree block has one or more owners. Active request tables own one reference per block. Each prefix entry owns one additional reference in this implementation. Completed/cancelled requests release their table once. Clearing retained cache entries must not invalidate active owners.

Shared blocks are immutable. Before changing a shared partial tail, allocate a private copy and update only the writing branch's table. Allocation failure must leave the old ownership intact.

## Debug a mismatch

1. Fix the seed, weights, dtype, device, and prompt. Disable stochastic sampling first.
2. Compare dense versus chunked logits at each position, beginning with one layer.
3. Check Q/K/V shapes, GQA mapping, and absolute positions.
4. Visualize a small offset mask; perturb forbidden future values.
5. Randomize physical block IDs and compare cache contents in logical order.
6. Add multiple requests and perturb only one to test sequence isolation.
7. Add scheduling, ensuring no sample occurs before the computed frontier reaches available IDs.
8. Add prefix hits, eviction, cancellation, and capacity pressure last.

A performance regression is not a numerical failure. A correct token output alone does not prove an optimization skipped work. Use operation/input-length traces and memory accounting as well as output assertions.
