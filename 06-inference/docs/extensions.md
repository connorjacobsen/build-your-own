# After the ten core stages

These projects extend your engine; they are not features claimed by the default gate. Ask for an independent extension grader when selecting one. The instructor should add acceptance cases without supplying the implementation.

## Incremental allocation and recompute preemption

The beginner engine reserves maximum declared capacity. Replace reservation with page growth as positions execute. Under pressure, reclaim a running victim's pages and requeue it while preserving token history and RNG state. Recompute already generated positions without sampling them again.

Acceptance requirements: force preemption with a tiny pool, preserve output versus the baseline, reclaim all ownership, and prove progress under two mutually competing requests. Repeated mutual eviction can create livelock even when no memory is leaked. Compare reservation slack and admitted concurrency on early-EOS workloads.

## A true paged attention kernel

Read K/V indirectly through the block table inside a CUDA/Triton kernel. Use online softmax to combine tiles with one common denominator. Begin with one decode query/head, then add GQA, multiple requests, and partial tails. Prefill adds the query-position dimension and offset mask.

Acceptance requirements: shuffled physical pages, nonmultiple lengths, partial tails, extreme logits, multiple KV heads, dtype-specific tolerances, bounded scratch storage, and benchmarks excluding compilation/warmup. Your Python paged implementation becomes the numerical oracle. Do not call a gather-plus-dense attention path a fused paged kernel.

## Streaming HTTP

Return incremental token events with a final finish event. Bound every per-request output queue and define overflow behavior. A slow client must not stall all other requests or grow memory without limit. Decode UTF-8 incrementally because one character can span multiple generated IDs.

Acceptance requirements: ordered interleaved streams, exactly one final event, disconnect cleanup, queue-bound enforcement, and correct split-Unicode behavior.

## Load a real checkpoint

Choose a tiny model and implement its exact architecture, tokenizer, normalization, rotary convention, and checkpoint mapping. Compare dense logits against its original library before enabling caching. Familiar names and matching tensor shapes do not prove equivalence.

Acceptance requirements: a pinned model revision, deterministic fixture prompts, dense/logit parity, cached parity, and the existing systems regression suite. External model downloads are outside the offline core course.

## Distributed execution

Begin with the algebra of splitting a linear map by output features versus input features. Output partitions can be concatenated; input partitions produce partial sums requiring a reduction. PyTorch weights have storage shape `[out,in]`, so map the algebra carefully to axes.

Extend into head placement, KV sharding/replication, and tensor-parallel collectives. Pipeline parallelism instead partitions layers; data-parallel serving replicates engines. Measure communication and small-batch overhead rather than assuming more devices are faster.

## Speculative decoding

For greedy verification, a target forward over prompt plus draft scores draft token zero at logit position P−1. Commit only the matching prefix and the target choice at the first mismatch; when all draft IDs match, a bonus target token can be emitted.

Stochastic verification requires probability-correct acceptance/rejection, not merely comparing sampled IDs. Read the [original speculative decoding method](https://arxiv.org/abs/2211.17192). Cache rollback, proposed versus committed frontiers, and preservation of sampling semantics belong in the test plan.
