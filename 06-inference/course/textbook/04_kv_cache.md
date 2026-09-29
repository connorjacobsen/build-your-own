> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 04 — Reuse the past

**Build:** cached `TinyLM.forward` and `generate_cached` in your existing modules.  
**Prerequisite:** stage 03. **Estimated effort:** 2–4 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## The invariant that permits caching

In a causal decoder, the representation at position i depends only on tokens through i. Appending a later token cannot change already computed representations when the weights and model settings are fixed. You may retain each layer's keys and values and reuse them on future calls.

Queries are usually transient: old queries are not needed to compute the next output. Each layer nevertheless needs its own historical K and V, because the new token passes through every layer. Caching only the final layer cannot reproduce the model's function.

This optimization exchanges memory for saved computation. It removes repeated historical projections and feed-forward work; attention still reads historical keys and values. Never describe cached decoding as constant-time with respect to context length.

## Extend the forward contract

`forward(ids,past=None)` keeps its existing return value. When `past` is provided, it contains one `(K,V)` pair per layer. Each tensor is `[S,Hkv,d]`, with the same cached length S across layers. `ids` is a nonempty chunk of **new** tokens. Return logits only for those new positions and a new cache containing S+T positions.

Do not modify the input cache tensors or token list. It is acceptable at this stage to concatenate tensors, even though that copies existing storage. Explicit copying makes the contract easy to test before you change the memory model in stage 06.

Absolute positions for the new queries and keys are S through S+T−1. New query row i may see key positions through S+i. The input chunk can contain multiple tokens: the grader tests multi-token append operations, not just one-token decoding.

Caching must preserve each layer's post-RoPE keys and unrotated values. Do not reapply RoPE to cached keys. Do not reset position to zero for each new chunk. Reject a combined context beyond `cfg.max_seq_len`.

## The first output is a prefill result

The final prompt logit row predicts the first continuation token. It does not require a separate one-token decode call first. For G outputs from P prompt tokens, cached generation processes P prompt positions followed by G−1 feedback positions. The last generated ID has not yet been fed through the model.

For P=5, G=3, the sequence is:

| Model call | Input positions | Result |
|---|---|---|
| Prefill | 0–4 | Sample output token 1 |
| Decode | 5 | Sample output token 2 |
| Decode | 6 | Sample output token 3; stop |

Seven positions have K/V, while the full prompt-plus-output history has eight IDs. This distinction later becomes the scheduler's computed frontier.

## Cached generation contract

Implement `generate_cached` with the same sampling and stopping behavior as `generate_naive`. On the first model call pass the prompt. On subsequent calls pass a one-token sequence containing the previous generated ID and the previous returned cache. Preserve per-request generator state, return generated IDs only, and make zero model calls for zero output.

The acceptance suite checks the **actual model input lengths** and attaches a hook to the first layer's Q projection. It expects one prompt-length call followed by one-row calls. Returning the right tokens while secretly recomputing the history fails the stage. You must use your stage 03 sampler, not a hardcoded greedy shortcut that fails stochastic generation.

## Prove that the cache is real state

Another test edits cached V values before appending. Your logits must change in the way the independent model predicts. This catches an implementation that accepts a `past` argument but ignores it or rebuilds state from stored IDs elsewhere.

Editing cached tensors is a test technique, not a normal client behavior. It makes the dependency observable. Similar tests later edit physical pages to prove that the block table actually participates in model execution.

## Debugging common failures

If full-prefill logits match but the first append fails, compare absolute RoPE positions. If one-token appends match but two-token chunks fail, inspect the offset causal mask. If one layer works and multiple layers fail, inspect the association between cache pairs and layers. If repeated calls corrupt earlier snapshots, inspect in-place mutation.

Numerical tests use the same checkpoint and float32 tolerances. Comparing two randomly initialized models is not a cache test. Disabling dropout or selecting eval mode is also part of holding the mathematical function fixed, though this tiny architecture has no dropout layer.

## Memory accounting

For L full-attention layers, N cached positions, Hkv KV heads, head width d, and b bytes per scalar, the cache uses

\[
2LNH_{kv}db\;\text{bytes}.
\]

The factor two is K plus V. The formula excludes weights, temporary attention scores, activations, allocator metadata, and quantization scales. For several requests, N is the sum of their cached positions. Do not multiply by batch size again after summing them.

With 32 layers, 8 KV heads, d=128, and bf16 at two bytes per scalar, each token needs 128 KiB of cache. An 8,192-token context needs 1 GiB before any other model memory. Small toy dimensions keep these allocations inexpensive while preserving the structure of the problem.

Caching becomes invalid when weights, adapter identity, or relevant positional settings change. This fact will determine prefix-cache identity later.

**Next:** [Stage 05 — Pack requests](05_batching.md).
