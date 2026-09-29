> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 05 — Pack multiple requests

**Build:** `TinyLM.forward_batch` in `src/toyvllm/model.py`.  
**Prerequisite:** stage 04. **Estimated effort:** 2–4 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## What batching is buying

A serving model often receives several sequences at once. Linear projections operate independently on rows, so you can combine useful rows from several requests into larger matrix operations. That can amortize dispatch overhead and improve hardware utilization. Attention has a different requirement: each request must see only its own context.

Distinguish three concepts. Sequential execution finishes one sequence before starting another. Static batching fixes a cohort of requests for a period of execution. Continuous batching changes membership between execution steps. Packing is the representation of useful token rows and boundaries; it is not itself an admission policy. This stage implements packing. Stage 07 implements changing membership.

## Contract

```text
forward_batch(sequences, pasts=None) -> (list_of_logits, list_of_caches)
```

`sequences` is a list of nonempty valid token sequences. If omitted, `pasts` means no cached history for every request. Otherwise it is a list of matching length, containing one cache or `None` per request. Each cache has the stage 04 format; different requests can have different cached lengths. An empty batch returns `([],[])`.

Outputs preserve request order. Each logits tensor has `[new_tokens_for_request,vocab_size]`. Each cache has one pair per layer and includes that request's old plus new positions. The output for request A must match an independent `forward(A,past_A)` under the same weights. Input caches remain unmodified.

**Work requirement:** at each layer, every `q/k/v/o/gate/up/down` projection receives all new rows in **one** `[sum(new_lengths),features]` call. Calling `forward` once per request is numerically plausible but does not pass this stage. Attention can still loop over requests; no fused GPU kernel is required.

## Three coordinate systems begin to appear

Suppose A has three new IDs and B has one. Their packed rows are 0–2 and 3. If A has four cached positions and B has twenty, the model positions are `[4,5,6,20]`, not `[0,1,2,3]` and not `[4,5,6,7]`.

Cumulative sequence lengths `[0,3,4]` define row boundaries. Absolute positions define RoPE and causal visibility. Neither tells you a physical cache address yet; stage 06 introduces that third coordinate.

For packed lengths `[3,1,6]`, ten useful rows replace a padded rectangle with eighteen rows. This is a statement about useful row count, not a guaranteed wall-clock speedup. The target workload and backend determine whether packing overhead is worthwhile.

## Separate positionwise work from context work

Embedding lookup, linear projections, RMSNorm over the channel axis, and the feed-forward branch can operate on packed rows. They do not mix token positions. Attention mixes positions and must respect sequence boundaries.

A single causal triangle over the concatenated stream is wrong. It would let B attend to all of A because A happens to precede B in memory. A physically adjacent row is not necessarily a logically previous token. For each request, combine its new projected K/V with its own past and apply its own absolute offset mask.

The residual stream must preserve the same row order when you reassemble per-request attention outputs. A correct attention result placed back into another request's rows is just as wrong as a bad mask. Split final logits and caches at the original sequence boundaries.

## Architecture choices you own

You may factor a decoder-layer helper that handles a list of contexts, or perform packing in `TinyLM` and keep per-layer helpers smaller. You may concatenate caches for clarity. You may construct position vectors using Python lists or tensor operations. The grader cares about results, isolation, and the declared projection boundary, not helper names or queue types.

Avoid duplicating the mathematical layer in a way that drifts from your dense implementation. Refactoring shared projection and residual work is often useful here. However, do not turn stage 04 back into a path that reprojects old cached rows just to share code with the batch path. The cumulative gate will catch that regression.

## Acceptance evidence

The grader uses unequal sequence lengths, mixed cached lengths, a batch containing both fresh and cached requests, and empty-batch behavior. It compares each result to independent execution. Projection hooks count calls and row counts for all seven linear modules in every layer.

A numerical mismatch can come from boundary bookkeeping even when individual attention calls are correct. Print a small table with request ID, packed start/end, cached length, and absolute positions. If only later requests fail, suspect an offset reset or incorrect split boundary. If all requests fail identically, revisit shared layer math.

## Reflection

Why does batching one-token decode requests potentially improve weight reuse? Each request needs the same model weights, and a larger matrix operation can process more rows per weight load. Why does that not make KV memory sharing automatic? The historical K/V belong to each request's context; they are distinct unless a later prefix-identity rule permits reuse.

Why is one Python thread per request not equivalent to this stage? Threads do not construct one packed tensor, establish attention boundaries, or coordinate device launches into one model invocation. Frontend concurrency and model batching address different parts of the system.

**Upstream reading:** [V1 model input preparation](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/worker/gpu_model_runner.py#L923). Follow the relationship between flattened inputs, positions, and attention metadata rather than reading the entire runner first.

**Next:** [Stage 06 — Physical pages](06_paging.md).
