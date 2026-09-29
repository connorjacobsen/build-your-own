> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 03 — Generate and sample

**Build:** `src/toyvllm/sampling.py`  
**Prerequisite:** stage 02. **Estimated effort:** 2–3 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## Objective and contract

Turn a forward pass into a generator. Keep the baseline intentionally simple: rerun the full growing token history on each iteration. This implementation becomes your own slow correctness oracle when you add caching. Implement:

```text
sample(logits, params=SamplingParams(), generator=None) -> int
generate_naive(model, prompt, max_new_tokens, params=SamplingParams()) -> list[int]
```

`sample` receives one finite one-dimensional logits vector. It returns a Python integer and does not modify the input. The supplied immutable `SamplingParams` defines temperature, top-k, top-p, seed, and optional EOS ID; it already validates the parameter range. Sampling happens on CPU float32 in this course so a request-local CPU generator works consistently across devices.

`generate_naive` receives a nonempty prompt of valid IDs and a nonnegative integer output limit. Return **generated IDs only**, preserve the caller's prompt, and perform no model calls for zero output. Include EOS in the returned sequence and stop immediately when it matches `params.eos_id`. With EOS disabled (`None`), produce exactly the requested count. Invalid empty prompts and negative output lengths raise `ValueError`.

## Why generation is a loop

The model computes next-token logits at every input position. Continuation uses the last row. A sampled token becomes input for the next decision; it cannot affect the probability distribution from which it was just drawn.

If the prompt has P tokens and you want G outputs, the baseline forwards lengths P, P+1, ..., P+G−1. That is `G*P+G*(G−1)/2` projected token positions. It is a work count, not a FLOP count: attention has additional length-dependent cost. The loop has a simple enough state that it is useful for diagnosing later system failures.

Keep these concerns separate: the model computes logits, the sampler chooses an ID, and the generator owns the evolving history and stop decision. Neither the model's weights nor its attention code should change when a user requests a different temperature.

## Sampling rules, in order

At temperature zero, choose `argmax` directly; do not divide by zero. Ties use the first maximum, matching PyTorch's argmax. Otherwise apply the following contract in order:

1. Convert logits to an independent CPU float32 working vector and divide by positive temperature.
2. If `top_k>0`, keep the largest `min(top_k,vocab_size)` entries and mask the rest to negative infinity. `top_k=0` disables this filter. Acceptance fixtures avoid ambiguous cutoff ties.
3. If `top_p<1`, sort the remaining scores descending, normalize them, and retain the shortest prefix whose cumulative probability reaches top-p. The threshold-crossing token stays.
4. Normalize the surviving distribution and draw one token using `torch.multinomial` with the supplied generator. Map back to original vocabulary order when necessary.

The order is part of the specification. Top-p after top-k operates on the renormalized top-k distribution, not the original full-vocabulary probabilities. A mathematically reasonable but different filtering order does not satisfy this contract.

Consider probabilities `[.50,.30,.15,.05]` and p=.60. The first two tokens survive because .50 alone is insufficient and .80 crosses the threshold. After renormalizing the retained mass, their probabilities become .625 and .375. Keeping only the first token silently turns a stochastic request into a deterministic one.

A tiny positive threshold must still keep a token. `p=1` does not filter. `top_k=1` with an unambiguous maximum behaves greedily even with positive temperature. Do not apply softmax twice to probabilities when your API promises logits.

## Randomness belongs to the request

Create one CPU `torch.Generator`, seeded from `params.seed`, for the generation request and reuse it across emissions. Do not reseed for every token; that repeats the same random draw pattern. Do not share one global generator among independent requests; a scheduling change would then change which request consumes which random numbers.

This is a systems property as much as a mathematical one. In a later stage, A may generate before or after B depending on admission and token budgets. Holding A's prompt, model, parameters, and random stream fixed should preserve its decisions on this reference execution path. Small numerical differences across devices can still change outputs near a probability boundary; per-request RNG does not promise universal bitwise reproducibility.

## Stopping and ownership

EOS and length limits are separate stop reasons. The baseline returns only IDs; the later request object will also record why it finished. The byte decoder omits special IDs when presenting text, so the string output need not visibly contain EOS even when the ID list does.

Copy the prompt before appending. Modifying the caller's list can break prefix-cache identity, benchmarks, and repeated tests in ways that look like model nondeterminism. The grader deliberately repeats generation after perturbing global RNG state and checks that request results remain stable.

Use `torch.inference_mode()` or `no_grad()` in generation to avoid building training graphs. Keep the dense model itself differentiable, as required in stage 02.

## What the gate proves

The grader compares baseline outputs with fixed weights, tests EOS/zero output, checks top-p crossing behavior across seeds, and verifies input immutability and RNG independence. These tests establish semantics. They do not claim the baseline is efficient.

Once this stage passes, try a short byte prompt in a REPL. Gibberish is expected from random weights. The important observation is that the model repeatedly consumes its own decisions and stops according to a defined contract.

**Next:** [Stage 04 — Reuse the past](04_kv_cache.md).
