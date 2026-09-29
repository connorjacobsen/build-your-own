# 03 · A slow, correct generation loop

**Time:** 90–120 minutes. **Prerequisite:** logits and causal decoding. **Deliverable:** a baseline generator and a reproducible sampler. **Build:** `nucleus_probabilities`.

Autoregressive generation repeatedly predicts one token and appends it to the history. A model forward pass computes logits for every input position; continuation uses the last row. With a prompt of `P` tokens and a request for `G` new tokens, the simplest loop reruns lengths `P, P+1, ..., P+G−1`.

This baseline is deliberately wasteful. It is also a valuable oracle because it has no block allocator, cache offsets, or scheduler. Keep it working throughout the course. When an optimized path disagrees, first compare logits at the earliest differing position, before comparing decoded text.

```python
from nanovllm_course.sampling import generate_naive, sample
model = make_model()
prompt = ByteTokenizer().encode("Hello")

def greedy_from_scratch(model, prompt, count):
    tokens = list(prompt)
    with torch.inference_mode():
        for _ in range(count):
            logits, _ = model(tokens)
            tokens.append(int(logits[-1].argmax()))
    return tokens[len(prompt):]

answer = greedy_from_scratch(model, prompt, 8)
assert answer == generate_naive(model, prompt, 8)
print("Generated IDs:", answer)
```

## Greedy, temperature, top-k, and top-p

Greedy decoding chooses the largest logit. Temperature sampling divides logits by a positive scalar before softmax; lower values concentrate probability and higher values flatten it. Temperature zero is handled as a separate greedy branch, never a division by zero.

Top-k keeps exactly the `k` largest entries in this course implementation. Top-p, or nucleus sampling, sorts probabilities and keeps the shortest prefix whose cumulative probability reaches the threshold. The token that **crosses** the threshold must stay. Removing it creates a smaller-than-requested nucleus and may leave no token for a very small threshold.

Filters compose in a chosen order. We apply temperature, then top-k, then top-p on the renormalized remaining distribution. Different orders can produce different distributions, so document yours. Production samplers may add repetition penalties, token masks, minimum lengths, and structured-output constraints. Those belong at the logits/sampling boundary rather than inside the cache manager.

```python
logits = torch.tensor([4., 2., 1., 0.])
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, temperature in zip(axes, [0.3, 1.0, 2.0]):
    p = (logits / temperature).softmax(-1)
    ax.bar(range(4), p.numpy()); ax.set_ylim(0, 1)
    ax.set_title(f"temperature={temperature}"); ax.set_xlabel("Token")
axes[0].set_ylabel("Probability"); plt.tight_layout(); plt.show()

p = torch.tensor([.50, .30, .15, .05])
remove = p.cumsum(0) - p >= .6
assert remove.tolist() == [False, False, True, True]
print("The .30 token crosses .6 and is retained.")
```

## Random state is request state

Using one global random generator couples requests. If request B arrives earlier, it consumes random numbers that request A would otherwise have used. Request A's output changes even though its prompt, seed, and model are unchanged.

Create a generator for each request and advance it only when that request samples. This stabilizes the random-number stream under scheduling changes. It does not guarantee bitwise reproducibility across every device or backend: floating-point differences near ties or sampling boundaries can still change outputs.

```python
params = SamplingParams(temperature=.8, top_k=20, top_p=.9, seed=123)
a = generate_naive(model, prompt, 10, params)
b = generate_naive(model, prompt, 10, params)
assert a == b
print(a)
# Greedy and top-k=1 agree for an unambiguous maximum.
assert sample(logits, SamplingParams(temperature=1, top_k=1), torch.Generator()) == int(logits.argmax())
```

## Stopping semantics

A request can finish by output-length limit, by EOS, or by explicit cancellation. These are different events. Decide whether EOS is included in returned IDs; our engine includes it and reports `finish_reason="eos"`. The byte tokenizer hides special IDs when decoding text.

The core defaults to no EOS stop to make timing experiments produce a fixed number of output tokens. To enable stopping, pass `SamplingParams(eos_id=257)`. A zero-token request returns immediately without a model call. Empty text becomes a BOS-only prompt; a truly empty list of input IDs is rejected.

```python
first_token = generate_naive(model, prompt, 1)[0]
stopped = generate_naive(model, prompt, 10, SamplingParams(eos_id=first_token))
assert stopped == [first_token]
assert generate_naive(model, prompt, 0) == []
print("EOS and zero-length checks passed")
```

## Count work correctly

The dense loop processes `G*P + G*(G−1)/2` token positions through projections. That is a count of positions, not FLOPs. Attention adds a sequence-length-dependent cost. A cached loop will process `P+G−1` new positions but still attend to older keys at every decode step. KV caching removes repeated historical projections; it does not make historical attention free.

```python
from nanovllm_course.metrics import projection_token_work
for output_length in [1, 8, 32]:
    print(output_length, projection_token_work(64, output_length))
```

## Build and check

Implement `nucleus_probabilities(logits, top_p)` returning a probability vector in original token order. Use a threshold of `.6` for probabilities `[.5,.3,.15,.05]` and prove that exactly the first two survive. Check `p=1`, a tiny positive `p`, and normalization after filtering.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("sampling")
else:
    print("Nucleus sampling exercise skipped.")
```

**Failure lab:** deliberately remove the crossing token. Predict which boundary test catches the bug. Then try a global generator for two interleaved requests and explain why per-request seeds are a systems concern, not just a sampling convenience.

Upstream bridge: inspect `Sampler` in the [pinned V1 sampler](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/sample/sampler.py). Identify the separation between logits processing, probability conversion, and random selection.

Next: [04 — KV caching](04_kv_cache.ipynb).
