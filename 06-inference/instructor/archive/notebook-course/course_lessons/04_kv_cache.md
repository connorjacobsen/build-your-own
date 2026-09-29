# 04 · Stop recomputing the past: a contiguous KV cache

**Time:** 2–3 hours. **Prerequisite:** the baseline generation loop. **Deliverable:** cached generation equal to dense generation. **Build:** `generate_with_cache`.

Causal attention creates a useful invariant: the hidden state at position `i` depends only on tokens through `i`. Appending a later token cannot change earlier states. With fixed weights, the earlier keys and values can therefore be reused.

Cache K and V separately at every layer. Caching only the final layer is insufficient: the new token passes through every layer and needs each layer's historical keys and values. Queries are usually transient because old queries are not needed to compute a new token's output.

A **prefill** computes the prompt's states. A **decode** step computes the next input position using a cache. The first generated token comes from the final prompt logits, so generating `G` output tokens needs one prompt prefill and `G−1` subsequent decode forwards. The last emitted token has not been fed back into the model yet.

## Make the cache contract explicit

Our model returns `(logits, past)`, where `past[layer]` is `(K,V)` with each tensor shaped `[cached_tokens,Hkv,d]`. To append a chunk:

1. Read `start` from the existing cache length.
2. Project only the new hidden states into Q/K/V.
3. Apply RoPE to Q and K using positions `start ... start+chunk−1`.
4. Concatenate old and new K/V.
5. Apply an offset causal mask for the new queries.
6. Return logits only for the new positions and the enlarged cache.

The reference uses `torch.cat` to make this contract visible. Repeated concatenation copies the old cache on every step; it is not the final memory design. Correctness comes before removing the copies.

```python
model = make_model()
ids = [256, 10, 20, 30, 40, 50, 60]
with torch.inference_mode():
    full, _ = model(ids)
    first, cache = model(ids[:3])
    second, cache = model(ids[3:5], cache)
    third, cache = model(ids[5:], cache)
chunked = torch.cat([first, second, third])
torch.testing.assert_close(chunked, full, rtol=2e-5, atol=2e-6)
assert cache[0][0].shape[0] == len(ids)
print("Maximum logit error:", (chunked - full).abs().max().item())
```

The two-token second chunk is intentional. A test that only decodes one token at a time can miss bugs in within-chunk causal masking. Also try chunk lengths that do not divide the total sequence length.

## Implement cached generation

The only difference from naive generation is what becomes the next model input. The first call receives the prompt; subsequent calls receive a one-token list. Sampling sees the same final-position logits up to floating-point tolerances.

```python
from nanovllm_course.sampling import sample, generate_naive, generate_cached

def cached_loop(model, prompt, count):
    past, inputs, output = None, list(prompt), []
    with torch.inference_mode():
        for _ in range(count):
            logits, past = model(inputs, past)
            token = sample(logits[-1])
            output.append(token)
            inputs = [token]
    return output

prompt = ByteTokenizer().encode("memory")
assert cached_loop(model, prompt, 12) == generate_naive(model, prompt, 12)
assert cached_loop(model, prompt, 12) == generate_cached(model, prompt, 12)
```

## Work saved versus memory spent

For a fixed prompt, increasing output length makes naive recomputation grow rapidly. The cache trades persistent memory for saved computation. Plot the projection-token counts rather than claiming a speedup from the formula. Real time also includes Python, allocation, attention, threading, and device overhead.

```python
from nanovllm_course.metrics import projection_token_work
lengths = np.arange(1, 65)
naive_work = [projection_token_work(64, int(g))["naive"] for g in lengths]
cached_work = [projection_token_work(64, int(g))["cached"] for g in lengths]
plt.plot(lengths, naive_work, label="Dense recomputation")
plt.plot(lengths, cached_work, label="Cached new positions")
plt.xlabel("Output tokens"); plt.ylabel("Positions projected")
plt.legend(); plt.show()

with torch.inference_mode():
    _, cache = model(prompt)
actual = sum(k.numel()*k.element_size() + v.numel()*v.element_size() for k,v in cache)
expected = model.cfg.kv_bytes_per_token() * len(prompt)
assert actual == expected
print("Actual KV bytes:", actual)
```

## Four failure modes to learn to recognize

**Position reset:** every new token rotates at position zero. Shapes and memory use look correct; logits drift immediately after prefill.

**Layer mismatch:** a key tensor from one layer is paired with values from another. Shapes can still agree. Keep the layer dimension explicit.

**Future exposure:** a multi-token cached chunk attends to all new keys without masking. Single-token decode tests pass; chunked prefill fails.

**Cache invalidation:** weights, adapter, positional convention, or relevant model settings change while old entries remain. Prompt identity alone cannot make those cached tensors valid.

A useful debugging order is: compare a single layer's Q/K/V, verify absolute positions, inspect the visible-key mask, and only then compare final logits. Compare identical weights and `eval()` mode. Do not compare two independently initialized models and blame the cache.

## Build and check

Implement `generate_with_cache(model,prompt,max_new_tokens)` for greedy generation in the exercise file. Do not call the reference generator. The checker verifies outputs and records the model's input lengths: after one prompt call, each call must have length one. This catches a fake cache implementation that happens to return the right tokens while recomputing the full history.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("decode")
else:
    print("Cached generation exercise skipped.")
```

**Stretch:** replace concatenation with a preallocated per-request `[max_tokens,Hkv,d]` buffer. Track logical length separately from capacity. Explain why this removes copying but can waste memory when requests have different lengths or finish early. That leads directly to paging.

**Checkpoint:** after a prompt of length 7 produces 4 output tokens, how many positions have been computed? **Answer:** 10: seven prompt positions plus three feedback tokens. The history contains 11 IDs, but the final output ID has no cached state yet.

Next: [05 — Batching and packed execution](05_batching.ipynb).
