# 02 · Build a tiny decoder-only transformer

**Time:** 2–3 hours. **Prerequisite:** causal attention and tensor shapes. **Deliverable:** a decoder whose logits have a clear causal meaning. **Build:** `apply_rope` and your own `TinyLM` variant.

The engine needs a real model so storage and scheduling changes can be checked against actual computation. Our decoder is inspired by common Llama-style components but is **not checkpoint-compatible with Llama**. We use byte tokens, small dimensions, separate untied embeddings, and an adjacent-pair RoPE convention. Loading an arbitrary pretrained checkpoint into these layers would not be correct.

A decoder layer consists of a normalized attention branch and a normalized feed-forward branch, each added to a residual stream:

\[
x' = x + W_o\operatorname{Attention}(W_q\operatorname{RMSNorm}(x), W_k\operatorname{RMSNorm}(x), W_v\operatorname{RMSNorm}(x)),
\]
\[
x'' = x' + W_d\big(\operatorname{SiLU}(W_g z)\odot W_u z\big),\quad z=\operatorname{RMSNorm}(x').
\]

The feed-forward expression is SwiGLU. It operates independently at each position; attention is what mixes information between positions. `Wq`, `Wk`, and `Wv` project the normalized hidden state, not the integer token IDs. An embedding lookup creates the initial hidden state.

## RMSNorm and residuals

RMSNorm rescales a vector by the reciprocal square root of its mean square, then applies a learned per-channel weight. Unlike LayerNorm, it does not subtract the mean. Epsilon prevents division by zero. Computing the statistic in float32 is useful when activations use lower precision.

```python
from nanovllm_course.model import RMSNorm, rope
x = torch.tensor([[1., 2., 3., 4.], [0., 0., 0., 0.]])
norm = RMSNorm(4)
y = norm(x)
assert torch.isfinite(y).all()
print("Normalized:", y.detach())
print("Nonzero row RMS:", y[0].square().mean().sqrt().item())
```

A residual connection does not mean the branch does nothing: it adds an update to a persistent stream. Pre-normalization means normalization happens before the branch projection. The final model also normalizes its hidden state before projecting to vocabulary logits.

## Rotary position embeddings, from two numbers to a head

Token order cannot be recovered from a bag of embeddings. RoPE rotates each adjacent pair of query/key channels by an angle determined by position. For one pair `(a,b)` and angle `θ`, the rotation is `(a cos θ − b sin θ, a sin θ + b cos θ)`. Multiple pairs use different frequencies.

Rotation preserves the norm of each pair. More importantly, query-key dot products acquire information about relative positions. Values are not rotated in this model. Cache keys **after** applying RoPE, and use the same absolute positions when appending. Resetting every decode token to position zero makes the cache numerically inconsistent with a dense forward pass.

```python
x = torch.randn(6, 2, 8)
positions = torch.arange(10, 16)
rotated = rope(x, positions)
torch.testing.assert_close(x.square().sum(-1), rotated.square().sum(-1))
torch.testing.assert_close(rope(x[:1], torch.tensor([0])), x[:1])
print("RoPE norm and position-zero checks passed")
```

## Read one forward pass

The package exposes `TinyLM.forward(ids, past=None)`. Without `past`, it computes all positions. With `past`, it appends a chunk; we will use that in lesson 04. The returned logits at input position `i` predict the token **after** that position. There is no softmax in the model's return value. Keeping logits lets the sampler choose temperature and filters separately.

```python
import inspect
from nanovllm_course.model import DecoderLayer
model = make_model()
tokens = ByteTokenizer().encode("cache")
with torch.inference_mode():
    logits, cache = model(tokens)
print("tokens:", len(tokens), "logits:", tuple(logits.shape))
print("layer-zero K/V:", tuple(cache[0][0].shape), tuple(cache[0][1].shape))
print("parameters:", sum(p.numel() for p in model.parameters()))
print(inspect.getsource(DecoderLayer.project))
print(inspect.getsource(DecoderLayer.finish))
assert logits.shape == (len(tokens), model.cfg.vocab_size)
```

Follow the source in this order: `TinyLM.forward`, `DecoderLayer.project`, `attention`, `DecoderLayer.finish`. Write the shape next to each variable. The architecture is small enough that you should be able to explain every learned matrix.

## Prove causality at model level

The attention mask may be correct while a later operation accidentally mixes positions. Perturb a suffix and verify that earlier logits are unchanged. This test spans embeddings, all layers, and the output projection.

```python
with torch.inference_mode():
    before, _ = model([256, 10, 20, 30, 40])
    after, _ = model([256, 10, 20, 99, 88])
torch.testing.assert_close(before[:3], after[:3])
assert not torch.allclose(before[3:], after[3:])
```

## Optional miniature training experiment

Training and serving solve different problems. Serving assumes weights are fixed; training changes them to improve a loss. Here is a complete tiny training loop on an original, repetitive local sequence. It teaches next-token label shifting and produces a model that overfits a pattern. It is not a pretrained language model or evidence of broad language ability.

This experiment is short enough to execute in the core CPU notebook. Cross entropy expects raw logits and target IDs. The target for `sequence[:-1]` is `sequence[1:]`. Shift only once.

```python
import torch.nn.functional as F
training_model = make_model(cfg=TinyConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=64))
training_model.train()
sequence = ByteTokenizer().encode("red blue red blue red blue red blue ")
optimizer = torch.optim.AdamW(training_model.parameters(), lr=0.008)
losses = []
for _ in range(60):
    prediction, _ = training_model(sequence[:-1])
    target = torch.tensor(sequence[1:])
    loss = F.cross_entropy(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
training_model.eval()
assert losses[-1] < losses[0]
plt.plot(losses); plt.xlabel("Training step"); plt.ylabel("Cross entropy")
plt.title("Overfitting one short pattern"); plt.show()
```

Do not reuse any KV cache across a weight update. Even with identical tokens, keys and values from the old weights are invalid. The same principle explains why production prefix keys must distinguish model/adaptor identity.

## Build and check

Implement `apply_rope(x, positions)` in the exercise file. The checker tests norm preservation, position zero, and a nonzero-position reference. Then recreate the decoder in a scratch module using the printed layer equations. Start with one head and one layer; add GQA only after matching attention outputs.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("rope")
else:
    print("RoPE exercise skipped; reference decoder and training experiment completed.")
```

**Questions:** Why must the head width be even here? Why does the KV memory formula use `Hkv`, while the output projection uses `Hq*d`? Which tensors change if the model weights change?

<details><summary>Answers</summary>

This RoPE implementation rotates channel pairs, so each head needs an even channel count. K/V are stored once per KV head, but attention produces one output per query head before concatenation. A weight update can change all hidden states, projected K/V, and logits; token IDs and position indices alone are not cached model state.
</details>

Upstream bridge: compare our layer with [vLLM's pinned Llama model](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/model_executor/models/llama.py). Find `LlamaAttention` and `LlamaDecoderLayer`. Notice tensor-parallel projections and a backend attention abstraction where our Python functions sit.

Next: [03 — Generation and sampling](03_generation_and_sampling.ipynb).
