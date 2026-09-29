# 01 · Tokens, tensor shapes, and causal attention

**Time:** 90–120 minutes. **Prerequisite:** lesson 00. **Deliverable:** a correct causal attention function and a KV memory estimator. **Build:** `causal_attention` and `kv_bytes` in `exercises/implementation.py`.

A language model consumes integer token IDs. A tokenizer is a mapping between text and sequences of IDs; it does not assign semantic vectors. The embedding table does that later. Production tokenizers usually merge common byte or character sequences. Our tokenizer maps each UTF-8 byte to an ID from 0 through 255 and reserves 256 for beginning-of-sequence (BOS), 257 for end-of-sequence (EOS).

Byte tokenization is intentionally simple and reversible. Python's `len(text)` counts Unicode code points; it need not equal the number of UTF-8 bytes or model tokens. This distinction affects context limits and benchmark denominators.

```python
tok = ByteTokenizer()
for text in ["cat", "café", "🦙"]:
    ids = tok.encode(text)
    print(repr(text), "characters:", len(text), "token IDs:", ids)
    assert tok.decode(ids) == text
assert tok.encode("") == [tok.bos_id]
```

## Learn a small shape vocabulary

Let `T` be token count, `D` model width, `Hq` query heads, `Hkv` KV heads, and `d=D/Hq` head width. A single sequence's hidden state is `[T,D]`. Its queries are `[T,Hq,d]`. Keys and values are `[T,Hkv,d]`. The weight matrix of `torch.nn.Linear(D,M)` has shape `[M,D]`; applying it to `[T,D]` produces `[T,M]`.

`view` changes the interpretation of contiguous storage. `transpose` changes strides and often creates a noncontiguous view. `reshape` can make a copy when needed. These are not interchangeable performance guarantees. Always distinguish **logical shape** from **physical layout**.

```python
T, D, Hq, Hkv = 5, 32, 4, 2
x = torch.randn(T, D)
projection = torch.nn.Linear(D, Hq * (D // Hq), bias=False)
q = projection(x).view(T, Hq, D // Hq)
print("hidden:", x.shape, "queries:", q.shape)
print("query strides:", q.stride(), "head-major strides:", q.transpose(0, 1).stride())
assert q.shape == (5, 4, 8)
```

## Derive attention before optimizing it

For one head, `Q` has shape `[Tq,d]` and `K,V` have shape `[Tk,d]`. Attention is

\[
S = QK^T / \sqrt{d},\qquad P = \operatorname{softmax}(S + M),\qquad O = PV.
\]

The scale controls score growth as head width increases. Softmax turns each query's scores into a distribution over keys. A mask places negative infinity in forbidden positions, making their probability zero. An output row is a weighted sum of value rows.

In a decoder, position `i` may attend only to positions `j <= i`. With a cached prefix of length `s`, local query row `i` represents **absolute position `s+i`**. Its valid keys satisfy `j <= s+i`. Writing a triangular mask only over the new chunk silently discards access to the prefix. Using no mask for a multi-token chunk silently exposes future tokens.

```python
def attention_from_scratch(q, k, v, query_start=0):
    # Equal query/KV head counts in this first implementation.
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(q.shape[-1])
    absolute_queries = query_start + torch.arange(q.shape[0])
    keys = torch.arange(k.shape[0])
    allowed = keys[None, :] <= absolute_queries[:, None]
    probabilities = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
    return torch.einsum("hts,shd->thd", probabilities, v), probabilities

q = torch.randn(3, 2, 8)
k, v = torch.randn(7, 2, 8), torch.randn(7, 2, 8)
out, probabilities = attention_from_scratch(q, k, v, query_start=4)
assert out.shape == (3, 2, 8)
assert probabilities[0, 0, 5:].sum() == 0
assert torch.allclose(probabilities.sum(-1), torch.ones(2, 3))
plt.imshow(probabilities[0].detach(), aspect="auto", cmap="Blues")
plt.xticks(range(7)); plt.yticks(range(3), ["position 4", "position 5", "position 6"])
plt.xlabel("Key position"); plt.title("One attention head: the offset causal mask")
plt.colorbar(label="Probability"); plt.show()
```

Predict which columns would become visible if `query_start` changed to 0. Then try it. A visual mask is an excellent debugging aid, but numerical assertions should remain the final check.

## Multi-head versus grouped-query attention

Multi-head attention gives each query head its own key and value head. Grouped-query attention (GQA) shares one KV head across several query heads. With `Hq=4` and `Hkv=2`, heads 0 and 1 use KV head 0, and heads 2 and 3 use KV head 1 in our convention. The reference uses `repeat_interleave` to make this mapping explicit. A fast kernel can reuse the shared head directly without materializing duplicates.

GQA reduces persistent KV memory. It does not reduce the number of query heads or eliminate attention over the historical sequence. For full attention layers, KV storage for `N` cached tokens is

\[
\text{bytes}=2 \times L \times N \times H_{kv} \times d \times b.
\]

The factor 2 is K plus V; `L` is layer count; `b` is bytes per scalar. This formula omits weights, activations, temporary attention scores, allocator overhead, and any scale tensors used in quantized caches. Batch size enters through the sum of cached tokens over all sequences, not a second hidden factor.

```python
cfg = TinyConfig()
print("Toy cache bytes per token:", cfg.kv_bytes_per_token())
# A hypothetical 32-layer, 8-KV-head, 128-head-width, bf16 model:
bytes_per_token = 2 * 32 * 8 * 128 * 2
print("Hypothetical KiB/token:", bytes_per_token / 1024)
print("8192-token sequence GiB:", bytes_per_token * 8192 / 2**30)
assert bytes_per_token * 8192 == 2**30
```

## Build and check

Implement `causal_attention(q,k,v,query_start)` for equal head counts first, then support GQA using `repeat_interleave`. Inputs use `[T,H,d]`. The checker compares with an independent per-head implementation and changes masked future values to test causality. Implement `kv_bytes(layers,tokens,kv_heads,head_dim,bytes_per_element)` as a separate unit-aware function.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("attention")
    check("memory")
else:
    print("Build exercises skipped: implement causal_attention and kv_bytes, then enable RUN_EXERCISES.")
```

**Debugging prompt:** you get the expected output shape but future values change earlier outputs. Which invariant failed? Shapes prove dimensional compatibility; they do not prove causal correctness. Test values, not just dimensions.

<details><summary>Design hints and completion criteria</summary>

Create query and key position vectors on `q.device`. Compare them by broadcasting. Add the head axis to the mask. Softmax over the key dimension, never the head dimension. Accumulate softmax in float32 for the reference implementation. Finish when both the offset case and the future-value perturbation case pass.
</details>

Next: [02 — A tiny decoder](02_tiny_transformer.ipynb).
