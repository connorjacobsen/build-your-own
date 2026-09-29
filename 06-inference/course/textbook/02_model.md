> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 02 — Build a causal transformer

**Build:** `src/toyvllm/model.py`  
**Prerequisite:** stage 01, basic matrix multiplication; the shape guide below introduces the tensor conventions.  
**Estimated effort:** 4–8 hours, split into attention, a decoder layer, and the full model.  
**Gate:** use the microstage number shown by `uv run course list`.

## Objective

Implement a real decoder-only transformer. The grader supplies fixed weights and checks your logits against an independent implementation. You build the layers and all forward computation; no pretrained library supplies the answer. The model is tiny enough to run on CPU and uses random weights by default. Plausible generated English is not a correctness criterion.

This stage deliberately fixes one small architecture so numerical results have a precise meaning. Later systems stages optimize execution of that same mathematical function. You remain free to choose helper methods and control flow, subject to the named module boundary used for checkpoint loading and execution instrumentation.

## Tensor vocabulary and required API

Let `T` be token count, `D` model width, `Hq` query heads, `Hkv` KV heads, and `d=D/Hq` head width. Hidden states use `[T,D]`; queries use `[T,Hq,d]`; keys and values use `[T,Hkv,d]`. All model inputs in this course represent individual unpadded sequences or explicitly packed rows, not an implicit leading batch dimension.

Implement these public entry points:

- `RMSNorm(dim, eps=1e-6)`: learned `weight` of shape `[dim]`, initialized to ones; forward preserves input shape.
- `rope(x, positions, base=10000.0)`: adjacent-pair rotation of `[T,H,d]` using absolute integer positions `[T]`.
- `attention(q,k,v,query_start=0)`: causal grouped-query attention, returning `[Tq,Hq,d]` for keys/values `[Tk,Hkv,d]`.
- `TinyLM(cfg)`: a `torch.nn.Module` storing `cfg`, model parameters, and a `device` property.
- `TinyLM.forward(ids, past=None)`: in this stage implement `past=None`; return `(logits, cache)`, where logits are `[T,vocab_size]` and cache is a list of `(K,V)` pairs, one per layer, each `[T,Hkv,d]`.

The supplied `TinyConfig` and `make_model` factory are setup plumbing. You need not rewrite them. `make_model` constructs your `TinyLM`, isolates initialization randomness, selects the device, and enters eval mode. Later stages add cached, packed, and paged paths to your model.

## Checkpoint and instrumentation contract

Use these parameter names and shapes. This allows the grader to inject identical weights instead of relying on framework initialization order.

| State-dict key | Shape |
|---|---|
| `embedding.weight` | `[vocab_size,D]` |
| `layers.i.attn_norm.weight` | `[D]` |
| `layers.i.q.weight` | `[Hq*d,D]` |
| `layers.i.k.weight`, `layers.i.v.weight` | `[Hkv*d,D]` |
| `layers.i.o.weight` | `[D,D]` |
| `layers.i.ffn_norm.weight` | `[D]` |
| `layers.i.gate.weight`, `layers.i.up.weight` | `[hidden_dim,D]` |
| `layers.i.down.weight` | `[D,hidden_dim]` |
| `norm.weight` | `[D]` |
| `lm_head.weight` | `[vocab_size,D]` |

Here `i` ranges from zero to `n_layers−1`. Use callable `nn.Linear` modules for `q/k/v/o/gate/up/down`, without biases. Keep embedding and output head untied. `layers` must be an indexable registered module sequence. Later tests attach hooks to these modules to count actual projected rows; do not route computation around their `forward` methods using direct access to weights.

## Derive the computation

RMSNorm divides by the root mean square and multiplies by a learned channel scale:

\[
\mathrm{RMSNorm}(x)=w\odot x\,(\operatorname{mean}(x^2)+\epsilon)^{-1/2}.
\]

It does not subtract a mean. Use float32 for the normalization statistic, then restore the activation dtype. Epsilon keeps an all-zero row finite.

For one query head, scaled dot-product attention is

\[
S=QK^T/\sqrt d,\quad P=\operatorname{softmax}(S+M),\quad O=PV.
\]

Softmax normalizes over keys. The mask is zero at allowed keys and negative infinity elsewhere. For local query row `i`, the absolute position is `query_start+i`; allowed keys satisfy `j <= query_start+i`. An output with the right shape can still be wrong if the softmax axis or absolute mask is wrong.

GQA maps query head `h` to KV head `h // (Hq/Hkv)`. For four query heads and two KV heads, the map is `[0,0,1,1]`. Stored K/V remain compact; attention may expand them for clarity.

For RoPE pair index `r`, the frequency is `10000^(-2r/d)`. At absolute position `p`, rotate the pair `(a,b)` by angle `p*frequency`:

\[
(a',b')=(a\cos\theta-b\sin\theta,\;a\sin\theta+b\cos\theta).
\]

Rotate Q and K, not V. Position zero must leave the vector unchanged, and rotation preserves its norm. This course uses adjacent pairs `(0,1),(2,3),...`; other model families may use another channel convention. Shape agreement alone does not establish checkpoint compatibility.

Each decoder layer computes normalized Q/K/V, applies RoPE and causal attention, concatenates head outputs, and adds an output-projection residual. It then applies a normalized SwiGLU branch:

\[
x'=x+W_o\operatorname{Attention}(Q,K,V),\qquad
z=\operatorname{RMSNorm}(x'),
\]
\[
x''=x'+W_{down}(\operatorname{SiLU}(W_{gate}z)\odot W_{up}z).
\]

After all layers, apply the final RMSNorm and `lm_head`. Return raw logits, not probabilities. Logits at input position `i` predict the token after `i`.

## Cache output in a dense stage

Although you will not consume `past` until stage 04, return each layer's **post-RoPE K** and unrotated V now. That forces you to identify exactly which tensors persist. Do not detach the entire forward path or decorate it with inference mode: the dense path must support gradients. Generation callers can disable gradients later.

The dense input domain is a nonempty one-dimensional sequence of valid IDs up to `cfg.max_seq_len`. Preserve inputs. The full engine later validates incoming IDs; you should also reject empty inputs and excessive context in the model with `ValueError` rather than producing malformed logits.

## Evidence and debugging

The grader checks individual operations, fixed-checkpoint logits/cache tensors, suffix perturbations, varied sequence lengths, and gradient flow. CPU float32 comparisons use tolerances rather than exact bitwise equality. Do not implement a special answer for one seed; checkpoint weights and inputs are test data.

A useful debugging sequence is one head, one layer, then GQA, then multiple layers. Compare intermediate shapes and norm preservation. If changing a future token alters an earlier logit, investigate causality before initialization. If the first token matches but later positions do not, investigate position rotations and masks.

**Reading connection:** [vLLM's pinned LlamaAttention and decoder layer](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/model_executor/models/llama.py). Our model is architecture-inspired and is not a Llama checkpoint loader.

**Next:** [Stage 03 — Generation](03_generation.md).
