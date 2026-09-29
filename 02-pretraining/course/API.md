# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `TinyConfig`

See the stage-specific architecture and method contract.

### `TinyConfig.__post_init__(self)`

Implement as specified by the owning stage.

### `TinyConfig.head_dim(self)`

Implement as specified by the owning stage.

### `TinyConfig.kv_bytes_per_token(self, bytes_per_element: int=4)`

Implement as specified by the owning stage.

## `RMSNorm`

See the stage-specific architecture and method contract.

### `RMSNorm.__init__(self, dim, eps=1e-06)`

Implement as specified by the owning stage.

### `RMSNorm.forward(self, x)`

Implement as specified by the owning stage.

## `rope`

Rotate adjacent pairs. x: [T,H,D], positions: [T] absolute positions.

Signature: `rope(x, positions, base=10000.0)`

## `attention`

Causal GQA with an explicit offset mask; handles cached multi-token chunks.

Signature: `attention(q, k, v, query_start=0)`

## `DecoderLayer`

See the stage-specific architecture and method contract.

### `DecoderLayer.__init__(self, cfg)`

Implement as specified by the owning stage.

### `DecoderLayer.project(self, x, positions)`

Implement as specified by the owning stage.

### `DecoderLayer.finish(self, x, attended)`

Implement as specified by the owning stage.

## `TinyLM`

See the stage-specific architecture and method contract.

### `TinyLM.__init__(self, cfg=TinyConfig())`

Implement as specified by the owning stage.

### `TinyLM.device(self)`

Implement as specified by the owning stage.

### `TinyLM._ids(self, ids)`

Implement as specified by the owning stage.

### `TinyLM.forward(self, ids, past=None)`

Return [new_tokens,vocab] logits and per-layer contiguous (K,V).

past=None is the dense baseline. Cache is returned, never mutated.
Calling this path with gradients enabled also supports the tiny training lab.

## `encode_bytes`

UTF-8 bytes with BOS=256 prepended and EOS=257 appended; return list[int].

Signature: `encode_bytes(text)`

## `next_token_batch`

From one 1D token stream, return long X,Y tensors [len(starts),context].
X starts at each specified offset; Y is shifted one position. Reject empty starts,
context<1, negative starts and windows extending past the available next token.

Signature: `next_token_batch(tokens, starts, context)`

## `token_loss`

Mean next-token cross entropy for logits [...,V] and same-prefix-shape targets.
Compute stable logsumexp minus selected logits; do not call F.cross_entropy.

Signature: `token_loss(logits, targets)`

## `adamw_step`

In-place AdamW over matching lists of tensors. state is initially {}, then contains
step (integer), m and v (tensor lists). Bias-correct both moments. Apply decoupled
decay p *= (1-lr*weight_decay). Do not mutate gradient tensors. No torch optimizer.
All parameters receive a dense gradient in this educational version.

Signature: `adamw_step(parameters, grads, state, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)`

## `cosine_lr`

0<=warmup<total; step>=0. With warmup>0, linear step/warmup * peak before warmup.
Cosine decay from peak at warmup to floor at total; clamp later steps to floor.
Reject invalid steps, bounds, negative floor or peak<floor.

Signature: `cosine_lr(step, warmup, total, peak, floor=0.0)`

## `clip_grad`

Clip the global L2 norm of existing gradients in place; return the preclip norm as float.
Ignore parameters whose grad is None. Reject max_norm<=0 and nonfinite gradient norm.

Signature: `clip_grad(parameters, max_norm)`

## `train_step`

One update on nonempty sequences, each at least two token IDs long.
Weight all target tokens equally, not all sequences equally. Clear old gradients,
set training mode, backpropagate the aggregate loss, optimizer.step(), return float loss.

Signature: `train_step(model, optimizer, sequences)`

## `save_checkpoint`

Save dict model, optimizer, step, rng (CPU torch RNG state) using torch.save.
This resume format is distinct from the portable inference export. Parent directory exists.

Signature: `save_checkpoint(path, model, optimizer, step)`

## `load_checkpoint`

Restore model, optimizer, CPU RNG from our trusted local checkpoint; return saved step.
Use torch.load(weights_only=True,map_location='cpu') and strict parameter matching.

Signature: `load_checkpoint(path, model, optimizer)`

## `train_run`

Initialize TinyLM under fork_rng, AdamW(lr=lr,weight_decay=.01), then train_step
on all supplied sequences for each update. Return model, list of pre-update losses.
Preserve the caller's CPU torch RNG state. Default config has dim=16, n_layers=1,
n_heads=2,n_kv_heads=1,hidden_dim=32; vocab and max length keep TinyConfig defaults.

Signature: `train_run(sequences, steps=30, seed=0, cfg=None, lr=0.01)`

## `export_model`

Write portable torch payload: format='toylm-v1', config=vars(model.cfg),
tokenizer={'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258},
state_dict=detached CPU cloned tensors, provenance=caller dict of JSON-safe metadata.
Require cfg.vocab_size=258. Return SHA256 of the exact file bytes. No optimizer in export.

Signature: `export_model(path, model, provenance)`
