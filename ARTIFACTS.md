# Shared artifact contracts

These formats keep the courses connected while preserving their independent environments. All examples are trusted local artifacts; do not load arbitrary third-party pickle payloads. Torch readers use `weights_only=True`. Record byte digests rather than relying on filenames.

## Dataset: toydata-v1

A directory contains `records.jsonl` and `manifest.json`. JSONL uses UTF-8, sorted keys, compact separators and one final newline per record. The manifest stores `format`, relative `file`, `sha256` and `records` count. Prepared records preserve `id` and original `text`, and add `split` (`train` or `validation`) and `content_sha256`. Train only on the training split. Reserved final evaluation cases must be separately identified; never repeatedly tune to them.

Course 03 produces this format. Course 02 consumes each document as `[BOS] + UTF-8 bytes + [EOS]`. The simple trainer takes separate sequences; packing metadata is a separate exercise and must be accompanied by an isolation mask before it is used by a transformer.

## Model: toylm-v1

A torch file contains `format='toylm-v1'`, `config`, `tokenizer`, `state_dict`, and `provenance`. Tokenizer metadata is exactly `{'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258}`. Byte IDs are 0–255. Training documents include EOS; ordinary generation prompts do not append EOS.

Configuration fields: vocab_size, dim, n_layers, n_heads, n_kv_heads, hidden_dim, max_seq_len. The architecture is the same pre-normalized RMSNorm / adjacent-pair RoPE / grouped-query attention / SwiGLU decoder in courses 02 and 06. Embedding and output weights are untied. There are no linear biases. Checkpoint keys:

- `embedding.weight`: [vocab,dim]
- `layers.i.attn_norm.weight`, `layers.i.ffn_norm.weight`: [dim]
- `layers.i.q.weight`: [n_heads*head_dim,dim]
- `layers.i.k.weight`, `layers.i.v.weight`: [n_kv_heads*head_dim,dim]
- `layers.i.o.weight`: [dim,dim]
- `layers.i.gate.weight`, `layers.i.up.weight`: [hidden_dim,dim]
- `layers.i.down.weight`: [dim,hidden_dim]
- `norm.weight`: [dim]; `lm_head.weight`: [vocab,dim]

Course 02 exports detached CPU tensors. Course 06 stage 30 loads them strictly and verifies metadata. Equality of dense logits is the first handoff test; only then compare cached/paged generation. This format is not an arbitrary pretrained-model loader.

## Adapter: toyadapter-v1

Course 04 stores `format`, `base_sha256`, `state` and `config`. State contains only qualified `A`/`B` tensors from LoRALinear modules. Per-module configuration contains `rank` and `alpha`. Base identity, target module paths, dimensions and scale must match before any restore changes a model. Merging adapters into ordinary linears yields a model that can be exported again as toylm-v1.

The fine-tuning API expects batched logits. The provided integration driver wraps the single-sequence training model by stacking per-sequence forward calls. This is an interface adapter, not a packed-training optimization. Right-padding does not influence preceding causal logits; padded targets are masked. Do not claim padding is computationally free.

## Evaluation: toyeval-v1

Course 05 returns `format`, `model_sha256`, `dataset_sha256`, `metrics={accuracy,n}` and `cases=[{id,prediction,score}]`. IDs are unique, n is the case count and accuracy is the mean score. The dataset digest identifies exact evaluation records, not the training corpus. Canonical JSON gives an evaluation file digest.

The score is a specified normalized exact-match metric; it is not a general semantic evaluator. More sophisticated metrics should receive a new explicit protocol version and retain per-case evidence. Course 08 verifies identity and aggregate consistency, not scientific validity or evaluator honesty.

## Release: toyrelease-v1 and local pointers

A release manifest contains selected relative paths, byte sizes, hashes and copied metadata. A content-addressed object store holds exact model/report bytes. An active pointer is `{'current': model_digest, 'previous': prior_digest_or_null}`. Compare-and-swap promotion prevents stale concurrent decisions. The local registry is a teaching substitute for a deployment control plane, not a live server deployment.

A release record returned by the integration step includes model and report digests plus the pointer state. Preserve it beside the experiment journal. Model bytes, evidence and policy decisions are distinct artifacts.
