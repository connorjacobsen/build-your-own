# Train a tiny language model

Build a byte-level causal transformer, understand its optimizer, resume training exactly, and export the same architecture used by the inference course. Sixteen stages separate mathematical components from training workflow. Use PyTorch autograd here; course 01 explains what it does.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/02-pretraining
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Define the token boundary](course/chapters/01_encode-bytes.md) | `encode_bytes` |
| 2 | [Shift inputs and targets](course/chapters/02_next-token-batch.md) | `next_token_batch` |
| 3 | [Compute stable token loss](course/chapters/03_token-loss.md) | `token_loss` |
| 4 | [Normalize residual activations](course/chapters/04_rmsnorm.md) | `RMSNorm` |
| 5 | [Encode relative position by rotation](course/chapters/05_rope.md) | `rope` |
| 6 | [Implement causal grouped-query attention](course/chapters/06_attention.md) | `attention` |
| 7 | [Build one decoder layer](course/chapters/07_decoderlayer.md) | `DecoderLayer` |
| 8 | [Assemble a trainable decoder](course/chapters/08_tinylm.md) | `TinyLM` |
| 9 | [Implement AdamW](course/chapters/09_adamw-step.md) | `adamw_step` |
| 10 | [Schedule the learning rate](course/chapters/10_cosine-lr.md) | `cosine_lr` |
| 11 | [Clip a global gradient norm](course/chapters/11_clip-grad.md) | `clip_grad` |
| 12 | [Perform one token-weighted update](course/chapters/12_train-step.md) | `train_step` |
| 13 | [Save resumable state](course/chapters/13_save-checkpoint.md) | `save_checkpoint` |
| 14 | [Resume without changing the trajectory](course/chapters/14_load-checkpoint.md) | `load_checkpoint` |
| 15 | [Train and diagnose a tiny model](course/chapters/15_train-run.md) | `train_run` |
| 16 | [Export a portable model artifact](course/chapters/16_export-model.md) | `export_model` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
