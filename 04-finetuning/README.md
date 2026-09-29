# Build a fine-tuning and preference system

Adapt models through fourteen stages covering conversation serialization, masked supervision, LoRA, supervised updates, DPO and portable adapters. The grader uses tiny controlled models; the capstone applies these components to the checkpoint you trained.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/04-finetuning
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Serialize supervised conversations](course/chapters/01_chat-tokens.md) | `chat_tokens` |
| 2 | [Collate variable-length examples](course/chapters/02_collate.md) | `collate` |
| 3 | [Shift and mask supervised loss](course/chapters/03_masked-loss.md) | `masked_loss` |
| 4 | [Implement low-rank adaptation](course/chapters/04_loralinear.md) | `LoRALinear` |
| 5 | [Inject adapters deliberately](course/chapters/05_inject-lora.md) | `inject_lora` |
| 6 | [Control the trainable boundary](course/chapters/06_freeze-except-adapters.md) | `freeze_except_adapters` |
| 7 | [Run one supervised update](course/chapters/07_sft-step.md) | `sft_step` |
| 8 | [Merge an adapter for inference](course/chapters/08_merge-lora.md) | `merge_lora` |
| 9 | [Score whole responses](course/chapters/09_sequence-logps.md) | `sequence_logps` |
| 10 | [Derive the DPO objective](course/chapters/10_dpo-loss.md) | `dpo_loss` |
| 11 | [Optimize a preference pair](course/chapters/11_dpo-step.md) | `dpo_step` |
| 12 | [Extract portable adapter weights](course/chapters/12_adapter-state.md) | `adapter_state` |
| 13 | [Bind an adapter to its base](course/chapters/13_save-adapter.md) | `save_adapter` |
| 14 | [Restore an adapter atomically](course/chapters/14_load-adapter.md) | `load_adapter` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
