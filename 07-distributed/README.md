# Build a distributed training system

Fourteen stages take you from partitioning to real multi-process gradient synchronization, sharded optimizer state, activation recomputation and a two-GPU NCCL gate. Stages 1–13 run on CPU; the final hardware gate is explicit and cannot pass by skipping.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/07-distributed
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Partition examples without duplication](course/chapters/01_partition-indices.md) | `partition_indices` |
| 2 | [Assign contiguous tensor shards](course/chapters/02_shard-bounds.md) | `shard_bounds` |
| 3 | [Flatten a communication bucket](course/chapters/03_flatten-tensors.md) | `flatten_tensors` |
| 4 | [Restore shapes through views](course/chapters/04_unflatten.md) | `unflatten` |
| 5 | [Accumulate uneven microbatches](course/chapters/05_accumulate-gradients.md) | `accumulate_gradients` |
| 6 | [Reduce weighted means across processes](course/chapters/06_weighted-allreduce.md) | `weighted_allreduce` |
| 7 | [Train with real synchronized gradients](course/chapters/07_distributed-step.md) | `distributed_step` |
| 8 | [Keep only local optimizer state](course/chapters/08_shard-state.md) | `shard_state` |
| 9 | [Update sharded momentum and gather weights](course/chapters/09_sharded-momentum-step.md) | `sharded_momentum_step` |
| 10 | [Unscale only finite gradients](course/chapters/10_unscale-gradients.md) | `unscale_gradients` |
| 11 | [Trade activation memory for recomputation](course/chapters/11_checkpointed.md) | `checkpointed` |
| 12 | [Consolidate checkpoint shards](course/chapters/12_consolidate-shards.md) | `consolidate_shards` |
| 13 | [Coordinate skipped updates](course/chapters/13_collective-finite.md) | `collective_finite` |
| 14 | [Run the same contract on two GPUs](course/chapters/14_distributed-step.md) | `distributed_step, collective_finite` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
