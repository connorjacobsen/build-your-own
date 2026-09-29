# Build a tiny autodiff framework

Build the differentiation engine underneath a neural network, using NumPy rather than delegating gradients to PyTorch. Fourteen small stages culminate in a trainable network. Prerequisites: Python, arrays, derivatives and matrix multiplication.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/01-autograd
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Own your tensors](course/chapters/01_tensor.md) | `Tensor` |
| 2 | [Undo broadcasting](course/chapters/02_unbroadcast.md) | `unbroadcast` |
| 3 | [Order a shared graph](course/chapters/03_topological.md) | `topological` |
| 4 | [Run reverse mode](course/chapters/04_tensor-backward.md) | `Tensor.backward` |
| 5 | [Differentiate addition](course/chapters/05_add.md) | `add` |
| 6 | [Differentiate multiplication](course/chapters/06_multiply.md) | `multiply` |
| 7 | [Differentiate matrix products](course/chapters/07_matmul.md) | `matmul` |
| 8 | [Differentiate reductions](course/chapters/08_summation.md) | `summation` |
| 9 | [Introduce a nonlinearity](course/chapters/09_relu.md) | `relu` |
| 10 | [Compose exponential and logarithm](course/chapters/10_exp.md) | `exp, log` |
| 11 | [Construct a loss from operations](course/chapters/11_mse.md) | `mse` |
| 12 | [Stabilize cross-entropy](course/chapters/12_cross-entropy.md) | `cross_entropy` |
| 13 | [Update shared parameters once](course/chapters/13_sgd.md) | `sgd` |
| 14 | [Train a composed network](course/chapters/14_train-mlp.md) | `train_mlp` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
