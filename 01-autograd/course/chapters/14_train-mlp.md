# 14. Train a composed network

A working training loop integrates ownership, graph construction, differentiation and optimization. Each iteration builds a fresh graph from current parameters. Retaining old graphs unnecessarily wastes memory and can expose stale values. A local random generator makes initialization reproducible without depending on earlier unrelated experiments. The small regression task verifies mechanics, not general language understanding.

## Implementation contract

Work in `src/learner/api.py`. Implement **train_mlp**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`train_mlp`**

Fit [N,D] x to [N,O] y with Linear-ReLU-Linear using only this engine.
Initialize w1,w2 using local NumPy normal(0,.3), biases zero. Return (parameters, losses),
parameters ordered [w1,b1,w2,b2], losses containing one PRE-update MSE per step.
Shapes are [D,H], [H], [H,O], [O]. Do not mutate inputs or use global NumPy RNG.

Follow the exact initialization and return contract. Use the engine functions and sgd; external autodiff is forbidden.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Compare train and held-out MSE for a linear target and then a curved target. Predict the effect of hidden width.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
