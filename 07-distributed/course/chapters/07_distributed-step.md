# 07. Train with real synchronized gradients

Every rank starts with the same weights, computes local contributions and participates in reductions before applying an identical update. Comparing the result with a single-process global batch is a strong correctness check. It detects normalization errors and inconsistent optimizer ordering. Collectives must occur in the same sequence even when one rank has an empty local batch.

## Implementation contract

Work in `src/learner/api.py`. Implement **distributed_step**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`distributed_step`**

One global mean-squared-error update with initialized group; model(x) and y both [N,1].
Unequal local N, including zero, are allowed; total N must be positive. Every rank executes
collectives in identical order. Sum local squared errors and gradients globally, divide by
GLOBAL example count, then step. Return identical global mean loss on every rank.

The teaching loss is scalar-output squared error. Production token training needs the same logic with supervised-token counts.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Predict the failure if gradients are divided by world size instead of global count.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
