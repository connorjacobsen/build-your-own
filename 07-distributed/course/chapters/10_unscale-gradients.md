# 10. Unscale only finite gradients

Loss scaling increases gradient magnitudes during low-precision backpropagation, then reverses that scaling before the optimizer sees them. A nonfinite gradient means the update should be skipped, not partially applied. Scan first, mutate second. The next coordination stage ensures all ranks agree about that decision.

## Implementation contract

Work in `src/learner/api.py`. Implement **unscale_gradients**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`unscale_gradients`**

Positive finite scale. If any present gradient is nonfinite, return False and change
nothing. Otherwise divide all present grads by scale and return True. Ignore None grads.
This stage is local; collective_finite later coordinates the skip decision across ranks.

This function is a local operation and does not execute collectives.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

Why would unscaling twice silently shrink every update?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
