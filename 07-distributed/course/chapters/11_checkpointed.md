# 11. Trade activation memory for recomputation

Activation checkpointing stores fewer intermediates and recomputes them during backward. It must preserve both gradients and stochastic behavior. This is unrelated to saving training state to disk despite the shared name. A mechanism test counts forward invocations so a plain function call cannot pass merely because its values match.

## Implementation contract

Work in `src/learner/api.py`. Implement **checkpointed**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`checkpointed`**

Call PyTorch non-reentrant activation checkpointing, preserving RNG state.
Forward outputs and backward gradients must match ordinary execution, while the function
executes again during backward when intermediates are required. This is activation
recomputation, not serialization of model weights. Use the framework checkpoint primitive.

Use non-reentrant PyTorch checkpointing with RNG preservation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

Measure the time-memory tradeoff for a longer stack of blocks.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
