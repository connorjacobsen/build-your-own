# 13. Coordinate skipped updates

If one rank skips an optimizer update while another applies it, replicated weights diverge. Nonfinite detection therefore becomes a collective decision. A minimum over boolean-like integer flags makes one failure visible everywhere. All ranks must reach this check before any state update, including momentum and update counters.

## Implementation contract

Work in `src/learner/api.py`. Implement **collective_finite**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`collective_finite`**

Initialized group and nonempty parameter list all on one device. Return bool true
on ALL ranks iff every present gradient on EVERY rank is finite. None is allowed.
Do not change gradients. Reduce a device-local int flag using MIN, so one bad rank
makes the same skip decision everywhere. Call before any optimizer update.

None gradients are allowed. The grader injects infinity on only one rank.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

What other state must remain unchanged when an update is skipped?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
