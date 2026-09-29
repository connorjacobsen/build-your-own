# 14. Run the same contract on two GPUs

Correct CPU collectives establish semantics but cannot establish GPU execution. The final gate launches one process per NVIDIA device using NCCL and compares the result to a CPU global-batch reference. Device assignment, collective placement and numerical tolerances become part of the systems contract. Passing a simulated or skipped hardware check would make a false claim.

## Implementation contract

Work in `src/learner/api.py`. Implement **distributed_step, collective_finite**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`distributed_step`**

One global mean-squared-error update with initialized group; model(x) and y both [N,1].
Unequal local N, including zero, are allowed; total N must be positive. Every rank executes
collectives in identical order. Sum local squared errors and gradients globally, divide by
GLOBAL example count, then step. Return identical global mean loss on every rank.

**`collective_finite`**

Initialized group and nonempty parameter list all on one device. Return bool true
on ALL ranks iff every present gradient on EVERY rank is finite. None is allowed.
Do not change gradients. Reduce a device-local int flag using MIN, so one bad rank
makes the same skip decision everywhere. Call before any optimizer update.

Requires two real NVIDIA GPUs. Run stages 1–13 locally; invoke the supplied Modal runner explicitly for stage 14. No paid job is launched by importing the runner.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Record GPU model, dtype, world size, throughput and communication time before drawing scaling conclusions.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
