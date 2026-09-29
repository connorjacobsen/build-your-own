# 06. Reduce weighted means across processes

An average of rank means is only correct when every rank has the same count. Reduce weighted sums and counts to handle unequal work. Zero-count ranks must still participate in every collective so other ranks can finish. The numerical payload and count must be on a device supported by the selected backend.

## Implementation contract

Work in `src/learner/api.py`. Implement **weighted_allreduce**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`weighted_allreduce`**

With initialized process group, return count-weighted global mean, leaving input intact.
local_count>=0; zero count contributes zero even if its local mean is arbitrary.
All ranks must call with equal tensor shape/dtype/device. Global count zero raises ValueError
on all ranks. Counts travel as float64 tensors on the input device. Use SUM collectives.

The grader launches two real CPU processes with Gloo. These are distributed tests, not a list-based simulation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Why can one rank skipping an all-reduce hang another rank?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
