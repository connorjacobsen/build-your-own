# 08. Keep only local optimizer state

Replicated optimizer moments can consume more memory than model weights. Sharding that state assigns each rank responsibility for only part of it. A tensor slice that still references the full backing storage does not deliver the intended memory saving; clone the owned range. This stage isolates that memory ownership contract before adding communication.

## Implementation contract

Work in `src/learner/api.py`. Implement **shard_state**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`shard_state`**

For a flat global momentum tensor return dict start,end,momentum containing only
an owned clone of this rank's balanced contiguous shard. Do not retain global storage.

Return only the owned momentum data and its range metadata.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

Estimate momentum memory per rank as world size increases.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
