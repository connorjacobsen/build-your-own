# 12. Consolidate checkpoint shards

Sharded state needs explicit ranges for recovery and changes in world size. Reconstructing a global tensor from validated shards allows repartitioning under a new topology. Missing, overlapping or malformed ranges are errors, not zeros to be guessed. Sorting by range supports arbitrary arrival order while preserving deterministic assembly.

## Implementation contract

Work in `src/learner/api.py`. Implement **consolidate_shards**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`consolidate_shards`**

Reconstruct 1D momentum from dicts start,end,momentum. Input order arbitrary.
Validate gap-free, nonoverlapping coverage [0,total), matching slice lengths and rank-one
tensors BEFORE concatenation. Empty slices allowed. Require at least one shard.
Return new owned tensor; reject malformed layouts with ValueError.

This consolidates momentum tensors in memory. The capstone writes shard files and a manifest around this operation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

How would you validate that shard files belong to the same training step?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
