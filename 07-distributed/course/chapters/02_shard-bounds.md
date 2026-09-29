# 02. Assign contiguous tensor shards

Optimizer state is easier to partition as contiguous flat ranges. Quotient and remainder determine balanced ownership, with at most one element difference between ranks. Empty shards are legitimate when tensors are small. An explicit half-open convention prevents overlapping endpoint ownership and makes gather reconstruction straightforward.

## Implementation contract

Work in `src/learner/api.py`. Implement **shard_bounds**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`shard_bounds`**

Balanced contiguous [start,end) ownership. First total%world_size ranks own one extra.
Require total>=0,world_size>=1, valid rank. Empty shards allowed.

The first remainder ranks own the extra elements.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Enumerate ranges for total=2 and world_size=4.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
