# 01. Partition examples without duplication

Data parallelism starts with an ownership rule for examples. Padding a partition to equal rank lengths can duplicate examples and subtly change the objective. This course begins with unpadded strided assignment so unequal local counts remain visible. Some ranks can have no work when the world is larger than the batch, which later collective code must handle deliberately.

## Implementation contract

Work in `src/learner/api.py`. Implement **partition_indices**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`partition_indices`**

Strided indices rank,rank+world_size,... below size. No duplication or padding.
Require size>=0, world_size>=1, 0<=rank<world_size. Return list[int].

This partitions one finite epoch. Shuffling is a separate deterministic permutation before partitioning.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

What fraction of examples would be duplicated by padding seven items to three equal ranks?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
