# 04. Restore shapes through views

A flat communication buffer loses tensor boundaries unless the layout is retained. Restoring views rather than copies lets one buffer back several shaped tensors. That makes mutation semantics important: changes through a view affect the shared bucket. Validating total element count prevents a subtly shifted parameter mapping.

## Implementation contract

Work in `src/learner/api.py`. Implement **unflatten**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`unflatten`**

Return views into rank-one flat with specified shapes, in order. Require exact element
count and all shape dimensions nonnegative. Scalar shape () consumes one element.
Mutating a returned view must mutate the corresponding flat segment.

Shapes may include scalars and zero-sized dimensions.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

Describe a bug caused by restoring parameters in a different order from flattening.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
