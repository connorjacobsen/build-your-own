# 03. Flatten a communication bucket

Collectives have latency overhead as well as bandwidth cost. Combining small tensors into one communication bucket amortizes that overhead. Tensor order becomes part of the bucket layout. A copied flat buffer can still retain autograd connectivity through concatenation; storage ownership and differentiability are different properties.

## Implementation contract

Work in `src/learner/api.py`. Implement **flatten_tensors**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`flatten_tensors`**

Concatenate flattened nonempty list of same-dtype, same-device tensors into a new 1D
tensor. Preserve autograd connectivity, tensor order and values. Zero-size tensors allowed.
Reject empty list, mixed dtypes or mixed devices. Do not modify inputs.

Require identical dtype and device rather than silently promoting or copying devices.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Why might one enormous bucket delay communication overlap?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
