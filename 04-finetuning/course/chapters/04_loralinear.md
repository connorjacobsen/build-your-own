# 04. Implement low-rank adaptation

LoRA represents a weight update as the product of two smaller matrices. Freezing the original weight reduces optimizer state and lets the same base support several adapters. A nonzero A and zero B start from exactly the base function while allowing B to receive a gradient. Initializing both factors to zero would prevent either from learning.

## Implementation contract

Work in `src/learner/api.py`. Implement **LoRALinear**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`LoRALinear`**

Wrap an nn.Linear as base. Require rank>=1 and alpha>0. Freeze base parameters.
A is nn.Parameter [rank,in], Kaiming-uniform initialized; B is zeros [out,rank].
Both match base dtype/device. Forward: base(x) + (x @ A.T @ B.T)*(alpha/rank).
Arbitrary leading input dimensions are supported. Do not modify base weights in forward.

Support leading batch and sequence dimensions. Preserve base bias and dtype/device.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

Which factor receives a gradient on the first update, and why?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
