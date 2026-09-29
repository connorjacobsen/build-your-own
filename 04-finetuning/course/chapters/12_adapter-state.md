# 12. Extract portable adapter weights

An adapter checkpoint should contain adaptation state rather than a second copy of the base model. Cloning detached CPU tensors prevents future updates from modifying a supposedly captured state. The qualified names describe where each matrix belongs and must agree with the receiving architecture.

## Implementation contract

Work in `src/learner/api.py`. Implement **adapter_state**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`adapter_state`**

Return dict qualified_name+'.A'/'.B' -> detached CPU cloned tensors for actual
LoRALinear modules only. Reject models with no adapters. No base weights in this state.

Only actual adapter objects count; ignore similarly named unrelated parameters.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Estimate storage for a rank-r adapter versus a full weight matrix.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
