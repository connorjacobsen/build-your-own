# 02. Collate variable-length examples

Padding makes examples rectangular, but padded positions must not become training targets. Attention visibility and loss supervision answer different questions: whether a token is readable and whether predicting it contributes to the objective. Keeping both masks makes that distinction visible. Labels remain aligned to original input positions at this stage.

## Implementation contract

Work in `src/learner/api.py`. Implement **collate**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`collate`**

examples are (ids,assistant_mask) pairs with equal nonzero lengths. Return dict
input_ids long [B,T], labels long [B,T] (original ID if mask true, else -100),
attention_mask bool [B,T]. Right-pad; padding labels=-100, attention=false.
Reject empty batches, empty examples and mismatched lengths. Do not shift labels here.

Right-pad only. Loss labels use -100; attention_mask is boolean.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Explain why a real prompt token can have attention=true but label=-100.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
