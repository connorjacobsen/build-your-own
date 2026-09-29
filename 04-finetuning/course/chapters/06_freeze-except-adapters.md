# 06. Control the trainable boundary

An optimizer only sees the parameters it is given, but autograd may still build gradients for other trainable parameters. Explicit freezing clarifies the intended training boundary and saves work. Parameter identity is stronger than a name suffix: an unrelated module could legitimately have a parameter called A.

## Implementation contract

Work in `src/learner/api.py`. Implement **freeze_except_adapters**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`freeze_except_adapters`**

Freeze all parameters except actual LoRALinear A/B objects. Return unique trainable
Parameters in model.parameters() order. Reject models without adapters. Names alone
must not cause unrelated parameters named A or B to become trainable.

Return the actual trainable Parameter objects, without duplicates.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Measure trainable parameter count before and after injection.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
