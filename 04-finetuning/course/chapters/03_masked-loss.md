# 03. Shift and mask supervised loss

A logit at position t predicts token t+1. Shifting in both the collator and loss would move targets two steps, while never shifting trains copying. The denominator is the number of supervised targets, not padded positions or sequences. An empty supervised batch should fail visibly rather than return NaN or a misleading zero.

## Implementation contract

Work in `src/learner/api.py`. Implement **masked_loss**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`masked_loss`**

Next-token mean CE: logits [B,T,V] at :-1 predict labels at 1:.
Ignore labels=-100, divide by total supervised tokens across batch. Raise ValueError
when no shifted target is supervised. Return differentiable scalar tensor.

Perform exactly one shift inside this function. Never supervise padding.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Manually mark the logit positions responsible for a two-token assistant answer.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
