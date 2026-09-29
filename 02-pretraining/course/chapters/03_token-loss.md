# 03. Compute stable token loss

Cross entropy compares unnormalized scores with observed classes. The selected target logit is subtracted from a stable log partition function. Flattening batch and sequence dimensions is valid only when every target is supervised equally; later courses introduce masks and counts. Autograd should flow through the numerical expression, so returning a Python number here would break training.

## Implementation contract

Work in `src/learner/api.py`. Implement **token_loss**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`token_loss`**

Mean next-token cross entropy for logits [...,V] and same-prefix-shape targets.
Compute stable logsumexp minus selected logits; do not call F.cross_entropy.

Return a scalar differentiable tensor. Only implement the expression; do not use the framework cross-entropy shortcut.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Why does adding a constant to every logit leave the loss unchanged?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
