# 12. Stabilize cross-entropy

Softmax probabilities can underflow and exponentials can overflow even when the final loss is finite. Subtracting the row maximum preserves normalized probabilities. Computing log probabilities through log-sum-exp avoids taking the log of a rounded zero. The gradient has a compact structure: predicted probability minus the target indicator, averaged across examples.

## Implementation contract

Work in `src/learner/api.py`. Implement **cross_entropy**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`cross_entropy`**

Stable mean cross-entropy for [N,C] logits and integer [N] labels. Return scalar Tensor.
Reject wrong label shape, empty N or out-of-range labels. Backward must remain finite
for logits of magnitude 1000. Compute derivatives yourself; do not delegate to torch.

Labels are integer class indices. Large-magnitude logits must give finite losses and gradients.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Add a constant 1000 to every class score in each row. What should change?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
