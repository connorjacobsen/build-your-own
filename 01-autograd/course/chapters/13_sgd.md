# 13. Update shared parameters once

Differentiation decides how the objective changes; optimization decides how parameters move. Parameter sharing introduces a new identity issue: several module paths can refer to one weight. The graph must accumulate every use, but the optimizer must update the unique weight once. This stage also makes gradient lifecycle explicit by clearing gradients after the step.

## Implementation contract

Work in `src/learner/api.py`. Implement **sgd**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`sgd`**

Update each unique Tensor exactly once in place using data -= lr*grad; then zero grad.
Reject negative lr. Parameters may contain repeated references to shared weights.

Mutate parameters in place; reject negative learning rates. Zero is allowed.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

Why do graph edge deduplication and optimizer parameter deduplication have different rules?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
