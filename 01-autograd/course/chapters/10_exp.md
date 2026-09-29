# 10. Compose exponential and logarithm

Exponentials and logarithms appear in normalized probabilities and likelihood losses. Their derivatives are simple, but their numerical domains matter. Taking log at zero is not an innocuous edge case. Checking log(exp(x)) on moderate inputs is a useful composition test, while extreme inputs teach why a stable fused loss is needed later.

## Implementation contract

Work in `src/learner/api.py`. Implement **exp, log**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`exp`**

Elementwise exponential with a reverse-mode derivative.

**`log`**

Natural logarithm; reject any nonpositive input with ValueError.

Raise ValueError for any nonpositive logarithm input. Inputs to exp in this stage are moderate.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

Why can mathematically equivalent log(exp(1000)) overflow?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
