# 08. Differentiate reductions

A sum sends the upstream derivative equally to each included input element. The output shape loses an axis unless keepdims is enabled. Backward must recover that axis before broadcasting. This is the reverse of forward broadcasting: here reduction creates a smaller output, and its derivative expands.

## Implementation contract

Work in `src/learner/api.py`. Implement **summation**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`summation`**

Sum all elements or one integer axis (negative axes allowed), with the corresponding VJP.

Support axis=None or a single integer, including negative axes.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

Compare sum and mean derivatives on batches of different sizes.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
