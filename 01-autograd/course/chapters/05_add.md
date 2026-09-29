# 05. Differentiate addition

Addition preserves each input’s influence. Broadcasting changes where that influence is collected. When both inputs are the same object, the same storage must receive two contributions. Assigning a parent gradient rather than adding to it silently breaks both shared operands and branched graphs.

## Implementation contract

Work in `src/learner/api.py`. Implement **add**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`add`**

Elementwise Tensor addition with NumPy broadcasting and accumulated parent gradients.

Return a new Tensor connected to both parents. Preserve input data.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Explain why add(x,x) has derivative two rather than one.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
