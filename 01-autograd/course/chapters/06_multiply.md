# 06. Differentiate multiplication

The derivative with respect to one factor depends on the other factor’s forward value. This local rule composes with arbitrary upstream weights through multiplication. A square is an especially useful adversarial example: both operand positions refer to one object, so both local derivatives must be accumulated.

## Implementation contract

Work in `src/learner/api.py`. Implement **multiply**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`multiply`**

Elementwise Tensor multiplication; support broadcasting and the same Tensor as both operands.

Support ordinary NumPy broadcasting; operands are Tensor instances.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Check a broadcast product with finite differences and a nonuniform output seed.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
