# 09. Introduce a nonlinearity

Without nonlinearities, composing linear layers produces another linear map. ReLU selects an active subset of coordinates. Its derivative is undefined at zero mathematically; implementations need an explicit convention. We choose zero there, and the grader checks it. Finite differences near the kink require care and should not be mistaken for smooth-function checks.

## Implementation contract

Work in `src/learner/api.py`. Implement **relu**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`relu`**

Elementwise max(x,0); derivative is zero at and below zero.

Derivative is zero for x<=0 and one for x>0, multiplied by the upstream seed.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Why can a ReLU unit stop learning if every input is negative?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
