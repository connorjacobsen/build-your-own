# 11. Construct a loss from operations

A training objective reduces many predictions to a scalar. Mean squared error divides by the number of output elements, not merely the batch count. Constructing it from earlier operations tests whether graph composition actually works. Both prediction and target tensors remain differentiable in this engine, even though a normal training target is treated as fixed.

## Implementation contract

Work in `src/learner/api.py`. Implement **mse**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`mse`**

Mean squared error across every broadcast result element, composed from this engine's ops.

Compose existing operations; do not write a separate shortcut gradient.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

How does changing mean reduction to sum affect the appropriate learning rate?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
