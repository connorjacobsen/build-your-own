# 03. Order a shared graph

A graph is not necessarily a tree. One parameter can influence two branches which later join. Visiting it twice can run its local derivative before all incoming contributions have arrived. A parents-before-children order, reversed at differentiation time, resolves this dependency. Node identity matters: equal numerical values can still be different parameters.

## Implementation contract

Work in `src/learner/api.py`. Implement **topological**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`topological`**

Return each reachable Tensor once, parents before children; shared edges retain contributions.

Graphs are acyclic; cycle detection is an optional extension. Return actual Tensor objects.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Draw a diamond graph and predict the error from a tree-only traversal.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
