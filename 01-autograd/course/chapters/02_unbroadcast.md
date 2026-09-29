# 02. Undo broadcasting

Broadcasting duplicates a value conceptually across output positions. The reverse operation adds the influence of every use back into the original input. Removing extra leading dimensions and reducing singleton axes are distinct steps. A bias shaped [D] added to [B,T,D] receives contributions from both B and T; averaging those contributions would change the derivative.

## Implementation contract

Work in `src/learner/api.py`. Implement **unbroadcast**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`unbroadcast`**

Sum a broadcast output gradient back to an original shape; return an ndarray of that shape.

Valid inputs are NumPy arrays and shapes that broadcast to their shape. Sum rather than average.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Trace a [1,4] bias expanded into [2,3,4]. How many contributions reach each bias element?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
