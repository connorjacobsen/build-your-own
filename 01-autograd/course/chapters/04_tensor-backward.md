# 04. Run reverse mode

Reverse mode computes a vector-Jacobian product, not an entire Jacobian. The seed specifies which weighted combination of outputs is being differentiated. A scalar loss conventionally has seed one. Every node must receive all contributions before its callback executes. This course deliberately resets reachable gradients at each backward call; optimizer accumulation is a separate policy.

## Implementation contract

Work in `src/learner/api.py`. Implement **Tensor.backward**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`Tensor.backward`**

See the method signatures in the learner scaffold.

Implement backward according to its scaffold docstring. Reject missing seeds for multi-element outputs and mismatched seed shapes.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

For y=[x²,3x], what derivative results from seed [2,4]?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
