# 01. Own your tensors

A tensor has numerical storage and participation in a graph. Those are different responsibilities. Owning the input array prevents a caller from silently changing a forward value after the graph was constructed. Gradient storage must have the same shape even for a scalar, whose shape is an empty tuple. This engine uses float64 so numerical derivative checks are easier to interpret.

## Implementation contract

Work in `src/learner/api.py`. Implement **Tensor**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`Tensor`**

Float64 tensor with owned data, zero grad, tuple parents and a no-argument _backward callback.
Constructor accepts data, parents=(), backward=None. Operations attach callbacks which
add vector-Jacobian products into parent.grad. No torch/autograd delegation is allowed.

Implement the constructor now; leave backward for stage 4. The callback is a no-argument callable. Parents retain identity, including repeated references.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

Why would sharing the caller’s array make an otherwise correct backward pass wrong?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
