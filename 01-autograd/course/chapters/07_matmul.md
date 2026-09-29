# 07. Differentiate matrix products

Matrix multiplication combines many scalar products. For A[M,K] and B[K,N], an upstream gradient G[M,N] implies gradients shaped exactly like A and B. Deriving one entry by its summation index is safer than memorizing transposes. Restricting the first implementation to matrices makes the algebra visible before adding batched broadcasting.

## Implementation contract

Work in `src/learner/api.py`. Implement **matmul**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`matmul`**

Rank-two matrix multiplication only. Reject non-matrices with ValueError.

Reject vectors and rank-three inputs. No torch autograd or numerical differentiation inside the implementation.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Derive dL/dA[i,k] as a sum over n.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Deep Learning Systems](https://dlsyscourse.org/)
- [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
