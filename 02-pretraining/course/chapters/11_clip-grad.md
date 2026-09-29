# 11. Clip a global gradient norm

Clipping all gradients together preserves their direction while limiting the magnitude of the proposed update. Clipping each tensor independently changes that direction. Reporting the preclip norm is useful for diagnosing unstable training. Nonfinite values must stop the step rather than turning parameters into NaNs.

## Implementation contract

Work in `src/learner/api.py`. Implement **clip_grad**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`clip_grad`**

Clip the global L2 norm of existing gradients in place; return the preclip norm as float.
Ignore parameters whose grad is None. Reject max_norm<=0 and nonfinite gradient norm.

The threshold applies over every parameter gradient combined. Ignore missing gradients.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

Show a two-parameter example where per-tensor clipping differs from global clipping.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
