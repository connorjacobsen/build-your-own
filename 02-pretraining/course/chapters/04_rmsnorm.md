# 04. Normalize residual activations

Residual streams accumulate information across layers. RMS normalization controls scale without subtracting a mean. A learned per-feature weight restores expressiveness. Computing squared magnitudes in float32 helps low-precision stability. The epsilon belongs inside the square root; moving it outside changes both forward values and gradients.

## Implementation contract

Work in `src/learner/api.py`. Implement **RMSNorm**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`RMSNorm`**



Allocate weight=ones(dim). Use eps=1e-6 by default. Preserve the input dtype after normalization.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

Compare all-zero input and a constant nonzero vector.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [AdamW](https://arxiv.org/abs/1711.05101)
