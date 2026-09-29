# 08. Estimate uncertainty with paired resampling

Two models evaluated on the same examples produce paired observations. Resampling example indices jointly preserves easy and hard cases across the comparison. Independently resampling each model throws away this relationship and changes uncertainty. A bootstrap interval estimates sampling variation under assumptions about these cases; it does not account for all dataset or judge biases.

## Implementation contract

Work in `src/learner/api.py`. Implement **paired_bootstrap**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`paired_bootstrap`**

Equal nonempty finite 1D arrays of per-example scores, larger=better. Resample paired
indices with local np.random.default_rng(seed).integers(0,N,size=(resamples,N)); return
dict delta=mean(candidate-baseline), low/high=quantiles at (1-confidence)/2 and complement.
resamples>=1 and 0<confidence<1. Never bootstrap the two models independently.

Use local seeded NumPy randomness and percentile intervals. Larger scores mean better performance.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

Why does a constant per-example improvement have a zero-width paired interval?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
