# 09. Sample a reproducible mixture

A training mixture changes how frequently each source contributes, regardless of raw dataset size. Sampling source then example makes that policy explicit. Replacement allows small sources to recur, which can overfit them. Stable source ordering and a local RNG ensure a harmless dictionary reordering does not change the sequence of training examples.

## Implementation contract

Work in `src/learner/api.py`. Implement **sample_mixture**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`sample_mixture`**

sources maps names to nonempty lists of records. weights has identical keys, finite
nonnegative values and positive total. Draw WITH replacement: choose sorted source name
via local random.Random(seed).choices, then a record via that RNG.choice; return copied
dicts with added source field. count>=0. Inputs remain unchanged; no global RNG use.

Weights are relative masses, not required to sum to one. Zero-weight sources never appear.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Estimate how often a one-record source repeats in a 1000-sample run.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
