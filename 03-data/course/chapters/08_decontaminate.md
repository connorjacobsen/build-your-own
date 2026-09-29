# 08. Decontaminate against held-out content

A model can appear capable because evaluation examples or close variants appeared in training. Decontamination checks training candidates against reserved content before fitting. This stage uses a strict overlap policy to make the mechanics explicit. It will over-remove common phrases, so an actual data pipeline needs an examined tradeoff and a versioned evaluation set.

## Implementation contract

Work in `src/learner/api.py`. Implement **decontaminate**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`decontaminate`**

Return copied records with no shared n-word shingle with any holdout text.
Also reject exact normalized matches, including short texts. Preserve order.
This is a strict toy filter; it can discard common phrases and is not a production detector.

Holdout content is a filter input only and must never be appended to training.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

Why is removing contaminated examples after model training too late?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
