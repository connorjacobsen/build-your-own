# 03. Make filtering explainable

A cheap heuristic can eliminate obvious repetition, but filtering changes the data distribution. A threshold that helps one language or document type can harm another. This toy filter exposes two measurable features: word count and maximum repetition fraction. Keeping the rule small makes false positives easy to examine before adopting more complex scoring models.

## Implementation contract

Work in `src/learner/api.py`. Implement **quality_filter**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`quality_filter`**

Return bool: normalized whitespace words count >= min_words and the most frequent
word's fraction <= max_repeat. Empty text always fails. Boundary thresholds are inclusive.
This toy heuristic is not a language detector or a universal measure of quality.

Use normalized whitespace-separated words; punctuation is not separately stripped.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 3 --only
uv run course check 3
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 3 --level 1`.

## Explain and investigate

Construct a legitimate document this filter rejects and describe the resulting bias.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
