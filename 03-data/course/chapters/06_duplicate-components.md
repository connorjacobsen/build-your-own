# 06. Group transitive near duplicates

Pairwise similarity is not transitive, but leakage groups need a transitive closure. If A resembles B and B resembles C, assigning A and C independently can leak related content through B. Connected components solve this grouping problem. This exact quadratic implementation prioritizes correctness; approximate candidate generation would be needed at web scale.

## Implementation contract

Work in `src/learner/api.py`. Implement **duplicate_components**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`duplicate_components`**

Return list of ID lists, components in first-record order, members in input order.
Add an edge when nonempty shingle-set Jaccard >= threshold, or normalized text is equal.
Connected components include transitive matches; distinct short documents do not match.
IDs must be unique; threshold in (0,1]. O(N^2) is intentional for this teaching dataset.

Use Jaccard intersection/union for nonempty unions. Equality of short normalized texts still creates an edge.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Construct a chain where A and C do not directly meet the threshold.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
