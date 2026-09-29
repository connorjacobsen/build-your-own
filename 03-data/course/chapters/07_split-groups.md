# 07. Split groups, not rows

A random row split can scatter related documents across training and validation. Group-level assignment keeps an entire connected component together. Hash-based assignment is reproducible across input ordering and allows streaming lookup once group identity is known. It gives an expected fraction rather than an exact count, particularly with a small number of uneven groups.

## Implementation contract

Work in `src/learner/api.py`. Implement **split_groups**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`split_groups`**

Return ID->'train'/'validation'. Hash salt + ':' + lexicographically smallest group ID;
SHA256 interpreted as big-endian int divided by 2**256; values below fraction go to validation.
Validate fraction in [0,1], nonempty groups and no ID repeated across or within groups.
Group and member input ordering must not affect assignment. This does not enforce exact counts.

Group identity is its lexicographically smallest member ID. A changing group can change its assignment.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

What happens to the validation fraction when one group contains half the corpus?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
