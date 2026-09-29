# 05. Represent local overlap

Word shingles capture consecutive local context. A bag of words can call reordered text identical; shingles preserve a limited amount of order. Larger windows are more specific but miss shorter overlaps. Sets ignore repeated occurrence counts, which is intentional for Jaccard similarity but means repetition is handled elsewhere.

## Implementation contract

Work in `src/learner/api.py`. Implement **shingles**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`shingles`**

Set of consecutive n-word tuples from normalized text. n>=1, else ValueError.
Documents with fewer than n words produce an empty set.

Short documents yield empty sets rather than a special match token.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Compare shingle sizes one, two and three on a lightly edited sentence.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
