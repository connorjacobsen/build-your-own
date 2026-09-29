# 04. Remove exact duplicates stably

Duplicate examples give repeated content extra weight. Removing duplicates before splitting reduces direct leakage and makes dataset size more meaningful. Stable first-occurrence retention provides a clear provenance rule and avoids nondeterminism from set iteration. Returning independent record dictionaries prevents downstream annotations from modifying the raw input collection.

## Implementation contract

Work in `src/learner/api.py`. Implement **deduplicate**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`deduplicate`**

Records are dicts with id,text. Return shallow copies of the first occurrence for
each normalized content ID, preserving input order. Never mutate caller records.

Copy dictionaries shallowly; records contain scalar fields in the acceptance cases.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 4 --only
uv run course check 4
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 4 --level 1`.

## Explain and investigate

When would you prefer retaining the newest rather than the first copy?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
