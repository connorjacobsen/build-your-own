# 02. Give content a stable identity

A dataset row ID describes a record; a content hash describes a chosen representation of its content. Stable hashes let independent runs agree about duplicates. Python’s built-in hash is process-dependent and inappropriate here. Hashing normalized text makes the matching policy an implicit part of content identity, so record that policy in a real dataset manifest.

## Implementation contract

Work in `src/learner/api.py`. Implement **document_id**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`document_id`**

SHA256 lowercase hex of normalize(text).encode('utf-8').

Use cryptographic SHA256 rather than Python hash.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Should changing the normalization policy produce a new dataset version?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
