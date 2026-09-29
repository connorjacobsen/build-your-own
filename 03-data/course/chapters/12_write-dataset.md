# 12. Write an immutable dataset artifact

Reproducibility begins with identifying the exact bytes used for training. Canonical JSON serialization makes a small dataset inspectable and repeatable. The manifest records a content digest and count; it is metadata, not proof that the content is good. Refusing to overwrite avoids silently invalidating an earlier experiment’s dataset reference.

## Implementation contract

Work in `src/learner/api.py`. Implement **write_dataset**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`write_dataset`**

Create directory; write records.jsonl as UTF-8 JSON, sort_keys=True, ensure_ascii=False,
separators=(',',':'), one newline per record. Write manifest.json containing format
'toydata-v1', file 'records.jsonl', sha256 of exact bytes, records count. Return manifest.
Empty datasets have zero bytes. Never overwrite an existing records.jsonl or manifest.json.

This educational writer is single-process and local. Concurrent writers and crash-atomic directory publication are later release concerns.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Why must the trailing newline convention be part of byte identity?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)
