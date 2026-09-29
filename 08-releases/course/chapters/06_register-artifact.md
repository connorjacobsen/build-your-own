# 06. Register immutable content

A content-addressed store names objects by their bytes. Repeated registration becomes idempotent, while corruption at an existing digest is a detectable inconsistency. Temporary writes and atomic replacement prevent readers from seeing a partially copied object. Checking the copied digest also catches source changes during registration.

## Implementation contract

Work in `src/learner/api.py`. Implement **register_artifact**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`register_artifact`**

Copy source to content-addressed store/<sha256>. Read/write by chunks; atomically
publish a temporary file with os.replace. Return digest. If object exists, verify it
matches digest and return without altering it. Corrupt existing objects raise ValueError.
The source remains unchanged. Clean temporary files on success or failure.

This is a local educational object store, not a distributed storage service.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

What extra guarantees would be needed for remote storage and multiple machines?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
