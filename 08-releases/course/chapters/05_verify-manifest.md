# 05. Verify integrity before use

A successful training run does not guarantee that later files are unchanged. Integrity verification checks the bundle against recorded byte identities before loading or promotion. Checking size alone misses equal-size corruption. Checking only one file misses a swapped tokenizer or configuration. Every listed artifact contributes to the release’s identity.

## Implementation contract

Work in `src/learner/api.py`. Implement **verify_manifest**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`verify_manifest`**

Validate format, nonempty unique file entries, path containment, exact byte sizes and
hashes. Return True; any mismatch raises ValueError. Unlisted files do not belong to bundle.

Unexpected unlisted files are ignored; only listed files form this bundle.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 5 --only
uv run course check 5
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 5 --level 1`.

## Explain and investigate

Demonstrate corruption that preserves byte length but changes predictions.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
