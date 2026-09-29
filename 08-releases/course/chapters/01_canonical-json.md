# 01. Make metadata byte-stable

Content identity requires an agreed serialization. Dictionary insertion order and whitespace should not change the digest of logically identical metadata. UTF-8 preserves human-readable text, while rejecting NaN and infinity prevents nonstandard numeric values from entering release evidence. The newline convention is part of the format.

## Implementation contract

Work in `src/learner/api.py`. Implement **canonical_json**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`canonical_json`**

Return UTF-8 bytes: JSON sorted keys, compact separators, ensure_ascii=False,
allow_nan=False, followed by one newline. Unsupported/nonfinite values must fail.

This is a local JSON convention, not a universal canonical-JSON standard for signatures.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 1 --only
uv run course check 1
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 1 --level 1`.

## Explain and investigate

Why is signing arbitrary pretty-printed JSON fragile?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
