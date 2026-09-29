# 02. Hash large artifacts incrementally

Weights can be much larger than available working memory. A streaming hash reads bounded chunks while producing the same digest as a one-shot hash. The digest identifies bytes, not semantics: two equivalent parameter serializations can have different hashes. That distinction is useful when tracing exactly which artifact was evaluated.

## Implementation contract

Work in `src/learner/api.py`. Implement **file_digest**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`file_digest`**

SHA256 hex of file bytes, read in chunks no larger than 1 MiB. Empty files allowed.

Do not read the whole file in one call.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 2 --only
uv run course check 2
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 2 --level 1`.

## Explain and investigate

Compare byte identity, parameter equality and prediction equivalence.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
