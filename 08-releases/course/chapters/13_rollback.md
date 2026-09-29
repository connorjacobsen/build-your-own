# 13. Restore the previous release

Rollback should use the same concurrency controls as promotion. Reading a prior version and then switching to it without checking the current version can undo someone else’s newer decision. Compare-and-swap preserves the expectation about what is being rolled back. Keeping the displaced release as previous makes an explicit undo possible.

## Implementation contract

Work in `src/learner/api.py`. Implement **rollback**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`rollback`**

Read previous digest and call promote(pointer,previous,expected_current). Reject no
prior release with ValueError. The compare-and-swap rejects a concurrent pointer change.
A successful rollback records the rolled-back release as previous, enabling explicit undo.

A first release has no prior target and must fail clearly.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 13 --only
uv run course check 13
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 13 --level 1`.

## Explain and investigate

What happens if a promotion occurs between reading the previous pointer and applying rollback?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
