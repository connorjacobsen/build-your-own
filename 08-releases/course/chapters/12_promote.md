# 12. Promote with compare-and-swap

Two evaluators can both decide to promote a candidate based on the same active model. A compare-and-swap detects stale decisions rather than allowing the last writer to win silently. A process lock protects the read-check-write transaction, while atomic replacement prevents partial JSON reads. The previous pointer supports recovery.

## Implementation contract

Work in `src/learner/api.py`. Implement **promote**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`promote`**

Local POSIX compare-and-swap pointer. JSON {current,previous}; absent means current=None.
Acquire exclusive flock on sibling <name>.lock, reread pointer, require current==expected
else RuntimeError, then atomically replace pointer with current=new_digest,previous=old.
New digest is 64 lowercase hex chars. Same-current promotion is idempotent and preserves
previous. Return resulting dict. Create parent directories. No remote deployment occurs.

Uses local POSIX flock, available on macOS and Linux. Atomic visibility is tested; full power-loss durability is outside this toy contract.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Why is atomic file replacement alone insufficient to prevent lost updates?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
