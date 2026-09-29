# 10. Summarize serving observations

A model can pass offline evaluations and still fail operationally. Error rate and tail latency expose different issues. Including failed requests in latency accounting prevents a fast-failing system from looking artificially healthy. The percentile definition matters on small samples, so this course uses an explicit nearest-rank rule.

## Implementation contract

Work in `src/learner/api.py`. Implement **summarize_requests**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`summarize_requests`**

Nonempty records {status:'ok'|'error',latency_seconds:finite nonnegative}. Return n,
error_rate, p95_seconds using nearest-rank ceil(.95*n)-1 on ALL requests including errors.
Reject malformed status or latency. Units remain seconds.

Use seconds consistently; do not infer missing observations.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

How can averaging latency hide a severe tail problem?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
