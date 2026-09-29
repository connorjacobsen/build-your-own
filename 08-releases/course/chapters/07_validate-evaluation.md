# 07. Validate evaluation evidence

A release decision must refer to the actual model and dataset being considered. Reports also need internal consistency: duplicate cases or an inflated aggregate can make an apparently valid number meaningless. Recomputing the aggregate from per-case scores catches bookkeeping defects. It cannot establish the honesty or relevance of the evaluator itself.

## Implementation contract

Work in `src/learner/api.py`. Implement **validate_evaluation**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`validate_evaluation`**

Validate toyeval-v1 identities, unique nonempty cases, finite [0,1] scores, metrics.n
equal case count, metrics.accuracy equal score mean (abs tolerance 1e-12). Both digests
must be 64 lowercase hex chars and match expected identities. Return True or ValueError.
This validates internal consistency, not whether an external evaluator was honest.

Scores are bounded [0,1] and n equals the number of unique cases.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Why does matching a model hash still not prove benchmark validity?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
