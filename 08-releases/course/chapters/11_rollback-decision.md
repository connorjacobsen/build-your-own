# 11. Decide when to roll back

An automated policy needs both failure thresholds and a minimum evidence rule. Very small samples can be noisy, but waiting can prolong impact. The policy here makes that tradeoff visible and testable. A decision is separated from its execution so a learner can inspect the evidence before changing active state.

## Implementation contract

Work in `src/learner/api.py`. Implement **rollback_decision**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`rollback_decision`**

Given summarize_requests output, return 'wait' below min_requests; else 'rollback'
if error_rate>max_error OR p95_seconds>max_p95; otherwise 'keep'. Threshold equality passes.
Validate finite metrics and policy, n>=0, rates in [0,1], latencies>=0,min_requests>=1.

Return exactly wait, keep or rollback; threshold equality passes.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

What risks arise from always waiting for a minimum request count?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
