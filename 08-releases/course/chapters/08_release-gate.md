# 08. Encode a release policy

Release policy combines absolute quality requirements, regression limits and evidence size. It is different from estimating statistical uncertainty. Returning ordered reasons makes failures actionable and auditable. Candidate and baseline must refer to comparable cases; otherwise a delta could reflect an easier dataset rather than an improved model.

## Implementation contract

Work in `src/learner/api.py`. Implement **release_gate**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`release_gate`**

Validate both reports and require same dataset and case-ID sets. Reject incomparable
evidence with ValueError. Return {eligible:bool,reasons:list[str]} in fixed order:
'too_few_cases', 'below_floor', 'regression'. A boundary exactly meeting a threshold passes.
Require thresholds in [0,1], min_cases>=1. This deterministic gate is policy, not a significance test.

The capstone adds paired uncertainty from course 05; this gate intentionally uses explicit deterministic thresholds.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 8 --only
uv run course check 8
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 8 --level 1`.

## Explain and investigate

How would you avoid tuning a candidate repeatedly against the final test set?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
