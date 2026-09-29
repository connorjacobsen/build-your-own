# 09. Route a stable canary cohort

A canary exposes a limited cohort to a candidate. Stable hashing keeps the same request identity in the same cohort across retries, making behavior reproducible. The salt defines the experiment assignment. Changing it reshuffles the cohort and should be recorded. Increasing the fraction under the same salt grows the candidate cohort predictably.

## Implementation contract

Work in `src/learner/api.py`. Implement **route_request**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`route_request`**

Deterministic SHA256(salt+':'+request_id) / 2**256 bucket. Return 'candidate' iff
bucket<fraction else 'baseline'. fraction in [0,1], including exact endpoints.
This routes one local simulation request; it does not contact a service.

This returns a routing label only; no live traffic is sent.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Should routing identity be per request or per user for a stateful application?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)
