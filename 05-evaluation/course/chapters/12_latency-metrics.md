# 12. Measure user-visible generation latency

Time to first token includes the delay before any answer appears. Inter-token latency describes the continuation experience. Counting the first token in decode throughput mixes prefill with decode work. Timestamp traces preserve enough detail to compute different summaries later and to detect non-monotonic measurement errors.

## Implementation contract

Work in `src/learner/api.py`. Implement **latency_metrics**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`latency_metrics`**

Monotonic finite timestamps in seconds, nonempty tokens, first>=arrival. Return
ttft, inter_token_seconds list, decode_tokens_per_second=(N-1)/(last-first) for N>1.
For one token decode throughput is None; zero decode duration for N>1 gives inf.
Do not count the prefill-generated first token in decode throughput.

The timestamps are supplied measurements; this function must not fabricate timing data.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 12 --only
uv run course check 12
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 12 --level 1`.

## Explain and investigate

Compare two systems with equal total latency but different first-token latency.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
