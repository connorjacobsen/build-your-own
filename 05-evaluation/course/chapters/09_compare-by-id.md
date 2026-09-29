# 09. Align comparisons by identity

Evaluation outputs often finish in a different order because of batching, retries or distributed workers. Zipping files can compare unrelated examples while producing plausible numbers. Stable IDs form the join key. Missing or duplicate cases should stop the comparison instead of silently reducing it to a convenient subset.

## Implementation contract

Work in `src/learner/api.py`. Implement **compare_by_id**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`compare_by_id`**

Each list contains {'id':str,'score':float}. Require identical nonempty unique ID sets.
Align sorted IDs, then paired_bootstrap with defaults and supplied seed. Return its dict.
Reordering input rows must never alter pairing or RNG interpretation.

Sort IDs before resampling so reordering files leaves the report unchanged.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 9 --only
uv run course check 9
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 9 --level 1`.

## Explain and investigate

Explain how silently dropping failed examples can inflate a benchmark score.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
