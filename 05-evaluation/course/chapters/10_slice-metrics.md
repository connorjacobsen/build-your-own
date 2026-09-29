# 10. Inspect performance slices

A model can improve overall while failing a small important group. Micro averaging follows the dataset’s group proportions; macro averaging weights groups equally. Reporting the worst slice helps expose concentrated failures, but small slice counts also mean higher uncertainty. Always keep counts beside slice metrics.

## Implementation contract

Work in `src/learner/api.py`. Implement **slice_metrics**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`slice_metrics`**

Records {'group':str,'correct':bool}. Return micro_accuracy, macro_accuracy,
worst_accuracy, groups={name:{n,accuracy}}. Require nonempty input; each row belongs to one
group. Macro weights groups equally; micro weights individual examples equally.

Each record has exactly one group in this teaching interface.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 10 --only
uv run course check 10
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 10 --level 1`.

## Explain and investigate

How would overlapping slices change the interpretation of a macro score?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
