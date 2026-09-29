# 07. Distinguish macro and micro behavior

Class imbalance makes aggregate accuracy easy to misread. Macro F1 gives every declared class equal influence, including absent classes under the explicit zero convention here. Zero denominators need a declared policy. Computing metrics from raw counts makes the weighting decisions auditable.

## Implementation contract

Work in `src/learner/api.py`. Implement **classification_metrics**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`classification_metrics`**

Given nonnegative square count matrix, return accuracy, macro_f1, per_class_f1 list.
Zero-denominator F1 is zero; macro includes ALL declared classes. Empty total accuracy=0.
Reject nonsquare/negative matrices.

F1 for class c is 2TP divided by predicted_count+true_count.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 7 --only
uv run course check 7
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 7 --level 1`.

## Explain and investigate

Build a classifier that gets high accuracy by ignoring a rare class.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
