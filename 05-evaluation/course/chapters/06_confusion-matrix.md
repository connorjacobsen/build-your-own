# 06. Count classification outcomes

A single accuracy number hides which classes are confused. The confusion matrix records truth on rows and predictions on columns, a convention that must remain consistent downstream. Empty declared classes still belong in the matrix. Fixed class dimensions make comparisons across runs meaningful even when a small sample omits a class.

## Implementation contract

Work in `src/learner/api.py`. Implement **confusion_matrix**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`confusion_matrix`**

Return int64 NumPy [C,C] counts, rows truth and columns prediction. Equal lengths,
classes>=1, IDs in [0,C). Empty inputs produce zeros. Invalid values raise ValueError.

Reject out-of-range IDs instead of resizing the matrix silently.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 6 --only
uv run course check 6
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 6 --level 1`.

## Explain and investigate

Which axis would you sum to compute predicted class frequency?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
