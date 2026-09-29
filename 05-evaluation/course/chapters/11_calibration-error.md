# 11. Measure calibration

Confidence and correctness are different quantities. Calibration asks whether outcomes occur at the rate the confidence predicts. Binned ECE is an approximation whose value depends on binning choices. Internal boundaries, confidence one and empty bins need explicit conventions. A low ECE does not by itself imply a model is accurate or useful.

## Implementation contract

Work in `src/learner/api.py`. Implement **calibration_error**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`calibration_error`**

Equal nonempty 1D lists; confidence finite in [0,1], bins>=1. ECE weighted absolute
gap between mean confidence and accuracy within equal-width bins. Internal boundaries
belong to the higher bin; confidence=1 belongs to final bin. Empty bins contribute zero.

Use confidence in the predicted answer, not arbitrary raw logits.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 11 --only
uv run course check 11
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 11 --level 1`.

## Explain and investigate

Can a constant-confidence model be calibrated but uninformative?

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
