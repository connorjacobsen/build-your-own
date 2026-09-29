# 14. Publish a finite immutable report

A machine-readable result is useful only if downstream consumers interpret it consistently. JSON NaN and infinity are nonstandard and can hide failed calculations. Canonical serialization and a file digest identify the exact evidence used by a release gate. Refusing overwrite preserves earlier reports for comparison.

## Implementation contract

Work in `src/learner/api.py`. Implement **write_report**. Earlier stages remain part of the cumulative gate. You may add private helpers; preserve the public signatures.

**`write_report`**

Serialize report as UTF-8 canonical JSON (sorted keys, compact separators,
ensure_ascii=False, allow_nan=False) followed by newline. Return exact-byte SHA256.
Refuse existing destination. Validate serialization before creating file.

This writes a local artifact only; it does not publish a model or call an external service.

## Work through it

Before coding, write down one ordinary input, one boundary input, and the state or tensor shapes at the output. Trace the ordinary case by hand. Then implement the smallest working version and run:

```bash
uv run course check 14 --only
uv run course check 14
```

The first command isolates this stage. The second verifies all stages through this one. A passing isolated test does not mark cumulative completion. Hints are available one level at a time with `uv run course hint 14 --level 1`.

## Explain and investigate

Trace how a report digest connects an evaluation to a model release.

Record your prediction before running the experiment, then explain any difference. Passing tests verifies the specified mechanics, not the quality of a model on arbitrary real-world data. The capstone and experiment journal ask you to connect those mechanics to measured behavior.

## Further reading

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)
