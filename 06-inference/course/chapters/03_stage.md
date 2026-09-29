# 03. RMS normalization

Build learned root-mean-square normalization and make an all-zero vector finite.

## Scope for this gate

Work in `src/toyvllm/model.py: RMSNorm`. Implement only the behavior described here and in the corresponding technical contract. Earlier cumulative gates must keep passing. This smaller gate is part of original textbook milestone 2; messages inside the preserved learner scaffold still use those original milestone numbers.

Read [the full technical contract](../textbook/02_model.md) for exact shapes, lifecycle rules and examples. `course read 3` also prints that contract below this stage introduction.

## Verify

```bash
uv run course check 3 --only
uv run course check 3
uv run course hint 3 --level 1
```

The grader checks observable invariants, including resource ownership and actual execution work where appropriate. Output agreement alone does not establish that an optimization is implemented. Before changing code, draw a minimal trace and predict what the gate should observe. Record the result and one remaining limitation in your experiment journal.

## Explain before moving on

Which invariant would a plausible but incorrect shortcut violate? Give a concrete input exposing that failure. Identify which state belongs to the caller, the request, or the shared engine. Explain why the next stage can safely build on this one.
