# 22. Emit truthful token events

Return exactly one event per actual new output. Partial work must not pretend to have produced a token.

## Scope for this gate

Work in `src/toyvllm/engine.py: Engine.step`. Implement only the behavior described here and in the corresponding technical contract. Earlier cumulative gates must keep passing. This smaller gate is part of original textbook milestone 7; messages inside the preserved learner scaffold still use those original milestone numbers.

Read [the full technical contract](../textbook/07_scheduler.md) for exact shapes, lifecycle rules and examples. `course read 22` also prints that contract below this stage introduction.

## Verify

```bash
uv run course check 22 --only
uv run course check 22
uv run course hint 22 --level 1
```

The grader checks observable invariants, including resource ownership and actual execution work where appropriate. Output agreement alone does not establish that an optimization is implemented. Before changing code, draw a minimal trace and predict what the gate should observe. Record the result and one remaining limitation in your experiment journal.

## Explain before moving on

Which invariant would a plausible but incorrect shortcut violate? Give a concrete input exposing that failure. Identify which state belongs to the caller, the request, or the shared engine. Explain why the next stage can safely build on this one.
