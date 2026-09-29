# 26. Expose a shared HTTP engine

Validate requests, return distinct IDs and send generation through one engine whose lifetime belongs to the application.

## Scope for this gate

Work in `src/toyvllm/server.py: create_app`. Implement only the behavior described here and in the corresponding technical contract. Earlier cumulative gates must keep passing. This smaller gate is part of original textbook milestone 9; messages inside the preserved learner scaffold still use those original milestone numbers.

Read [the full technical contract](../textbook/09_serving.md) for exact shapes, lifecycle rules and examples. `course read 26` also prints that contract below this stage introduction.

## Verify

```bash
uv run course check 26 --only
uv run course check 26
uv run course hint 26 --level 1
```

The grader checks observable invariants, including resource ownership and actual execution work where appropriate. Output agreement alone does not establish that an optimization is implemented. Before changing code, draw a minimal trace and predict what the gate should observe. Record the result and one remaining limitation in your experiment journal.

## Explain before moving on

Which invariant would a plausible but incorrect shortcut violate? Give a concrete input exposing that failure. Identify which state belongs to the caller, the request, or the shared engine. Explain why the next stage can safely build on this one.
