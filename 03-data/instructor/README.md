# Instructor maintenance — contains solutions

Do not read this directory during the default learning path. Acceptance tests never fall back to this code.

```bash
uv run python -m pytest grader --implementation instructor.reference -q
```

This validates the reference against the independent grader. It does not complete learner stages and creates no learner completion receipt. Use a temporary copy for fault injection; never replace learner files. See the suite validation report for checks actually run and hardware limitations.
