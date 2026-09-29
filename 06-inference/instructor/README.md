# Instructor maintenance — not the learner workflow

The learner builds `src/toyvllm/` and uses `uv run course check N`. Do not substitute
reference imports for learner work or reveal complete solutions in routine hints.

The oracle now lives under `grader/oracle/` so acceptance tests can load fixed weights
and compute independent expected results. It is not a public learner dependency.
Original notebook materials were preserved under `archive/` before conversion.

Maintenance checks:

```bash
uv run pytest grader --implementation grader.oracle -m 'not gpu' -q
uv run pytest instructor/tests -q
uv run ruff check src/toyvllm grader course_cli instructor/tests scripts
```

The first command validates CPU acceptance cases, not learner progress. The second
runs reference regressions and mutant rejection checks. GPU gates require a real
CUDA host. Do not change absence of CUDA to a passing skip.

When changing a contract, update the chapter, scaffolds, tests, and hints coherently.
Preserve current learner implementations. The historical validation documents in
the archive describe the previous completed notebook course, not this unfinished
learner package. The current validation record is `docs/validation.md`.
