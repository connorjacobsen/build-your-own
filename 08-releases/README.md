# Build a model release and monitoring system

Fourteen stages turn model artifacts and evaluation reports into a local reproducible release workflow. Build integrity checks, release gates, canary routing, monitoring decisions and concurrency-safe promotion/rollback. All operations stay local; no production service is changed.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/08-releases
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Make metadata byte-stable](course/chapters/01_canonical-json.md) | `canonical_json` |
| 2 | [Hash large artifacts incrementally](course/chapters/02_file-digest.md) | `file_digest` |
| 3 | [Bound artifact paths](course/chapters/03_safe-artifact-path.md) | `safe_artifact_path` |
| 4 | [Describe an explicit bundle](course/chapters/04_build-manifest.md) | `build_manifest` |
| 5 | [Verify integrity before use](course/chapters/05_verify-manifest.md) | `verify_manifest` |
| 6 | [Register immutable content](course/chapters/06_register-artifact.md) | `register_artifact` |
| 7 | [Validate evaluation evidence](course/chapters/07_validate-evaluation.md) | `validate_evaluation` |
| 8 | [Encode a release policy](course/chapters/08_release-gate.md) | `release_gate` |
| 9 | [Route a stable canary cohort](course/chapters/09_route-request.md) | `route_request` |
| 10 | [Summarize serving observations](course/chapters/10_summarize-requests.md) | `summarize_requests` |
| 11 | [Decide when to roll back](course/chapters/11_rollback-decision.md) | `rollback_decision` |
| 12 | [Promote with compare-and-swap](course/chapters/12_promote.md) | `promote` |
| 13 | [Restore the previous release](course/chapters/13_rollback.md) | `rollback` |
| 14 | [Release from verified evidence](course/chapters/14_release-model.md) | `release_model` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/)
- [Python os.replace](https://docs.python.org/3/library/os.html#os.replace)
- [Python file locks](https://docs.python.org/3/library/fcntl.html)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
