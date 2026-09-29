# Build a model evaluation harness

Learn to make and audit model-quality claims through fourteen small stages. Implement likelihood scoring, classification metrics, paired uncertainty, slices, calibration, latency summaries and artifact-backed reports. Statistical evidence remains distinct from passing software tests.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/05-evaluation
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Specify answer normalization](course/chapters/01_normalize-answer.md) | `normalize_answer` |
| 2 | [Score acceptable references](course/chapters/02_exact-match.md) | `exact_match` |
| 3 | [Score a continuation at the right offset](course/chapters/03_continuation-logp.md) | `continuation_logp` |
| 4 | [Make length normalization explicit](course/chapters/04_rank-choices.md) | `rank_choices` |
| 5 | [Aggregate perplexity by token count](course/chapters/05_perplexity.md) | `perplexity` |
| 6 | [Count classification outcomes](course/chapters/06_confusion-matrix.md) | `confusion_matrix` |
| 7 | [Distinguish macro and micro behavior](course/chapters/07_classification-metrics.md) | `classification_metrics` |
| 8 | [Estimate uncertainty with paired resampling](course/chapters/08_paired-bootstrap.md) | `paired_bootstrap` |
| 9 | [Align comparisons by identity](course/chapters/09_compare-by-id.md) | `compare_by_id` |
| 10 | [Inspect performance slices](course/chapters/10_slice-metrics.md) | `slice_metrics` |
| 11 | [Measure calibration](course/chapters/11_calibration-error.md) | `calibration_error` |
| 12 | [Measure user-visible generation latency](course/chapters/12_latency-metrics.md) | `latency_metrics` |
| 13 | [Run an auditable evaluation](course/chapters/13_evaluate.md) | `evaluate` |
| 14 | [Publish a finite immutable report](course/chapters/14_write-report.md) | `write_report` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [CS336 evaluation curriculum](https://cs336.stanford.edu/)
- [ARENA evaluations](https://www.arena.education/curriculum)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
