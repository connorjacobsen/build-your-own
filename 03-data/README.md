# Build a training-data pipeline

Construct reproducible, leakage-aware datasets in thirteen stages. Work on tiny local records so you can inspect every filtering and splitting decision. No web crawl or external dataset download is required.

This is a build-first course: you implement the unfinished `src/learner/api.py`, read the chapters, and run independently authored tests. There is no default reference fallback. Read [the suite guide](../README.md) for the learning order and artifact handoffs.

Read [the conceptual foundation](course/TEXTBOOK.md) and [API reference](course/API.md), then work stage by stage.

## Start

```bash
cd ~/Code/ai/courses/03-data
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The initial failure is intentional. Implement one small stage, use conceptual hints when stuck, then run the cumulative check. Foundations use CPU and tiny deterministic data; nothing downloads a model or starts paid compute. See [the capstone](course/CAPSTONE.md) for an applied experiment after the acceptance stages.

## Stages

| Stage | Chapter | Implement |
|---|---|---|
| 1 | [Separate matching text from training text](course/chapters/01_normalize.md) | `normalize` |
| 2 | [Give content a stable identity](course/chapters/02_document-id.md) | `document_id` |
| 3 | [Make filtering explainable](course/chapters/03_quality-filter.md) | `quality_filter` |
| 4 | [Remove exact duplicates stably](course/chapters/04_deduplicate.md) | `deduplicate` |
| 5 | [Represent local overlap](course/chapters/05_shingles.md) | `shingles` |
| 6 | [Group transitive near duplicates](course/chapters/06_duplicate-components.md) | `duplicate_components` |
| 7 | [Split groups, not rows](course/chapters/07_split-groups.md) | `split_groups` |
| 8 | [Decontaminate against held-out content](course/chapters/08_decontaminate.md) | `decontaminate` |
| 9 | [Sample a reproducible mixture](course/chapters/09_sample-mixture.md) | `sample_mixture` |
| 10 | [Pack without erasing boundaries](course/chapters/10_pack-documents.md) | `pack_documents` |
| 11 | [Build isolated causal visibility](course/chapters/11_attention-mask.md) | `attention_mask` |
| 12 | [Write an immutable dataset artifact](course/chapters/12_write-dataset.md) | `write_dataset` |
| 13 | [Integrate the preparation pipeline](course/chapters/13_prepare-dataset.md) | `prepare_dataset` |

## Learning loop

`course check N` runs stages 1–N. `--only` is a debugging aid; `--seed 91` varies seeded cases. `course status` reports the latest run and detects source changes. Any skipped test prevents completion. The grader owns expected answers and tests invariants, edge cases, and integration. Local tests are readable, not secure hidden tests. Keep `instructor/` out of your normal reading path; it contains worked references for course maintenance.

Use [EXPERIMENTS.md](course/EXPERIMENTS.md) to record predictions, measurements and limitations. Keep holdout data untouched while making choices. GPU timing and model quality are separate from unit correctness. A passing toy exercise does not establish production readiness.

## Sources

- [CS336 data curriculum](https://cs336.stanford.edu/)
- [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html)

The curriculum and tests here are original teaching material inspired by these concepts; they are not university assignments or official product certifications.
