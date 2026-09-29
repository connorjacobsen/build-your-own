# Capstone: Prepare and audit a small corpus

## Build

Create at least 50 self-authored records including exact duplicates, lightly edited copies, repeated spam, Unicode variants and reserved evaluation examples. Predict each filter outcome before running preparation.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Raw records, a rejection audit, normalized-content IDs, connected duplicate groups, split assignments, token counts and toydata-v1 manifest.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Verify no duplicate component crosses splits, reserved exact content is absent, and rerunning into a new directory produces the same JSONL bytes.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Compare two normalization or shingle policies and manually inspect disagreements. Report which source types each policy disproportionately removes.
