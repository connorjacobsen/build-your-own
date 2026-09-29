# Capstone: Compare models without changing the question

## Build

Define a small task with stable case IDs and multiple acceptable references where needed. Compare the base and adapted models on identical held-out cases. Select the normalization and decoding policy before collecting final results.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Per-case predictions and scores, model/dataset digests, paired interval, slice counts, failure examples and a canonical toyeval-v1 report.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Shuffle output order and confirm identical comparisons; remove a case and confirm rejection; deliberately change continuation offsets and observe a regression.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Add a second metric and explain where it disagrees with exact match. Treat any model-based judge as another fallible measurement system.
