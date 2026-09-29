# Capstone: Train a model you can later serve

## Build

Consume the JSONL training split from course 03. Start with a tiny repeated sequence to diagnose the loop, then train on a small self-authored corpus. Choose configuration and learning rate using validation loss. Export toylm-v1 and save a resume checkpoint separately.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Dataset digest, configuration, parameter count, training tokens, loss curves, held-out loss, checkpoint-resume equivalence and a portable export digest.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Interrupt training at an update boundary and compare the next update with uninterrupted execution. Load the exported artifact in course 06 and compare dense logits.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Add a budgeted scaling experiment over model width and tokens; extrapolation is an estimate, not a guarantee.
