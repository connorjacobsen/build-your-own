# Capstone: Recover and scale a tiny training update

## Build

Match a global-batch update across two CPU ranks, then save each rank’s momentum shard plus step, range, dtype and artifact identity metadata. Reconstruct and repartition state for a different world size in a controlled experiment.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Single-process/distributed parameter comparisons, local state sizes, checkpoint manifest, missing-shard failure, and a resumed update comparison.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Inject a nonfinite gradient on one rank and verify a unanimous skipped update. Check that no rank changes weights or optimizer state when the update is skipped.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Run the two-GPU gate, then measure communication and compute separately. The supplied stages teach components; assembling a crash-recovery orchestrator is a capstone integration task.
