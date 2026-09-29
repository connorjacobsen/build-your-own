# Capstone: Release and roll back a model locally

## Build

Use actual model artifacts and toyeval-v1 reports from earlier courses. Declare a quality floor, regression allowance, minimum evidence count and monitoring policy before trying candidates. Simulate a canary workload.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Verified bundle manifest, object digests, gate reasons, active-pointer history, concurrent-promotion conflict and rollback trace.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Reject a report for another model, corrupt one byte, inject a latency spike, and race two promotions. Confirm the intended prior state survives each failed operation.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Connect a local server’s model selection to the pointer while preserving in-flight request ownership. Live deployment is a separate explicitly authorized project.
