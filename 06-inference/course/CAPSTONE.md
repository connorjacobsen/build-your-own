# Capstone: Serve the model you trained

## Build

Complete CPU gates, load a toylm-v1 checkpoint from course 02 or merged course 04 output, then compare dense, cached and paged generation on the same requests. Add concurrent clients and cancellation.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Checkpoint digest, numerical equivalence, projected-row counts, page ownership traces, latency samples and a cancellation/resource-cleanup demonstration.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Run the cumulative CPU gate through stage 30. For GPU claims run stages 31–32 on real CUDA. Compare identical outputs and workload under each execution path.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Implement a specialized paged kernel or quantization as a separate experiment after correctness, and report error and performance rather than only speed.
