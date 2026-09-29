# Capstone: Adapt the checkpoint from pretraining

## Build

Load a trusted course 02 checkpoint and wrap it in the batched interface described in ARTIFACTS.md. Build coherent instruction examples, apply LoRA to selected linears, run supervised tuning, then a small DPO experiment with a fixed reference.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

Base and adapter digests, chat template, masks, trainable parameter count, optimization budget, held-out task and regression scores, and merged-model export.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Check frozen base tensors before and after training, merged/unmerged logits, wrong-base adapter rejection, and preferred-margin direction on a controlled pair.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Compare output-head adaptation with attention projection adaptation using matched budgets. Explain results with evidence rather than assuming more targets are always better.
