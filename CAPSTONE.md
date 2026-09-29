# Own one model from data to rollback

The eight courses become one project when the artifacts cross their boundaries. Build the components first; use this capstone to understand the interfaces, failure modes and evidence needed to operate them together.

## Required path

1. Prepare a small, licensed or self-authored corpus in course 03. Keep duplicates in one split, document filtering decisions and export the dataset manifest. Reserve an evaluation set before optimizing the model.
2. Train course 02's decoder on the training split. Overfit a tiny diagnostic subset first, then measure held-out loss. Export weights with configuration, tokenizer identity and data provenance. Save a resumable checkpoint separately.
3. Build the evaluation protocol with course 05. Choose metrics and case identities before choosing a favorite model. Store predictions and per-case scores, not only a headline number.
4. Adapt the model with course 04. Compare a frozen base, supervised fine-tuning and a small preference experiment. Check which parameters changed. Save adapters bound to the base digest, merge for serving and export a new model artifact.
5. Load the checkpoint through course 06's artifact loader. Verify dense logits before comparing cached and paged generation. Serve concurrent callers through one shared engine and inspect actual cache ownership and projected-row counts.
6. Use course 07 to reproduce a single-process update across two real processes. Then test optimizer-state sharding and recovery from a set of saved shards. Reserve GPU performance claims for actual accelerator measurements.
7. Register the model and evidence in course 08. Apply a predeclared release policy, simulate a stable canary cohort, inject failures, and perform a compare-and-swap rollback. Explain what remains necessary before replacing the local registry with a production control plane.

Course 01 supplies the conceptual foundation for gradients rather than a mandatory runtime dependency of the PyTorch trainer. Explain a training failure using the computational graph, and compare one small layer's gradients between your autodiff engine and PyTorch.

## Provided integration driver

After the relevant stages are implemented:

```bash
cd ~/Code/ai/courses/02-pretraining
uv run python ../tools/lifecycle.py --output /tmp/my-model-lifecycle
```

This loads each course by an explicit path, avoiding collisions between their independent `learner` packages. It prepares a tiny dataset, trains, exports, adapts, merges, checks training-to-serving logits and paged generation, evaluates, promotes locally and rolls back. It writes a summary plus the actual artifacts. Existing output directories are refused.

The fixture deliberately uses tiny data and a permissive release policy to test connectivity. **Its result is not evidence that a model is useful or improved.** In particular, the supervised chat task and pretraining text task need not improve each other. A useful experiment chooses a task, template, data distribution, held-out cases and policy coherently. Replace the smoke workload through your own experiment driver; do not weaken acceptance tests.

`--reference` is an instructor-only maintenance option outside the learning CLI. It invokes complete reference implementations to test course compatibility, never marks stages completed, and must not be used as a substitute for your implementation.

## Evidence to deliver

Write an experiment report with the dataset and model digests, exact configurations, seeds, compute budget, training curves, validation selection rule, final held-out metrics and paired uncertainty where appropriate. Include representative failures and subgroup counts. Keep a separate systems table containing hardware, dtype, workload, warmup, timing samples, throughput, latency and memory.

Include a replayable artifact handoff and a rollback trace. Make explicit which observations come from deterministic tests, statistical experiments, CPU processes or real GPUs. The strongest outcome is a system whose failures you can explain and whose results another person can reproduce.
