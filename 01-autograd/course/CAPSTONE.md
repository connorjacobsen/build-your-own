# Capstone: Train and diagnose a nonlinear regressor

## Build

Use train_mlp on y=x*x over a bounded interval. Split inputs before tuning width or learning rate. Compare hidden widths 4, 8 and 16 using identical train/validation examples and at least three initialization seeds.

Complete the relevant acceptance stages first. Keep implementation changes in learner-owned source and experiment drivers. Do not modify acceptance tests to fit a preferred result.

## Evidence to produce

A gradient-check report on a smooth composed expression, train/validation MSE curves, final parameters and an explanation of one failed learning-rate choice.

Record hypothesis, configuration, seed, input identity, measurements and limitations in EXPERIMENTS.md. A failed hypothesis is a valid learning result if the experiment and explanation are sound. Separate debugging examples from validation choices and final held-out evidence.

## Integration checks

Compare gradients of your network with an independently constructed PyTorch network loaded with identical arrays. Do not import the instructor engine.

These checks complement the automated stage gates. The capstone is an experimental report, not an automatic certification of model quality. Use the suite [artifact contract](../../ARTIFACTS.md) to connect outputs with other courses and the [lifecycle driver](../../CAPSTONE.md) for a provided smoke test.

## Extension after the core project

Add a nonlinear classification task; implement additional operations only after documenting their derivatives and shape domains.
