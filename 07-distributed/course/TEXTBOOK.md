# Preserve the objective while changing where work happens

Distributed training should initially compute the same update as a single-process global batch. Only after establishing that equivalence should you optimize communication, memory or throughput. Many failures are weighting and ownership errors rather than exotic network problems.

## A rank-mean trap

Rank 0 sees one example with gradient 10. Rank 1 sees three examples whose mean gradient is 2. The global mean gradient is (1*10+3*2)/4=4. Averaging rank means gives 6. Dividing by world size is valid only under equal local denominators. For language models, the denominator may be supervised tokens rather than examples.

All ranks must execute collectives in compatible order. A rank with no examples still contributes zeros and participates. If it returns early, another rank may wait forever. The CPU grader launches actual Gloo processes and includes an empty-rank case to expose this failure.

## State placement

Parameters, gradients, optimizer moments and activations are different memory categories. Replicated data parallelism duplicates parameters and optimizer state while splitting examples. The educational sharded momentum step stores only one rank's optimizer slice but still replicates parameters and gradients. Do not label that full parameter sharding.

Flattening gives tensors a shared communication layout. Gather reconstruction must use the same order, shape metadata and balanced ownership ranges. Padding communication payloads is a transport detail; padded values must not become model parameters.

## Precision and recovery

Mixed precision introduces numerical concerns independent of communication. Loss scaling can prevent small gradients from underflowing, but overflow must cause a consistent skipped update across all ranks. Momentum buffers and counters must follow the same decision as weights.

Activation checkpointing trades saved intermediates for recomputation. Disk checkpointing saves persistent training state. Their names are similar but their purposes and correctness tests differ. Sharded disk recovery requires all pieces to refer to the same model and update, with exact range coverage before repartitioning.

## Hardware claims

CPU process tests validate collective semantics. They cannot establish NCCL behavior, GPU memory savings or scaling speedups. The final gate requires two CUDA devices and compares their update against a CPU reference. A benchmark additionally needs synchronized timing, identical workloads, warmup and multiple measurements. Faster execution that changes the objective or drops work is not a valid improvement.
