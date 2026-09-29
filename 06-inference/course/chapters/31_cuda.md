# 31. Run causal attention on CUDA

This is a real NVIDIA CUDA hardware gate. Implement device-preserving behavior in the same model and cache code already tested on CPU. Tensor factories for positions, masks and physical KV storage must use the model's device. CPU numerical correctness remains a prerequisite.

Run `uv run course check 31` on a CUDA host or explicitly invoke the supplied Modal runner. Missing CUDA fails the gate; it cannot produce completion by skipping. Gate 31 checks attention with grouped KV heads and nonzero query offsets. Gate 32 runs the complete engine, compares output tokens with a CPU model, and verifies KV storage remains on the GPU.

Read [the measurement contract](../textbook/10_acceleration.md). Record hardware, dtype, model dimensions and repeated synchronized timing samples. A passing gate establishes numerical/device behavior, not a speedup guarantee. The toy gathered-attention implementation is deliberately easier to understand than a production paged kernel.
