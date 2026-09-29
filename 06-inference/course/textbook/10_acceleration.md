> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 10 — Run on CUDA and measure honestly

**Build:** device portability in your model/cache and `metrics.benchmark`.  
**Prerequisite:** stage 09. **Estimated effort:** 3–6 hours; kernel extensions take longer.  
**Gate:** use the microstage number shown by `uv run course list`.

## Why the GPU appears here

You can learn causality, cache identity, block ownership, and scheduling using small CPU tensors. Those tests should stay fast and local while you edit. This stage changes the question: does the same engine work on a real CUDA device, and are its timing claims meaningful?

A GPU stage must execute on a GPU to count as complete. The gate fails with an actionable message when CUDA is absent; it does not award a green result for skipping every accelerator test. You can run CPU-only measurement examples during development, but that is not the full stage gate.

## Device contract

Your attention functions, model, and paged K/V pool must honor the requested device. Position vectors, masks, indexing tensors, and temporary buffers must be created on compatible devices. The output of CUDA attention remains on CUDA. The engine's K and V storage remain on CUDA throughout generation.

Sampling may intentionally move a logit vector to CPU and use its per-request CPU generator, as specified earlier. That is a known synchronization cost of this teaching design, not a claim of a fully asynchronous production runner. Python ownership checks can also introduce host/device synchronization.

The gate checks offset causal GQA on CUDA, then runs the paged engine and compares its generated IDs with a CPU oracle for fixed, well-separated fixture decisions. Kernel output tolerances are looser than CPU float32 tolerances. Numerical closeness does not guarantee every possible prompt has identical argmax behavior across devices; ties and near-ties require special care.

## Benchmark contract

Implement `benchmark(fn,device="cpu",warmup=2,repeats=5)`. Invoke `fn` for the specified warmups, then record one elapsed-seconds sample for each measured repetition. Return a dictionary with `samples_seconds` and its `median_seconds`. Reject negative warmup or fewer than one repeat with `ValueError`.

Use a monotonic high-resolution clock. For CUDA, synchronize the specified device immediately before starting the timer and after the work completes, before stopping the timer. For MPS, use its synchronization API. CPU work needs no device synchronization. Warmups are outside measured samples.

The grader spies on synchronization calls independently of GPU availability to verify timer ordering. The actual CUDA tests remain separate. Measuring only launch time can report an impressive number that omits most execution; the timer must bracket completed work.

## Running on Modal

The repository provides `scripts/modal_grade.py`, following Modal's [GPU pytest pattern](https://modal.com/docs/examples/ci-on-modal). It provisions a fixed L4 GPU, installs the locked environment, uploads only the course runner, grader, chapters, and your source, and executes the same cumulative gate. It does not upload your home directory, local credentials, notebook archive, or unrelated files.

Install the optional client and authenticate your own account once:

```bash
uv sync --locked --group gpu
uv run --group gpu modal setup
```

Then explicitly start a remote grading run:

```bash
uv run --group gpu modal run scripts/modal_grade.py --stage 10
```

This command uses your Modal account and can incur GPU charges. No remote run is started by local `course check`, imports, notebook execution, or the default sync. The supplied runner returns the test exit status and saves a source-fingerprinted receipt under `.course/`. It has a bounded runtime and maximum one container. Credentials stay with Modal's client environment.

The runner is optional: the same stage can execute on your own CUDA machine with `uv run course check 10`. Apple MPS is not CUDA and does not satisfy the CUDA gate. The SDK/image definition can be checked locally without claiming that remote execution occurred.

## Record a defensible comparison

After correctness passes, run the benchmark driver against your implementation:

```bash
uv run python scripts/benchmark.py --device cpu
# On a CUDA host:
uv run python scripts/benchmark.py --device cuda
```

It compares dense generation, contiguous caching, and a cold paged engine on identical weights and prompts. The benchmark calls your timing helper. It saves raw samples, versions, device identity, dtype, workload sizes, and timing boundaries under `artifacts/`. Output equality is checked before timing.

The cold paged case includes engine/pool creation; model construction is excluded from all cases. A long-lived warmed-engine benchmark would answer a different question. Label that boundary if you add it. Never combine a warm prefix hit in one method with a cold prompt in another and call it a pure kernel speedup.

No hardware-independent latency threshold is part of correctness grading. A tiny paged Python engine can be slower because gathers, bookkeeping, and synchronization outweigh saved work. Report that result rather than changing correctness criteria to make a chart look better.

## Understand which optimization comes next

A fused attention implementation can reduce intermediate memory traffic. PyTorch SDPA offers an optimized backend boundary, but its selected kernel depends on dtype, device, shapes, and masks. For a cached rectangular query, preserve the absolute-position mask; do not assume `is_causal=True` has the alignment you intend. SDPA boolean masks use `True` for allowed positions, and inference requires `dropout_p=0.0`. See [PyTorch 2.7 SDPA](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html).

A custom paged kernel reads through block tables inside the device kernel rather than gathering the entire history in Python. Online softmax maintains a common maximum, denominator, and weighted numerator across tiles. Independently softmaxing each page and averaging page outputs is mathematically wrong because it uses separate denominators.

CUDA Graphs reduce suitable launch overhead by replaying captured operations with stable memory addresses. Dynamic Python lists, CPU sampling, and variable shapes mean your whole engine cannot simply be wrapped in a capture context. Begin with a fixed-shape device subgraph, then design staging buffers and shape buckets. See [PyTorch's CUDA Graph notes](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs).

Neither a fused custom kernel nor whole-engine graph capture is required or falsely claimed as shipped by this stage. They are substantial follow-on builds, specified in [extensions](../../docs/extensions.md).

## Completion artifact

Submit your passing cumulative receipt and a benchmark report explaining: workload, correctness evidence, raw timing spread, memory capacity, known synchronizations, and the dominant measured overhead. Include a comparison with vLLM's richer execution design and a clear statement of what your engine does not implement.

Finishing this stage means you built the core components yourself and demonstrated their behavior under independent tests, including real GPU execution. That is a stronger outcome than having run a finished tutorial engine.
