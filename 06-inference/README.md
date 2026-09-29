# Build a tiny vLLM inference engine

The original course is preserved at `~/Code/ai/nanovllm`. This independent copy preserves its learner code and acceptance cases, and splits ten large milestones into **32 smaller gates**. It adds a portable trained-checkpoint loader shared with course 02. Nothing was moved out of the original directory.

```bash
cd ~/Code/ai/courses/06-inference
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

Implement `src/toyvllm/`. The chapters explain the concepts; the grader supplies expected behavior and tests real execution work. The local grader is inspectable, not a hidden-test service. Complete reference implementations remain under `grader/oracle/` for instructor validation; do not use them as learner implementations.

Stages 1–30 run on CPU. Stages 31–32 require actual NVIDIA CUDA. The optional Modal runner is explicit and incurs provider compute charges when invoked. See [GPU instructions](../GPU.md). Existing scaffold error messages and archived notebooks retain original milestone numbers; follow `course list` for the new gates.

| Stage | Build / verify |
|---|---|
| 1 | [UTF-8 and special tokens](course/chapters/01_stage.md) |
| 2 | [Round trips and invalid bytes](course/chapters/02_stage.md) |
| 3 | [RMS normalization](course/chapters/03_stage.md) |
| 4 | [Rotary positions](course/chapters/04_stage.md) |
| 5 | [Causal grouped-query attention](course/chapters/05_stage.md) |
| 6 | [Assemble the decoder](course/chapters/06_stage.md) |
| 7 | [Prove causality and gradient flow](course/chapters/07_stage.md) |
| 8 | [Greedy generation and EOS](course/chapters/08_stage.md) |
| 9 | [Nucleus sampling](course/chapters/09_stage.md) |
| 10 | [Own each request random stream](course/chapters/10_stage.md) |
| 11 | [Append a contiguous KV cache](course/chapters/11_stage.md) |
| 12 | [Generate with real cache reuse](course/chapters/12_stage.md) |
| 13 | [Batch isolated contexts](course/chapters/13_stage.md) |
| 14 | [Pack the linear work](course/chapters/14_stage.md) |
| 15 | [Allocate pages atomically](course/chapters/15_stage.md) |
| 16 | [Map logical slots to physical storage](course/chapters/16_stage.md) |
| 17 | [Execute from real paged KV](course/chapters/17_stage.md) |
| 18 | [Schedule partial prefills](course/chapters/18_stage.md) |
| 19 | [Handle terminal request states](course/chapters/19_stage.md) |
| 20 | [Preserve stochastic request semantics](course/chapters/20_stage.md) |
| 21 | [Make progress under capacity pressure](course/chapters/21_stage.md) |
| 22 | [Emit truthful token events](course/chapters/22_stage.md) |
| 23 | [Retain and evict full prefixes](course/chapters/23_stage.md) |
| 24 | [Copy a shared tail before writing](course/chapters/24_stage.md) |
| 25 | [Reuse prefixes in the engine](course/chapters/25_stage.md) |
| 26 | [Expose a shared HTTP engine](course/chapters/26_stage.md) |
| 27 | [Serve concurrent callers](course/chapters/27_stage.md) |
| 28 | [Bound pending work and cancellation](course/chapters/28_stage.md) |
| 29 | [Measure completed device work](course/chapters/29_stage.md) |
| 30 | [Load a trained portable artifact](course/chapters/30_artifact.md) |
| 31 | [Run causal attention on CUDA](course/chapters/31_cuda.md) |
| 32 | [Run the complete engine on CUDA](course/chapters/32_cuda.md) |

The ten [textbook chapters](course/textbook) retain the detailed mathematical and systems contracts. `course read` prints the relevant textbook alongside each microstage. The architecture uses a tiny random-weight decoder, and becomes a trained model when you import course 02's artifact. Paging gathers ordinary PyTorch attention; it is a correctness-oriented teaching implementation, not vLLM performance equivalence.

Read [the capstone](course/CAPSTONE.md) and [artifact contract](../ARTIFACTS.md) to connect training, evaluation and serving. The original optional notebooks remain supporting visual experiments. The default learning path is code, contracts and tests.
