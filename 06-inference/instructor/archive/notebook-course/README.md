# Build a tiny vLLM — an executable course

**15 Jupyter notebooks · 11 graded build milestones · a working PyTorch inference engine · uv-managed environment**

Build a small transformer, then turn it into a multi-request inference engine with KV caching, packed execution, paged storage, continuous scheduling, prefix reuse, and a local HTTP service. The course includes derivations, runnable experiments, plots, failure cases, learner scaffolds, worked solutions, tests, benchmarks, and a capstone.

The core runs on **CPU, including Apple Silicon**, with no model downloads or GPU requirement. Optional MPS/CUDA experiments come later. Assume ordinary Python knowledge; tensor shapes and transformer components are introduced before the systems work. Allow **25–40 hours**, with more time for extension projects.

## Start here

From this folder:

```bash
uv sync --locked
uv run jupyter lab --ServerApp.root_dir=. notebooks/00_start_here.ipynb
```

If `uv` is not installed, follow [uv's installation instructions](https://docs.astral.sh/uv/getting-started/installation/). The first sync requires internet to install dependencies; core lessons need no network afterward. Python 3.11 is selected by `.python-version`; uv can provision it if necessary. The locked environment has been exercised on this Apple Silicon Mac. Linux/Windows and CUDA-specific execution have not been verified here.

Open [lesson 00](notebooks/00_start_here.ipynb) and use **Restart Kernel and Run All**. Each notebook is independent. There is no need to run prior notebooks to create hidden variables or download weights.

## What you will build

```text
caller → tokenizer → waiting queue → admission + token budget
                                            ↓
                              packed tokens / positions / block tables
                                            ↓
                              tiny transformer + physical KV pool
                                            ↓
                                  logits → sampler → token events
                                            ↓
                                finish/cancel → release → admit more
```

The included model is a real, small decoder with RMSNorm, RoPE, grouped-query attention, SwiGLU, and causal masking. It has **random weights by default**, so its generated text is not meaningful language. Lesson 02 trains a small pattern locally to explain the difference between learning weights and serving them.

The implementation demonstrates inference mechanics. Its paged attention gathers K/V into ordinary PyTorch tensors, and its scheduler reserves worst-case capacity at admission. It does not claim vLLM's performance, dynamic allocation/preemption policy, pretrained-model compatibility, or production API coverage. These differences are part of the lessons.

## Course map

| Lesson | Topic | Build or experiment | Time |
|---|---|---|---|
| [00](notebooks/00_start_here.ipynb) | Environment and engine tour | Verify the kernel; run token events | 45–60 min |
| [01](notebooks/01_tokens_and_tensors.ipynb) | Tokens, tensors, attention | Offset causal attention; KV memory formula | 1.5–2 h |
| [02](notebooks/02_tiny_transformer.ipynb) | Tiny transformer | RoPE, RMSNorm, GQA, SwiGLU; local training | 2–3 h |
| [03](notebooks/03_generation_and_sampling.ipynb) | Autoregressive generation | Greedy and nucleus sampling; per-request RNG | 1.5–2 h |
| [04](notebooks/04_kv_cache.ipynb) | Contiguous KV cache | Cached generation; dense/chunk equivalence | 2–3 h |
| [05](notebooks/05_batching.ipynb) | Packing and batching | Token packing, boundaries, static slot waste | 1.5–2 h |
| [06](notebooks/06_paged_memory.ipynb) | Paged allocation | Logical/physical mapping, free lists, refs | 2–3 h |
| [07](notebooks/07_paged_attention.ipynb) | Paged execution | Real paged K/V, shuffled pages, online softmax | 2–3 h |
| [08](notebooks/08_scheduler.ipynb) | Continuous scheduling | Chunked prefill, budgets, admission, cancellation | 3–4 h |
| [09](notebooks/09_prefix_caching.ipynb) | Prefix sharing | Full-block reuse, eviction, copy on write | 2–3 h |
| [10](notebooks/10_serving.ipynb) | Local serving | Shared-engine HTTP API and concurrent clients | 2 h |
| [11](notebooks/11_benchmarking.ipynb) | Measurement | TTFT, ITL, throughput, parameter sweeps | 2–3 h |
| [12](notebooks/12_accelerators.ipynb) | Accelerators | SDPA, MPS/CUDA equivalence, optional CUDA Graph | 2–4 h |
| [13](notebooks/13_capstone.ipynb) | Capstone | Assemble and grade your own engine | 4–8 h |
| [14](notebooks/14_vllm_source_and_extensions.ipynb) | Upstream and extensions | Source trace, tensor parallelism, speculation | 2–4 h |

Time estimates overlap when you build the capstone incrementally. Completing every optional extension will take longer than the core estimate.

### A six-week route

1. **Foundations:** 00–02; write attention and RoPE, explain all tensor shapes.
2. **Reuse:** 03–05; implement sampling, cached generation, and packing.
3. **Memory:** 06–07; implement pages and connect them to real model execution.
4. **Scheduling:** 08–09; manage lifecycle, memory pressure, and prefix sharing.
5. **Serving and evidence:** 10–12; run concurrent clients, benchmark, inspect hardware limits.
6. **Capstone and upstream:** 13–14; assemble your engine and trace a request in vLLM.

If you already know transformers, run 01–03 and pass their exercises before moving directly to lesson 04.

## Learn by implementing

Write your answers in [exercises/implementation.py](exercises/implementation.py). The capstone scaffold is [exercises/starter_engine.py](exercises/starter_engine.py). Reference examples run with `RUN_EXERCISES=False`; skipped learner exercises are clearly reported and are not counted as passing.

```bash
uv run python scripts/check_exercises.py attention memory
uv run python scripts/check_exercises.py rope sampling decode
uv run python scripts/check_exercises.py packing slots paged scheduler prefix
uv run python scripts/check_exercises.py engine
```

After implementing a lesson's named functions, set `RUN_EXERCISES=True` in that notebook to run its checks. Restart the kernel after editing imported Python modules. See [the exercise guide](docs/exercises.md) for contracts, hints, and completion criteria. Worked solutions are in [exercises/reference_solutions.py](exercises/reference_solutions.py); the finished engine is in [src/nanovllm_course](src/nanovllm_course).

## Run and verify

```bash
# Short multi-request demo
uv run python scripts/demo.py

# Engine regression tests
uv run pytest -q

# Check supplied solutions, not your unfinished exercises
uv run python scripts/check_exercises.py --reference

# Execute every notebook in a separate fresh kernel
uv run python scripts/execute_notebooks.py

# Or just one lesson (quote the glob)
uv run python scripts/execute_notebooks.py '09*'

# Save timing samples and environment metadata
uv run python scripts/benchmark.py
```

Executed notebooks and reports are saved under `artifacts/`, leaving the learner notebooks untouched. The delivered verification record is [docs/validation.md](docs/validation.md). Runtime artifacts are ignored by Git.

### Local HTTP service

```bash
uv run uvicorn nanovllm_course.server:create_app --factory --host 127.0.0.1 --port 8000
```

Then visit [the local API docs](http://127.0.0.1:8000/docs), or send:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_new_tokens":8}'
```

This is a custom local endpoint with complete JSON responses. It is not OpenAI API compatible, and HTTP streaming is an extension exercise.

## vLLM source basis

I inspected the V1 scheduler, KV manager, block pool, request state, engine core, GPU model runner, sampler, Llama model, and design documents from **vLLM v0.11.0**, pinned at [`b8b302cde434df8c9289a2b465406b47ebab1c2d`](https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d). This is an intentionally fixed reading baseline, not the latest-release claim.

[The source map](docs/source_map.md) links functions to lessons and spells out differences. [The upstream manifest](docs/upstream_manifest.json) records URLs and SHA-256 hashes. To fetch the small reading set locally:

```bash
uv run python scripts/fetch_reference.py
```

This writes into `.reference/`; it does not install or build vLLM. The course code is an original educational implementation, not a vendored vLLM fork and not affiliated with the vLLM project or the separate nano-vLLM project.

## Project layout

```text
notebooks/                 15 learner-facing notebooks
course/lessons/             Markdown authoring sources for the notebooks
src/nanovllm_course/        finished tokenizer, model, cache, engine, sampler, metrics, server
exercises/                 learner stubs, engine scaffold, checkers, worked solutions
tests/                     numerical and lifecycle regression tests
scripts/                   demo, benchmark, notebook execution/build, reference fetch
docs/                      source map, invariants, glossary, troubleshooting, validation
artifacts/                 generated execution evidence and benchmark results
pyproject.toml + uv.lock    project environment and locked dependencies
```

The notebooks are generated from Markdown authoring files by `scripts/build_notebooks.py`. **Do not regenerate notebooks after editing them without saving your work**: regeneration intentionally replaces their contents. For ordinary learning, edit the exercise Python files and use copies of notebooks for personal notes.

For help, see [troubleshooting](docs/troubleshooting.md), [invariants](docs/invariants.md), and [the glossary](docs/glossary.md).
