# Validation record

Validated September 27, 2026, on this local Apple Silicon machine.

- Platform: `macOS-15.7.7-arm64-arm-64bit`
- Python: `3.11.13`; PyTorch: `2.7.1`
- Core device/dtype: CPU / float32, one PyTorch thread.
- Environment: `uv sync --locked`; resolved versions are in `uv.lock`.
- Regression tests: **23 passed** (`uv run pytest -q`).
- Reference exercise checks: **11 passed** (`uv run python scripts/check_exercises.py --reference`).
- Ruff: lint and formatting checks passed for source, tests, exercises, and scripts.
- Fresh-kernel notebook execution: **15/15 passed**, 26.01 seconds total including kernel startup.
- Learner exercises remain deliberately unimplemented and are skipped by default in notebooks.
- MPS float32 model comparison executed: maximum logit difference from CPU approximately `8.34e-7`.
- CUDA was unavailable: the CUDA Graph branch was skipped and is **not claimed as tested**.
- In-memory HTTP serving checks covered health, generation, zero output, invalid limits, long prompts, and four concurrent clients.
- Representative scheduler and benchmark plots were visually inspected.

## Notebook evidence

| Notebook | Result | Execution seconds |
|---|---|---:|
| 00_start_here.ipynb | Passed | 3.29 |
| 01_tokens_and_tensors.ipynb | Passed | 1.50 |
| 02_tiny_transformer.ipynb | Passed | 3.83 |
| 03_generation_and_sampling.ipynb | Passed | 1.49 |
| 04_kv_cache.ipynb | Passed | 1.23 |
| 05_batching.ipynb | Passed | 1.37 |
| 06_paged_memory.ipynb | Passed | 1.36 |
| 07_paged_attention.ipynb | Passed | 1.30 |
| 08_scheduler.ipynb | Passed | 1.37 |
| 09_prefix_caching.ipynb | Passed | 1.36 |
| 10_serving.ipynb | Passed | 1.38 |
| 11_benchmarking.ipynb | Passed | 1.45 |
| 12_accelerators.ipynb | Passed | 2.60 |
| 13_capstone.ipynb | Passed | 1.14 |
| 14_vllm_source_and_extensions.ipynb | Passed | 1.34 |

Executed notebooks are under `artifacts/executed/`; raw execution summary is `artifacts/notebook-report.json`. They are generated local artifacts, excluded from Git. Repeat with `uv run python scripts/execute_notebooks.py`.

## A small measured comparison

57 prompt tokens, 16 generated tokens, fixed random tiny model, greedy sampling without EOS, CPU float32. Two warmup calls and five measured calls per method. Model construction is excluded from all methods; the paged method includes fresh engine/pool construction and disables prefix caching. All generated token IDs were equal.

| Method | Median milliseconds |
|---|---:|
| naive | 8.067 |
| contiguous_cache | 4.656 |
| paged_engine_cold | 6.919 |

These timings describe this tiny workload on this machine, not vLLM performance or a universal speedup. Raw samples and full metadata are in `artifacts/benchmark.json`.

## Scope and remaining limits

The core CPU path, notebook execution, and small MPS comparison were exercised. Windows/Linux portability, full MPS paged serving, CUDA Graph execution, custom GPU kernels, pretrained checkpoints, distributed execution, and streaming HTTP were not validated. The latter capabilities are explicitly extensions rather than implemented core features.

A dependency deprecation warning from Starlette/AnyIO may appear during tests; it did not fail the tests. No warning filter hides it.
