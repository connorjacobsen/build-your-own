# Validation record — 2026-09-28

Host: macOS Apple Silicon, Python 3.11.13, PyTorch 2.7.1. This host has no NVIDIA CUDA devices. Every course was locked and installed in its own uv virtual environment.

## Course acceptance tests against instructor references

| Course | CPU cases passed | Hardware cases excluded from this sweep |
|---|---:|---:|
| Autodiff | 33 | 0 |
| Pretraining | 29 | 0 |
| Data | 16 | 0 |
| Fine-tuning | 25 | 0 |
| Evaluation | 21 | 0 |
| Inference | 47 | 2 CUDA |
| Distributed training | 18 | 1 two-GPU NCCL |
| Releases | 15 | 0 |
| **Total** | **204** | **3** |

The CPU sweep passed with zero skipped tests. Distributed cases launched real pairs of Gloo worker processes, including uneven and empty local batches, weighted reductions, sharded momentum updates and coordinated finite-gradient decisions. The pretraining and fine-tuning gates were also rerun after strengthening projection-value and nonzero-reference-margin assertions.

Commands: `01-autograd/.venv/bin/python tools/validate.py`; machine-readable JUnit and summary files are under `instructor/validation/`. Instructor-reference passes are course-maintenance evidence and do not mark learner stages completed.

## Grader and integration checks

- **19 suite maintenance tests passed:** representative deliberately broken implementations were rejected, cumulative receipts became stale after nested source changes, isolated checks did not complete a course, skips/failures could not produce completion, and every catalog gate/chapter existed.
- **27 inherited inference regression and mutation tests passed.** The old CLI-specific maintenance tests were replaced by tests of the new shared CLI.
- Original inference learner Python files and original stage acceptance tests were compared byte-for-byte with `~/Code/ai/nanovllm`. They were preserved. The new artifact loader is an additional learner exercise.
- The reference lifecycle ran both from the authoring runtime and from the isolated pretraining environment: dataset preparation → training → export → LoRA adaptation → merge → serving loader → paged engine → evaluation → local promotion → rollback.
- That lifecycle used 22 training documents and 2 held-out cases. Its permissive policy tests compatibility, not useful generalization or an improvement claim. The report explicitly records `quality_claim: false`.
- The seven new courses fail their initial learner gate with the expected NotImplementedError. The inference copy preserves existing tokenizer work: its first microstage passes, while the next gate exposes an existing decoding edge case. No learner solution was filled in during this course-building task.
- Active generated Python was formatted and linted. Preserved learner files and original acceptance cases were not reformatted. Active Markdown links were checked locally.

## Hardware and remote limits

The three hardware tests were separately invoked on this host and rejected missing CUDA with nonzero results; none skipped into a passing completion. This verifies the absence-of-hardware failure path only.

Both Modal runners import successfully against the locked Modal 1.5.5 SDK and construct their app/image definitions. **No remote image build, paid GPU job, CUDA kernel execution, NCCL GPU execution or performance benchmark was run.** Those remain explicit learner/instructor hardware checks. The distributed runner requests two L4 GPUs in one container; inference requests one L4.

An inherited Starlette/AnyIO deprecation warning remains visible during HTTP tests. It did not prevent passing tests and has not been hidden.

## Revalidation

Run `uv sync --locked` in each course after moving the suite. Then use `tools/validate.py` for CPU instructor checks and `instructor/test_suite.py` for grader maintenance. The learning-path smoke script intentionally assumes the seven new courses remain unfinished and should be revised rather than rerun blindly after the learner begins solving them.

A changed implementation, grader, catalog or lockfile invalidates earlier learner receipts. Future reports must distinguish reference maintenance, learner progress, statistical experiments and actual hardware measurements.
