# Troubleshooting

## Imports fail or the kernel uses the wrong Python

Start Jupyter from the project root with `uv run jupyter lab`, or use the README command with `--ServerApp.root_dir=.`. The explicit root keeps links to sibling `docs/` and `exercises/` accessible even when opening a notebook by filename. Print `sys.executable`; it should point into this project's `.venv`. Run `uv sync --locked` if dependencies are missing. If using an external Jupyter server, register and select the kernel described in lesson 00. Restart the kernel after editing library code.

The source package is installed in editable mode, but Python caches imported modules. Installing correctly does not automatically refresh a running kernel. Restarting is more reliable for this course than mixing old class instances with reloaded definitions.

## uv installation or dependency resolution

Use the official uv installer linked in the README. The project requires Python 3.11 or 3.12 and selects 3.11. The lock file is part of the deliverable; do not delete it as a first troubleshooting step. `uv sync --locked` checks that it agrees with `pyproject.toml`.

The initial environment install needs network access and enough disk for PyTorch and Jupyter. No model checkpoints are downloaded. On Linux, PyTorch's package may bring accelerator-related dependencies even when you run the CPU course; CPU-wheel specialization is an optional environment customization, not required to understand the code. If changing indexes or dependency versions, preserve the original lock and rerun the course checks.

## Generated text looks broken

Random weights generate arbitrary byte IDs. Some sequences are invalid UTF-8 and decode with replacement characters. This is expected. Compare token IDs, logits, and cache invariants. Lesson 02's small training loop overfits a pattern; it does not turn the model into a general assistant.

## A notebook passed in pieces but fails under Run All

Look for variables from old cells, a model whose weights were changed, or an engine reused without clearing state. Every distributed notebook starts from independent setup. Restart the kernel, use Run All, and compare with the clean notebook or Markdown authoring source. Learner exercise failures are intentional until implemented.

## A learner exercise reports UNFINISHED

Edit `exercises/implementation.py`. The supplied worked solutions are separate. `RUN_EXERCISES=False` skips your code; `--reference` validates the supplied solutions. Neither means your exercise is complete.

## My toy paged engine is slower

The reference has Python loops, explicit checks, K/V gathers, and scheduler overhead. Small matrix workloads can be dominated by these costs. Cache savings in projected positions are not equivalent to a guaranteed wall-clock gain. Use lesson 11 to establish equivalent workloads and report raw samples. Do not remove correctness checks in order to “fix” a numerical problem with timing.

## GPU or MPS problems

The core defaults to CPU. Lesson 12 checks available hardware and skips CUDA Graph work when CUDA is absent. MPS is not CUDA and cannot execute CUDA Graphs or CUDA/Triton kernels. If testing another device or dtype, first compare logits with the CPU reference and use synchronization around timing. Accelerator paths are exploratory; only hardware actually present during validation is claimed as tested.

## Cache capacity errors

A request's declared prompt plus output limit must fit the model context. Its worst-case computed positions must fit the total pool. Reduce the output limit, increase `num_blocks`, or use fewer/shorter requests. Waiting requests can be delayed by reservations held by active requests. The toy does not automatically preempt or swap them.

## Server port already in use

Choose another port, such as `--port 8001`, and update the client URL. Stop a terminal server with Ctrl-C. The notebook uses TestClient and cleans up its in-memory application, so it does not need an open network port. Use a single process for this teaching service: multiple uvicorn workers create separate model/engine instances.

## Where did my notebook edits go?

`scripts/build_notebooks.py` regenerates notebooks from `course/lessons/*.md`. It is an authoring tool and overwrites notebooks. Use it only after saving learner changes. The execution script instead writes copies under `artifacts/executed/` and does not overwrite the originals.
