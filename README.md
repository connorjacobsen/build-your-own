# Build the model stack

Eight independent, build-first courses teach the model lifecycle through **131 small stages**. You implement learner code; instructor-owned tests verify it. Each course has its own `uv` project, lockfile, chapters, conceptual hints, grader, experiment journal and capstone. Nothing completes itself by importing a reference implementation.

## Choose a starting point

| Course | Stages | Main artifact | Hardware |
|---|---:|---|---|
| [01 — Autodiff](01-autograd/README.md) | 14 | A trainable NumPy neural-network engine | CPU |
| [02 — Pretraining](02-pretraining/README.md) | 16 | A trained byte-level transformer and portable checkpoint | CPU; optional larger GPU experiments |
| [03 — Data](03-data/README.md) | 13 | Deduplicated, split, versioned JSONL dataset | CPU |
| [04 — Fine-tuning](04-finetuning/README.md) | 14 | SFT/DPO training code and portable LoRA adapters | CPU; optional larger GPU experiments |
| [05 — Evaluation](05-evaluation/README.md) | 14 | Reproducible per-case evidence and comparisons | CPU |
| [06 — Inference](06-inference/README.md) | 32 | Cached, paged, continuously batched serving engine | CPU through 30; NVIDIA GPU for 31–32 |
| [07 — Distributed training](07-distributed/README.md) | 14 | Real process collectives and sharded optimizer state | CPU through 13; two NVIDIA GPUs for 14 |
| [08 — Releases](08-releases/README.md) | 14 | Local artifact registry, release gates and rollback | CPU |

A useful first pass is **01 → 03 → 02 → 05 → 04 → 06 → 07 → 08**. Read evaluation before interpreting fine-tuning improvements. If you are already comfortable with gradients, begin with the inference course you originally requested and return to the foundations as needed. Prerequisites are solid Python, basic arrays, matrix multiplication, derivatives and elementary probability. The textbook introductions connect those ideas to each implementation.

## The learning loop

```bash
cd ~/Code/ai/courses/01-autograd
uv sync --locked
uv run course list
uv run course read 1
uv run course check 1
```

The first check intentionally fails at an unfinished learner function. Implement it in `src/learner/api.py` (or `src/toyvllm/` for inference), then rerun. Work through as many small stages as you need; there is no requirement to finish a whole course in one sitting.

- `uv run course check N` verifies stages 1–N cumulatively.
- `uv run course check N --only` isolates a stage for debugging and cannot establish cumulative completion.
- `uv run course check N --seed 91` varies seeded inputs where a gate uses randomized cases.
- `uv run course hint N --level 1` reveals one conceptual hint; levels 2 and 3 narrow the reasoning.
- `uv run course status` checks the latest receipt against current source, grader, catalog and lockfile.

No global installation is required. Each course is editable and self-contained. Use its own working directory; the `learner` package name is intentionally scoped to that course’s virtual environment. `uv sync` downloads dependencies but never starts a GPU job or downloads model weights.

## What gets verified

Acceptance tests use independent formulas, PyTorch baselines, adversarial examples and lifecycle invariants. They test gradients and parameter changes as well as outputs. Systems gates inspect actual cache reuse, packed projection work, process communication and cleanup. Every stage has a runnable gate; later stages integrate earlier components.

Correct mechanics, statistical model quality and hardware performance are separate claims. Controlled training examples prove a training path works. They do not prove useful generalization. Capstones require held-out comparisons and experiment records. GPU gates fail on missing hardware; they cannot claim completion by skipping. See [GRADING.md](GRADING.md) and [VALIDATION.md](VALIDATION.md).

The local grader is inspectable, not a secure hidden-test service. Treat tests as instructor-owned acceptance criteria. Complete references are deliberately outside learner source, under each course’s `instructor/` (inference retains `grader/oracle/`). Do not read these during the default path. [AGENTS.md](AGENTS.md) tells coding assistants to coach unless you request a worked solution.

## Connect the courses

[ARTIFACTS.md](ARTIFACTS.md) defines the shared dataset, model, adapter, evaluation and release contracts. [CAPSTONE.md](CAPSTONE.md) describes the end-to-end project. `tools/lifecycle.py` is a provided integration driver that invokes your completed APIs; it is not a replacement for implementing them. [GPU.md](GPU.md) explains optional remote execution and its limits.

The original `~/Code/ai/nanovllm` remains untouched. `06-inference` is a copy with preserved learner files, a new artifact-loader exercise and smaller grading gates. Work in one copy at a time; changes are not automatically synchronized between them.

These are substantial courses, not a promise of expertise after a fixed number of hours. Early stages may take under an hour; graph differentiation, decoder assembly, scheduling and distributed recovery can take several sessions. Follow the contracts, keep an experiment journal and explain a failure before reaching for another hint.
