# 00 · Build an inference engine you can explain

**Time:** 45–60 minutes. **Prerequisites:** Python functions, classes, and basic algebra. Tensor operations are introduced in lesson 01. **Deliverable:** a working uv/Jupyter environment and your first multi-request generation run.

This course builds a small, real autoregressive transformer and the system around it. By the end you will be able to follow a token from an incoming request, through scheduling and physical cache allocation, into a model forward pass, and back to a caller. Every optimization has a correctness oracle: the slow implementation that came before it.

The core runs on CPU, including Apple Silicon, without external model weights, a Hugging Face account, CUDA, or the `vllm` package. The model has random weights by default. Its outputs are not meaningful language. That is useful: numerical equivalence, ownership, causal masking, and scheduling do not require a pretrained model. Lesson 02 includes a small local training experiment if you want to see learning happen.

## What does vLLM add to a transformer?

A transformer answers: given tokens, what are the logits for the next token? An inference engine also has to answer: whose tokens should run now, where should their cached tensors live, when can memory be reused, and what happens when a request finishes or disappears?

Our path is:

```text
text → token IDs → waiting queue → admission / scheduling
                                    ↓
                        (tokens, positions, block tables)
                                    ↓
                         packed transformer execution
                                    ↓
                          logits → sampling → events
                                    ↓
                     stop / free blocks / admit more work
```

The course uses vLLM **v0.11.0**, commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`, as a deliberately fixed V1 reading baseline. This is not a claim that v0.11.0 is the newest release. Current vLLM has additional features and moving internals. The [pinned architecture document](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/docs/design/arch_overview.md) provides the upstream context; our local [source map](../docs/source_map.md) supplies concrete reading targets and differences.

## Start with uv

Run these commands in a terminal from the repository root, not in a Python notebook cell:

```bash
uv sync --locked
uv run jupyter lab --ServerApp.root_dir=. notebooks/00_start_here.ipynb
```

`pyproject.toml` declares dependencies. `uv.lock` records their resolved versions. `.python-version` selects Python 3.11. The editable project installation means changes under `src/` are available to your environment without reinstalling. A running kernel still holds imported modules in memory: restart it after editing library code.

Jupyter has two processes worth distinguishing: the web application and the Python kernel. Launching Jupyter with `uv run` keeps both in this project's environment. The notebook below prints the interpreter path; it should contain this project's `.venv`. If you use an existing Jupyter installation, register an explicit kernel with:

```bash
uv run python -m ipykernel install --user --name nanovllm-course --display-name "nano-vLLM course"
```

Then select that kernel. Do not install packages with bare `pip` in a notebook: that can modify a different interpreter. All required packages are already declared. The [official uv Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/) explains the server/kernel distinction.

```python
print("Interpreter:", sys.executable)
print("PyTorch:", torch.__version__)
print("Project:", ROOT)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
assert (ROOT / "uv.lock").exists()
```

## Take the engine for a short drive

This is the finished reference implementation. Do not try to memorize it yet. Watch the interface: callers submit requests, the engine advances by a bounded step, and each step can yield zero or more token events. A step with no emitted tokens can still make progress by computing a prompt chunk.

```python
tokenizer = ByteTokenizer()
model = make_model()
engine = Engine(model, token_budget=8, prefill_chunk=4, block_size=4)
engine.add_request("short", tokenizer.encode("Hi"), max_new_tokens=4)
engine.add_request("long", tokenizer.encode("What is a KV cache?"), max_new_tokens=4)
events = []
while engine.has_work:
    events.extend(engine.step())
for event in events:
    print(event)
assert len(events) == 8
print("First three scheduler steps:", engine.history[:3])
engine.clear_prefix_cache()
assert engine.pool.n_free == engine.pool.num_blocks
```

**Observe:** generated IDs can be bytes that do not decode into valid UTF-8 text. The engine works on IDs; decoded text is a presentation layer. The final sampled token does not need its own KV state unless another token will be generated from it. That off-by-one matters later.

## How to use this course

Read the notebooks in order the first time. Each is independently executable in a fresh kernel. Use **Restart Kernel and Run All** to expose hidden state and out-of-order execution. Do not treat a green cell from yesterday as evidence that today's edit works.

Each lesson combines a derivation, a small implementation or trace, numerical assertions, a build exercise, and a connection to upstream code. The exercises live in `exercises/implementation.py`; the finished source in `src/nanovllm_course/` is available for comparison. `RUN_EXERCISES=False` means reference examples execute while unfinished learner code is skipped. After implementing the named function, set that flag to `True` in that notebook, or run its checker from the terminal. Skipped exercises are not passed exercises.

For each milestone, predict the result before running it. Then change one input that attacks the assumption: a nonzero cache offset, a prompt crossing a block boundary, a one-token budget, or two identical prefixes with different continuations. Write down what failed and which invariant explains it.

| Part | Lessons | You will build |
|---|---|---|
| Model foundations | 01–03 | Byte tokenizer, attention, tiny decoder, generation and sampling |
| Reuse and layout | 04–07 | Contiguous cache, packed batches, block pool, paged model execution |
| Serving system | 08–10 | Scheduler, prefix cache, cancellation, local HTTP service |
| Evidence and extensions | 11–14 | Benchmarks, GPU experiments, capstone, source-reading bridge |

Budget approximately **25–40 hours** for reading and implementation, longer if tensor programming is new. A fast systems track can skim 01–03 after passing the corresponding checks. A six-week schedule is in the README.

## First checkpoint

Without looking at the code, answer: Why can one engine step yield no tokens? What persists between steps? What would be wrong with treating each incoming HTTP request as a separate model instance?

<details><summary>Suggested answers</summary>

A partial prompt must be fully processed before sampling its continuation. Request metadata, token histories, per-request random generators, and KV tensors persist. A model per HTTP request duplicates weights and prevents one shared scheduler from combining work. A shared model without a shared scheduler also does not automatically provide continuous batching.
</details>

Next: [01 — Tokens and tensor mechanics](01_tokens_and_tensors.ipynb).
