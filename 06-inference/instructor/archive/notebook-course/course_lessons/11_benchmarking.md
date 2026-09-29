# 11 · Measure the thing you actually changed

**Time:** 2–3 hours. **Prerequisite:** all core engine components. **Deliverable:** a reproducible benchmark report with correctness checks. **Build:** a benchmark matrix, not a single headline number.

Performance claims need a workload, a measurement boundary, and a comparison with equivalent outputs. “Tokens per second” is ambiguous unless you say whether you count prompt tokens, generated tokens, useful work, or padded positions. A microbenchmark and a serving benchmark answer different questions.

Three useful measurements are:

- **Time to first token (TTFT):** first emitted token time minus arrival time. It includes queueing and prompt work inside the chosen boundary.
- **Inter-token latency (ITL):** elapsed time between consecutive output events for one request. A request with one output token has no ITL samples.
- **Aggregate output throughput:** total output tokens divided by the workload's wall-clock interval. State whether the interval includes admission, tokenization, warmup, or network time.

Our engine timestamps arrivals in `add_request` and emissions inside `step`. These are host-side engine timestamps, not client-observed network latency. Time to final response is a separate measurement.

## Benchmark naive versus contiguous-cache generation

Use the same model, prompt, output limit, precision, device, and sampler. Warm up before collecting repeated samples. Accelerator work is asynchronous, so synchronize at timing boundaries. The helper supports CPU, CUDA, and MPS synchronization; core measurements below use CPU.

```python
from nanovllm_course.sampling import generate_naive, generate_cached
from nanovllm_course.metrics import benchmark, projection_token_work
model = make_model()
prompt = ByteTokenizer().encode("Measure the same workload. "*2)
count = 12
assert generate_naive(model,prompt,count) == generate_cached(model,prompt,count)
naive = benchmark(lambda: generate_naive(model,prompt,count), warmup=1, repeats=5)
cached = benchmark(lambda: generate_cached(model,prompt,count), warmup=1, repeats=5)
print({"naive_seconds":naive["median_seconds"], "cached_seconds":cached["median_seconds"]})
print("Observed ratio (this machine only):", naive["median_seconds"]/cached["median_seconds"])
plt.boxplot([naive["samples_seconds"], cached["samples_seconds"]], tick_labels=["Naive","Cached"])
plt.ylabel("Seconds per generation"); plt.title("Equal output length; CPU wall clock"); plt.show()
```

Do not fail a correctness test because the cached version is slower for a tiny workload. Python calls, thread startup, concatenation, and gathers can overwhelm the saved math. Report the result honestly and investigate which cost dominates. The course does not assert a universal speedup.

## Compare three execution paths fairly

The paged engine adds scheduler and allocator overhead to the model path. Construct a new engine inside each timed invocation for a cold end-to-end comparison. A warm-prefix benchmark should deliberately reuse an engine and report the number of cached positions. Mixing those two experiments makes a speedup hard to interpret.

```python
def paged_cold():
    engine = Engine(model, num_blocks=64, block_size=8, token_budget=32, prefill_chunk=16,
                    prefix_caching=False)
    engine.add_request("one", prompt, count)
    engine.run()
    return engine.requests["one"].output
assert paged_cold() == generate_naive(model,prompt,count)
paged = benchmark(paged_cold, warmup=1, repeats=5)
print("Paged engine cold median:", paged["median_seconds"])
```

The benchmark intentionally includes pool creation for the cold engine. A long-lived serving engine amortizes that cost. Add a second benchmark that reuses a warmed pool if you want to measure only request execution, and label it accordingly.

## Sweep a real scheduling parameter

A token budget controls how many new positions can execute per step. Larger budgets may amortize overhead and reduce prompt completion time. They can also make individual steps longer. Smaller prompt chunks allow more interleaving but create more calls. Our round-robin policy and CPU runner will not reproduce production GPU tradeoffs exactly.

```python
from nanovllm_course.metrics import engine_metrics
records = []
for budget in [4, 8, 16, 32]:
    engine = Engine(model, token_budget=budget, prefill_chunk=8, prefix_caching=False)
    prompts = [ByteTokenizer().encode("a"*n) for n in [4,12,24,40]]
    for i, ids in enumerate(prompts):
        engine.add_request(str(i), ids, 8)
    engine.run()
    metrics = engine_metrics(engine)
    for i, ids in enumerate(prompts):
        assert engine.requests[str(i)].output == generate_cached(model,ids,8)
    records.append({"budget":budget, "throughput":metrics["output_tokens_per_second"],
                    "median_ttft_ms":float(np.median(metrics["ttft_seconds"])*1000),
                    "steps":len(engine.history)})
for row in records:
    print(row)
fig, axes = plt.subplots(1,2,figsize=(10,3))
axes[0].plot([r["budget"] for r in records], [r["throughput"] for r in records], marker="o")
axes[0].set_ylabel("Output tokens / second")
axes[1].plot([r["budget"] for r in records], [r["median_ttft_ms"] for r in records], marker="o")
axes[1].set_ylabel("Median engine TTFT (ms)")
for ax in axes: ax.set_xlabel("Token budget")
plt.tight_layout(); plt.show()
```

These are small single-run serving observations, not statistically strong capacity estimates. Repeat each configuration with shuffled workload order and enough samples to characterize noise before drawing conclusions. Four requests cannot estimate p99 latency credibly.

## Open-loop and closed-loop load

A closed-loop client sends another request after a previous one completes. This can hide overload because arrival rate drops when the server slows down. An open-loop workload schedules arrivals independently of completion. As offered load exceeds service capacity, queue delay grows even if model execution speed is unchanged.

For a stationary, stable system, Little's law relates average in-system requests `L`, arrival rate `λ`, and average time in system `W`: `L=λW`. It is a consistency check on measurement boundaries, not a substitute for a latency distribution. Do not apply the steady-state interpretation to an overloaded queue that grows without bound.

## Build a report

Write a table with device, Python/PyTorch versions, model dimensions and dtype, prompt/output-length distributions, number of requests, arrival pattern, warmup/repeats, prefix hit rate, memory pool capacity, and exact timing boundary. Include output-equivalence checks and raw timing samples. Count cancellations separately from completed requests.

The script `uv run python scripts/benchmark.py` saves a small machine-readable CPU report to `artifacts/benchmark.json`. It is a repeatable starting point, not a production vLLM benchmark.

**Failure lab:** omit synchronization on an accelerator and compare reported times. Then include model construction in only one of two compared paths. Explain why both comparisons are misleading even when the timer code itself is syntactically correct.

**Stretch:** add scheduled arrivals, track queue wait separately from model service time, and plot offered load against TTFT and throughput. Stop making throughput claims when the queue no longer reaches a stable regime.

Next: [12 — Accelerator lab](12_accelerators.ipynb).
