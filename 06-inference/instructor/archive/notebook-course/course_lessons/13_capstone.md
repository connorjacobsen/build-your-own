# 13 · Capstone: assemble your own tiny inference engine

**Time:** 4–8 hours. **Prerequisite:** lessons 01–12. **Deliverable:** your implementation, a correctness report, and a benchmark report. **Build:** `build_engine` backed by `exercises/starter_engine.py`.

So far you have worked on isolated contracts: causal attention, absolute positions, cached generation, packing, physical slots, scheduling, and prefix boundaries. The capstone connects them into a system that keeps all those invariants true at once.

Use the finished package only as a reference. Write your engine in the starter module, retaining the same external interface so the behavioral checks can compare implementations. You may reuse the supplied tokenizer, transformer, sampler, and block pool for the minimum capstone. The main work is request state, admission, scheduling, execution metadata, completion, and cleanup. For the full build track, replace each reused component with your earlier exercise implementation after its checks pass.

## Milestone A: one request, one cache

Support `add_request(id,prompt,max_new_tokens,params=None)`, `step()`, `run()`, and `has_work`. `run()` returns a mapping from request ID to generated IDs. Maintain `requests[id].output` and `requests[id].status` for inspection. A step computes at most the configured token budget.

Before adding queues, write a table for a prompt of length 5 and three output tokens. Record available IDs, computed positions, emitted IDs, and cache capacity after each step. With prompt chunks of 2, the first two steps emit nothing; the third finishes prefill and emits the first continuation. Two further decode calls produce the remaining outputs. Your implementation should reproduce the table exactly.

## Milestone B: multiple requests and arrivals

Separate waiting and running queues. Use atomic admission and reserve worst-case capacity first. Rotate execution order, cap active sequences, and construct one packed model call per engine step. Submit another request after the first step to prove admission is continuous.

Do not sample partial prompt chunks. Do not accidentally replace a request's absolute position with its packed offset. Do not allow a completed request's stale table to be used in a later call. Sample each request using its own generator.

## Milestone C: completion and pressure

Handle EOS, zero requested output, cancellation before admission, cancellation while running, repeated cancellation, duplicate IDs, invalid token IDs, and requests larger than the entire pool. Explain which cases are accepted immediately, rejected immediately, or transition to a terminal state.

After all requests finish and you clear retained prefix entries, every physical block must be free. While requests are live, total references must match explicit owners. Test memory pressure with a pool small enough to serialize some admissions. “The program did not crash” is not a progress proof; bound the number of steps in your tests.

## Milestone D: prefix reuse

Add a cache scoped to one immutable model. Publish only computed full prompt blocks. Acquire references before any eviction can reclaim a candidate. Preserve at least one input position for final logits. Confirm outputs are unchanged and count skipped prompt positions.

A request can finish while its prefix remains reusable. Distinguish request release from cache eviction. Clearing the prefix cache while active requests use shared pages should remove cache ownership without invalidating request ownership.

## Inspect the finished reference on a capstone workload

This code runs independently of your unfinished capstone. It serves as an executable specification and provides a pattern for your own report.

```python
from nanovllm_course.sampling import generate_naive
model = make_model()
engine = Engine(model, num_blocks=12, block_size=4, token_budget=5, prefill_chunk=3,
                max_sequences=3, prefix_caching=True)
workload = {
    "shared-a": ([256,1,2,3,4,5,6,7,8], 6),
    "short": ([256,42], 2),
    "long": ([256]+list(range(20)), 7),
}
for rid,(prompt,count) in workload.items():
    engine.add_request(rid,prompt,count)
engine.step()
engine.add_request("cancel-me", [256,7,7,7], 4)
engine.cancel("cancel-me")
for _ in range(500):
    if not engine.has_work:
        break
    engine.step()
assert not engine.has_work, "Progress failure"
for rid,(prompt,count) in workload.items():
    assert engine.requests[rid].output == generate_naive(model,prompt,count)
engine.add_request("shared-b", workload["shared-a"][0], 6)
engine.run()
assert engine.requests["shared-b"].output == engine.requests["shared-a"].output
assert engine.requests["cancel-me"].status == "cancelled"
assert all(step["tokens"] <= 5 for step in engine.history)
engine.clear_prefix_cache()
assert engine.pool.n_free == engine.pool.num_blocks
print("Capstone reference workload passed in", len(engine.history), "steps")
```

## Grade your implementation

Implement `build_engine(model, **kwargs)` in `exercises/implementation.py` to return your starter-engine implementation. Run `uv run python scripts/check_exercises.py engine`. The checker tests behavior against dense generation, late arrivals, cancellation, budget bounds, zero output, and final ownership cleanup. It is a minimum contract check, not exhaustive proof.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("engine")
else:
    print("Learner capstone skipped. The finished reference workload above is verified separately.")
```

The package has a broader regression suite under `tests/`. Once your public interfaces match, adapt those tests to construct your engine. Keep independent expected values; do not modify expected answers merely to make an incorrect implementation pass.

## Suggested assessment rubric

| Dimension | Evidence | Points |
|---|---|---:|
| Model/cache correctness | Dense, cached, and paged logits agree; offset masks and GQA work | 25 |
| Memory ownership | Atomic allocation, shuffled pages, no double-free, clean final pool | 20 |
| Scheduling | Bounded work, late arrival, no sampling partial prompts, progress under pressure | 20 |
| Prefix reuse | Correct identity, immutable shared blocks, eviction, measured skipped work | 15 |
| Serving and lifecycle | Shared engine, cancellation, limits, clean shutdown | 10 |
| Measurement and explanation | Raw timings, stated boundaries, honest limitations, source mapping | 10 |

Passing means you can explain the implementation as well as execute it. Include one bug you deliberately introduced, the smallest failing test, and the invariant that fixes it.

## Extension projects with concrete acceptance criteria

**On-demand allocation and recompute preemption.** Grow pages only as positions are scheduled. Force eviction of a running request under pressure, then resume it without changing its already generated outputs. Preserve RNG state, avoid resampling historical IDs, and demonstrate progress on two mutually competing requests. Compare utilization with reservation under early EOS.

**Blockwise paged attention.** Implement the lesson 07 online softmax directly over physical pages. Test all block boundaries and partial tails against dense attention. Report numerical tolerances, memory temporaries, and timings. A slower Python version is still a useful correctness milestone.

**Checkpoint loading.** Choose a tiny supported model and implement its exact architecture, tokenizer, positional convention, and weight mapping. Compare logits with its original library before enabling caching. A familiar model name alone does not establish compatibility. Do this as an optional network/download project; the core course remains independent of external artifacts.

**Fairness and admission policy.** Add shortest-prompt bypass or explicit priorities. Measure whether short requests improve and whether long requests starve. State the policy's progress assumptions and test them with repeated arrivals.

**Streaming HTTP.** Implement the previous lesson's bounded queues and disconnect handling. Test a slow consumer and split Unicode bytes. Keep request cleanup in one ownership path.

## Write the final report

Your report should contain the supported behavior, a diagram of request state transitions, a memory-accounting example, the tested invariants, a table of benchmark results, and an explicit list of differences from production vLLM. Link to the exact source revision you read. A report that calls gathered PyTorch attention an optimized paged kernel, or calls random weights a language-capable model, is misleading even if all tests pass.

Next: [14 — Read vLLM and plan further work](14_vllm_source_and_extensions.ipynb).
