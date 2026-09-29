# 08 · Build the scheduler and continuous engine loop

**Time:** 3–4 hours. **Prerequisite:** paged execution. **Deliverable:** a bounded-work engine supporting arrivals, completion, and cancellation. **Build:** `schedule_round` and the capstone engine scaffold.

The model runner knows how to execute supplied work. The scheduler decides which work exists and whether memory is available. Keep these responsibilities separate enough that a scheduler can be tested using metadata traces without analyzing matrix multiplication.

Each request has a prompt, a growing token history, a count of positions already computed, generated output IDs, sampling parameters and random state, a block table, and a lifecycle status. In this course the lifecycle is:

```text
waiting → running → finished
    ↘        ↘
       cancelled
```

Cancellation must release active ownership and remove the request from its queue exactly once. Completion frees request-owned pages; cached prefix ownership can remain. A request whose declared maximum cannot fit in the entire pool is rejected at submission rather than left waiting forever.

## Unify prompt and decode work

Let `N` be the number of available token IDs and `C` the number of computed positions. Pending work is `N−C`. At first, all prompt IDs are available and `C=0`. When the prompt is fully computed, sample one output ID and append it; now `N−C=1`. This token-deficit view supports both prompt chunks and decode steps.

Our step assigns at most `min(pending,prefill_chunk,remaining_budget)` positions to each visited request. It emits a token only after `computed == len(tokens)` for that request. Sampling from an intermediate prompt chunk would predict the rest of the prompt instead of its continuation.

The V1 [scheduler's `schedule` method](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L179) uses computed-token accounting as a central abstraction. Our version leaves out speculative tokens, multimodal inputs, distributed workers, and asynchronous execution.

```python
model = make_model()
engine = Engine(model, token_budget=3, prefill_chunk=3, prefix_caching=False)
request = engine.add_request("A", [256,1,2,3,4,5,6,7], max_new_tokens=3)
rows = []
while engine.has_work:
    events = engine.step()
    rows.append((engine.step_index, request.computed, len(request.tokens), list(request.output)))
for row in rows:
    print("step, computed, available IDs, output:", row)
assert rows[0][3] == [] and rows[1][3] == []
assert len(request.output) == 3
assert request.computed == len(request.prompt) + len(request.output) - 1
```

## Admission and memory pressure

The reference reserves enough blocks for `P+G−1` computed positions when admitting a request. Shared prefix blocks count toward this reservation. The remaining blocks are private. Once admitted, the request can finish without requesting more capacity. This gives a simple progress guarantee: a running request cannot deadlock waiting for its next page.

The cost is lower concurrency when output limits are conservative or requests stop early. Upstream vLLM allocates incrementally and can preempt/recompute work under pressure. Do not describe our reservation policy as vLLM's algorithm.

Admission is first-come-first-served. If the waiting head does not fit, later waiting requests do not bypass it. This is head-of-line blocking. Execution rotates admitted requests between steps, preventing a tiny token budget from always favoring the same first request. It gives opportunities to run, not a latency SLA under unbounded incoming load.

```python
engine = Engine(model, num_blocks=6, block_size=4, token_budget=4,
                prefill_chunk=3, max_sequences=3, prefix_caching=False)
engine.add_request("long", [256] + [9]*11, 8)
engine.step()
engine.add_request("late-short", [256,2], 2)
engine.run()
for rid, req in engine.requests.items():
    print(rid, req.status, "output:", req.output, "TTFT:", round(req.ttft,4))
assert all(h["tokens"] <= 4 for h in engine.history)
assert engine.pool.n_free == engine.pool.num_blocks
```

## Draw an actual scheduling trace

This plot uses the engine's real schedule history. Bar width is the number of new positions computed, and the x-axis is an iteration index, not elapsed time. Unequal token counts can have unequal runtimes; do not read this as a device utilization chart.

```python
fig, ax = plt.subplots(figsize=(10,3))
ids = list(engine.requests)
for step in engine.history:
    for row in step["scheduled"]:
        color = "#2563eb" if row["phase"] == "prefill" else "#16a34a"
        y = ids.index(row["request_id"])
        ax.barh(y, .85, left=step["step"], color=color)
        ax.text(step["step"]+.42, y, str(row["count"]), ha="center", va="center", color="white")
ax.set_yticks(range(len(ids)), ids); ax.set_xlabel("Engine step")
ax.set_title("Blue: prompt work; green: decode work; label: scheduled token count")
plt.tight_layout(); plt.show()
```

## Cancellation and zero output

A cancelled waiting request has no pages to release. A cancelled running request does. Make cancellation idempotent so duplicate cleanup calls do not double-free. This matters when an HTTP disconnect and a normal completion race at the application boundary.

```python
engine = Engine(model, max_sequences=1, prefill_chunk=2, token_budget=2)
engine.add_request("active", [1]*20, 5)
engine.add_request("waiting", [2]*3, 5)
engine.step()
engine.cancel("active"); engine.cancel("active")
engine.cancel("waiting")
engine.clear_prefix_cache()
assert not engine.has_work
assert engine.pool.n_free == engine.pool.num_blocks
zero = engine.add_request("zero", [1], 0)
assert zero.status == "finished" and zero.output == []
```

## Build and check

Implement `schedule_round(pending,budget,chunk)` returning counts for the supplied request order. Counts cannot exceed deficits, the per-request chunk limit, or the total budget. This pure helper deliberately leaves rotation and admission to the caller. Then study `exercises/starter_engine.py` and sketch how `step` will call your helper, construct paged inputs, and sample only complete frontiers.

```python
if RUN_EXERCISES:
    from exercises.checks import check
    check("scheduler")
else:
    print("Scheduler exercise skipped.")
```

**Pressure experiment:** lower `num_blocks` until only one request fits. Confirm every admitted request finishes and later ones eventually enter. Submit a request larger than the entire pool and verify immediate rejection. These are progress properties, not merely output-equivalence tests.

**Stretch: recompute preemption.** Replace full reservation with on-demand allocation. On allocation failure, select a victim, release its active pages, preserve its token history, reset its computed frontier, and requeue it. Recomputing already-generated positions must not resample them or advance the random generator. Add a policy that prevents repeated mutual eviction from producing livelock. See the capstone for acceptance tests.

Next: [09 — Prefix reuse and copy on write](09_prefix_caching.ipynb).
