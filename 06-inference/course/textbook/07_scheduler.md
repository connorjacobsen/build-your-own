> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 07 — Schedule a living batch

**Build:** `src/toyvllm/engine.py`.  
**Prerequisite:** stage 06. **Estimated effort:** 4–8 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## A model runner does not decide what should run

Your paged model accepts supplied work. The engine must decide which requests own capacity and which token positions execute now. Requests can arrive at different times, produce different output lengths, and finish or cancel between iterations. This is where a collection of numerical functions becomes a serving system.

Keep two decisions distinct: **admission** moves a waiting request into the set with resources to run; **scheduling** assigns bounded work to admitted requests for one iteration. A larger token budget cannot make a request fit into insufficient cache memory.

## Engine contract

Preserve the supplied constructor signature. `prefix_caching=False` is used for all stage 07 cases; implement prefix reuse in stage 08. Maintain one `pool`, one model, a `requests` mapping, and a `history` list.

| API | Behavior |
|---|---|
| `add_request(id,prompt,max_new_tokens=16,params=None)` | Validate and store independent request state; return an inspectable request object |
| `has_work` | True while a waiting or running request remains |
| `step()` | Admit eligible work, execute at most one packed paged model call, update state, return newly emitted token events |
| `run()` | Advance to completion and return all known request IDs mapped to copied output-ID lists |
| `cancel(id)` | Mark a waiting/running request cancelled, release ownership, remove it from active queues; repeated cancellation is harmless |
| `clear_prefix_cache()` | No-op when caching is disabled; stage 08 extends it |

IDs are unique for the engine lifetime, including completed requests. Duplicate IDs raise `ValueError`. Prompts must be nonempty lists of Python integer IDs in the vocabulary. Negative/noninteger output limits are invalid. Prompt length plus requested output must not exceed the configured context limit. If a positive-output request's maximum computed positions cannot fit in the entire pool, reject it immediately. These checks use `ValueError`.

A zero-output request completes immediately with an empty output and `finish_reason="length"`, without allocating pages or making model calls. EOS is included in output IDs and has finish reason `"eos"`. Ordinary length completion uses `"length"`; cancellation uses status and reason `"cancelled"`.

## Observable request state

The representation may be a dataclass or an ordinary object. Expose the following fields:

- `request_id`, `prompt`, `max_new_tokens`, `params`;
- `tokens`: prompt followed by sampled IDs; `output`: generated IDs only;
- `computed`: number of positions with valid KV; `table`: current physical block table;
- `status`: `waiting`, `running`, `finished`, or `cancelled`;
- `finish_reason`, initially `None`; `cached_tokens`, initially zero;
- `arrival`, `token_times`, and `finished_at`: host monotonic timestamps for measurement.

A `TokenEvent` may be any object exposing `request_id`, `token_id`, `finished`, and `finish_reason`. Return exactly one event for each new ID emitted during that step. Partial prompt work yields no event. Event ordering across requests is your choice; ordering within one request is causal.

Each nonempty execution adds a history entry with `tokens` (actual scheduled count) and `scheduled` (rows containing request ID, `start`, `count`, `phase`, and `emitted`). `phase` is `prefill` if the start is still inside the prompt, otherwise `decode`. `emitted` is the new ID or `None`. These records should describe real work, not predicted or padded positions.

## The computed frontier unifies prefill and decode

At any moment the history has N available IDs and C computed positions. Pending work is N−C. Initially N is prompt length and C is zero. After the final prompt chunk, a sample extends N by one. There is now one uncomputed ID to feed back on the next decode.

For each visited request, schedule no more than its pending work, `prefill_chunk`, or the remaining global `token_budget`. The global count cannot exceed the budget. Sample only once the computed frontier reaches the end of available IDs. A partial prompt chunk must never sample its intermediate logits as the continuation.

For a seven-token prompt and a chunk limit of two, the first step computes two positions and emits nothing. This is useful progress even though the client has not received a token. The grader checks that distinction.

## Start with reservation-based admission

Reserve enough logical blocks for P+G−1 computed positions when a request with P prompt tokens and G>0 outputs is admitted. This guarantees an admitted request can finish without asking for additional cache capacity. Do not reserve cache for the final output ID unless it is going to be fed back.

Use first-come-first-served admission and respect `max_sequences`. If the waiting head cannot fit, it can wait until a running request releases capacity. Among admitted requests, rotate opportunities so a one-token budget does not permanently favor the same first request. The exact trace need not match the instructor's implementation; the gate checks bounds, output equivalence, arrivals, and progress.

The progress argument relies on positive budgets, finite requests, and full reservation. It does **not** prove an on-demand allocator safe. Production vLLM can allocate incrementally and preempt/recompute; that more involved policy is an extension after this stage.

## Finishing is an ownership transition

When a request ends, release its request-owned table once and prevent future scheduling. Waiting cancellation has no table to free; active cancellation does. The same logical transition can be requested twice by different cleanup paths, so make cancellation idempotent.

Do not recreate a random generator each time the request is scheduled. Preserve the per-request stream from stage 03. A change from budget one to budget thirty-two should not alter that request's seed progression.

The grader submits late arrivals, cancels requests, uses tiny pools, compares stochastic outputs across budgets, and runs randomized finite workloads with bounded step counts. A loop that never makes progress fails instead of hanging the course indefinitely. It also measures actual paged inputs and compares them with your history; logging a small budget while executing extra positions does not pass.

**Reading:** [V1 scheduler](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L179). Notice computed-token accounting, then distinguish its allocation/preemption choices from the simpler reservation contract here.

**Next:** [Stage 08 — Prefix sharing](08_prefix.md).
