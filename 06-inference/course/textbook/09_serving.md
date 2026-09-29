> This is a technical textbook chapter using original milestone numbers. The current course has 32 smaller gates; use `course list` for their numbering.

# Stage 09 — Serve concurrent callers

**Build:** `src/toyvllm/server.py`.  
**Prerequisite:** stage 08. **Estimated effort:** 3–5 hours.  
**Gate:** use the microstage number shown by `uv run course list`.

## Objective

Expose your engine through a small HTTP application. The route is a frontend to one shared engine; it must not create a new model or independent generation loop for every request. You have already solved scheduling and cache ownership. The server must preserve those decisions while callers arrive, wait, finish, and disappear.

Implement `create_app(model=None,max_pending=64,engine=None)` returning a FastAPI application. Construct the default model/engine once per application, unless an engine is explicitly supplied. Expose that exact shared instance as `app.state.engine`. The optional injection point allows deterministic lifecycle tests; it is also a useful separation of frontend and compute ownership.

Do not start a server on import. The application factory can be used by a terminal server or by an in-memory test client.

## HTTP contract

`GET /health` returns HTTP 200 with JSON containing `{"status":"ok"}`. Additional fields are allowed.

`POST /generate` accepts:

| Field | Type and default | Validation |
|---|---|---|
| `prompt` | Required string | Empty text is allowed and tokenizes to BOS |
| `max_new_tokens` | Integer, default 16 | Inclusive range 0–128 |
| `temperature` | Number, default 0 | Inclusive range 0–5 |
| `seed` | Integer, default 0 | Passed into request-local sampling |

Use schema validation with HTTP 422 for invalid field types/ranges. If tokenized prompt plus requested output exceeds engine capacity/context, return HTTP 400. If accepting another caller would exceed `max_pending`, return HTTP 429 immediately. Error bodies may follow FastAPI's normal schema; their exact prose is not graded.

Successful responses use HTTP 200 and include `request_id` (unique string), `token_ids` (generated integer list), `text` (decoded generated IDs), `finish_reason`, `cached_prompt_tokens`, and `ttft_seconds`. EOS is not enabled by this endpoint's default sampling parameters, so ordinary completion is by length. A zero-output request returns an empty list/string and may use `null` TTFT.

This is a custom teaching API. It is not OpenAI-compatible. Streaming HTTP, authorization, and external checkpoint loading are extension projects rather than silently implied features.

## One compute owner, many waiting callers

A straightforward architecture gives each caller a future and lets one background owner advance the engine. The route validates and submits a uniquely identified request. The compute loop performs bounded engine steps and resolves callers whose request state has become terminal. It yields between steps so new arrivals can reach the queue.

The exact choice of futures, conditions, or channels is yours. The important boundaries are one owner of engine mutation, stable request identity, bounded pending state, and cleanup after every terminal path. A synchronous tiny CPU step can run on the event loop in this reference architecture, but it blocks other work during that step. Moving expensive execution into a worker process is a natural later extension.

Do not mistake declaring a function `async` for making a long PyTorch call nonblocking. An async function can still monopolize the event loop until it reaches an actual yield point. Likewise, multiple Python threads do not automatically establish safe concurrent mutation of your pool.

## Lifecycle contract

The application lifespan starts the background loop and stops it during shutdown. Shutdown must release active request ownership and clear retained prefixes. A caller that disconnects or whose route task is cancelled must not leave active work or consume pending capacity indefinitely. Put cleanup on a path that runs even when awaiting the result is interrupted.

Remove per-caller bookkeeping after a response or cancellation. You may retain completed engine request records for diagnostics if they are bounded or intentionally part of this small local model, but do not count completed records against the pending-call limit. Pending capacity describes live frontend requests, not all IDs ever seen.

Exceptions from engine execution must not leave every waiting caller blocked forever. Surface an error and clean up affected ownership. You do not need a production recovery system; you do need well-defined completion and cancellation paths.

## How backpressure is tested

The grader injects a gate around your engine. While the gate is paused, one caller occupies a single available pending slot. A second caller must receive 429. Cancelling the first route must release its pending slot and engine work. When the gate opens, a later caller must succeed.

This avoids testing queue behavior through arbitrary sleeps or assumptions about a machine's speed. Respect the injected engine; constructing an unrelated engine inside the route defeats both the architecture and the test. The normal concurrent-client case also checks that every caller's ID appears in the one shared engine's execution history.

## Run your completed server

After passing the gate, start it from the project root:

```bash
uv run uvicorn toyvllm.server:create_app --factory --host 127.0.0.1 --port 8000
```

Send a request from another terminal:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_new_tokens":8}'
```

Use a single process for this course. Multiple uvicorn workers each construct another model and engine; that is replication, not one shared scheduler. Stop the terminal server with Ctrl-C. The grader uses in-memory transports and should not leave a network port open.

## Measurement boundaries

An engine-side TTFT begins at `add_request` and ends at the first emission. It does not include every network hop or necessarily frontend validation/tokenization. Client-observed latency begins before sending the HTTP request and ends when data arrives. Whole-response time differs from TTFT because this endpoint returns the completed result.

If you later implement streaming, token boundaries are not necessarily text-character boundaries. Byte tokens can split UTF-8 characters, so use an incremental decoder or a carefully computed cumulative text difference. A bounded per-request output queue is also necessary to prevent a slow consumer from growing memory without limit.

**Next:** [Stage 10 — GPU execution and measurement](10_acceleration.md).
