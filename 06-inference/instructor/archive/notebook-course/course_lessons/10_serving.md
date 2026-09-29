# 10 · Serve multiple callers through one engine

**Time:** 2 hours. **Prerequisite:** lifecycle management and token events. **Deliverable:** a local JSON endpoint backed by a shared engine. **Build:** a concurrent client experiment and a cancellation extension.

An HTTP wrapper should submit work to the engine, not secretly create a second scheduler for every caller. A useful single-process teaching design has one model, one engine, one background loop that advances it, and one future per caller awaiting completion.

The frontend validates text and generation limits, tokenizes the prompt, assigns a unique request ID, and enqueues it. The background loop calls `engine.step()`, then resolves futures for completed requests. It yields control between steps so new arrivals can join. Cache ownership remains with the engine.

Our service is deliberately a small custom API: `POST /generate` and `GET /health`. It is not OpenAI API compatible. HTTP responses contain complete outputs; in-process `TokenEvent`s demonstrate incremental generation. Streaming HTTP is an extension below, not a feature hidden behind the word “serving.”

## Start it locally

From a terminal at the project root:

```bash
uv run uvicorn nanovllm_course.server:create_app --factory --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain caching","max_new_tokens":12,"temperature":0.0}'
```

The response includes token IDs, decoded text, finish reason, reused prompt-token count, and engine-side time to first token. Random model weights mean the decoded text is generally gibberish. Use IDs and invariants to judge the engine.

For automated notebooks we can exercise the same application in memory. This avoids leaving a server or background task running when you choose Run All.

```python
from fastapi.testclient import TestClient
from nanovllm_course.server import create_app
app = create_app(make_model())
with TestClient(app) as client:
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/generate", json={"prompt":"hello", "max_new_tokens":4})
    assert response.status_code == 200
    print(response.json())
    assert len(response.json()["token_ids"]) == 4
assert app.state.engine.pool.n_free == app.state.engine.pool.num_blocks
```

## Trace ownership across awaits

`server.py` creates the engine once in `create_app`. The lifespan context starts the pump and stops it during shutdown. The route stores a future keyed by request ID. While waiting it checks for disconnects; its `finally` block cancels any still-active request and removes per-caller bookkeeping. Duplicate cancellation is safe because the engine is idempotent.

The model call itself is synchronous and blocks the event loop during each small CPU step. Chunking bounds how much work a step admits, but this is not a production responsiveness guarantee. A real deployment would isolate compute in a worker/process and design a clear command/output protocol. Adding threads without assigning a single owner of cache mutation can introduce races.

The service limits pending requests to 64 by default and returns 429 when full. It validates generation limits and rejects prompt-plus-output lengths exceeding the model context. This is a simple form of backpressure: reject excess work instead of allowing an unbounded queue to exhaust memory.

```python
app = create_app(make_model())
with TestClient(app) as client:
    cases = [
        ({"prompt":"", "max_new_tokens":0}, 200),
        ({"prompt":"x", "max_new_tokens":-1}, 422),
        ({"prompt":"x"*600, "max_new_tokens":4}, 400),
    ]
    for payload, expected_status in cases:
        response = client.post("/generate", json=payload)
        assert response.status_code == expected_status
        print(expected_status, "for prompt length", len(payload["prompt"]))
```

## Run concurrent requests

A concurrent client is necessary to exercise overlapping admissions. Sending one request and waiting before sending the next tests serial serving and prefix reuse, not continuous batching.

The in-process test below starts several callers. It proves that the shared application returns valid responses for overlapping clients; it is not a network latency benchmark. Inspect the recorded schedule to see how many requests actually shared a step. Very short requests or host timing may reduce overlap, so do not assert that every step contains every request.

```python
from concurrent.futures import ThreadPoolExecutor
app = create_app(make_model())
with TestClient(app) as client:
    def submit(i):
        response = client.post("/generate", json={"prompt":"prefix " + str(i), "max_new_tokens":8})
        assert response.status_code == 200
        return response.json()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(submit, range(4)))
    assert len({r["request_id"] for r in results}) == 4
    assert all(len(r["token_ids"]) == 8 for r in results)
    max_shared = max(len(h["scheduled"]) for h in app.state.engine.history)
    print("Most requests in one observed step:", max_shared)
```

## Streaming extension: specify the protocol first

Implement a separate `POST /generate_stream` using server-sent events if you want the next systems challenge. Replace a single result future with a bounded per-request event queue. The pump publishes token events without blocking every other request on one slow client. The response consumes events and sends a final event with the finish reason.

Define these behaviors before coding: how to represent token IDs, how to signal completion and errors, what to do when the queue fills, and how a disconnect cancels the request. Do not decode each byte token independently; a multibyte UTF-8 character can span several tokens. An incremental decoder or cumulative-output diff is needed for text chunks.

**Acceptance tests:** two overlapping streams preserve each request's order; a disconnect releases its blocks; a slow consumer cannot grow an unbounded queue; the final event is emitted once; Unicode decoding does not introduce replacement characters for a valid split sequence.

## A realistic deployment boundary

This application is for local learning. It has no checkpoint loader, authentication, distributed workers, persistent request store, or production telemetry. The core server and notebooks use fixed random weights. The important result is a working boundary between callers and an engine whose memory and scheduling behavior you can inspect.

Upstream bridge: compare the engine loop with [V1 `EngineCore.step`](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/engine/core.py#L272), then use the [architecture guide](https://docs.vllm.ai/en/latest/design/arch_overview/) to locate frontend, core, and worker responsibilities. Current process architecture may differ from the pinned baseline; preserve the conceptual boundaries when reading newer code.

Next: [11 — Benchmarks](11_benchmarking.ipynb).
