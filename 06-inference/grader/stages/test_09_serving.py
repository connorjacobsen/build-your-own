import asyncio
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.mark.examples
def test_http_contract_validation_and_shared_engine(api, model):
    app = api.server.create_app(model)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = {"prompt": "hello", "max_new_tokens": 4, "temperature": 0.0}
        a, b = client.post("/generate", json=body), client.post("/generate", json=body)
        assert a.status_code == b.status_code == 200
        a, b = a.json(), b.json()
        assert a["request_id"] != b["request_id"]
        assert a["token_ids"] == b["token_ids"] and len(a["token_ids"]) == 4
        from grader.oracle.sampling import generate_naive

        assert a["token_ids"] == generate_naive(model, [256] + list(b"hello"), 4)
        assert a["text"] == bytes(t for t in a["token_ids"] if 0 <= t < 256).decode(
            "utf-8", errors="replace"
        )
        assert a["finish_reason"] == "length"
        assert (
            client.post("/generate", json={"prompt": "", "max_new_tokens": 0}).json()["token_ids"]
            == []
        )
        assert (
            client.post("/generate", json={"prompt": "x", "max_new_tokens": -1}).status_code == 422
        )
        assert client.post("/generate", json={"prompt": "x" * 600}).status_code == 400
    assert app.state.engine.pool.n_free == app.state.engine.pool.num_blocks


@pytest.mark.extended
def test_concurrent_clients_use_one_engine(api, model):
    app = api.server.create_app(model)
    with TestClient(app) as client:
        engine_id = id(app.state.engine)

        def call(i):
            response = client.post(
                "/generate", json={"prompt": "prefix " + str(i), "max_new_tokens": 8}
            )
            assert response.status_code == 200
            return response.json()

        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(call, range(4)))
        assert len({r["request_id"] for r in results}) == 4
        assert all(len(r["token_ids"]) == 8 for r in results)
        assert id(app.state.engine) == engine_id
        scheduled = {
            row["request_id"] for step in app.state.engine.history for row in step["scheduled"]
        }
        assert {r["request_id"] for r in results} <= scheduled, (
            "All callers must pass through the shared engine"
        )


@pytest.mark.extended
def test_backpressure_and_cancelled_caller_release_resources(api, model):
    # The gate makes admission/backpressure deterministic without relying on host speed.
    class Gate:
        def __init__(self):
            self.inner = api.engine.Engine(model, prefix_caching=False)
            self.ready = False

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def step(self):
            return self.inner.step() if self.ready else []

    gate = Gate()
    app = api.server.create_app(model, max_pending=1, engine=gate)

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/generate", json={"prompt": "a", "max_new_tokens": 8})
                )
                for _ in range(200):
                    if gate.requests:
                        break
                    await asyncio.sleep(0.001)
                assert gate.requests, "Route never submitted to its injected engine"
                second = await client.post("/generate", json={"prompt": "b", "max_new_tokens": 1})
                assert second.status_code == 429, "Full pending queue must reject additional work"
                first.cancel()
                try:
                    await first
                except asyncio.CancelledError:
                    pass
                for _ in range(50):
                    if not gate.has_work:
                        break
                    await asyncio.sleep(0.001)
                assert not gate.has_work, "Cancelled route left an orphaned request"
                gate.ready = True
                response = await client.post("/generate", json={"prompt": "c", "max_new_tokens": 1})
                assert response.status_code == 200, (
                    "Cancelled caller did not release pending capacity"
                )
        assert gate.pool.n_free == gate.pool.num_blocks

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
