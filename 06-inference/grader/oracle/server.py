"""Minimal local JSON service with one background owner of a shared engine.

Not an OpenAI-compatible endpoint. Streaming and network auth are extension labs.
"""

import asyncio
from contextlib import asynccontextmanager
import uuid
from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from pydantic import BaseModel, Field
from .engine import Engine
from .model import make_model
from .sampling import SamplingParams
from .tokenizer import ByteTokenizer


class GenerateBody(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=16, ge=0, le=128)
    seed: int = 0
    temperature: float = Field(default=0.0, ge=0, le=5)


def create_app(model=None, max_pending=64, engine=None):
    tokenizer = ByteTokenizer()
    engine = engine if engine is not None else Engine(model or make_model(), num_blocks=128)
    futures = {}

    async def pump():
        while True:
            if engine.has_work:
                try:
                    engine.step()  # Intentionally synchronous CPU teaching runner.
                except Exception as exc:
                    for rid, future in list(futures.items()):
                        engine.cancel(rid)
                        if not future.done():
                            future.set_exception(exc)
            for rid, future in list(futures.items()):
                req = engine.requests[rid]
                if req.status in {"finished", "cancelled"} and not future.done():
                    future.set_result(req)
            await asyncio.sleep(0.001)  # Yield so arrivals can join the next batch.

    @asynccontextmanager
    async def lifespan(app):
        worker = asyncio.create_task(pump())
        yield
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        for rid in list(futures):
            engine.cancel(rid)
        engine.clear_prefix_cache()

    app = FastAPI(title="Tiny vLLM teaching service", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "random-tiny-transformer"}

    @app.post("/generate")
    async def generate(body: GenerateBody, request: HTTPRequest):
        if len(futures) >= max_pending:
            raise HTTPException(429, "request queue is full")
        rid = uuid.uuid4().hex
        try:
            engine.add_request(
                rid,
                tokenizer.encode(body.prompt),
                body.max_new_tokens,
                SamplingParams(temperature=body.temperature, seed=body.seed),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        future = asyncio.get_running_loop().create_future()
        futures[rid] = future
        try:
            while not future.done():
                if await request.is_disconnected():
                    engine.cancel(rid)
                    raise HTTPException(499, "client disconnected")
                await asyncio.wait({future}, timeout=0.02)
            result = future.result()
            return {
                "request_id": rid,
                "token_ids": result.output,
                "text": tokenizer.decode(result.output),
                "finish_reason": result.finish_reason,
                "cached_prompt_tokens": result.cached_tokens,
                "ttft_seconds": result.ttft,
            }
        finally:
            engine.cancel(rid)
            futures.pop(rid, None)
            engine.requests.pop(rid, None)

    return app
