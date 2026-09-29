"""Synchronous, single-owner engine; continuous admission and token-budget steps.

Worst-case reservation on admission avoids allocation deadlock. Unlike vLLM,
this reference does not dynamically grow/preempt running requests.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from time import perf_counter
import torch
from .cache import BlockPool, PrefixCache
from .sampling import SamplingParams, sample


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    params: SamplingParams
    generator: torch.Generator
    arrival: float
    tokens: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    table: list[int] = field(default_factory=list)
    computed: int = 0
    cached_tokens: int = 0
    status: str = "waiting"
    finish_reason: str | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None

    @property
    def ttft(self):
        return self.token_times[0] - self.arrival if self.token_times else None

    @property
    def inter_token_latencies(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    finished: bool
    finish_reason: str | None


class Engine:
    def __init__(
        self,
        model,
        num_blocks=64,
        block_size=8,
        token_budget=32,
        max_sequences=8,
        prefill_chunk=16,
        prefix_caching=True,
    ):
        if min(token_budget, max_sequences, prefill_chunk) <= 0:
            raise ValueError("scheduler limits must be positive")
        self.model = model.eval()
        self.pool = BlockPool(
            model.cfg, num_blocks, block_size, model.device, model.embedding.weight.dtype
        )
        self.prefix = PrefixCache(self.pool) if prefix_caching else None
        self.token_budget, self.max_sequences = token_budget, max_sequences
        self.prefill_chunk = prefill_chunk
        self.requests = {}
        self.waiting, self.running = deque(), deque()
        self.history = []
        self.step_index = 0

    def add_request(self, request_id, prompt, max_new_tokens=16, params=None):
        prompt = list(prompt)
        if request_id in self.requests:
            raise ValueError("request_id must be unique, including completed requests")
        if not prompt or any(
            type(t) is not int or not 0 <= t < self.model.cfg.vocab_size for t in prompt
        ):
            raise ValueError("prompt must contain valid token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if len(prompt) + max_new_tokens > self.model.cfg.max_seq_len:
            raise ValueError("prompt plus requested output exceeds context limit")
        needed = math.ceil((len(prompt) + max_new_tokens - 1) / self.pool.block_size)
        if max_new_tokens and needed > self.pool.num_blocks:
            raise ValueError("request cannot fit in the entire KV pool")
        params = params or SamplingParams()
        req = Request(
            request_id,
            prompt,
            max_new_tokens,
            params,
            torch.Generator().manual_seed(params.seed),
            perf_counter(),
            tokens=prompt.copy(),
        )
        self.requests[request_id] = req
        if max_new_tokens == 0:
            req.status, req.finish_reason, req.finished_at = "finished", "length", perf_counter()
        else:
            self.waiting.append(request_id)
        return req

    @property
    def has_work(self):
        return bool(self.waiting or self.running)

    def _admit(self):
        # FCFS admission: a blocked head waits; later requests do not bypass it.
        while self.waiting and len(self.running) < self.max_sequences:
            req = self.requests[self.waiting[0]]
            shared = self.prefix.acquire(req.prompt) if self.prefix else []
            total = math.ceil((len(req.prompt) + req.max_new_tokens - 1) / self.pool.block_size)
            private = total - len(shared)
            if self.prefix:
                self.prefix.make_room(private)
            if self.pool.n_free < private:
                self.pool.release(shared)
                break
            req.table = shared + self.pool.allocate(private)
            req.computed = req.cached_tokens = len(shared) * self.pool.block_size
            req.status = "running"
            self.running.append(self.waiting.popleft())

    def _finish(self, req, reason):
        self.pool.release(req.table)
        req.table = []
        req.status = "cancelled" if reason == "cancelled" else "finished"
        req.finish_reason, req.finished_at = reason, perf_counter()
        if req.request_id in self.running:
            self.running.remove(req.request_id)
        if req.request_id in self.waiting:
            self.waiting.remove(req.request_id)

    def cancel(self, request_id):
        req = self.requests[request_id]
        if req.status in {"waiting", "running"}:
            self._finish(req, "cancelled")

    @torch.inference_mode()
    def step(self):
        self._admit()
        if not self.running:
            if self.waiting:
                raise RuntimeError("no progress: admission invariant broken")
            return []
        budget, selected = self.token_budget, []
        # Rotate the queue so a tiny budget cannot permanently starve later requests.
        for request_id in list(self.running):
            if budget == 0:
                break
            req = self.requests[request_id]
            pending = len(req.tokens) - req.computed
            count = min(pending, self.prefill_chunk, budget)
            assert count > 0
            selected.append((req, count, req.computed))
            budget -= count
        self.running.rotate(-len(selected))
        items = [(r.tokens[start : start + n], r.table, start) for r, n, start in selected]
        logits_by_request = self.model.forward_paged_batch(items, self.pool)
        events, rows = [], []
        for (req, count, start), logits in zip(selected, logits_by_request):
            phase = "prefill" if start < len(req.prompt) else "decode"
            req.computed += count
            if self.prefix:
                self.prefix.publish(req.prompt, req.table, req.computed)
            emitted = None
            # Partial prompt chunks must never sample from intermediate logits.
            if req.computed == len(req.tokens):
                emitted = sample(logits[-1], req.params, req.generator)
                req.tokens.append(emitted)
                req.output.append(emitted)
                req.token_times.append(perf_counter())
                reason = (
                    "eos"
                    if emitted == req.params.eos_id
                    else "length"
                    if len(req.output) == req.max_new_tokens
                    else None
                )
                if reason:
                    self._finish(req, reason)
                events.append(TokenEvent(req.request_id, emitted, reason is not None, reason))
            rows.append(
                {
                    "request_id": req.request_id,
                    "phase": phase,
                    "start": start,
                    "count": count,
                    "emitted": emitted,
                }
            )
        self.history.append(
            {
                "step": self.step_index,
                "scheduled": rows,
                "tokens": self.token_budget - budget,
                "free_blocks": self.pool.n_free,
                "waiting": len(self.waiting),
                "running": len(self.running),
            }
        )
        self.step_index += 1
        self.pool.check()
        return events

    def run(self):
        while self.has_work:
            self.step()
        return {rid: r.output.copy() for rid, r in self.requests.items()}

    def clear_prefix_cache(self):
        if self.prefix:
            self.prefix.clear()
