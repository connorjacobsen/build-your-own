"""Capstone scaffold; implement the lifecycle yourself.

You may reuse the model, BlockPool, PrefixCache, Request, TokenEvent, and sampler.
First implement without prefix reuse; add sharing once baseline checks pass.
Required inspection attributes: requests, pool, history (each entry has 'tokens').
The checker uses add_request, step, run, cancel, clear_prefix_cache, has_work.
"""

from collections import deque
from nanovllm_course.cache import BlockPool, PrefixCache


class LearnerEngine:
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
        self.model = model.eval()
        self.pool = BlockPool(
            model.cfg, num_blocks, block_size, model.device, model.embedding.weight.dtype
        )
        self.prefix = PrefixCache(self.pool) if prefix_caching else None
        self.token_budget, self.max_sequences, self.prefill_chunk = (
            token_budget,
            max_sequences,
            prefill_chunk,
        )
        self.requests = {}
        self.waiting, self.running = deque(), deque()
        self.history = []

    def add_request(self, request_id, prompt, max_new_tokens=16, params=None):
        raise NotImplementedError(
            "Validate input; create state/RNG; enqueue or finish zero-output request"
        )

    @property
    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self):
        raise NotImplementedError("Admit; budget; pack; execute; advance; sample; finish; record")

    def run(self):
        while self.has_work:
            self.step()
        return {rid: request.output.copy() for rid, request in self.requests.items()}

    def cancel(self, request_id):
        raise NotImplementedError("Remove from the appropriate queue and release ownership once")

    def clear_prefix_cache(self):
        if self.prefix:
            self.prefix.clear()
