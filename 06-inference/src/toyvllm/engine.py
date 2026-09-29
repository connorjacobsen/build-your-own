"""Stage 07 implements lifecycle/scheduling; stage 08 adds prefix caching.

Read the contract for required observable request fields and history records.
Internal queue types, helper methods, and request representation are yours.
"""


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
        raise NotImplementedError("Stage 07: engine state and one shared physical pool")

    def add_request(self, request_id, prompt, max_new_tokens=16, params=None):
        raise NotImplementedError("Stage 07: validate, create request state, enqueue")

    @property
    def has_work(self):
        raise NotImplementedError("Stage 07: unfinished queued or running work")

    def step(self):
        raise NotImplementedError("Stage 07: select bounded work, execute, sample and release")

    def run(self):
        raise NotImplementedError("Stage 07: advance until no work remains; return outputs by ID")

    def cancel(self, request_id):
        raise NotImplementedError("Stage 07: idempotent terminal transition")

    def clear_prefix_cache(self):
        raise NotImplementedError("Stage 08: clear retained entries; no-op with caching disabled")
