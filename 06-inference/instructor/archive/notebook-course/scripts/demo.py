"""Run with uv run python scripts/demo.py."""

import torch
from nanovllm_course import ByteTokenizer, Engine, make_model

torch.set_num_threads(1)
tok = ByteTokenizer()
engine = Engine(make_model(), token_budget=12, prefill_chunk=6)
for i, prompt in enumerate(["Hello", "Explain KV caching", "Hello again"]):
    engine.add_request(str(i), tok.encode(prompt), max_new_tokens=8)
while engine.has_work:
    for event in engine.step():
        print(event)
print("Random weights: token IDs demonstrate execution, not language quality.")
engine.clear_prefix_cache()
assert engine.pool.n_free == engine.pool.num_blocks
