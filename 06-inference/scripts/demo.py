"""After stage 07: a client of YOUR engine, with no reference fallback."""

import torch
from toyvllm import ByteTokenizer, Engine, make_model

torch.set_num_threads(1)
tok = ByteTokenizer()
engine = Engine(make_model(), token_budget=8, prefill_chunk=4, prefix_caching=False)
for rid, text in [("a", "Hello"), ("b", "What is a KV cache?")]:
    engine.add_request(rid, tok.encode(text), 8)
while engine.has_work:
    for event in engine.step():
        print(event)
assert engine.pool.n_free == engine.pool.num_blocks
