"""Worked solutions. Try the matching exercise before reading this file."""

import math
import torch
from nanovllm_course.model import attention as causal_attention, rope as apply_rope
from nanovllm_course.sampling import generate_cached as generate_with_cache
from nanovllm_course.cache import BlockPool
from nanovllm_course.engine import Engine as build_engine

__all__ = [
    "causal_attention",
    "apply_rope",
    "generate_with_cache",
    "build_engine",
    "kv_bytes",
    "nucleus_probabilities",
    "pack_tokens",
    "physical_slots",
    "run_paged",
    "schedule_round",
    "reusable_prefix_tokens",
]


def kv_bytes(layers, tokens, kv_heads, head_dim, bytes_per_element):
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per_element


def nucleus_probabilities(logits, top_p):
    sorted_logits, order = logits.sort(descending=True)
    p = sorted_logits.softmax(-1)
    remove = p.cumsum(-1) - p >= top_p
    p[remove] = 0
    p /= p.sum()
    result = torch.zeros_like(p)
    result[order] = p
    return result


def pack_tokens(sequences, starts):
    ids, positions, boundaries = [], [], [0]
    for sequence, start in zip(sequences, starts):
        ids.extend(sequence)
        positions.extend(range(start, start + len(sequence)))
        boundaries.append(len(ids))
    return ids, positions, boundaries


def physical_slots(table, block_size, start, count):
    if block_size <= 0 or start < 0 or count < 0 or start + count > len(table) * block_size:
        raise ValueError("invalid logical range")
    return (
        [table[p // block_size] for p in range(start, start + count)],
        [p % block_size for p in range(start, start + count)],
    )


@torch.inference_mode()
def run_paged(model, prompt, max_new_tokens):
    if max_new_tokens == 0:
        return []
    block_size = 4
    count = math.ceil((len(prompt) + max_new_tokens - 1) / block_size)
    pool = BlockPool(
        model.cfg,
        num_blocks=count,
        block_size=block_size,
        device=model.device,
        dtype=model.embedding.weight.dtype,
    )
    table = list(reversed(pool.allocate(count)))
    output, inputs, computed = [], list(prompt), 0
    try:
        for _ in range(max_new_tokens):
            logits = model.forward_paged_batch([(inputs, table, computed)], pool)[0]
            computed += len(inputs)
            token = int(logits[-1].argmax())
            output.append(token)
            inputs = [token]
        return output
    finally:
        pool.release(table)
        pool.check()
        assert pool.n_free == count


def schedule_round(pending, budget, chunk):
    result = []
    for count in pending:
        scheduled = min(count, budget, chunk)
        result.append(scheduled)
        budget -= scheduled
    return result


def reusable_prefix_tokens(prompt_length, block_size):
    if prompt_length < 1 or block_size < 1:
        raise ValueError("positive lengths required")
    return ((prompt_length - 1) // block_size) * block_size
