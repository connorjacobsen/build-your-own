import pytest
import torch
from grader.oracle.model import rope
from grader.oracle.cache import BlockPool
from grader.oracle.sampling import generate_cached, generate_naive, SamplingParams


@torch.inference_mode()
@pytest.mark.parametrize("cuts", [[1, 2, 3, 4, 5, 6, 7], [3, 5, 7], [7]])
def test_cached_chunks_equal_dense(model, cuts):
    tokens = [256, 8, 11, 21, 4, 19, 90]
    full, _ = model(tokens)
    past, start, pieces = None, 0, []
    for end in cuts:
        logits, past = model(tokens[start:end], past)
        pieces.append(logits)
        start = end
    torch.testing.assert_close(torch.cat(pieces), full, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_no_future_leak(model):
    a, _ = model([1, 2, 3, 4, 5])
    b, _ = model([1, 2, 3, 8, 9])
    torch.testing.assert_close(a[:3], b[:3])


def test_rope_preserves_norm():
    x = torch.randn(8, 4, 16)
    y = rope(x, torch.arange(8))
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))


@torch.inference_mode()
def test_paged_batch_unequal_lengths_and_permuted_blocks(model):
    pool = BlockPool(model.cfg, num_blocks=8, block_size=2)
    blocks = pool.allocate(8)
    ta, tb = [blocks[i] for i in [5, 1, 7, 3]], [blocks[i] for i in [0, 6, 2, 4]]
    a, b = [1, 3, 5, 7, 9], [2, 4, 6, 8, 10, 12, 14]
    initial = model.forward_paged_batch([(a[:3], ta, 0), (b[:4], tb, 0)], pool)
    later = model.forward_paged_batch([(a[3:], ta, 3), (b[4:], tb, 4)], pool)
    for ids, first, second in zip([a, b], initial, later):
        dense, _ = model(ids)
        torch.testing.assert_close(torch.cat([first, second]), dense, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "params", [SamplingParams(), SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=19)]
)
def test_generation_paths(model, params):
    assert generate_naive(model, [256, 3, 8], 12, params) == generate_cached(
        model, [256, 3, 8], 12, params
    )
