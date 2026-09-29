import math
import pytest
import torch
from grader.oracle.model import rope as reference_rope


@pytest.mark.examples
def test_rmsnorm_learned_weight_and_zero(api):
    norm = api.model.RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected)


@pytest.mark.examples
def test_rope_absolute_positions(api):
    x = torch.arange(48).float().view(3, 2, 8) / 10
    positions = torch.tensor([0, 4, 19])
    actual = api.model.rope(x, positions)
    torch.testing.assert_close(actual, reference_rope(x, positions))
    torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))


@pytest.mark.examples
def test_attention_offset_gqa_and_future_mask(api):
    q, k, v = torch.randn(3, 4, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8)
    expected = torch.empty_like(q)
    for i in range(3):
        for h in range(4):
            scores = k[: 5 + i, h // 2] @ q[i, h] / math.sqrt(8)
            expected[i, h] = scores.softmax(0) @ v[: 5 + i, h // 2]
    actual = api.model.attention(q, k, v, query_start=4)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    changed = v.clone()
    changed[5:] += 100
    torch.testing.assert_close(api.model.attention(q, k, changed, 4)[0], actual[0])


@pytest.mark.examples
@torch.inference_mode()
def test_dense_model_matches_fixed_checkpoint(model, oracle):
    ids = [256, 12, 7, 99, 3]
    actual, cache = model(ids)
    expected, reference_cache = oracle(ids)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert len(cache) == model.cfg.n_layers
    for (k, v), (rk, rv) in zip(cache, reference_cache):
        torch.testing.assert_close(k, rk)
        torch.testing.assert_close(v, rv)


@pytest.mark.extended
@torch.inference_mode()
def test_causality_and_varied_sequences(model, oracle, case_seed):
    g = torch.Generator().manual_seed(case_seed)
    for length in [1, 2, 9, 23]:
        ids = torch.randint(0, 258, (length,), generator=g).tolist()
        a, _ = model(ids)
        b, _ = oracle(ids)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a, _ = model([1, 2, 3, 4, 5])
    b, _ = model([1, 2, 3, 90, 91])
    torch.testing.assert_close(a[:3], b[:3])


@pytest.mark.extended
def test_dense_gradients_reach_the_parameters(model):
    model.train()
    logits, _ = model([256, 1, 2, 3])
    logits.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"No finite gradient for {name}"
