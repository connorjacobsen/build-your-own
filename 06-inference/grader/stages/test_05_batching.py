import pytest
import torch


@pytest.mark.examples
@torch.inference_mode()
def test_unequal_sequences_are_independent(model, oracle):
    sequences = [[256, 1, 2], [256, 7], [3, 4, 5, 6, 7, 8]]
    logits, caches = model.forward_batch(sequences)
    assert len(logits) == len(caches) == 3
    for ids, result, cache in zip(sequences, logits, caches):
        expected, past = oracle(ids)
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        for (k, v), (rk, rv) in zip(cache, past):
            torch.testing.assert_close(k, rk)
            torch.testing.assert_close(v, rv)


@pytest.mark.extended
@torch.inference_mode()
def test_mixed_cached_lengths_and_empty_batch(model, oracle):
    _, past_a = oracle([1, 2, 3, 4])
    _, past_b = oracle([6])
    sequences, pasts = [[8, 9], [2, 3, 4], [5, 7]], [past_a, past_b, None]
    logits, caches = model.forward_batch(sequences, pasts)
    for ids, past, result in zip(sequences, pasts, logits):
        expected, _ = oracle(ids, past)
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    assert [cache[0][0].shape[0] for cache in caches] == [6, 4, 2]
    assert model.forward_batch([]) == ([], [])


@pytest.mark.extended
@torch.inference_mode()
def test_projections_are_packed_not_sequential_calls(model):
    calls = {}
    hooks = []
    for i, layer in enumerate(model.layers):
        for name in ["q", "k", "v", "o", "gate", "up", "down"]:
            key = (i, name)
            calls[key] = []
            hooks.append(
                getattr(layer, name).register_forward_pre_hook(
                    lambda module, args, key=key: calls[key].append(args[0].shape[0])
                )
            )
    try:
        model.forward_batch([[1, 2, 3], [4], [5, 6]])
    finally:
        for hook in hooks:
            hook.remove()
    assert all(rows == [6] for rows in calls.values()), (
        f"Expected one packed six-row call per projection; observed {calls}"
    )
