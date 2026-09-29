import pytest
import torch
from grader.oracle.sampling import generate_naive as reference_generate


@pytest.mark.examples
@pytest.mark.parametrize("ends", [[3, 5, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
@torch.inference_mode()
def test_cached_chunks_match_dense_without_mutation(model, oracle, ends):
    ids = [256, 3, 9, 12, 15, 2, 8, 17, 4]
    expected, _ = oracle(ids)
    past, start, pieces = None, 0, []
    for end in ends:
        snapshots = [(k.clone(), v.clone()) for k, v in past] if past else []
        logits, new_past = model(ids[start:end], past)
        if past:
            for pair, saved in zip(past, snapshots):
                for a, b in zip(pair, saved):
                    torch.testing.assert_close(a, b)
        assert new_past[0][0].shape[0] == end
        pieces.append(logits)
        past, start = new_past, end
    torch.testing.assert_close(torch.cat(pieces), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.examples
def test_cache_reuses_work_not_just_outputs(api, model, oracle):
    forward_sizes, projection_sizes = [], []
    original = model.forward

    def traced(ids, past=None):
        forward_sizes.append(len(ids))
        return original(ids, past)

    model.forward = traced
    hook = model.layers[0].q.register_forward_pre_hook(
        lambda module, args: projection_sizes.append(args[0].shape[0])
    )
    try:
        got = api.sampling.generate_cached(model, [256, 4, 9, 2], 6)
    finally:
        hook.remove()
    assert got == reference_generate(oracle, [256, 4, 9, 2], 6)
    assert forward_sizes == [4, 1, 1, 1, 1, 1], "Feed back only the most recently generated ID"
    assert projection_sizes == [4, 1, 1, 1, 1, 1], (
        "Cached positions must not be reprojected internally"
    )


@pytest.mark.extended
@torch.inference_mode()
def test_cached_values_are_actually_consumed(model, oracle):
    _, past = model([1, 2, 3, 4])
    edited = [(k.clone(), v.clone() + 2) for k, v in past]
    actual, _ = model([5, 6], edited)
    expected, _ = oracle([5, 6], edited)
    original, _ = oracle([1, 2, 3, 4, 5, 6])
    assert not torch.allclose(expected, original[-2:])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.extended
def test_cached_sampling_matches_seeded_baseline(api, model, oracle):
    params = api.sampling.SamplingParams(temperature=0.7, top_k=12, top_p=0.9, seed=92)
    assert api.sampling.generate_cached(model, [256, 5, 9], 12, params) == reference_generate(
        oracle, [256, 5, 9], 12, params
    )
