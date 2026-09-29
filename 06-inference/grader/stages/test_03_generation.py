import pytest
import torch
from grader.oracle.sampling import generate_naive as reference_generate, sample as reference_sample


@pytest.mark.examples
def test_greedy_and_length_zero(api, model, oracle):
    prompt = [256, 8, 2]
    assert api.sampling.generate_naive(model, prompt, 6) == reference_generate(oracle, prompt, 6)
    assert prompt == [256, 8, 2]

    class NoCalls:
        def __call__(self, *args):
            raise AssertionError("A zero-output request must not execute the model")

    assert api.sampling.generate_naive(NoCalls(), [1], 0) == []


@pytest.mark.examples
def test_eos_is_included_and_stops(api, model, oracle):
    eos = reference_generate(oracle, [4, 8], 1)[0]
    params = api.sampling.SamplingParams(eos_id=eos)
    assert api.sampling.generate_naive(model, [4, 8], 9, params) == [eos]


@pytest.mark.extended
def test_temperature_filters_and_request_rng(api, model, oracle):
    params = api.sampling.SamplingParams(temperature=0.8, top_k=9, top_p=0.8, seed=51)
    expected = reference_generate(oracle, [256, 4, 3], 12, params)
    torch.manual_seed(999)
    assert api.sampling.generate_naive(model, [256, 4, 3], 12, params) == expected
    torch.rand(100)
    assert api.sampling.generate_naive(model, [256, 4, 3], 12, params) == expected


@pytest.mark.extended
def test_nucleus_crossing_token_and_logit_immutability(api):
    logits = torch.tensor([0.50, 0.30, 0.15, 0.05]).log()
    saved = logits.clone()
    params = api.sampling.SamplingParams(temperature=1.0, top_p=0.6)
    observed = set()
    for seed in range(50):
        g1, g2 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed)
        got = api.sampling.sample(logits, params, g1)
        assert got == reference_sample(logits, params, g2), "Nucleus must retain the crossing token"
        observed.add(got)
    assert observed == {0, 1}
    torch.testing.assert_close(logits, saved)
