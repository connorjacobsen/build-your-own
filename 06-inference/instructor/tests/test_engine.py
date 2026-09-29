import random
import pytest
from grader.oracle import Engine, SamplingParams
from grader.oracle.sampling import generate_naive


@pytest.mark.parametrize(
    "budget,chunk,blocks,size,prefix",
    [(1, 1, 32, 2, True), (5, 3, 12, 4, True), (32, 16, 32, 8, False)],
)
def test_mixed_requests_match_baseline(model, budget, chunk, blocks, size, prefix):
    engine = Engine(
        model,
        token_budget=budget,
        prefill_chunk=chunk,
        num_blocks=blocks,
        block_size=size,
        prefix_caching=prefix,
    )
    prompts = [[256, 1, 2, 3, 4, 5], [256, 1, 2, 3, 4, 6], [4], list(range(17))]
    for i, p in enumerate(prompts):
        engine.add_request(str(i), p, 6)
    result = engine.run()
    for i, p in enumerate(prompts):
        assert result[str(i)] == generate_naive(model, p, 6)
    assert all(0 < step["tokens"] <= budget for step in engine.history)
    engine.clear_prefix_cache()
    assert engine.pool.n_free == blocks


def test_late_arrival_and_no_sampling_partial_prompt(model):
    engine = Engine(model, token_budget=2, prefill_chunk=2)
    r = engine.add_request("long", [1] * 8, 5)
    assert engine.step() == []
    assert r.computed == 2 and r.output == []
    engine.add_request("late", [2], 1)
    engine.run()
    assert engine.requests["late"].finished_at < r.finished_at


def test_prefix_reuse_exact_block_boundary(model):
    engine = Engine(model, block_size=4, num_blocks=6)
    prompt = [256, 1, 2, 3, 4, 5, 6, 7]
    engine.add_request("cold", prompt, 3)
    engine.run()
    engine.add_request("warm", prompt, 3)
    engine.run()
    assert engine.requests["warm"].cached_tokens == 4
    assert engine.requests["warm"].output == engine.requests["cold"].output
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 6


def test_seed_independent_of_schedule(model):
    params = SamplingParams(temperature=0.8, top_k=7, top_p=0.9, seed=42)
    prompt = [256, 5, 2, 1]
    baseline = generate_naive(model, prompt, 12, params)
    for budget in [1, 7, 32]:
        engine = Engine(model, token_budget=budget, prefill_chunk=3)
        engine.add_request("other", [6] * 9, 9, params)
        engine.add_request("target", prompt, 12, params)
        engine.run()
        assert engine.requests["target"].output == baseline


def test_zero_eos_cancel_and_invalid_requests(model):
    engine = Engine(model, max_sequences=1)
    r = engine.add_request("zero", [1], 0)
    assert r.status == "finished" and not engine.has_work
    eos = generate_naive(model, [1], 1)[0]
    engine.add_request("eos", [1], 8, SamplingParams(eos_id=eos))
    engine.run()
    assert engine.requests["eos"].output == [eos]
    assert engine.requests["eos"].finish_reason == "eos"
    engine.add_request("active", [2] * 12, 9)
    engine.add_request("waiting", [3], 1)
    engine.step()
    engine.cancel("active")
    engine.cancel("active")
    engine.cancel("waiting")
    engine.clear_prefix_cache()
    assert not engine.has_work and engine.pool.n_free == engine.pool.num_blocks
    for prompt, n in [([], 1), ([999], 1), ([1], -1), ([1], 1000)]:
        with pytest.raises(ValueError):
            engine.add_request("invalid", prompt, n)
    with pytest.raises(ValueError):
        engine.add_request("zero", [1], 1)


def test_pressure_admission_and_eviction(model):
    engine = Engine(model, num_blocks=4, block_size=4, token_budget=3, prefill_chunk=2)
    for i in range(10):
        engine.add_request(str(i), [i] * 9, 5)
    engine.run()
    assert all(r.status == "finished" for r in engine.requests.values())
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 4


def test_randomized_arrivals_cancellations(model):
    rng = random.Random(3)
    engine = Engine(model, num_blocks=12, block_size=4, token_budget=7, prefill_chunk=3)
    for step in range(30):
        prompt = [rng.randrange(20) for _ in range(rng.randrange(1, 12))]
        engine.add_request(str(step), prompt, rng.randrange(1, 6))
        engine.step()
        if step % 4 == 0:
            engine.cancel(str(step))
        engine.pool.check()
    engine.run()
    for r in engine.requests.values():
        if r.status == "finished":
            assert r.output == generate_naive(model, r.prompt, r.max_new_tokens)
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 12
