import random
import pytest
from grader.oracle.sampling import generate_naive as reference_generate


def finish_bounded(engine, limit=500):
    for _ in range(limit):
        if not engine.has_work:
            return
        engine.step()
    pytest.fail("Engine made no bounded progress: waiting/running requests remain after 500 steps")


@pytest.mark.examples
def test_partial_prefill_and_measured_token_budget(api, model, oracle):
    engine = api.engine.Engine(model, token_budget=3, prefill_chunk=2, prefix_caching=False)
    request = engine.add_request("long", [256, 1, 2, 3, 4, 5, 6], 4)
    calls = []
    original = model.forward_paged_batch

    def traced(items, pool):
        calls.append(sum(len(ids) for ids, _, _ in items))
        assert calls[-1] <= 3, "Actual model input exceeds the scheduler's token budget"
        return original(items, pool)

    model.forward_paged_batch = traced
    assert engine.step() == [], "A partial prompt must make progress without sampling"
    assert request.computed == 2 and request.output == []
    engine.add_request("late", [256, 9], 2)
    finish_bounded(engine)
    assert request.output == reference_generate(oracle, request.prompt, 4)
    assert engine.requests["late"].output == reference_generate(oracle, [256, 9], 2)
    assert len(calls) == len(engine.history), "One paged model invocation per nonempty step"
    assert [h["tokens"] for h in engine.history] == calls, "Trace must describe actual execution"
    assert engine.pool.n_free == engine.pool.num_blocks


@pytest.mark.examples
def test_zero_eos_cancellation_and_ids(api, model, oracle):
    engine = api.engine.Engine(
        model, token_budget=2, prefill_chunk=2, max_sequences=1, prefix_caching=False
    )
    zero = engine.add_request("zero", [1], 0)
    assert zero.status == "finished" and zero.output == [] and not engine.has_work
    eos = reference_generate(oracle, [1], 1)[0]
    engine.add_request("eos", [1], 8, api.sampling.SamplingParams(eos_id=eos))
    assert engine.run()["eos"] == [eos]
    assert engine.requests["eos"].finish_reason == "eos"
    engine.add_request("active", [2] * 12, 8)
    engine.add_request("waiting", [3], 2)
    engine.step()
    engine.cancel("active")
    engine.cancel("active")
    engine.cancel("waiting")
    assert engine.requests["active"].status == engine.requests["waiting"].status == "cancelled"
    assert not engine.has_work and engine.pool.n_free == engine.pool.num_blocks
    with pytest.raises(ValueError):
        engine.add_request("zero", [1], 1)


@pytest.mark.extended
@pytest.mark.parametrize("budget", [1, 5, 32])
def test_scheduling_preserves_per_request_random_stream(api, model, oracle, budget):
    params = api.sampling.SamplingParams(temperature=0.8, top_k=7, top_p=0.9, seed=39)
    engine = api.engine.Engine(model, token_budget=budget, prefill_chunk=3, prefix_caching=False)
    engine.add_request("a", [256, 1, 2, 3, 4, 5], 8, params)
    engine.add_request("b", [256, 9], 6, params)
    finish_bounded(engine)
    for req in engine.requests.values():
        assert req.output == reference_generate(oracle, req.prompt, req.max_new_tokens, params)


@pytest.mark.extended
def test_impossible_requests_rejected_and_pressure_makes_progress(api, model):
    engine = api.engine.Engine(
        model, num_blocks=4, block_size=4, token_budget=3, prefill_chunk=2, prefix_caching=False
    )
    for prompt, count in [([], 1), ([999], 1), ([1], -1), ([1] * 20, 1), ([1], 600)]:
        with pytest.raises(ValueError):
            engine.add_request("invalid", prompt, count)
    for i in range(8):
        engine.add_request(str(i), [i] * 9, 5)
    finish_bounded(engine)
    assert all(r.status == "finished" for r in engine.requests.values())
    assert engine.pool.n_free == 4


@pytest.mark.extended
def test_randomized_arrivals_cancel_and_progress(api, model, oracle, case_seed):
    rng = random.Random(case_seed)
    engine = api.engine.Engine(
        model, num_blocks=12, block_size=4, token_budget=5, prefill_chunk=3, prefix_caching=False
    )
    for i in range(20):
        engine.add_request(
            str(i), [rng.randrange(20) for _ in range(rng.randrange(1, 13))], rng.randrange(1, 6)
        )
        engine.step()
        if i % 4 == 0:
            engine.cancel(str(i))
        engine.pool.check()
    finish_bounded(engine)
    for req in engine.requests.values():
        if req.status == "finished":
            assert req.output == reference_generate(oracle, req.prompt, req.max_new_tokens)
    assert engine.pool.n_free == 12


@pytest.mark.extended
def test_step_events_correspond_to_actual_new_outputs(api, model):
    engine = api.engine.Engine(model, token_budget=5, prefill_chunk=2, prefix_caching=False)
    engine.add_request("a", [1, 2, 3], 3)
    engine.add_request("b", [4, 5], 2)
    for _ in range(100):
        if not engine.has_work:
            break
        before = {rid: len(r.output) for rid, r in engine.requests.items()}
        events = engine.step()
        expected = {
            (rid, r.output[-1]) for rid, r in engine.requests.items() if len(r.output) > before[rid]
        }
        assert {(e.request_id, e.token_id) for e in events} == expected
        assert len(events) == len(expected), "Duplicate or omitted token event"
        for event in events:
            req = engine.requests[event.request_id]
            assert event.finished == (req.status == "finished")
            assert event.finish_reason == req.finish_reason
    assert not engine.has_work
