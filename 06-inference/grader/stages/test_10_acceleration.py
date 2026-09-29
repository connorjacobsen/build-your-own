import pytest
import torch
from grader.oracle.model import attention as reference_attention


@pytest.mark.examples
def test_benchmark_warmup_sample_count_and_validation(api):
    calls = []
    result = api.metrics.benchmark(lambda: calls.append(1), warmup=2, repeats=5)
    assert len(calls) == 7
    assert len(result["samples_seconds"]) == 5
    assert all(t >= 0 for t in result["samples_seconds"])
    assert result["median_seconds"] == sorted(result["samples_seconds"])[2]
    with pytest.raises(ValueError):
        api.metrics.benchmark(lambda: None, repeats=0)


@pytest.mark.extended
def test_device_timing_brackets_work_with_synchronization(api, monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: events.append("sync"))
    api.metrics.benchmark(lambda: events.append("work"), device="cuda", warmup=0, repeats=2)
    assert events == ["sync", "work", "sync", "sync", "work", "sync"], (
        "Timed accelerator work must be bracketed by synchronization"
    )


@pytest.fixture
def require_cuda():
    if not torch.cuda.is_available():
        pytest.fail(
            "Stage 10 requires NVIDIA CUDA: run the supplied Modal runner or a CUDA host. This is not a passing skip."
        )


@pytest.mark.gpu
@pytest.mark.timeout(60)
@pytest.mark.examples
@torch.inference_mode()
def test_cuda_attention_gqa_and_offset(api, require_cuda):
    for start, count, total in [(0, 5, 5), (6, 3, 9), (22, 1, 23)]:
        q, k, v = (
            torch.randn(count, 4, 16, device="cuda"),
            torch.randn(total, 2, 16, device="cuda"),
            torch.randn(total, 2, 16, device="cuda"),
        )
        actual = api.model.attention(q, k, v, start)
        assert actual.device.type == "cuda"
        torch.testing.assert_close(
            actual, reference_attention(q, k, v, start), atol=2e-5, rtol=2e-4
        )


@pytest.mark.gpu
@pytest.mark.timeout(60)
@pytest.mark.extended
@torch.inference_mode()
def test_cuda_engine_matches_cpu_and_keeps_kv_on_device(api, model, oracle, require_cuda):
    from grader.oracle.sampling import generate_naive

    model = model.cuda()
    engine = api.engine.Engine(model, num_blocks=12, block_size=4, token_budget=5, prefill_chunk=3)
    for rid, prompt in [("a", [256, 1, 2, 3, 4]), ("b", [256, 7])]:
        engine.add_request(rid, prompt, 5)
    result = engine.run()
    assert engine.pool.k.device.type == engine.pool.v.device.type == "cuda"
    for rid, req in engine.requests.items():
        assert result[rid] == generate_naive(oracle, req.prompt, 5)
    engine.clear_prefix_cache()
    assert engine.pool.n_free == 12
