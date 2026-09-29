"""Explicit timing boundaries and shape/work estimates used by the labs."""

from statistics import median
from time import perf_counter
import torch


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(fn, device="cpu", warmup=2, repeats=5):
    if repeats < 1 or warmup < 0:
        raise ValueError("invalid timing repetition count")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()
        fn()
        synchronize(device)
        samples.append(perf_counter() - start)
    return {"median_seconds": median(samples), "samples_seconds": samples}


def projection_token_work(prompt_length, output_length):
    """Number of token positions whose projections run (not FLOPs)."""
    if prompt_length < 1 or output_length < 0:
        raise ValueError("invalid lengths")
    return {
        "naive": output_length * prompt_length + output_length * (output_length - 1) // 2,
        "cached": prompt_length + output_length - 1 if output_length else 0,
    }


def engine_metrics(engine):
    completed = [r for r in engine.requests.values() if r.status == "finished" and r.output]
    if not completed:
        return {"output_tokens": 0, "ttft_seconds": [], "itl_seconds": []}
    end = max(r.finished_at for r in completed)
    start = min(r.arrival for r in completed)
    tokens = sum(len(r.output) for r in completed)
    return {
        "output_tokens": tokens,
        "elapsed_seconds": end - start,
        "output_tokens_per_second": tokens / (end - start),
        "ttft_seconds": [r.ttft for r in completed],
        "itl_seconds": [gap for r in completed for gap in r.inter_token_latencies],
        "cached_prompt_tokens": sum(r.cached_tokens for r in completed),
    }
