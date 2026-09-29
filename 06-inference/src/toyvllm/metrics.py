"""Stage 10: measurement is separate from the engine's correctness contract."""


def benchmark(fn, device="cpu", warmup=2, repeats=5):
    """Return median_seconds and samples_seconds. Synchronize timed device work."""
    raise NotImplementedError("Stage 10: warm up, synchronize, record repeated timings")
