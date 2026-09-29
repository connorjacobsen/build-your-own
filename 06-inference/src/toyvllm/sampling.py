"""Stage 03: implement sampling and dense generation; stage 04: reuse K/V."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    eos_id: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not isinstance(self.top_k, int) or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("top_k >= 0 and 0 < top_p <= 1 required")


def sample(logits, params=SamplingParams(), generator=None):
    """One finite [vocab] vector -> Python token ID; see stage 03 for filter order."""
    raise NotImplementedError("Stage 03: implement greedy / temperature / top-k / top-p sampling")


def generate_naive(model, prompt, max_new_tokens, params=SamplingParams()):
    """Return generated IDs only; reference behavior starts with full-history recomputation."""
    raise NotImplementedError("Stage 03: implement autoregressive generation")


def generate_cached(model, prompt, max_new_tokens, params=SamplingParams()):
    """Same output contract; after prefill, model inputs must have length one."""
    raise NotImplementedError("Stage 04: implement incremental generation")
