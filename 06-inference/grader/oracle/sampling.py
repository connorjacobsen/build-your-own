"""Sampling is separate from both the model and the scheduler."""

from dataclasses import dataclass
import math
import torch


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
    # Sampling on CPU keeps a request-local generator independent of device and schedule.
    scores = logits.detach().float().cpu().clone()
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("expected one finite logits vector")
    if params.temperature == 0:
        return int(scores.argmax())
    scores /= params.temperature
    if params.top_k:
        k = min(params.top_k, scores.numel())
        keep = scores.topk(k).indices
        filtered = torch.full_like(scores, float("-inf"))
        filtered[keep] = scores[keep]
        scores = filtered
    if params.top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        probabilities = sorted_scores.softmax(-1)
        # Keep the token that crosses p; remove only tokens after it.
        remove = probabilities.cumsum(-1) - probabilities >= params.top_p
        scores[order[remove]] = float("-inf")
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


@torch.inference_mode()
def generate_naive(model, prompt, max_new_tokens, params=SamplingParams()):
    if not prompt or max_new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative output length required")
    tokens, output = list(prompt), []
    generator = torch.Generator().manual_seed(params.seed)
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        token = sample(logits[-1], params, generator)
        output.append(token)
        tokens.append(token)
        if token == params.eos_id:
            break
    return output


@torch.inference_mode()
def generate_cached(model, prompt, max_new_tokens, params=SamplingParams()):
    if not prompt or max_new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative output length required")
    output, past, inputs = [], None, list(prompt)
    generator = torch.Generator().manual_seed(params.seed)
    for _ in range(max_new_tokens):
        logits, past = model(inputs, past)
        token = sample(logits[-1], params, generator)
        output.append(token)
        inputs = [token]
        if token == params.eos_id:
            break
    return output
