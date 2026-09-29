"""Portable artifact loading: learner stage 30."""

import torch
from .model import TinyLM, TinyConfig


def load_export(path, device="cpu"):
    """Load trusted toylm-v1 export with weights_only=True. Validate format and exact byte
    tokenizer metadata, config.vocab_size=258, then construct TinyLM and strict-load state_dict.
    Return eval-mode model on requested device. Preserve caller CPU RNG state.
    Incompatible metadata raises ValueError; incompatible parameter keys raise RuntimeError.
    """
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if (
        payload.get("format") != "toylm-v1"
        or payload.get("tokenizer")
        != dict(kind="utf8-byte", bos_id=256, eos_id=257, vocab_size=258)
        or payload["config"]["vocab_size"] != 258
    ):
        raise ValueError("Incompatible artifact")
    with torch.random.fork_rng(devices=[]):
        model = TinyLM(TinyConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()
