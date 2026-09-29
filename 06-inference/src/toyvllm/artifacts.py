"""Portable artifact loading: learner stage 30."""
import torch
from .model import TinyLM, TinyConfig

def load_export(path, device='cpu'):
    """Load trusted toylm-v1 export with weights_only=True. Validate format and exact byte
    tokenizer metadata, config.vocab_size=258, then construct TinyLM and strict-load state_dict.
    Return eval-mode model on requested device. Preserve caller CPU RNG state.
    Incompatible metadata raises ValueError; incompatible parameter keys raise RuntimeError.
    """
    raise NotImplementedError('Stage 30: load a trained artifact')
