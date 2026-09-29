"""Supplied configuration contract; no model execution logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 258
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 128
    max_seq_len: int = 512

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.dim,
                self.n_layers,
                self.n_heads,
                self.n_kv_heads,
                self.hidden_dim,
                self.max_seq_len,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("dim must divide by heads; query heads must divide by KV heads")
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    def kv_bytes_per_token(self, bytes_per_element: int = 4):
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_element
