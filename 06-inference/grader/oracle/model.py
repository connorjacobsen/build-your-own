"""A real, small decoder: RMSNorm, RoPE, GQA, SwiGLU, and three execution paths.

Dense, contiguous-cache, and paged paths use exactly the same model weights.
Shapes use token-major [tokens, heads, head_dim] order outside attention.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rope(x, positions, base=10000.0):
    """Rotate adjacent pairs. x: [T,H,D], positions: [T] absolute positions."""
    frequency = base ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
    angles = positions.float()[:, None, None] * frequency[None, None, :]
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


def attention(q, k, v, query_start=0):
    """Causal GQA with an explicit offset mask; handles cached multi-token chunks."""
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(q.shape[-1])
    query_positions = query_start + torch.arange(q.shape[0], device=q.device)
    key_positions = torch.arange(k.shape[0], device=q.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    scores = scores.float().masked_fill(~visible[None], float("-inf"))
    probabilities = scores.softmax(dim=-1).to(v.dtype)
    return torch.einsum("hts,shd->thd", probabilities, v)


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def project(self, x, positions):
        z = self.attn_norm(x)
        cfg = self.cfg
        q = rope(self.q(z).view(-1, cfg.n_heads, cfg.head_dim), positions)
        k = rope(self.k(z).view(-1, cfg.n_kv_heads, cfg.head_dim), positions)
        v = self.v(z).view(-1, cfg.n_kv_heads, cfg.head_dim)
        return q, k, v

    def finish(self, x, attended):
        x = x + self.o(attended.reshape(-1, self.cfg.dim))
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class TinyLM(nn.Module):
    def __init__(self, cfg=TinyConfig()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    @property
    def device(self):
        return self.embedding.weight.device

    def _ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("expected a nonempty, one-dimensional token sequence")
        return ids

    def forward(self, ids, past=None):
        """Return [new_tokens,vocab] logits and per-layer contiguous (K,V).

        past=None is the dense baseline. Cache is returned, never mutated.
        Calling this path with gradients enabled also supports the tiny training lab.
        """
        ids = self._ids(ids)
        start = 0 if past is None else past[0][0].shape[0]
        if start + len(ids) > self.cfg.max_seq_len:
            raise ValueError("context length exceeded")
        if past is not None and len(past) != self.cfg.n_layers:
            raise ValueError("one KV pair per layer is required")
        positions = torch.arange(start, start + len(ids), device=self.device)
        x, cache = self.embedding(ids), []
        for i, layer in enumerate(self.layers):
            q, k, v = layer.project(x, positions)
            if past is not None:
                k = torch.cat((past[i][0], k), dim=0)
                v = torch.cat((past[i][1], v), dim=0)
            x = layer.finish(x, attention(q, k, v, start))
            cache.append((k, v))
        return self.lm_head(self.norm(x)), cache

    @torch.inference_mode()
    def forward_batch(self, sequences, pasts=None):
        if not sequences:
            return [], []
        ids = [self._ids(tokens) for tokens in sequences]
        pasts = pasts if pasts is not None else [None] * len(ids)
        if len(pasts) != len(ids):
            raise ValueError("one cache per sequence required")
        starts = [0 if past is None else past[0][0].shape[0] for past in pasts]
        if any(start + len(tokens) > self.cfg.max_seq_len for start, tokens in zip(starts, ids)):
            raise ValueError("context exceeded")
        sizes = [len(tokens) for tokens in ids]
        positions = torch.cat(
            [
                torch.arange(start, start + size, device=self.device)
                for start, size in zip(starts, sizes)
            ]
        )
        x = self.embedding(torch.cat(ids))
        caches = [[] for _ in ids]
        for layer_id, layer in enumerate(self.layers):
            q, k, v = layer.project(x, positions)
            outputs, offset = [], 0
            for i, (size, start, past) in enumerate(zip(sizes, starts, pasts)):
                sl = slice(offset, offset + size)
                ki, vi = k[sl], v[sl]
                if past is not None:
                    ki = torch.cat([past[layer_id][0], ki])
                    vi = torch.cat([past[layer_id][1], vi])
                outputs.append(attention(q[sl], ki, vi, start))
                caches[i].append((ki, vi))
                offset += size
            x = layer.finish(x, torch.cat(outputs))
        return list(self.lm_head(self.norm(x)).split(sizes)), caches

    @torch.inference_mode()
    def forward_paged_batch(self, items, pool):
        """Packed projections, separate per-request attention, real paged KV storage.

        items contains (new_token_ids, block_table, num_computed_tokens).
        Gathering K/V makes this a correctness reference, not a fast paged kernel.
        """
        if not items:
            return []
        ids = [self._ids(item[0]) for item in items]
        for tokens, (_, table, start) in zip(ids, items):
            if start < 0 or start + len(tokens) > self.cfg.max_seq_len:
                raise ValueError("invalid context length")
            if len(table) * pool.block_size < start + len(tokens):
                raise ValueError("insufficient reserved blocks")
        positions = torch.cat(
            [
                torch.arange(start, start + len(tokens), device=self.device)
                for tokens, (_, _, start) in zip(ids, items)
            ]
        )
        sizes = [len(t) for t in ids]
        x = self.embedding(torch.cat(ids))
        for layer_id, layer in enumerate(self.layers):
            q, k, v = layer.project(x, positions)
            outputs, offset = [], 0
            for size, (_, table, start) in zip(sizes, items):
                sl = slice(offset, offset + size)
                pool.write(layer_id, table, start, k[sl], v[sl])
                all_k, all_v = pool.read(layer_id, table, start + size)
                outputs.append(attention(q[sl], all_k, all_v, start))
                offset += size
            x = layer.finish(x, torch.cat(outputs))
        return list(self.lm_head(self.norm(x)).split(sizes))


def make_model(seed=7, cfg=None, device="cpu"):
    """Seed initialization without changing the caller's global CPU random stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyLM(cfg or TinyConfig())
    return model.to(device).eval()
