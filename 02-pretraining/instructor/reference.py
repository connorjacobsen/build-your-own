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


def encode_bytes(text):
    """UTF-8 bytes with BOS=256 prepended and EOS=257 appended; return list[int]."""
    return [256] + list(text.encode("utf-8")) + [257]


def next_token_batch(tokens, starts, context):
    """From one 1D token stream, return long X,Y tensors [len(starts),context].
    X starts at each specified offset; Y is shifted one position. Reject empty starts,
    context<1, negative starts and windows extending past the available next token.
    """
    if context < 1 or not starts or any(s < 0 or s + context >= len(tokens) for s in starts):
        raise ValueError("Invalid window")
    x = torch.tensor([tokens[s : s + context] for s in starts], dtype=torch.long)
    y = torch.tensor([tokens[s + 1 : s + context + 1] for s in starts], dtype=torch.long)
    return x, y


def token_loss(logits, targets):
    """Mean next-token cross entropy for logits [...,V] and same-prefix-shape targets.
    Compute stable logsumexp minus selected logits; do not call F.cross_entropy.
    """
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - selected).mean()


def adamw_step(parameters, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """In-place AdamW over matching lists of tensors. state is initially {}, then contains
    step (integer), m and v (tensor lists). Bias-correct both moments. Apply decoupled
    decay p *= (1-lr*weight_decay). Do not mutate gradient tensors. No torch optimizer.
    All parameters receive a dense gradient in this educational version.
    """
    if not state:
        state.update(
            step=0,
            m=[torch.zeros_like(p) for p in parameters],
            v=[torch.zeros_like(p) for p in parameters],
        )
    state["step"] += 1
    b1, b2 = betas
    with torch.no_grad():
        for p, g, m, v in zip(parameters, grads, state["m"], state["v"]):
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.mul_(1 - lr * weight_decay)
            p.addcdiv_(
                m / (1 - b1 ** state["step"]),
                (v / (1 - b2 ** state["step"])).sqrt() + eps,
                value=-lr,
            )


def cosine_lr(step, warmup, total, peak, floor=0.0):
    """0<=warmup<total; step>=0. With warmup>0, linear step/warmup * peak before warmup.
    Cosine decay from peak at warmup to floor at total; clamp later steps to floor.
    Reject invalid steps, bounds, negative floor or peak<floor.
    """
    if not 0 <= warmup < total or step < 0 or floor < 0 or peak < floor:
        raise ValueError("Invalid schedule")
    if step < warmup:
        return peak * step / warmup
    ratio = min(1.0, (step - warmup) / (total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * ratio))


def clip_grad(parameters, max_norm):
    """Clip the global L2 norm of existing gradients in place; return the preclip norm as float.
    Ignore parameters whose grad is None. Reject max_norm<=0 and nonfinite gradient norm.
    """
    if max_norm <= 0:
        raise ValueError("Positive threshold required")
    ps = [p for p in parameters if p.grad is not None]
    norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps))
    if not math.isfinite(norm):
        raise ValueError("Nonfinite gradient")
    if norm > max_norm:
        for p in ps:
            p.grad.mul_(max_norm / norm)
    return norm


def train_step(model, optimizer, sequences):
    """One update on nonempty sequences, each at least two token IDs long.
    Weight all target tokens equally, not all sequences equally. Clear old gradients,
    set training mode, backpropagate the aggregate loss, optimizer.step(), return float loss.
    """
    if not sequences or any(len(s) < 2 for s in sequences):
        raise ValueError("Each sequence needs an input and target")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = sum(len(s) - 1 for s in sequences)
    loss = 0
    for seq in sequences:
        logits, _ = model(seq[:-1])
        targets = torch.tensor(seq[1:], device=model.device)
        loss = loss + token_loss(logits, targets) * (len(seq) - 1) / count
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def save_checkpoint(path, model, optimizer, step):
    """Save dict model, optimizer, step, rng (CPU torch RNG state) using torch.save.
    This resume format is distinct from the portable inference export. Parent directory exists.
    """
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            step=step,
            rng=torch.get_rng_state(),
        ),
        path,
    )


def load_checkpoint(path, model, optimizer):
    """Restore model, optimizer, CPU RNG from our trusted local checkpoint; return saved step.
    Use torch.load(weights_only=True,map_location='cpu') and strict parameter matching.
    """
    record = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(record["model"], strict=True)
    optimizer.load_state_dict(record["optimizer"])
    torch.set_rng_state(record["rng"])
    return record["step"]


def train_run(sequences, steps=30, seed=0, cfg=None, lr=0.01):
    """Initialize TinyLM under fork_rng, AdamW(lr=lr,weight_decay=.01), then train_step
    on all supplied sequences for each update. Return model, list of pre-update losses.
    Preserve the caller's CPU torch RNG state. Default config has dim=16, n_layers=1,
    n_heads=2,n_kv_heads=1,hidden_dim=32; vocab and max length keep TinyConfig defaults.
    """
    cfg = cfg or TinyConfig(dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        losses = [train_step(model, optimizer, sequences) for _ in range(steps)]
    return model, losses


def export_model(path, model, provenance):
    """Write portable torch payload: format='toylm-v1', config=vars(model.cfg),
    tokenizer={'kind':'utf8-byte','bos_id':256,'eos_id':257,'vocab_size':258},
    state_dict=detached CPU cloned tensors, provenance=caller dict of JSON-safe metadata.
    Require cfg.vocab_size=258. Return SHA256 of the exact file bytes. No optimizer in export.
    """
    import hashlib

    if model.cfg.vocab_size != 258:
        raise ValueError("Byte vocabulary required")
    torch.save(
        dict(
            format="toylm-v1",
            config=vars(model.cfg),
            tokenizer=dict(kind="utf8-byte", bos_id=256, eos_id=257, vocab_size=258),
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            provenance=provenance,
        ),
        path,
    )
    return hashlib.sha256(__import__("pathlib").Path(path).read_bytes()).hexdigest()
