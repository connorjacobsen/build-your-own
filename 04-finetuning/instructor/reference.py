"""Instructor reference: instruction tuning, adapters and preference optimization."""

import hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def chat_tokens(messages):
    """Return (ids, assistant_mask) lists. Start with BOS=256 (mask false).
    For each {'role': system|user|assistant, 'content':str}, append UTF-8 bytes of
    '<|ROLE|>\n' (mask false), UTF-8 content (mask true iff assistant), then EOS=257
    (mask true iff assistant). Reject empty message lists, unknown roles or non-string content.
    Literal role headers are ordinary bytes, not additional vocabulary IDs.
    """
    if not messages:
        raise ValueError("Messages required")
    ids = [256]
    mask = [False]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Invalid message")
        header = list(f"<|{role}|>\n".encode())
        body = list(content.encode()) + [257]
        ids.extend(header + body)
        mask.extend([False] * len(header) + [role == "assistant"] * len(body))
    return ids, mask


def collate(examples, pad_id=0):
    """examples are (ids,assistant_mask) pairs with equal nonzero lengths. Return dict
    input_ids long [B,T], labels long [B,T] (original ID if mask true, else -100),
    attention_mask bool [B,T]. Right-pad; padding labels=-100, attention=false.
    Reject empty batches, empty examples and mismatched lengths. Do not shift labels here.
    """
    if not examples or any(not ids or len(ids) != len(mask) for ids, mask in examples):
        raise ValueError("Invalid examples")
    size = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), size), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    for i, (tokens, mask) in enumerate(examples):
        n = len(tokens)
        ids[i, :n] = torch.tensor(tokens)
        attention[i, :n] = True
        labels[i, :n] = torch.where(torch.tensor(mask, dtype=torch.bool), ids[i, :n], -100)
    return dict(input_ids=ids, labels=labels, attention_mask=attention)


def masked_loss(logits, labels):
    """Next-token mean CE: logits [B,T,V] at :-1 predict labels at 1:.
    Ignore labels=-100, divide by total supervised tokens across batch. Raise ValueError
    when no shifted target is supervised. Return differentiable scalar tensor.
    """
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any():
        raise ValueError("No supervised target tokens")
    logp = logits[:, :-1].log_softmax(-1)
    selected = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -selected[mask].mean()


class LoRALinear(nn.Module):
    """Wrap an nn.Linear as base. Require rank>=1 and alpha>0. Freeze base parameters.
    A is nn.Parameter [rank,in], Kaiming-uniform initialized; B is zeros [out,rank].
    Both match base dtype/device. Forward: base(x) + (x @ A.T @ B.T)*(alpha/rank).
    Arbitrary leading input dimensions are supported. Do not modify base weights in forward.
    """

    def __init__(self, base, rank=2, alpha=2.0):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("Invalid adapter")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * (self.alpha / self.rank)


def inject_lora(model, targets, rank=2, alpha=2.0):
    """In-place replace exact qualified names of nn.Linear modules with LoRALinear.
    Validate every name and type before changing any module. Reject duplicates, missing names,
    empty targets and non-linear targets. Return the same model. Other modules stay untouched.
    """
    modules = dict(model.named_modules())
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(not n or not isinstance(modules.get(n), nn.Linear) for n in targets)
    ):
        raise ValueError("Invalid targets")
    for name in targets:
        parent, _, leaf = name.rpartition(".")
        setattr(modules[parent], leaf, LoRALinear(modules[name], rank, alpha))
    return model


def freeze_except_adapters(model):
    """Freeze all parameters except actual LoRALinear A/B objects. Return unique trainable
    Parameters in model.parameters() order. Reject models without adapters. Names alone
    must not cause unrelated parameters named A or B to become trainable.
    """
    allowed = {id(p) for m in model.modules() if isinstance(m, LoRALinear) for p in [m.A, m.B]}
    if not allowed:
        raise ValueError("No adapters found")
    selected = []
    for p in model.parameters():
        p.requires_grad_(id(p) in allowed)
        if p.requires_grad:
            selected.append(p)
    return selected


def sft_step(model, optimizer, batch):
    """One optimizer update using model(input_ids)->[B,T,V] logits and masked_loss.
    Set train mode, clear old grads, backpropagate, step; return pre-update float loss.
    The teaching model interface needs no attention mask because test fixtures are causal
    positionwise networks; capstone wrappers must handle padding and document isolation.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = masked_loss(model(batch["input_ids"]), batch["labels"])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def merge_lora(layer):
    """Return a new ordinary nn.Linear on the same device/dtype with W + alpha/rank * B@A
    and the same bias. Preserve the input adapter and all its parameters. Merging twice
    from the same original gives the same result; do not add the delta into base in place.
    """
    base = layer.base
    out = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        out.weight.copy_(base.weight + (layer.B @ layer.A) * (layer.alpha / layer.rank))
        if base.bias is not None:
            out.bias.copy_(base.bias)
    return out


def sequence_logps(logits, labels):
    """Return [B] SUM of next-token log probabilities for labels[:,1:] excluding -100.
    An unsupervised row contributes zero. Keep gradients; do not length-normalize for DPO.
    """
    target = labels[:, 1:]
    valid = target != -100
    chosen = (
        logits[:, :-1].log_softmax(-1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    )
    return (chosen * valid).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """Mean -logsigmoid(beta*((pc-pr)-(rc-rr))). All arguments [B]; beta>0.
    Detach reference terms so the frozen baseline receives no gradient. Stable for large margins.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected).detach()
    return -F.logsigmoid(beta * margin).mean()


def dpo_step(policy, reference, optimizer, chosen, rejected, beta=0.1):
    """chosen/rejected are collate-style batches of equal batch size. One policy update
    from summed response log probabilities. Run reference in eval mode under no_grad;
    clear policy gradients, leave reference parameters unchanged, return float DPO loss.
    """
    policy.train()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    pc = sequence_logps(policy(chosen["input_ids"]), chosen["labels"])
    pr = sequence_logps(policy(rejected["input_ids"]), rejected["labels"])
    with torch.no_grad():
        rc = sequence_logps(reference(chosen["input_ids"]), chosen["labels"])
        rr = sequence_logps(reference(rejected["input_ids"]), rejected["labels"])
    loss = dpo_loss(pc, pr, rc, rr, beta)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def adapter_state(model):
    """Return dict qualified_name+'.A'/'.B' -> detached CPU cloned tensors for actual
    LoRALinear modules only. Reject models with no adapters. No base weights in this state.
    """
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            state[name + ".A"] = m.A.detach().cpu().clone()
            state[name + ".B"] = m.B.detach().cpu().clone()
    if not state:
        raise ValueError("No adapters")
    return state


def save_adapter(path, model, base_sha256):
    """Save trusted local torch payload format='toyadapter-v1', base_sha256 (64 hex chars),
    state=adapter_state(model), config={module_name:{rank,alpha}}. Return SHA256 file digest.
    Reject malformed base digest. The base digest identifies a toylm-v1 artifact externally.
    """
    if len(base_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_sha256):
        raise ValueError("Invalid digest")
    config = {
        n: dict(rank=m.rank, alpha=m.alpha)
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    }
    torch.save(
        dict(
            format="toyadapter-v1",
            base_sha256=base_sha256,
            state=adapter_state(model),
            config=config,
        ),
        path,
    )
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(path, model, expected_base_sha256):
    """Load onto an already injected matching model. Validate format, base digest, config,
    all adapter keys and shapes BEFORE modifying anything; mismatch raises ValueError.
    Copy tensors in place under no_grad, converting to destination dtype/device. Base stays fixed.
    """
    record = torch.load(path, weights_only=True, map_location="cpu")
    current = adapter_state(model)
    config = {
        n: dict(rank=m.rank, alpha=m.alpha)
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    }
    if (
        record.get("format") != "toyadapter-v1"
        or record.get("base_sha256") != expected_base_sha256
        or record.get("config") != config
        or set(record.get("state", {})) != set(current)
        or any(record["state"][k].shape != v.shape for k, v in current.items())
    ):
        raise ValueError("Incompatible adapter")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in record["state"].items():
            params[k].copy_(v)
