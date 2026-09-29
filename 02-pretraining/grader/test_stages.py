import copy
import hashlib
import math
import pytest
import torch


def make(api, seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return api.TinyLM(
            api.TinyConfig(dim=16, n_layers=1, n_heads=2, n_kv_heads=1, hidden_dim=32)
        )


def test_01(api):
    assert api.encode_bytes("é") == [256, 195, 169, 257]
    assert api.encode_bytes("") == [256, 257]


def test_02(api):
    x, y = api.next_token_batch([1, 2, 3, 4, 5], [0, 2], 2)
    assert x.tolist() == [[1, 2], [3, 4]] and y.tolist() == [[2, 3], [4, 5]]
    assert x.dtype == y.dtype == torch.long
    for starts, ctx in [([], 2), ([-1], 2), ([3], 2), ([0], 0)]:
        with pytest.raises(ValueError):
            api.next_token_batch([1, 2, 3, 4, 5], starts, ctx)


def test_03(api, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(2, 3, 7, generator=g) * 100).requires_grad_()
    y = torch.randint(7, (2, 3), generator=g)
    got = api.token_loss(x, y)
    ref = torch.nn.functional.cross_entropy(x.flatten(0, 1), y.flatten())
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(torch.autograd.grad(got, x)[0], torch.autograd.grad(ref, x)[0])


def test_04(api):
    n = api.RMSNorm(4)
    n.weight.data.copy_(torch.tensor([1, 2, 3, 4.0]))
    x = torch.tensor([[0.0, 0, 0, 0], [1.0, 2, 3, 4]])
    torch.testing.assert_close(
        n(x), x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6) * n.weight
    )


def test_05(api):
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0]]])
    p = torch.tensor([0, 3])
    y = api.rope(x, p)
    angles = p[:, None, None] * torch.tensor([1.0, 0.01])[None, None, :]
    expected = torch.stack(
        [
            x[..., ::2] * angles.cos() - x[..., 1::2] * angles.sin(),
            x[..., 1::2] * angles.cos() + x[..., ::2] * angles.sin(),
        ],
        -1,
    ).flatten(-2)
    torch.testing.assert_close(y, expected)


def test_06(api, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(3, 4, 4, generator=g)
    k = torch.randn(5, 2, 4, generator=g)
    v = torch.randn(5, 2, 4, generator=g)
    out = api.attention(q, k, v, 2)
    for i in range(3):
        for h in range(4):
            expected = (k[: i + 3, h // 2] @ q[i, h] / 2).softmax(0) @ v[: i + 3, h // 2]
            torch.testing.assert_close(out[i, h], expected)


def test_07(api, seed):
    torch.manual_seed(seed)
    cfg = api.TinyConfig(dim=16, n_heads=2, n_kv_heads=1, hidden_dim=32)
    b = api.DecoderLayer(cfg)
    x = torch.randn(5, 16)
    q, k, v = b.project(x, torch.arange(5))
    assert q.shape == (5, 2, 8) and k.shape == v.shape == (5, 1, 8)
    z = b.attn_norm(x)
    torch.testing.assert_close(q, api.rope(b.q(z).view(5, 2, 8), torch.arange(5)))
    torch.testing.assert_close(k, api.rope(b.k(z).view(5, 1, 8), torch.arange(5)))
    torch.testing.assert_close(v, b.v(z).view(5, 1, 8))
    attended = torch.randn(5, 2, 8)
    residual = x + b.o(attended.flatten(1))
    z = b.ffn_norm(residual)
    torch.testing.assert_close(
        b.finish(x, attended), residual + b.down(torch.nn.functional.silu(b.gate(z)) * b.up(z))
    )


def test_08(api, seed):
    m = make(api, seed)
    a, _ = m([256, 1, 2, 3])
    b, _ = m([256, 1, 9, 8])
    torch.testing.assert_close(a[:2], b[:2])
    assert a.shape == (4, 258)
    x = m.embedding(torch.tensor([256, 1, 2, 3]))
    pos = torch.arange(4)
    for layer in m.layers:
        q, k, v = layer.project(x, pos)
        x = layer.finish(x, api.attention(q, k, v))
    torch.testing.assert_close(a, m.lm_head(m.norm(x)))
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_09(api, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(3, 4, generator=g)
    ref = torch.nn.Parameter(p.clone())
    opt = torch.optim.AdamW([ref], lr=0.02, weight_decay=0.1)
    state = {}
    for _ in range(4):
        grad = torch.randn(3, 4, generator=g)
        saved = grad.clone()
        ref.grad = grad.clone()
        opt.step()
        api.adamw_step([p], [grad], state, 0.02, weight_decay=0.1)
        torch.testing.assert_close(p, ref)
        torch.testing.assert_close(grad, saved)
    assert state["step"] == 4


def test_10(api):
    assert api.cosine_lr(0, 2, 10, 1) == 0 and api.cosine_lr(2, 2, 10, 1) == 1
    assert api.cosine_lr(6, 2, 10, 1, 0.1) == pytest.approx(0.55)
    assert api.cosine_lr(99, 2, 10, 1, 0.1) == pytest.approx(0.1)
    assert api.cosine_lr(0, 0, 10, 1) == 1
    with pytest.raises(ValueError):
        api.cosine_lr(1, 3, 3, 1)


def test_11(api):
    a = torch.nn.Parameter(torch.ones(2))
    a.grad = torch.tensor([3.0, 4.0])
    b = torch.nn.Parameter(torch.ones(1))
    assert api.clip_grad([a, b], 2) == pytest.approx(5)
    torch.testing.assert_close(a.grad, torch.tensor([1.2, 1.6]))
    a.grad[0] = float("nan")
    with pytest.raises(ValueError):
        api.clip_grad([a], 1)


def test_12(api):
    a = make(api)
    b = copy.deepcopy(a)
    seqs = [[1, 2, 3], [4, 5, 6, 7, 8]]
    opt = torch.optim.SGD(a.parameters(), lr=0.03)
    refopt = torch.optim.SGD(b.parameters(), lr=0.03)
    terms = []
    for s in seqs:
        logits, _ = b(s[:-1])
        terms.append(
            torch.nn.functional.cross_entropy(logits, torch.tensor(s[1:]), reduction="sum")
        )
    expected = sum(terms) / 6
    expected.backward()
    refopt.step()
    got = api.train_step(a, opt, seqs)
    assert got == pytest.approx(expected.item())
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)
    with pytest.raises(ValueError):
        api.train_step(a, opt, [[1]])


def test_13(api, tmp_path):
    m = make(api)
    opt = torch.optim.AdamW(m.parameters())
    api.train_step(m, opt, [[1, 2, 3]])
    path = tmp_path / "resume.pt"
    api.save_checkpoint(path, m, opt, 7)
    r = torch.load(path, weights_only=True)
    assert r["step"] == 7 and r["optimizer"]["state"]
    torch.testing.assert_close(r["rng"], torch.get_rng_state())
    assert set(r["model"]) == set(m.state_dict())


def test_14(api, tmp_path):
    m = make(api)
    opt = torch.optim.AdamW(m.parameters(), lr=0.01)
    api.train_step(m, opt, [[1, 2, 3]])
    path = tmp_path / "resume.pt"
    api.save_checkpoint(path, m, opt, 1)
    expected_rng = torch.rand(3)
    expected = api.train_step(m, opt, [[2, 3, 4]])
    other = make(api, 9)
    otheropt = torch.optim.AdamW(other.parameters(), lr=0.9)
    assert api.load_checkpoint(path, other, otheropt) == 1
    torch.testing.assert_close(torch.rand(3), expected_rng)
    assert api.train_step(other, otheropt, [[2, 3, 4]]) == pytest.approx(expected)
    for a, b in zip(m.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)


def test_15(api, seed):
    state = torch.get_rng_state().clone()
    m, loss = api.train_run([[256, 1, 2, 3, 257]] * 2, steps=25, seed=seed)
    assert len(loss) == 25 and all(math.isfinite(v) for v in loss) and loss[-1] < loss[0] * 0.3
    torch.testing.assert_close(torch.get_rng_state(), state)
    _, again = api.train_run([[256, 1, 2, 3, 257]] * 2, steps=25, seed=seed)
    assert loss == again


def test_16(api, tmp_path):
    m = make(api)
    path = tmp_path / "model.pt"
    digest = api.export_model(path, m, {"dataset_sha256": "abc", "seed": 7})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    record = torch.load(path, weights_only=True)
    assert record["format"] == "toylm-v1"
    assert record["tokenizer"] == dict(kind="utf8-byte", bos_id=256, eos_id=257, vocab_size=258)
    other = api.TinyLM(api.TinyConfig(**record["config"]))
    other.load_state_dict(record["state_dict"], strict=True)
    torch.testing.assert_close(m([1, 2, 3])[0], other([1, 2, 3])[0])
    assert record["provenance"]["seed"] == 7
