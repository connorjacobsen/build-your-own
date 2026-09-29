import copy
import hashlib
import math
import pytest
import torch
from torch import nn


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(258, 8)
        self.head = nn.Linear(8, 258)

    def forward(self, x):
        return self.head(self.embedding(x))


def test_01(api):
    ids, mask = api.chat_tokens(
        [dict(role="user", content="é"), dict(role="assistant", content="ok")]
    )
    assert ids == [256] + list("<|user|>\né".encode()) + [257] + list(b"<|assistant|>\nok") + [257]
    assert [x for x, m in zip(ids, mask) if m] == list(b"ok") + [257]
    with pytest.raises(ValueError):
        api.chat_tokens([dict(role="tool", content="x")])


def test_02(api):
    b = api.collate([([256, 1, 2], [False, False, True]), ([256, 3], [False, True])])
    assert b["labels"].tolist() == [[-100, -100, 2], [-100, 3, -100]]
    assert b["input_ids"].tolist() == [[256, 1, 2], [256, 3, 0]] and b[
        "attention_mask"
    ].tolist() == [[True] * 3, [True, True, False]]
    with pytest.raises(ValueError):
        api.collate([([1], [True, False])])


def test_03(api, seed):
    x = torch.randn(2, 4, 7, generator=torch.Generator().manual_seed(seed), requires_grad=True)
    y = torch.tensor([[-100, -100, 2, 3], [-100, 4, -100, -100]])
    ref = torch.nn.functional.cross_entropy(
        x[:, :-1].reshape(-1, 7), y[:, 1:].reshape(-1), ignore_index=-100
    )
    loss = api.masked_loss(x, y)
    torch.testing.assert_close(loss, ref)
    grad = torch.autograd.grad(loss, x)[0]
    assert torch.count_nonzero(grad[:, -1]) == 0 and torch.count_nonzero(grad[0, 0]) == 0
    with pytest.raises(ValueError):
        api.masked_loss(x, torch.full_like(y, -100))


def test_04(api, seed):
    torch.manual_seed(seed)
    base = nn.Linear(4, 3, dtype=torch.float64)
    layer = api.LoRALinear(base, 2, 6)
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    torch.testing.assert_close(layer(x), base(x))
    assert layer.A.shape == (2, 4) and layer.B.shape == (3, 2)
    assert not base.weight.requires_grad and not base.bias.requires_grad
    layer.B.data.normal_()
    before = base.weight.clone()
    torch.testing.assert_close(layer(x), base(x) + (x @ layer.A.T @ layer.B.T) * 3)
    layer(x).sum().backward()
    assert layer.A.grad is not None and layer.B.grad is not None and base.weight.grad is None
    torch.testing.assert_close(before, base.weight)


def test_05(api):
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 2)))
    old = model[0]
    assert api.inject_lora(model, ["1.0"]) is model
    assert model[0] is old and isinstance(model[1][0], api.LoRALinear)
    fresh = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    before = fresh[0]
    with pytest.raises(ValueError):
        api.inject_lora(fresh, ["0", "1"])
    assert fresh[0] is before


def test_06(api):
    model = TokenModel()
    api.inject_lora(model, ["head"])
    selected = api.freeze_except_adapters(model)
    assert {id(p) for p in selected} == {id(model.head.A), id(model.head.B)}
    assert not model.embedding.weight.requires_grad
    with pytest.raises(ValueError):
        api.freeze_except_adapters(TokenModel())


def test_07(api, seed):
    torch.manual_seed(seed)
    model = TokenModel()
    api.inject_lora(model, ["head"])
    ps = api.freeze_except_adapters(model)
    batch = api.collate([([256, 1, 2, 257], [False, False, True, True])])
    before = {n: p.clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(ps, lr=0.1)
    loss = api.sft_step(model, opt, batch)
    assert math.isfinite(loss)
    assert not torch.equal(model.head.B, before["head.B"])
    for n, p in model.named_parameters():
        if n not in ["head.A", "head.B"]:
            torch.testing.assert_close(p, before[n])


def test_08(api, seed):
    torch.manual_seed(seed)
    layer = api.LoRALinear(nn.Linear(4, 3), 2, 3)
    layer.B.data.normal_()
    x = torch.randn(7, 4)
    before = layer.base.weight.clone()
    merged = api.merge_lora(layer)
    assert isinstance(merged, nn.Linear)
    torch.testing.assert_close(merged(x), layer(x))
    torch.testing.assert_close(api.merge_lora(layer)(x), merged(x))
    torch.testing.assert_close(layer.base.weight, before)


def test_09(api):
    x = torch.zeros(2, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 2, -100]])
    out = api.sequence_logps(x, labels)
    torch.testing.assert_close(out, torch.tensor([-3 * math.log(5), -math.log(5)]))
    out.sum().backward()
    assert torch.count_nonzero(x.grad[1, 0]) == 0
    assert api.sequence_logps(x, torch.full_like(labels, -100)).tolist() == [0, 0]


def test_10(api):
    pc = torch.tensor([2.0, -1000.0], requires_grad=True)
    pr = torch.tensor([0.0, 1000.0], requires_grad=True)
    rc = torch.tensor([1.0, 0.0], requires_grad=True)
    rr = torch.zeros(2, requires_grad=True)
    loss = api.dpo_loss(pc, pr, rc, rr, 0.5)
    expected = (torch.nn.functional.softplus(torch.tensor(-0.5)) + torch.tensor(1000.0)) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all() and rc.grad is None and rr.grad is None
    with pytest.raises(ValueError):
        api.dpo_loss(pc, pr, rc, rr, 0)


def test_11(api, seed):
    torch.manual_seed(seed)
    model = TokenModel()
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    c = api.collate([([256, 1, 2], [False, False, True])])
    r = api.collate([([256, 1, 3], [False, False, True])])
    before = {k: v.clone() for k, v in reference.state_dict().items()}
    loss = api.dpo_step(model, reference, opt, c, r, beta=1)
    assert loss == pytest.approx(math.log(2))
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(p.grad is None for p in reference.parameters())
    margin = (
        api.sequence_logps(model(c["input_ids"]), c["labels"])
        - api.sequence_logps(model(r["input_ids"]), r["labels"])
    ).item()
    baseline = (
        api.sequence_logps(reference(c["input_ids"]), c["labels"])
        - api.sequence_logps(reference(r["input_ids"]), r["labels"])
    ).item()
    assert margin > baseline
    expected_second = torch.nn.functional.softplus(torch.tensor(-(margin - baseline))).item()
    second = api.dpo_step(model, reference, opt, c, r, beta=1)
    assert second == pytest.approx(expected_second, rel=1e-5)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_12(api):
    model = TokenModel()
    api.inject_lora(model, ["head"])
    state = api.adapter_state(model)
    assert set(state) == {"head.A", "head.B"} and all(
        not x.requires_grad and x.device.type == "cpu" for x in state.values()
    )
    saved = model.head.A.clone()
    state["head.A"].add_(3)
    torch.testing.assert_close(model.head.A, saved)


def test_13(api, tmp_path):
    model = TokenModel()
    api.inject_lora(model, ["head"])
    path = tmp_path / "adapter.pt"
    digest = api.save_adapter(path, model, "a" * 64)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    r = torch.load(path, weights_only=True)
    assert (
        r["format"] == "toyadapter-v1"
        and r["base_sha256"] == "a" * 64
        and r["config"] == {"head": {"rank": 2, "alpha": 2.0}}
    )
    with pytest.raises(ValueError):
        api.save_adapter(path, model, "bad")


def test_14(api, tmp_path):
    model = TokenModel()
    api.inject_lora(model, ["head"])
    model.head.B.data.fill_(2)
    path = tmp_path / "adapter.pt"
    api.save_adapter(path, model, "a" * 64)
    other = TokenModel()
    api.inject_lora(other, ["head"])
    base = other.head.base.weight.clone()
    api.load_adapter(path, other, "a" * 64)
    torch.testing.assert_close(other.head.B, model.head.B)
    torch.testing.assert_close(base, other.head.base.weight)
    before = {k: v.clone() for k, v in other.state_dict().items()}
    with pytest.raises(ValueError):
        api.load_adapter(path, other, "b" * 64)
    for k, v in other.state_dict().items():
        torch.testing.assert_close(v, before[k])
    r = torch.load(path, weights_only=True)
    r["state"]["head.B"] = torch.ones(999, 2)
    torch.save(r, path)
    with pytest.raises(ValueError):
        api.load_adapter(path, other, "a" * 64)
    for k, v in other.state_dict().items():
        torch.testing.assert_close(v, before[k])
