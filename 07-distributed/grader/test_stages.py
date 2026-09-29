import copy
import importlib
from datetime import timedelta
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp


def test_01(api):
    parts = [api.partition_indices(7, r, 3) for r in range(3)]
    assert parts == [[0, 3, 6], [1, 4], [2, 5]] and sorted(sum(parts, [])) == list(range(7))
    assert api.partition_indices(1, 2, 3) == []
    with pytest.raises(ValueError):
        api.partition_indices(5, 3, 3)


def test_02(api):
    assert [api.shard_bounds(8, r, 3) for r in range(3)] == [(0, 3), (3, 6), (6, 8)]
    assert [api.shard_bounds(1, r, 3) for r in range(3)] == [(0, 1), (1, 1), (1, 1)]
    with pytest.raises(ValueError):
        api.shard_bounds(-1, 0, 2)


def test_03(api):
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    flat = api.flatten_tensors([x, y])
    torch.testing.assert_close(flat, torch.tensor([1.0, 2.0, 3.0]))
    flat.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    flat.detach()[0] = 99
    assert x[0, 0] == 1
    with pytest.raises(ValueError):
        api.flatten_tensors([x, torch.ones(2, dtype=torch.float64)])


def test_04(api):
    flat = torch.arange(7.0)
    pieces = api.unflatten(flat, [(2, 2), (), (2,)])
    assert [p.shape for p in pieces] == [(2, 2), (), (2,)]
    pieces[1].fill_(99)
    assert flat[4] == 99
    with pytest.raises(ValueError):
        api.unflatten(flat, [(3, 3)])


def test_05(api, seed):
    torch.manual_seed(seed)
    a = nn.Linear(3, 1)
    b = copy.deepcopy(a)
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)
    expected = (b(x) - y).square().mean()
    expected.backward()
    loss = api.accumulate_gradients(
        a, [(x[:1], y[:1]), (x[1:], y[1:])], lambda p, t: (p - t).square().sum()
    )
    assert loss == pytest.approx(expected.item())
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def _worker(rank, namespace, directory, mode, backend):
    api = importlib.import_module(namespace)
    directory = Path(directory)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    dist.init_process_group(
        backend,
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        if mode == "reduce":
            original = (
                torch.tensor([2.0, 4.0], device=device)
                if rank == 0
                else torch.tensor([8.0, 10.0], device=device)
            )
            result = api.weighted_allreduce(original, 1 if rank == 0 else 3)
            empty = api.weighted_allreduce(original, 0 if rank == 0 else 2)
            payload = dict(value=result.cpu(), original=original.cpu(), empty=empty.cpu())
        elif mode == "step":
            torch.manual_seed(31)
            model = nn.Linear(3, 1).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.03)
            x = torch.arange(15.0, device=device).reshape(5, 3) / 10
            y = x.sum(-1, keepdim=True)
            sl = slice(0, 1) if rank == 0 else slice(1, 5)
            loss = api.distributed_step(model, opt, x[sl], y[sl])
            # A second update includes an entirely empty rank.
            sl2 = slice(0, 0) if rank == 0 else slice(None)
            loss2 = api.distributed_step(model, opt, x[sl2], y[sl2])
            payload = dict(
                loss=loss, loss2=loss2, weights={k: v.cpu() for k, v in model.state_dict().items()}
            )
        elif mode == "sharded":
            p = nn.Parameter(torch.arange(5.0, device=device))
            state = {}
            for i in [1, 2]:
                p.grad = torch.full_like(p, float(i))
                api.sharded_momentum_step([p], state, 0.1, 0.9)
            payload = dict(
                value=p.detach().cpu(),
                start=state["start"],
                end=state["end"],
                momentum=state["momentum"].cpu(),
            )
        elif mode == "finite":
            p = nn.Parameter(torch.ones(2, device=device))
            p.grad = torch.ones_like(p)
            first = api.collective_finite([p])
            p.grad[0] = float("inf") if rank == 1 else 1
            second = api.collective_finite([p])
            payload = dict(first=first, second=second)
        torch.save(payload, directory / f"{rank}.pt")
    finally:
        dist.destroy_process_group()


def run_workers(api, tmp_path, mode, backend="gloo"):
    mp.spawn(_worker, args=(api.__name__, str(tmp_path), mode, backend), nprocs=2, join=True)
    return [torch.load(tmp_path / f"{rank}.pt", weights_only=True) for rank in range(2)]


@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_06(api, tmp_path):
    rows = run_workers(api, tmp_path, "reduce")
    for r in rows:
        torch.testing.assert_close(r["value"], torch.tensor([6.5, 8.5]))
        torch.testing.assert_close(r["empty"], torch.tensor([8.0, 10.0]))
    torch.testing.assert_close(rows[0]["original"], torch.tensor([2.0, 4.0]))


@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_07(api, tmp_path):
    rows = run_workers(api, tmp_path, "step")
    torch.manual_seed(31)
    model = nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    x = torch.arange(15.0).reshape(5, 3) / 10
    y = x.sum(-1, keepdim=True)
    losses = []
    for _ in range(2):
        opt.zero_grad()
        loss = (model(x) - y).square().mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    for row in rows:
        assert row["loss"] == pytest.approx(losses[0]) and row["loss2"] == pytest.approx(losses[1])
        for k, v in model.state_dict().items():
            torch.testing.assert_close(row["weights"][k], v)


def test_08(api):
    x = torch.arange(7.0)
    r = api.shard_state(x, 1, 3)
    assert r["start"] == 3 and r["end"] == 5
    torch.testing.assert_close(r["momentum"], torch.tensor([3.0, 4.0]))
    r["momentum"].zero_()
    assert x[3] == 3


@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_09(api, tmp_path):
    rows = run_workers(api, tmp_path, "sharded")
    p = nn.Parameter(torch.arange(5.0))
    opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
    for i in [1, 2]:
        p.grad = torch.full_like(p, float(i))
        opt.step()
    for rank, r in enumerate(rows):
        torch.testing.assert_close(r["value"], p)
        assert r["momentum"].numel() == (3 if rank == 0 else 2)
        torch.testing.assert_close(r["momentum"], torch.full_like(r["momentum"], 2.9))


def test_10(api):
    a = nn.Parameter(torch.ones(2))
    b = nn.Parameter(torch.ones(1))
    a.grad = torch.tensor([4.0, 8.0])
    b.grad = torch.tensor([float("inf")])
    assert api.unscale_gradients([a, b], 4) is False
    torch.testing.assert_close(a.grad, torch.tensor([4.0, 8.0]))
    b.grad = None
    assert api.unscale_gradients([a, b], 4) is True
    torch.testing.assert_close(a.grad, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        api.unscale_gradients([a], 0)


def test_11(api, seed):
    torch.manual_seed(seed)
    x = torch.randn(5, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    calls = []

    def fn(v):
        calls.append(1)
        return v.sin().square()

    out = api.checkpointed(fn, x)
    out.sum().backward()
    y.sin().square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    assert len(calls) >= 2, "Must recompute intermediates during backward"


def test_12(api):
    shards = [
        dict(start=3, end=5, momentum=torch.tensor([3.0, 4.0])),
        dict(start=0, end=3, momentum=torch.tensor([0.0, 1.0, 2.0])),
    ]
    full = api.consolidate_shards(shards, 5)
    torch.testing.assert_close(full, torch.arange(5.0))
    full[0] = 99
    assert shards[1]["momentum"][0] == 0
    with pytest.raises(ValueError):
        api.consolidate_shards(shards[:1], 5)
    with pytest.raises(ValueError):
        api.consolidate_shards([shards[0], shards[1], shards[1]], 5)


@pytest.mark.distributed
@pytest.mark.timeout(90)
def test_13(api, tmp_path):
    rows = run_workers(api, tmp_path, "finite")
    assert rows == [dict(first=True, second=False)] * 2


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.timeout(120)
def test_14(api, tmp_path):
    assert torch.cuda.device_count() >= 2, (
        "Final hardware gate requires TWO NVIDIA GPUs; use scripts/modal_grade.py explicitly. CPU gates 1-13 are separate."
    )
    rows = run_workers(api, tmp_path, "step", backend="nccl")
    for key in rows[0]["weights"]:
        torch.testing.assert_close(rows[0]["weights"][key], rows[1]["weights"][key])
    torch.manual_seed(31)
    m = nn.Linear(3, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.03)
    x = torch.arange(15.0).reshape(5, 3) / 10
    y = x.sum(-1, keepdim=True)
    for _ in range(2):
        opt.zero_grad()
        loss = (m(x) - y).square().mean()
        loss.backward()
        opt.step()
    for k, v in m.state_dict().items():
        torch.testing.assert_close(rows[0]["weights"][k], v, atol=2e-5, rtol=2e-4)
