import numpy as np
import pytest
import torch


def test_01(api):
    a = np.array([1.0, 2.0])
    t = api.Tensor(a)
    a[0] = 99
    assert t.data.dtype == np.float64 and t.data[0] == 1
    np.testing.assert_array_equal(t.grad, [0, 0])
    assert t.parents == ()
    assert api.Tensor(2).data.shape == ()


def test_02(api, seed):
    g = np.random.default_rng(seed).normal(size=(2, 3, 4))
    np.testing.assert_allclose(api.unbroadcast(g, (1, 4)), g.sum((0, 1), keepdims=False)[None, :])
    np.testing.assert_allclose(api.unbroadcast(g, ()), g.sum())
    np.testing.assert_allclose(api.unbroadcast(g, (2, 3, 4)), g)


def test_03(api):
    a = api.Tensor(1)
    b = api.Tensor(2, (a,))
    c = api.Tensor(3, (a, b))
    d = api.Tensor(4, (c, c))
    nodes = api.topological(d)
    assert len(nodes) == 4 and len({id(n) for n in nodes}) == 4
    assert nodes.index(a) < nodes.index(b) < nodes.index(c) < nodes.index(d)


def test_04(api):
    a = api.Tensor([2.0, 3.0])
    out = api.Tensor([4.0, 9.0], (a,))

    def vjp():
        a.grad.__iadd__(out.grad * 2 * a.data)

    out._backward = vjp
    out.backward(np.array([3.0, 4.0]))
    np.testing.assert_allclose(a.grad, [12, 24])
    out.backward(np.array([3.0, 4.0]))
    np.testing.assert_allclose(a.grad, [12, 24])
    with pytest.raises(ValueError):
        out.backward()
    with pytest.raises(ValueError):
        out.backward(np.ones(1))


def test_05(api, seed):
    rng = np.random.default_rng(seed)
    a = api.Tensor(rng.normal(size=(3, 2)))
    b = api.Tensor([1.0, 2.0])
    out = api.add(a, b)
    out.backward(np.ones((3, 2)))
    np.testing.assert_allclose(out.data, a.data + b.data)
    np.testing.assert_allclose(b.grad, [3, 3])
    api.add(a, a).backward(np.ones((3, 2)))
    np.testing.assert_allclose(a.grad, 2)


def test_06(api, seed):
    a = api.Tensor(np.random.default_rng(seed).normal(size=(3, 2)))
    b = api.Tensor([2.0, -1.0])
    api.multiply(a, b).backward(np.ones((3, 2)))
    np.testing.assert_allclose(a.grad, np.broadcast_to(b.data, (3, 2)))
    np.testing.assert_allclose(b.grad, a.data.sum(0))
    api.multiply(a, a).backward(np.ones((3, 2)))
    np.testing.assert_allclose(a.grad, 2 * a.data)


def test_07(api, seed):
    rng = np.random.default_rng(seed)
    a = api.Tensor(rng.normal(size=(3, 4)))
    b = api.Tensor(rng.normal(size=(4, 2)))
    g = rng.normal(size=(3, 2))
    out = api.matmul(a, b)
    out.backward(g)
    np.testing.assert_allclose(out.data, a.data @ b.data)
    np.testing.assert_allclose(a.grad, g @ b.data.T)
    np.testing.assert_allclose(b.grad, a.data.T @ g)
    with pytest.raises(ValueError):
        api.matmul(api.Tensor([1, 2]), b)


def test_08(api, seed):
    x = api.Tensor(np.random.default_rng(seed).normal(size=(2, 3)))
    for axis, keep in [(None, False), (0, False), (-1, True)]:
        y = api.summation(x, axis, keep)
        np.testing.assert_allclose(y.data, x.data.sum(axis=axis, keepdims=keep))
        y.backward(np.ones_like(y.data))
        np.testing.assert_allclose(x.grad, 1)


def test_09(api):
    x = api.Tensor([-2.0, 0.0, 3.0])
    y = api.relu(x)
    y.backward(np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(y.data, [0, 0, 3])
    np.testing.assert_allclose(x.grad, [0, 0, 4])


def test_10(api, seed):
    x = api.Tensor(np.random.default_rng(seed).uniform(0.2, 2, (2, 3)))
    y = api.log(api.exp(x))
    y.backward(np.full((2, 3), 2.0))
    np.testing.assert_allclose(y.data, x.data)
    np.testing.assert_allclose(x.grad, 2)
    with pytest.raises(ValueError):
        api.log(api.Tensor([0.0]))
    z = api.log(x)
    z.backward(np.ones((2, 3)))
    np.testing.assert_allclose(x.grad, 1 / x.data)


def test_11(api, seed):
    rng = np.random.default_rng(seed)
    x = api.Tensor(rng.normal(size=(3, 2)))
    y = api.Tensor(rng.normal(size=(3, 2)))
    loss = api.mse(x, y)
    loss.backward()
    np.testing.assert_allclose(loss.data, ((x.data - y.data) ** 2).mean())
    np.testing.assert_allclose(x.grad, 2 * (x.data - y.data) / 6)
    np.testing.assert_allclose(y.grad, -x.grad)


def test_12(api, seed):
    data = np.random.default_rng(seed).normal(size=(4, 5)) * 1000
    labels = np.array([0, 4, 1, 3])
    x = api.Tensor(data)
    loss = api.cross_entropy(x, labels)
    loss.backward()
    ref = torch.tensor(data, requires_grad=True)
    expected = torch.nn.functional.cross_entropy(ref, torch.tensor(labels))
    expected.backward()
    np.testing.assert_allclose(loss.data, expected.item(), atol=1e-9)
    np.testing.assert_allclose(x.grad, ref.grad.numpy(), atol=1e-9)
    with pytest.raises(ValueError):
        api.cross_entropy(x, np.array([0, 1, 2, 5]))


def test_13(api):
    p = api.Tensor([2.0, 4.0])
    p.grad[:] = [1, 2]
    api.sgd([p, p], 0.1)
    np.testing.assert_allclose(p.data, [1.9, 3.8])
    np.testing.assert_allclose(p.grad, 0)
    with pytest.raises(ValueError):
        api.sgd([p], -0.1)


def test_14(api, seed):
    x = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2 * x + 0.5
    old = x.copy()
    p, loss = api.train_mlp(x, y, hidden=12, steps=400, lr=0.05, seed=seed)
    assert (
        len(loss) == 400
        and np.isfinite(loss).all()
        and loss[-1] < 0.03
        and loss[-1] < loss[0] * 0.1
    )
    assert [v.data.shape for v in p] == [(1, 12), (12,), (12, 1), (1,)]
    np.testing.assert_array_equal(x, old)
    _, again = api.train_mlp(x, y, hidden=12, steps=400, lr=0.05, seed=seed)
    np.testing.assert_array_equal(loss, again)
    # Mechanism guard: the training loop must exercise the learner's backward and optimizer.
    calls = []
    original = api.sgd

    def tracked(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    api.sgd = tracked
    try:
        api.train_mlp(x, y, steps=4, seed=seed)
    finally:
        api.sgd = original
    assert len(calls) == 4
