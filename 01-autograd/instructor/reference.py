"""Instructor reference: a deliberately small NumPy reverse-mode engine."""

import numpy as np


class Tensor:
    """Float64 tensor with owned data, zero grad, tuple parents and a no-argument _backward callback.
    Constructor accepts data, parents=(), backward=None. Operations attach callbacks which
    add vector-Jacobian products into parent.grad. No torch/autograd delegation is allowed.
    """

    def __init__(self, data, parents=(), backward=None):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data)
        self.parents = tuple(parents)
        self._backward = backward if backward is not None else lambda: None

    def backward(self, grad=None):
        """Reset reachable gradients, seed this output, propagate in reverse topological order.
        Without grad the output must have exactly one element. Explicit grad must match shape.
        Repeated backward calls replace the previous gradients, rather than accumulate them.
        """
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar output requires a seed")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.data.shape:
            raise ValueError("Seed shape mismatch")
        nodes = topological(self)
        for node in nodes:
            node.grad = np.zeros_like(node.data)
        self.grad = grad.copy()
        for node in reversed(nodes):
            node._backward()


def unbroadcast(grad, shape):
    """Sum a broadcast output gradient back to an original shape; return an ndarray of that shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def topological(root):
    """Return each reachable Tensor once, parents before children; shared edges retain contributions."""
    seen, ordered = set(), []

    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for parent in node.parents:
            visit(parent)
        ordered.append(node)

    visit(root)
    return ordered


def add(a, b):
    """Elementwise Tensor addition with NumPy broadcasting and accumulated parent gradients."""
    out = Tensor(a.data + b.data, (a, b))

    def backward():
        a.grad += unbroadcast(out.grad, a.data.shape)
        b.grad += unbroadcast(out.grad, b.data.shape)

    out._backward = backward
    return out


def multiply(a, b):
    """Elementwise Tensor multiplication; support broadcasting and the same Tensor as both operands."""
    out = Tensor(a.data * b.data, (a, b))

    def backward():
        a.grad += unbroadcast(out.grad * b.data, a.data.shape)
        b.grad += unbroadcast(out.grad * a.data, b.data.shape)

    out._backward = backward
    return out


def matmul(a, b):
    """Rank-two matrix multiplication only. Reject non-matrices with ValueError."""
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError("Only matrices are supported")
    out = Tensor(a.data @ b.data, (a, b))

    def backward():
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad

    out._backward = backward
    return out


def summation(x, axis=None, keepdims=False):
    """Sum all elements or one integer axis (negative axes allowed), with the corresponding VJP."""
    out = Tensor(x.data.sum(axis=axis, keepdims=keepdims), (x,))

    def backward():
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        x.grad += np.broadcast_to(g, x.data.shape)

    out._backward = backward
    return out


def relu(x):
    """Elementwise max(x,0); derivative is zero at and below zero."""
    out = Tensor(np.maximum(x.data, 0), (x,))
    out._backward = lambda: None

    def backward():
        x.grad += out.grad * (x.data > 0)

    out._backward = backward
    return out


def exp(x):
    """Elementwise exponential with a reverse-mode derivative."""
    out = Tensor(np.exp(x.data), (x,))

    def backward():
        x.grad += out.grad * out.data

    out._backward = backward
    return out


def log(x):
    """Natural logarithm; reject any nonpositive input with ValueError."""
    if np.any(x.data <= 0):
        raise ValueError("Log requires positive values")
    out = Tensor(np.log(x.data), (x,))

    def backward():
        x.grad += out.grad / x.data

    out._backward = backward
    return out


def mse(pred, target):
    """Mean squared error across every broadcast result element, composed from this engine's ops."""
    delta = add(pred, multiply(target, Tensor(-1.0)))
    square = multiply(delta, delta)
    return multiply(summation(square), Tensor(1.0 / square.data.size))


def cross_entropy(logits, labels):
    """Stable mean cross-entropy for [N,C] logits and integer [N] labels. Return scalar Tensor.
    Reject wrong label shape, empty N or out-of-range labels. Backward must remain finite
    for logits of magnitude 1000. Compute derivatives yourself; do not delegate to torch.
    """
    y = np.asarray(labels)
    if (
        logits.data.ndim != 2
        or len(logits.data) == 0
        or y.shape != (len(logits.data),)
        or y.dtype.kind not in "iu"
        or np.any(y < 0)
        or np.any(y >= logits.data.shape[1])
    ):
        raise ValueError("Invalid labels or logits")
    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    z = np.exp(shifted).sum(axis=-1, keepdims=True)
    out = Tensor((np.log(z[:, 0]) - shifted[np.arange(len(y)), y]).mean(), (logits,))

    def backward():
        g = np.exp(shifted) / z
        g[np.arange(len(y)), y] -= 1
        logits.grad += out.grad * g / len(y)

    out._backward = backward
    return out


def sgd(parameters, lr):
    """Update each unique Tensor exactly once in place using data -= lr*grad; then zero grad.
    Reject negative lr. Parameters may contain repeated references to shared weights.
    """
    if lr < 0:
        raise ValueError("Negative learning rate")
    seen = set()
    for p in parameters:
        if id(p) not in seen:
            p.data -= lr * p.grad
            p.grad.fill(0)
            seen.add(id(p))


def train_mlp(x, y, hidden=8, steps=300, lr=0.03, seed=0):
    """Fit [N,D] x to [N,O] y with Linear-ReLU-Linear using only this engine.
    Initialize w1,w2 using local NumPy normal(0,.3), biases zero. Return (parameters, losses),
    parameters ordered [w1,b1,w2,b2], losses containing one PRE-update MSE per step.
    Shapes are [D,H], [H], [H,O], [O]. Do not mutate inputs or use global NumPy RNG.
    """
    rng = np.random.default_rng(seed)
    w1, b1 = Tensor(rng.normal(0, 0.3, (x.shape[1], hidden))), Tensor(np.zeros(hidden))
    w2, b2 = Tensor(rng.normal(0, 0.3, (hidden, y.shape[1]))), Tensor(np.zeros(y.shape[1]))
    params = [w1, b1, w2, b2]
    losses = []
    for _ in range(steps):
        pred = add(matmul(relu(add(matmul(Tensor(x), w1), b1)), w2), b2)
        loss = mse(pred, Tensor(y))
        losses.append(float(loss.data))
        loss.backward()
        sgd(params, lr)
    return params, losses
