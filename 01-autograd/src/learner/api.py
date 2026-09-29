# ruff: noqa: F401
"""Instructor reference: a deliberately small NumPy reverse-mode engine."""

import numpy as np


class Tensor:
    """Float64 tensor with owned data, zero grad, tuple parents and a no-argument _backward callback.
    Constructor accepts data, parents=(), backward=None. Operations attach callbacks which
    add vector-Jacobian products into parent.grad. No torch/autograd delegation is allowed.
    """

    def __init__(self, data, parents=(), backward=None):
        raise NotImplementedError("Implement Tensor.__init__; see course list/read")

    def backward(self, grad=None):
        """Reset reachable gradients, seed this output, propagate in reverse topological order.
        Without grad the output must have exactly one element. Explicit grad must match shape.
        Repeated backward calls replace the previous gradients, rather than accumulate them."""
        raise NotImplementedError("Implement Tensor.backward; see course list/read")


def unbroadcast(grad, shape):
    """Sum a broadcast output gradient back to an original shape; return an ndarray of that shape."""
    raise NotImplementedError("Implement unbroadcast; see course list/read")


def topological(root):
    """Return each reachable Tensor once, parents before children; shared edges retain contributions."""
    raise NotImplementedError("Implement topological; see course list/read")


def add(a, b):
    """Elementwise Tensor addition with NumPy broadcasting and accumulated parent gradients."""
    raise NotImplementedError("Implement add; see course list/read")


def multiply(a, b):
    """Elementwise Tensor multiplication; support broadcasting and the same Tensor as both operands."""
    raise NotImplementedError("Implement multiply; see course list/read")


def matmul(a, b):
    """Rank-two matrix multiplication only. Reject non-matrices with ValueError."""
    raise NotImplementedError("Implement matmul; see course list/read")


def summation(x, axis=None, keepdims=False):
    """Sum all elements or one integer axis (negative axes allowed), with the corresponding VJP."""
    raise NotImplementedError("Implement summation; see course list/read")


def relu(x):
    """Elementwise max(x,0); derivative is zero at and below zero."""
    raise NotImplementedError("Implement relu; see course list/read")


def exp(x):
    """Elementwise exponential with a reverse-mode derivative."""
    raise NotImplementedError("Implement exp; see course list/read")


def log(x):
    """Natural logarithm; reject any nonpositive input with ValueError."""
    raise NotImplementedError("Implement log; see course list/read")


def mse(pred, target):
    """Mean squared error across every broadcast result element, composed from this engine's ops."""
    raise NotImplementedError("Implement mse; see course list/read")


def cross_entropy(logits, labels):
    """Stable mean cross-entropy for [N,C] logits and integer [N] labels. Return scalar Tensor.
    Reject wrong label shape, empty N or out-of-range labels. Backward must remain finite
    for logits of magnitude 1000. Compute derivatives yourself; do not delegate to torch."""
    raise NotImplementedError("Implement cross_entropy; see course list/read")


def sgd(parameters, lr):
    """Update each unique Tensor exactly once in place using data -= lr*grad; then zero grad.
    Reject negative lr. Parameters may contain repeated references to shared weights."""
    raise NotImplementedError("Implement sgd; see course list/read")


def train_mlp(x, y, hidden=8, steps=300, lr=0.03, seed=0):
    """Fit [N,D] x to [N,O] y with Linear-ReLU-Linear using only this engine.
    Initialize w1,w2 using local NumPy normal(0,.3), biases zero. Return (parameters, losses),
    parameters ordered [w1,b1,w2,b2], losses containing one PRE-update MSE per step.
    Shapes are [D,H], [H], [H,O], [O]. Do not mutate inputs or use global NumPy RNG."""
    raise NotImplementedError("Implement train_mlp; see course list/read")
