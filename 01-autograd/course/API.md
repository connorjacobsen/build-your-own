# Public API and input domains

These are scaffold contracts, not implementations. Earlier stages remain dependencies of later stages.

## `Tensor`

Float64 tensor with owned data, zero grad, tuple parents and a no-argument _backward callback.
Constructor accepts data, parents=(), backward=None. Operations attach callbacks which
add vector-Jacobian products into parent.grad. No torch/autograd delegation is allowed.

### `Tensor.__init__(self, data, parents=(), backward=None)`

Implement as specified by the owning stage.

### `Tensor.backward(self, grad=None)`

Reset reachable gradients, seed this output, propagate in reverse topological order.
Without grad the output must have exactly one element. Explicit grad must match shape.
Repeated backward calls replace the previous gradients, rather than accumulate them.

## `unbroadcast`

Sum a broadcast output gradient back to an original shape; return an ndarray of that shape.

Signature: `unbroadcast(grad, shape)`

## `topological`

Return each reachable Tensor once, parents before children; shared edges retain contributions.

Signature: `topological(root)`

## `add`

Elementwise Tensor addition with NumPy broadcasting and accumulated parent gradients.

Signature: `add(a, b)`

## `multiply`

Elementwise Tensor multiplication; support broadcasting and the same Tensor as both operands.

Signature: `multiply(a, b)`

## `matmul`

Rank-two matrix multiplication only. Reject non-matrices with ValueError.

Signature: `matmul(a, b)`

## `summation`

Sum all elements or one integer axis (negative axes allowed), with the corresponding VJP.

Signature: `summation(x, axis=None, keepdims=False)`

## `relu`

Elementwise max(x,0); derivative is zero at and below zero.

Signature: `relu(x)`

## `exp`

Elementwise exponential with a reverse-mode derivative.

Signature: `exp(x)`

## `log`

Natural logarithm; reject any nonpositive input with ValueError.

Signature: `log(x)`

## `mse`

Mean squared error across every broadcast result element, composed from this engine's ops.

Signature: `mse(pred, target)`

## `cross_entropy`

Stable mean cross-entropy for [N,C] logits and integer [N] labels. Return scalar Tensor.
Reject wrong label shape, empty N or out-of-range labels. Backward must remain finite
for logits of magnitude 1000. Compute derivatives yourself; do not delegate to torch.

Signature: `cross_entropy(logits, labels)`

## `sgd`

Update each unique Tensor exactly once in place using data -= lr*grad; then zero grad.
Reject negative lr. Parameters may contain repeated references to shared weights.

Signature: `sgd(parameters, lr)`

## `train_mlp`

Fit [N,D] x to [N,O] y with Linear-ReLU-Linear using only this engine.
Initialize w1,w2 using local NumPy normal(0,.3), biases zero. Return (parameters, losses),
parameters ordered [w1,b1,w2,b2], losses containing one PRE-update MSE per step.
Shapes are [D,H], [H], [H,O], [O]. Do not mutate inputs or use global NumPy RNG.

Signature: `train_mlp(x, y, hidden=8, steps=300, lr=0.03, seed=0)`
