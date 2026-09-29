# Reverse-mode differentiation, from values to learning

A neural network is a composition of functions. Training asks how a scalar objective changes when each parameter changes. Reverse mode answers this efficiently by reusing the intermediate values computed in the forward pass.

For a node y=f(x), the local backward rule maps an upstream gradient g to J_f(x)^T g. This is a vector-Jacobian product. You almost never need to construct the full Jacobian. A matrix product's local rule uses two matrix multiplications, and an elementwise function's local rule uses elementwise multiplication.

## A shared-value example

Let a=2, b=a*a and L=b+b. The forward values are b=4 and L=8. Starting with dL/dL=1, addition contributes one to b through each edge, so dL/db=2. Multiplication contributes a*dL/db through each operand position. Both positions refer to a, yielding dL/da=8.

Two separate mistakes can yield the wrong answer. Deduplicating edges loses contributions, while failing to deduplicate node traversal can execute a callback too early or too often. Graph traversal visits a node once; local derivative rules account for every operand occurrence.

## Broadcasting is a linear map

Adding a bias [D] to activations [B,D] uses each bias element B times. Its backward gradient is the sum over B. For [B,T,D], sum over both leading axes. The correct rule depends on the original shape, not on guessing which output dimensions look like batches.

Reduction reverses the direction: summing an input axis sends one upstream value back to every element on that axis. Together, broadcasting and reduction form a useful pair of adjoint operations.

## Numerical checks

A central finite difference estimates a derivative as (f(x+h)-f(x-h))/(2h). Very large h measures curvature rather than the local derivative; very small h loses precision through cancellation. Float64 and h around 1e-5 are useful starting points for these tiny smooth examples, not universal guarantees. Avoid checking ReLU exactly at its nondifferentiable kink.

The acceptance grader also uses algebraic invariants and a separate PyTorch baseline. Independent checks reduce the chance that the implementation and its test repeat the same mistake. Explain why each test is informative before treating a green result as understanding.

The course engine intentionally omits broadcasting matmul, higher-order derivatives, mutation tracking and memory-efficient graph disposal. These are extensions after you can reason about the existing graph and its ownership rules.
