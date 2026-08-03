# Implicit Backward Guide

This guide explains the package gradient path through equilibrium solves, what
`implicit_adjoint_solve` computes, and the scope of the public engine.

## Forward Equation

Every DEQ-style layer solves

$$
z^\star=f_\theta(z^\star,x).
$$

The damped solver actually iterates

$$
T_\alpha(z)=(1-\alpha)z+\alpha f_\theta(z,x).
$$

At convergence, \(T_\alpha(z^\star)=z^\star\), so the fixed point is unchanged
by damping.

## Differentiating The Fixed Point

Let a loss \(L(z^\star)\) depend on the equilibrium. The implicit function
theorem gives

$$
(I-J_f(z^\star))\frac{\partial z^\star}{\partial \theta}
=
\frac{\partial f_\theta(z^\star,x)}{\partial \theta}.
$$

For reverse mode, solve the adjoint system

$$
(I-J_f(z^\star)^\top)u
=
\frac{\partial L}{\partial z^\star}.
$$

Then

$$
\frac{\partial L}{\partial \theta}
=
u^\top \frac{\partial f_\theta(z^\star,x)}{\partial \theta}.
$$

For the damped transition \(T_\alpha\), the adjoint matrix becomes

$$
I-J_{T_\alpha}(z^\star)^\top
=
I-\left((1-\alpha)I+\alpha J_f(z^\star)\right)^\top.
$$

## What The Public Engine Does Today

`SILVADEQEngine` uses package fixed-point solvers in the forward pass. Its
`backward_solver` configuration field is reserved for implicit-adjoint
diagnostics and future custom-autograd integration.

The current public training behavior is:

1. Solve the equilibrium with the selected finite solver.
2. Reconstruct the state structure if it was packed.
3. Optionally run one differentiable transition, controlled by `reengage`.
4. Let ordinary PyTorch autograd differentiate through the differentiable
   transition that remains connected to the graph.

This is intentionally honest. It is a practical package-native training path,
not a claim of memory-constant custom implicit backward for every module.

## When To Use `implicit_adjoint_solve`

Use `implicit_adjoint_solve` when you need the adjoint vector

$$
u=(I-J_T(z^\star)^\top)^{-1}g
$$

for diagnostics or a manual implicit-gradient experiment.

```python
import torch
from silva_networks import implicit_adjoint_solve

z_star = torch.randn(4)
grad_output = torch.ones_like(z_star)

def transition(z):
    return torch.tanh(0.4 * z)

adjoint = implicit_adjoint_solve(
    transition,
    z_star,
    grad_output,
    alpha=0.7,
    max_iter=30,
    tol=1e-6,
)
print(adjoint.x, adjoint.residuals)
```

The helper uses VJP-backed GMRES and does not materialize the full Jacobian.

## What A Full Custom Backward Would Add

A memory-constant DEQ backward would wrap the solve in a custom autograd
function:

| Step | Custom-backward responsibility |
| --- | --- |
| Forward | Solve \(z^\star=f_\theta(z^\star,x)\) without storing the whole trajectory |
| Backward | Recompute \(f_\theta(z^\star,x)\) under grad tracking |
| Linear solve | Solve \((I-J_f^\top)u=g\) with GMRES, Anderson, or Broyden |
| Parameter grads | Call `torch.autograd.grad(f_theta, parameters, grad_outputs=u)` |
| Input grads | Call `torch.autograd.grad(f_theta, x, grad_outputs=u)` |

That path is compatible with the math already exposed by the package, but it is
not what `SILVADEQEngine` claims today.

## Practical Guidance

| Use case | Recommended path |
| --- | --- |
| Training package examples | Use the default reengage path |
| Comparing solver residuals | Use `return_result=True` and inspect `SolverResult` |
| Checking implicit-gradient conditioning | Use `implicit_adjoint_solve` and GMRES residuals |
| Proving stability | Use `stability_report` and spectral radius diagnostics |
| Building a custom memory-constant layer | Start from this guide and implement a custom autograd wrapper |
