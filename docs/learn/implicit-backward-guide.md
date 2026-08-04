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

## Three Gradient Modes

`SolverConfig.backward_mode` selects the gradient estimator independently of
the forward solver:

| Mode | Forward graph | Backward calculation |
| --- | --- | --- |
| `"unrolled"` | retains differentiable finite solver operations | ordinary reverse differentiation through the retained trajectory |
| `"implicit"` | solves under `no_grad` | custom backward solves \((I-J_{T_\alpha}^\top)u=g\) and applies the adjoint to tracked parameters and inputs |
| `"phantom"` | solves under `no_grad` | differentiates through `phantom_steps` damped transitions from the detached numerical state |

The exact implicit path is active in `solve_equilibrium`; it is not only a
diagnostic helper. Package layers pass their trainable parameters through
`params` and differentiable non-state inputs through `tensors`, then the custom
backward returns the corresponding sensitivities.

```python
from silva_networks import SolverConfig

config = SolverConfig(
    solver="anderson",
    max_iter=30,
    tol=1e-5,
    backward_mode="implicit",
    backward_solver="gmres",
    backward_max_iter=40,
    backward_tol=1e-7,
)
```

After `loss.backward()`, the solver result records `backward_solver`,
`backward_iterations`, `backward_residual`, and `backward_converged` in
`result.info`. Those values describe the linear adjoint solve and should be
reported separately from the forward fixed-point residual.

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

## What Implicit Mode Executes

The custom backward follows these steps:

| Step | Responsibility |
| --- | --- |
| Forward | Solve \(z^\star=f_\theta(z^\star,x)\) without storing the whole trajectory |
| Backward | Recompute \(f_\theta(z^\star,x)\) under grad tracking |
| Linear solve | Solve \((I-J_f^\top)u=g\) with GMRES, Anderson, or Broyden |
| Parameter grads | Call `torch.autograd.grad(f_theta, parameters, grad_outputs=u)` |
| Input grads | Call `torch.autograd.grad(f_theta, x, grad_outputs=u)` |

The implementation uses the damped transition Jacobian, so the gradient passed
to transition parameters carries the corresponding factor \(\alpha\). The
state can be a tensor or a packed tuple/list; packed multi-state systems are
treated as one coupled vector during both solves.

## Practical Guidance

| Use case | Recommended path |
| --- | --- |
| Small, shallow validation where trajectory memory is acceptable | Use `backward_mode="unrolled"` |
| Equilibrium training with a matrix-free exact adjoint | Use `backward_mode="implicit"` and inspect backward residuals |
| Short approximate gradient from the solved state | Use `backward_mode="phantom"` and report steps/tau |
| Comparing solver residuals | Use `return_result=True` and inspect `SolverResult` |
| Checking implicit-gradient conditioning | Use `implicit_adjoint_solve` and GMRES residuals |
| Proving stability | Use `stability_report` and spectral radius diagnostics |

## Tensor Contract and Sources

The forward transition, its state input, and its output must have identical
shape, dtype, and device. The adjoint right-hand side \(g\), solution \(u\),
and Jacobian-vector products use that same state contract.

See [Solvers](../api/solvers.md) for every configuration field and
[Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers)
for the implicit-function and DEQ sources. GMRES is listed under
[Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra).

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is implicit differentiation derived? | [Mathematical Foundations](mathematical-foundations.md#implicit-differentiation) |
| Which solver options control the backward system? | [Solvers API](../api/solvers.md) |
| Can I run a minimal implicit-gradient notebook? | [Implicit Autodiff Notebook](../implicit-bridge-notebooks/02_implicit_autodiff.ipynb) |
