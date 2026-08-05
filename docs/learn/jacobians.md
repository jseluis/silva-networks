# Jacobians and Stability

The Jacobian describes how a small perturbation of the state changes one
application of the transition map:

$$
J_f(z,x)
=
\frac{\partial f_\theta}{\partial z}(z,x).
$$

At a fixed point \(z^\star=f_\theta(z^\star,x)\), this matrix controls the
local behavior of the solver and the sensitivity of the equilibrium.

## Residual Linearization

Define the residual

$$
r(z,x)=f_\theta(z,x)-z.
$$

Perturb the state by a small vector \(\delta z\). A first-order Taylor expansion
gives

$$
f_\theta(z^\star+\delta z,x)
\approx
f_\theta(z^\star,x)+J_f(z^\star,x)\delta z.
$$

Because \(f_\theta(z^\star,x)=z^\star\),

$$
f_\theta(z^\star+\delta z,x)-z^\star
\approx
J_f(z^\star,x)\delta z.
$$

If repeated solver steps shrink perturbations, the equilibrium is locally
attractive. A sufficient local contraction condition is

$$
\rho(J_f(z^\star,x))<1,
$$

where \(\rho\) is the spectral radius.

## Full Jacobian

For small states, the package can materialize the full matrix:

```python
from silva_networks import full_jacobian

J = full_jacobian(f, z_star)
```

If `z_star` has shape `(entities, hidden_dim)`, the flattened Jacobian has shape

$$
(N d)\times(N d).
$$

This is excellent for tests, toy derivations, and verifying custom operators.
It is not the right tool for large images or large graphs.

## Matrix-Free Products

Most useful diagnostics only need products. The Jacobian-vector product is

$$
Jv
=
\left.\frac{d}{d\epsilon}
f_\theta(z^\star+\epsilon v,x)
\right|_{\epsilon=0}.
$$

The vector-Jacobian product is

$$
J^\top v
=
\nabla_z \langle f_\theta(z^\star,x),v\rangle.
$$

Package calls:

```python
from silva_networks import jvp, vjp

_, Jv = jvp(f, z_star, probe)
Jtv = vjp(f, z_star, probe)
```

These products keep memory under control on GPU because the matrix is never
stored explicitly.

## Damped Solver Jacobian

A damped Picard step is

$$
T_\alpha(z)
=
(1-\alpha)z+\alpha f_\theta(z,x).
$$

Differentiate with respect to \(z\):

$$
J_{T_\alpha}(z)
=
(1-\alpha)I+\alpha J_f(z,x).
$$

This is why the same transition \(f_\theta\) can behave differently under
different damping values. The helper below estimates the damped spectral
radius:

```python
from silva_networks import damped_spectral_radius

rho = damped_spectral_radius(f, z_star, alpha=0.5)
```

## Stability Report

`stability_report` collects the common checks:

```python
from silva_networks import stability_report

report = stability_report(f, z_star, samples=4, iters=12)
report.residual, report.spectral_radius
```

The reported quantities are numerical diagnostics, not formal certificates.
They are useful for comparing solvers, damping values, local/global branches,
and custom operators on the same problem.

## Lyapunov-Style Energy

The residual norm gives a natural energy-like quantity:

$$
E(z)=\frac12\|f_\theta(z,x)-z\|_2^2.
$$

Along a successful solve, this value often decreases:

```python
import torch
from silva_networks import SolverConfig, solve_with_energy

trace = solve_with_energy(
    f,
    z0,
    energy_fn=lambda z: 0.5 * torch.linalg.norm((f(z) - z).reshape(-1)).square(),
    config=SolverConfig(max_iter=20, alpha=0.5),
)
trace.energies
```

When it rises sharply, the configuration may need stronger damping, spectral
normalization, smaller learning rate, fewer stacked interactions, or a different
solver.

## Minimal Verification Pattern

For a new SILVA branch:

1. Test a tiny state and call `full_jacobian`.
2. Compare `J @ v` with `jvp`.
3. Compare `J.T @ v` with `vjp`.
4. Estimate \(\rho(J_f)\) and \(\rho(J_{T_\alpha})\).
5. Train a small CPU validation case before running larger device experiments.

The notebook [Solvers and Jacobians](../package-notebooks/02_solvers_and_jacobians.ipynb)
executes this pattern with a small transition map.

## Hutchinson Derivation

For a Rademacher probe \(v\), with independent entries in \(\{-1,+1\}\),

$$
\mathbb E[vv^\top]=I.
$$

Therefore

$$
\mathbb E\|J^\top v\|_2^2
=
\mathbb E[v^\top JJ^\top v]
=
\operatorname{tr}(JJ^\top)
=
\|J\|_F^2.
$$

This is the estimator behind `hutchinson_jacobian_norm`.

## Claim Boundaries

| Measurement | Supports | Does not prove |
| --- | --- | --- |
| \(\|f(z_K)-z_K\|\) small | the computed state is near a fixed point for the sampled input | global existence |
| \(\rho(J_{T_\alpha})<1\) | local linear stability near the state | global contraction |
| low Hutchinson norm | smaller sampled Jacobian energy | exact spectral bound |
| finite-difference agreement | local autograd sanity check | full training correctness |

The Jacobian-regularized equilibrium source and Hutchinson estimator source are
listed in [Equilibrium and Implicit Layers](../paper/references.md#equilibrium-and-implicit-layers)
and [Solvers and Linear Algebra](../paper/references.md#solvers-and-linear-algebra).

## Where to Go Next

| Question | Page |
| --- | --- |
| How do solver updates alter the effective Jacobian? | [Solver Derivation Lab](solver-derivation-lab.md) |
| Which Jacobian estimators are public? | [Jacobians API](../api/jacobians.md) |
| Where can I vary stability diagnostics interactively? | [Interactive Diagnostics Lab](interactive-diagnostics-lab.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
