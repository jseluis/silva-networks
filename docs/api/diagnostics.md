# Diagnostics

SILVA layers are useful when the fixed-point dynamics can be inspected. The
package exposes residual curves, Jacobian products, spectral-radius estimates,
and Lyapunov-style energy traces as ordinary Python functions.

The stochastic Jacobian estimate follows Hutchinson
[[14]](../paper/references.md#ref-14){ .silva-cite }, and the equilibrium
regularization interpretation follows Jacobian-regularized DEQ
[[6]](../paper/references.md#ref-6){ .silva-cite }.

## Residual

For a transition \(f\), the fixed-point residual is

$$
r(z)=f(z)-z,
\qquad
\|r(z)\|_2=\|f(z)-z\|_2.
$$

`residual_curve` records this norm during a solve:

```python
from silva_networks import SolverConfig, residual_curve

curve = residual_curve(f, z0, SolverConfig(max_iter=20, alpha=0.5))
```

## Damped Spectral Radius

The executed damped update is

$$
T_\alpha(z)=(1-\alpha)z+\alpha f(z).
$$

Its local Jacobian is

$$
J_{T_\alpha}(z^\star)
=
(1-\alpha)I+\alpha J_f(z^\star).
$$

`damped_spectral_radius` estimates

$$
\rho(J_{T_\alpha}(z^\star))
$$

with VJP-based power iteration. This is the quantity used for local
fixed-point stability diagnostics.

## Full Diagnostic Record

For a layer transition \(f\), a reproducible experiment record should contain:

$$
\left(
z_K,\,
\{\|f(z_k)-z_k\|_2\}_{k=0}^{K},\,
\rho(J_{T_\alpha}(z_K)),\,
\widehat{\|J_f(z_K)\|_F},\,
\{E(z_k)\}_{k=0}^{K}
\right).
$$

In package terms:

```python
from silva_networks import (
    SolverConfig,
    damped_spectral_radius,
    hutchinson_jacobian_norm,
    lyapunov_quadratic_energy,
    solve_with_energy,
)

config = SolverConfig(alpha=0.5, max_iter=20, tol=1e-6)
report = solve_with_energy(
    f,
    z0,
    energy_fn=lambda z: lyapunov_quadratic_energy(z, interaction(z)),
    config=config,
    include_stability=True,
)
rho = damped_spectral_radius(f, report.result.z, alpha=config.alpha)
fro = hutchinson_jacobian_norm(f, report.result.z, samples=8, squared=False)
```

## Lyapunov-Style Energy

The package provides the quadratic alignment diagnostic

$$
E_i(z)=\|z_i\|_2^2-z_i^\top h_i,
$$

where \(h_i\) is a local, global, or combined interaction evaluated at the
same state. In code:

```python
from silva_networks import lyapunov_quadratic_energy

energy = lyapunov_quadratic_energy(z, local_update + global_update)
```

This is a diagnostic proxy. A rigorous Lyapunov certificate requires the
assumptions of the specific dynamical system being studied.

## Solve With Energy

`solve_with_energy` runs a fixed-point solve while evaluating an energy
function on each iterate:

```python
from silva_networks import solve_with_energy

report = solve_with_energy(
    f,
    z0,
    energy_fn=lambda z: lyapunov_quadratic_energy(z, interaction(z)),
    include_stability=True,
)

report.result.z
report.energies
report.energy_deltas
report.stability.spectral_radius
```

`descent_fraction(report.energies)` returns the fraction of consecutive energy
steps that did not increase.

## Failure Modes

| Symptom | Mathematical sign | First response |
| --- | --- | --- |
| residual grows | \(\|r_{k+1}\|>\|r_k\|\) repeatedly | lower `alpha`, inspect operator scale |
| residual alternates | linearized modes near unit radius | increase damping, try Anderson with ridge |
| adjoint solve is ill-conditioned | \(I-J_f^\top\) nearly singular | inspect \(\rho(J_f)\), regularize or simplify |
| energy rises | \(E_{k+1}-E_k>0\) often | treat energy as warning, not proof |
| task metric improves but residual is high | finite-depth behavior dominates | report finite-solve budget honestly |
| graph batch leaks context | global term ignores `batch` | validate `batch` and use package global operators |

## Claim Scale

| Evidence | Claim scale |
| --- | --- |
| one residual curve | this input solved under this budget |
| residuals over a validation split | this configuration is numerically reliable on sampled data |
| residuals plus \(\rho(J_{T_\alpha})<1\) | local stability evidence near computed states |
| full ablation plus diagnostics | model mechanism claim |
| theorem-level assumptions plus diagnostics | certificate or guarantee |

::: silva_networks.diagnostics

## Where to Go Next

| Question | Page |
| --- | --- |
| How should residual and stability traces be interpreted? | [Interactive Diagnostics Lab](../learn/interactive-diagnostics-lab.md) |
| Which Jacobian estimates support the diagnostics? | [Jacobians API](jacobians.md) |
| Which solver result fields supply the traces? | [Solvers API](solvers.md) |
