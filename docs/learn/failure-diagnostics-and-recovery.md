# Failure Diagnostics and Recovery

Failure examples are part of the experiment contract. A solver that returns a tensor has
not necessarily found a useful equilibrium.

## Local Error Dynamics

Near a fixed point,

$$
e_{k+1}\approx J_T(z^\star)e_k.
$$

Slow decay indicates a radius near one; alternating residuals suggest a negative dominant
mode; growth indicates an expansive direction; a plateau may indicate scaling, precision,
or an incompatible tolerance. The fixed-point formulation follows the equilibrium-model
foundation [[4]](../paper/references.md#ref-4){ .silva-cite }; Jacobian control and
diagnostic motivation are developed further in
[[6]](../paper/references.md#ref-6){ .silva-cite }.

## Diagnostic Matrix

| Symptom | Measure | Controlled response |
| --- | --- | --- |
| Residual grows | Branch norms, Jacobian radius, finite values | Reduce recurrent scale, normalize inputs, or enforce a certificate |
| Residual alternates | Signed state differences and damped Picard | Reduce damping before adding acceleration |
| Residual stalls | Absolute/relative curves and precision | Rescale states, revise tolerance, or increase precision |
| Forward works but gradients fail | Adjoint residual and finite differences | Tighten backward solve or revise the local Jacobian |
| Constraint drifts | Boundary, positivity, feasibility, or conservation error | Project inside every transition rather than only at readout |
| Training is unstable | Per-branch gradients and spectral diagnostics | Isolate the offending branch and add it back incrementally |

## Executable Residual Check

This compact program compares a stable map with a near-critical one while retaining the
complete residual history:

```python
import torch
from silva_networks import SolverConfig, solve_equilibrium

def run_case(factor: float):
    transition = lambda state: factor * state + 1.0
    return solve_equilibrium(
        transition,
        torch.zeros(1),
        SolverConfig(
            solver="picard",
            max_iter=40,
            tol=1e-7,
            alpha=1.0,
            return_best=True,
        ),
    )

stable = run_case(0.70)
slow = run_case(0.98)
assert stable.residual < slow.residual
print("stable:", stable.iterations, stable.residual)
print("near critical:", slow.iterations, slow.residual)
```

Plot \`stable.residuals\` and \`slow.residuals\`, not only their final values. The curve
distinguishes geometric decay from a budget-limited near-critical solve.

## Required Failure Record

Retain the input and seed, complete solver configuration, residual history, returned-versus-
best state choice, branch norms, certificate or radius estimate, backward residual, task loss,
and the exact recovery change. Notebook 47 provides executable stable, slow, oscillatory, and
damped cases with a 300-dpi diagnostic figure.

## Recovery Is an Experiment

Change one factor at a time. Damping, normalization, recurrent scaling, solver choice,
tolerance, and backward method answer different questions. Record each as an ablation rather
than silently changing several controls until the run succeeds.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are solver residuals defined? | [Solvers](../api/solvers.md) |
| Which diagnostics are available programmatically? | [Diagnostics](../api/diagnostics.md) |
| Where can I run the failure cases? | [Failure Diagnostics Workshop](../package-notebooks/47_failure_diagnostics_workshop.ipynb) |
