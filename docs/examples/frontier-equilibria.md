# Recent Equilibrium Examples

This page runs one small, inspectable case for each new SILVA family. The
complete script is `examples/frontier_equilibria.py`; the executable notebook
adds derivations, plots, gradient checks, and architecture variations.
All four cases remain specializations of the SILVA equilibrium
[[1]](../paper/references.md#ref-1){ .silva-cite } and cite the mechanism they
adapt: FNO-DEQ [[43]](../paper/references.md#ref-43){ .silva-cite },
physics-guided graph DEQ [[44]](../paper/references.md#ref-44){ .silva-cite },
DDEQ [[45]](../paper/references.md#ref-45){ .silva-cite }, or HomoODE
[[46]](../paper/references.md#ref-46){ .silva-cite }.

$$
z^\star=T_\theta(z^\star;x).
$$

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/16_frontier_equilibrium_families/">Open executable notebook</a>
  <a class="md-button" href="../../package-notebooks/16_frontier_equilibrium_families/16_frontier_equilibrium_families.ipynb" download>Download notebook</a>
</div>

## Run Every Case

```bash
python examples/frontier_equilibria.py
```

The script reports shapes and numerical residuals for a smooth Fourier field,
a directed transport graph, an analytic homotopy problem, and an empirical
particle measure.

All snippets below use these imports:

```python
import torch
from torch import nn

from silva_networks import (
    SILVAFNODEQ,
    SILVADistributionalDEQ,
    SILVAHomotopyEquilibrium,
    SILVAPhysicsGuidedGraphDEQ,
    SolverConfig,
    make_affine_homotopy_dataset,
    make_graph_transport_dataset,
    make_periodic_elliptic_dataset,
    make_variable_measure_dataset,
)
```

## Fourier Field

```python
data = make_periodic_elliptic_dataset(
    samples=1, height=8, width=8, modes=2, seed=31
)

model = SILVAFNODEQ(
    1,
    4,
    1,
    modes_height=3,
    modes_width=3,
    state_scale=0.05,
    config=SolverConfig(max_iter=12, tol=1e-6, alpha=1.0),
)
result = model(data.forcing, return_result=True)
dataset_residual = data.equation_residual().abs().max()
```

The test checks the fixed-point residual. This untrained run validates the
operator and solver contract, while the generated target checks the periodic
elliptic equation. This untrained call is not a learned PDE benchmark.

## Directed Transport Graph

```python
data = make_graph_transport_dataset(samples=1, nodes=6, seed=32)

model = SILVAPhysicsGuidedGraphDEQ(3, 5, 1)
result = model(
    data.x,
    data.edge_index,
    edge_weight=data.edge_weight,
    edge_velocity=data.edge_velocity,
    return_result=True,
)
dataset_residual = data.equation_residual().abs().max()
```

Opposite edge directions receive opposite signed velocities. A separate unit
test relabels every node and edge and verifies that the predictions relabel in
the same way.

## Analytic Homotopy

```python
class AffineTransition(nn.Module):
    def forward(self, state, condition):
        return 0.5 * state + condition

model = SILVAHomotopyEquilibrium(
    1,
    1,
    1,
    transition=AffineTransition(),
    readout=nn.Identity(),
    steps=48,
    horizon=10.0,
    learnable_initial=False,
)
data = make_affine_homotopy_dataset(
    samples=2, dimension=1, contraction=0.5, seed=33
)
result = model(data.condition, return_result=True)
analytic_error = torch.max(torch.abs(result.output - data.target))
```

Because $z^\star=2x$ is known, this case checks both terminal residual and
state error.

## Empirical Measure

```python
data = make_variable_measure_dataset(
    samples=1,
    min_particles=4,
    max_particles=6,
    dimension=2,
    seed=34,
)
model = SILVADistributionalDEQ(
    2,
    4,
    particles=5,
    heads=2,
    kernel="gaussian",
    step_size=0.2,
    max_iter=5,
)
result = model(
    data.context,
    context_mask=data.context_mask,
    return_result=True,
)
```

The script compares the initial and final empirical-measure discrepancy. For a
task with padded sets, pass `context_mask` and `latent_mask`. For completion
where observed particles must stay exact, pass `fixed_mask`.

## Expected Scale

With the repository seed, the four cases complete in a few seconds on a CPU.
Exact predictions depend on parameter initialization, while these properties
must remain stable:

| Case | Stable expectation |
| --- | --- |
| Fourier | output shape matches the input grid; fixed-point residual is finite |
| graph physics | one output per node; fixed-point residual is finite |
| homotopy | terminal residual is smaller than initial velocity norm |
| distributional | state has the selected latent particle count; discrepancy is finite |

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the evolving or terminal physical state**, the condition is **time, initial condition, and external forcing**, and the
repeated map is **an explicit flow step or residual field T(z, x) - z**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **task, equation, invariance, and fixed-point residuals for four operator classes**. The
invariants that must remain true are **time-step shape, initial condition, and integration consistency**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/frontier_equilibria.py"
```

### Run the Complete Example

```bash
python examples/frontier_equilibria.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
SILVA Fourier equilibrium: {'shape': (1, 1, 8, 8), 'residual': 1.6093609644940443e-07, 'dataset_equation_residual': 5.960464477539062e-07}
SILVA physics graph equilibrium: {'shape': (6, 1), 'residual': 5.127419058226224e-07, 'dataset_equation_residual': 5.960464477539063e-08}
SILVA homotopy equilibrium: {'shape': (2, 1), 'terminal_residual': 0.00807332992553711, 'analytic_error': 0.01614689826965332}
SILVA distributional equilibrium: {'shape': (1, 5, 4), 'initial_discrepancy': 0.48991382122039795, 'final_discrepancy': 0.453036904335022}
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: frontier-equilibria
  state: the evolving or terminal physical state
  condition: time, initial condition, and external forcing
  repeated_transition: an explicit flow step or residual field T(z, x) - z
  invariant_checks: time-step shape, initial condition, and integration consistency
  compact_evidence: task, equation, invariance, and fixed-point residuals for four operator classes
  scale_axes: time horizon, step count, state dimension, and stiffness
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **the complete PDE, graph, homotopy, or measure benchmark**. Increase only one of
**time horizon, step count, state dimension, and stiffness** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the equations and generated datasets derived? | [Dataset-Backed Equilibrium Labs](../learn/frontier-dataset-labs.md) |
| What arguments and result fields are public? | [Recent Equilibrium API](../api/frontier.md) |
| How do the broader ODE/PDE cases work? | [Scientific Operators Example](scientific-operators.md) |
| Can I execute every derivation and check? | [Recent Equilibrium Families Notebook](../package-notebooks/16_frontier_equilibrium_families.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
