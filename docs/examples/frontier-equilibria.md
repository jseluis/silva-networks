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

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the equations and generated datasets derived? | [Dataset-Backed Equilibrium Labs](../learn/frontier-dataset-labs.md) |
| What arguments and result fields are public? | [Recent Equilibrium API](../api/frontier.md) |
| How do the broader ODE/PDE cases work? | [Scientific Operators Example](scientific-operators.md) |
| Can I execute every derivation and check? | [Recent Equilibrium Families Notebook](../package-notebooks/16_frontier_equilibrium_families.ipynb) |
