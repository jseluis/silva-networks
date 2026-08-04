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
)
```

## Fourier Field

```python
axis = torch.linspace(0.0, 1.0, 8)
y, x = torch.meshgrid(axis, axis, indexing="ij")
forcing = (torch.sin(torch.pi * x) * torch.sin(torch.pi * y))[None, None]

model = SILVAFNODEQ(
    1,
    4,
    1,
    modes_height=3,
    modes_width=3,
    state_scale=0.05,
    config=SolverConfig(max_iter=12, tol=1e-6, alpha=1.0),
)
result = model(forcing, return_result=True)
```

The test checks the fixed-point residual. This untrained run validates the
operator and solver contract; it is not a learned PDE solution benchmark.

## Directed Transport Graph

```python
coordinates = torch.linspace(0.0, 1.0, 6)
x = torch.stack([coordinates, torch.sin(torch.pi * coordinates)], dim=-1)
forward = torch.stack([torch.arange(5), torch.arange(1, 6)])
edge_index = torch.cat([forward, forward.flip(0)], dim=1)
velocity = torch.cat([torch.ones(5), -torch.ones(5)])

model = SILVAPhysicsGuidedGraphDEQ(2, 5, 1)
result = model(
    x,
    edge_index,
    edge_velocity=velocity,
    return_result=True,
)
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
condition = torch.tensor([[0.25], [-0.4]])
result = model(condition, return_result=True)
analytic_error = torch.max(torch.abs(result.output - 2.0 * condition))
```

Because $z^\star=2x$ is known, this case checks both terminal residual and
state error.

## Empirical Measure

```python
context = torch.tensor(
    [[[-1.0, 0.0], [-0.3, 0.5], [0.4, -0.2], [1.0, 0.1]]]
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
result = model(context, return_result=True)
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
| Why are these the correct equations? | [Recent Equilibrium Families Inside SILVA](../learn/frontier-equilibrium-families.md) |
| What arguments and result fields are public? | [Recent Equilibrium API](../api/frontier.md) |
| How do the broader ODE/PDE cases work? | [Scientific Operators Example](scientific-operators.md) |
| Can I execute every derivation and check? | [Recent Equilibrium Families Notebook](../package-notebooks/16_frontier_equilibrium_families.ipynb) |
