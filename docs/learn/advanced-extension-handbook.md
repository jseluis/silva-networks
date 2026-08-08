# Advanced Extension Handbook

A SILVA extension begins with a state contract, not a family label. Define the state,
conditioning variables, branch decomposition, constraints, and readout before adding a
constructor.

## Derivation Contract

Start from

$$
z^\star=T_\theta(z^\star;x,E,b,c),\qquad
T_\theta=\Pi_{\mathcal C}\circ\sigma\circ(S+H+L+G+P),
$$

where $P$ collects optional physics or proximal terms and
$\Pi_{\mathcal C}$ enforces hard constraints. Every term must return the same state shape.

## Six Construction Steps

1. Implement and test each primitive branch independently.
2. Evaluate the assembled transition directly and compare it with the written equation.
3. Establish contraction, monotonicity, positivity, projection, or another well-posedness route.
4. Solve a deterministic analytic case and validate gradients independently.
5. Train a tiny task, reload its checkpoint, and retain measured diagnostics.
6. Register source data, metrics, scale defaults, tests, notebooks, and a reproduction dossier.

## Granularity Rules

Keep encoders, stimulus, self-interaction, local interaction, global context, constraints,
solver, backward method, and readout independently replaceable when their contracts differ.
A configured composition does not need a new canonical family. Add a family when a stable
mechanism, constructor, evidence path, and source relationship recur across experiments.

## Equivalence Test

For a primitive implementation $T_{\mathrm{primitive}}$ and a public composition
$T_{\mathrm{public}}$, check

$$
\epsilon_T=\max_i\left\|T_{\mathrm{primitive}}(z_i,x_i)-
T_{\mathrm{public}}(z_i,x_i)\right\|_\infty,
$$

followed by equilibrium, readout, and gradient agreement. Notebook 46 executes the complete
process with a custom self branch and a trained compact task.

## Executable Contract Check

The SILVA construction [[1]](../paper/references.md#ref-1){ .silva-cite }
requires the transition to preserve its state space. Validate that property before
selecting a solver:

```python
import torch
from torch import nn
from silva_networks import validate_silva_transition

class ContractiveBranch(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(0.1 * torch.eye(width))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(state))

state = torch.zeros(8, 6, requires_grad=True)
report = validate_silva_transition(ContractiveBranch(6), state)
assert report.preserves_shape
assert report.finite
assert report.differentiable
print(report)
```

The report checks the generic state contract. Add family-specific tests for graph
equivariance, boundary projection, positivity, conservation, or multiscale coupling
before the module is treated as a reusable family.

## Registration Checklist

- Public constructor and complete signature
- Canonical family key and aliases
- Compact and full scale defaults
- Equation, source relation, references, datasets, and metrics
- Primitive and assembled equivalence tests
- Solver, gradient, shape, device, and serialization tests
- Executed notebook with results and figures
- Family dossier and editable source-scale configuration

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the replaceable branch contracts introduced? | [Custom Layers](custom-layers.md) |
| Which lab builds and verifies a custom family? | [Extension Builder Workshop](../package-notebooks/46_extension_builder_workshop.ipynb) |
| Where are all family-scale experiment plans? | [Family Dossiers](../families/index.md) |
