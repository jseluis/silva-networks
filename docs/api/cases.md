# Generalized Cases API

The case architectures are SILVA-native implementations of major DEQ
application families. They accept `SolverConfig` rather than embedding paper
recipes.

All families preserve the same contract:

$$
z^\star=f_\theta(z^\star,x),
\qquad
r^\star=f_\theta(z^\star,x)-z^\star.
$$

What changes is the meaning and layout of \(z\). A sequence uses token states,
a multiscale model uses a tuple of feature maps, a graph uses node states, an
implicit representation uses query-aligned states, and diffusion uses a packed
trajectory. The return dataclasses keep the task output and `solver_result`
together so callers can report predictions and convergence evidence from the
same forward pass.

## Public Families

| Family | Main public objects |
| --- | --- |
| Sequence DEQ | `SILVASequenceDEQ`, `SILVASequenceTransition`, `SILVARelativeSelfAttention`, `SILVAAdaptiveEmbedding`, `SILVAProjectedAdaptiveLogSoftmax`, `SILVASequenceOutput` |
| Multiscale vision DEQ | `SILVAMultiscaleDEQ`, learned MDEQ fusion, `SILVAMultiscaleResidualBlock`, `SILVAMultiscaleClassificationHead`, classifier/segmenter, `SILVAMultiscaleOutput` |
| Implicit graph | `SILVAImplicitGraphNetwork`, `SILVAGraphEquilibriumOutput` |
| Implicit neural representation | `SILVAImplicitNeuralRepresentation`, `SILVACoordinateInjection`, `SILVAINROutput` |
| Diffusion equilibrium | `SILVADiffusionEquilibrium`, `SILVADiffusionOutput` |

## Minimal Graph Case

```python
import torch
from silva_networks import SILVAImplicitGraphNetwork, SolverConfig

x = torch.randn(6, 4)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])

model = SILVAImplicitGraphNetwork(
    in_dim=4,
    hidden_dim=12,
    out_dim=3,
    config=SolverConfig(solver="anderson", max_iter=20, tol=1e-5),
)
result = model(x, edge_index, return_result=True)

assert result.output.shape == (6, 3)
print(result.solver_result.residual, result.solver_result.converged)
```

The node-state transition can be read as

$$
Z^\star
=
\Phi\{S_\theta(X)+L_\theta(Z^\star,E)+G_\theta(Z^\star)\},
$$

which makes the graph case a direct SILVA specialization rather than a
separate solving mechanism.

## Result Contracts

| Result | Primary output | Equilibrium evidence |
| --- | --- | --- |
| `SILVASequenceOutput` | token features or vocabulary scores | `solver_result` and solved state |
| `SILVAMultiscaleOutput` | tuple of scale states | one packed-state `solver_result` |
| `SILVAGraphEquilibriumOutput` | node or graph predictions | node state and `solver_result` |
| `SILVAINROutput` | query-aligned field values | hidden field and `solver_result` |
| `SILVADiffusionOutput` | final sample | trajectory and `solver_result` |

For derivations, tensor layouts, and complete small runs, use
[Paper Families as SILVA Configurations](../learn/paper-family-adaptations.md)
and [Paper Family Cases](../examples/paper-family-cases.md). Primary method
sources are collected in [Paper and References](../paper/references.md).

## API

::: silva_networks.cases

## Where to Go Next

| Question | Page |
| --- | --- |
| How does each case connect to a research architecture? | [Paper Family Adaptations](../learn/paper-family-adaptations.md) |
| Where are the compact cases executed? | [Paper Family Cases](../examples/paper-family-cases.md) |
| How can a case be selected from one interface? | [Family Selection API](families.md) |
