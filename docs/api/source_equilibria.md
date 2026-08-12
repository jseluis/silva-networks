# Source-Aligned Equilibrium Families

This module contains fourteen independently implemented SILVA families that
preserve a distinct mechanism from a published equilibrium architecture. Every
family exposes its transition, accepts replaceable internal modules where the
method permits them, and uses the shared SILVA solver configuration unless the
published construction is an equation-residual method rather than a hidden-state
root solve.

The compact implementations establish equations, tensor contracts, gradients,
constraints, and extension points. Published benchmark numbers require the
source dataset, preprocessing, model width, training schedule, checkpoint, and
evaluation protocol listed in the corresponding reproduction dossier.

| Family | Public class | Solved object | Primary reference |
| --- | --- | --- | --- |
| Lipschitz MDEQ | `SILVALipschitzMultiscaleEquilibrium` | packed multiscale state | [[99]](../paper/references.md#ref-99) |
| SubDEQ | `SILVASubhomogeneousEquilibrium` | positive normalized state | [[100]](../paper/references.md#ref-100) |
| algorithmic reasoner | `SILVAAlgorithmicReasoner` | graph processor state | [[101]](../paper/references.md#ref-101) |
| DEQH | `SILVAHamiltonianEquilibrium` | symmetric Hamiltonian | [[102]](../paper/references.md#ref-102) |
| inverse imaging | `SILVAInverseImagingEquilibrium` | reconstructed image | [[103]](../paper/references.md#ref-103) |
| snapshot compressive imaging | `SILVASnapshotCompressiveEquilibrium` | video volume | [[104]](../paper/references.md#ref-104) |
| magnetic-particle imaging | `SILVAMagneticParticleEquilibrium` | primal/split/dual state | [[105]](../paper/references.md#ref-105) |
| hyperspectral sparse representation | `SILVASparseHyperspectralEquilibrium` | sparse code and cube | [[106]](../paper/references.md#ref-106) |
| serialized smoothing | `SILVASerializedSmoothingEquilibrium` | noisy-sample equilibria and certificate | [[107]](../paper/references.md#ref-107) |
| diffusion restoration | `SILVADiffusionRestorationEquilibrium` | joint reverse trajectory | [[108]](../paper/references.md#ref-108) |
| recurrent equilibrium network | `SILVARecurrentEquilibriumNetwork` | algebraic state per time step | [[109]](../paper/references.md#ref-109) |
| Lipschitz robust equilibrium | `SILVALipschitzRobustEquilibrium` | logits, bound, and radius | [[110]](../paper/references.md#ref-110) |
| image matting | `SILVAImageMattingEquilibrium` | trimap-constrained alpha matte | [[111]](../paper/references.md#ref-111) |
| dynamic economic equilibrium | `SILVADynamicEconomicEquilibrium` | feasible policy functions | [[112]](../paper/references.md#ref-112) |

## Common Result Contract

Most families return a tensor by default and a `SILVASourceEquilibriumResult`
when `return_result=True`. The expanded result contains the solved state, task
output, and complete `SolverResult`, including residual history and termination
information. Specialized results retain additional quantities such as smoothing
certificates, recurrent trajectories, robust radii, and economic residuals.

```python
from silva_networks import (
    SILVALipschitzMultiscaleEquilibrium,
    SolverConfig,
)

model = SILVALipschitzMultiscaleEquilibrium(
    input_dim=32,
    scale_dims=(64, 32, 16),
    output_dim=10,
    contraction=0.8,
    config=SolverConfig(
        solver="anderson",
        max_iter=40,
        tol=1e-5,
        backward_mode="implicit",
        backward_solver="gmres",
    ),
)
result = model(features, return_result=True)
scale_states = model.split_state(result.state)
```

## Replaceable Internals

Replacement modules must preserve the documented shape and mathematical
contract. For example, inverse imaging accepts any differentiable forward and
adjoint pair, snapshot imaging accepts any shape-preserving volumetric prior,
and Hamiltonian prediction accepts an invariant or equivariant interaction
backbone that returns a square atom-orbital matrix.

```python
model = SILVAInverseImagingEquilibrium(
    channels=2,
    forward_operator=undersampled_fourier,
    adjoint_operator=undersampled_fourier_adjoint,
    prior=multiscale_image_prior,
    step_size=0.2,
    config=solver_config,
)
```

The [Source-Aligned Family Deep Dive](../learn/source-equilibrium-families.md)
derives every transition, gives compact and source-scale routes, and links each
family to its executable notebook and reproduction dossier.

## API

::: silva_networks.source_equilibria
    options:
      show_root_heading: true
      members_order: source
      show_source: false

## Where to Go Next

| Question | Page |
| --- | --- |
| How is each transition derived and extended? | [Source-Aligned Family Deep Dive](../learn/source-equilibrium-families.md) |
| Which data, storage, and compute routes are prepared? | [Experiment Protocols](experiment_protocols.md) |
| How do I inspect one complete family dossier? | [Family Reproduction Dossiers](../families/index.md) |
| Where are the full citations? | [Paper and References](../paper/references.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
