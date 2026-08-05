# Scientific Operators API

`silva_networks.scientific` provides reusable numerical and learned components
for ODEs, PDEs, and function-to-function models. The module connects Neural
ODEs [[7]](../paper/references.md#ref-7){ .silva-cite }, Fourier Neural
Operators [[31]](../paper/references.md#ref-31){ .silva-cite }, neural-operator
theory [[32]](../paper/references.md#ref-32){ .silva-cite }, and the structured
SILVA transition [[1]](../paper/references.md#ref-1){ .silva-cite } without
identifying these distinct constructions as the same object.

## Four Separate Roles

| Role | Public API | Mathematical object |
| --- | --- | --- |
| spatial discretization | `finite_difference_gradient_1d`, `finite_difference_laplacian_1d`, `finite_difference_laplacian_2d` | sampled derivative or Laplacian |
| physical diagnostic | `poisson_residual_2d`, `boundary_error_2d`, `relative_residual_norm` | equation or boundary error |
| implicit numerical step | `SILVAImplicitTimeStep` | backward-Euler fixed point |
| learned solution operator | `SILVAOperatorModel`, `SILVAFourierNeuralOperator` | parameterized map between sampled functions |

Keeping these roles separate matters. A low fixed-point residual says that the
numerical transition has settled. It does not establish a low PDE residual or a
low prediction error. The learning material shows how to report all three.

## Implicit Time Step

Given a semidiscrete system

$$
\frac{du}{dt}=R(u,c),
$$

`SILVAImplicitTimeStep` solves the backward-Euler equation

$$
u^{n+1}=u^n+\Delta t\,R(u^{n+1},c).
$$

The previous state is the SILVA stimulus, while the right-hand side supplies
the recurrent field. A projector can impose a hard boundary condition or state
constraint after every transition evaluation.

```python
from silva_networks import (
    SILVADirichletBoundary2D,
    SILVAImplicitTimeStep,
    SILVAReactionDiffusionRHS2D,
    SolverConfig,
)

rhs = SILVAReactionDiffusionRHS2D(
    diffusion=0.01,
    spacing=1.0 / 31.0,
)
step = SILVAImplicitTimeStep(
    rhs,
    step_size=0.005,
    projector=SILVADirichletBoundary2D(0.0),
    config=SolverConfig(max_iter=40, tol=1e-6, alpha=0.8),
)
next_field = step(previous_field)
```

## Learned Operator

For an input function sampled as `field`, `SILVAOperatorModel` lifts, solves,
and reads out

$$
a\xrightarrow{R_\phi}s,
\qquad
z^\star=\Psi\left[s+B_\theta(z^\star)+L_\theta(z^\star)+G_\theta(z^\star)\right],
\qquad
\widehat u=Q_\omega(z^\star).
$$

`SILVAFourierNeuralOperator` selects the Fourier point architecture for
$B_\theta$. It retains the same learned spectral weights when the sampled grid
resolution changes, subject to the number of available modes.

```python
import torch
from silva_networks import SILVAFourierNeuralOperator, SolverConfig

model = SILVAFourierNeuralOperator(
    in_channels=2,       # coefficient field and source field
    state_channels=16,
    out_channels=1,
    modes_height=6,
    modes_width=6,
    config=SolverConfig(max_iter=20, tol=1e-5, alpha=0.4),
)

problem = torch.randn(4, 2, 32, 32)
result = model(problem, return_result=True)
print(result.output.shape, result.solver_result.residual)
```

## API

::: silva_networks.scientific
    options:
      show_root_heading: true
      members_order: source
      show_signature_annotations: true

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the equations derived and mapped into SILVA branches? | [Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md) |
| Where do all constructions run together? | [Scientific Operators Example](../examples/scientific-operators.md) |
| Can I execute the derivations cell by cell? | [Neural Operators, ODEs, and PDEs Notebook](../package-notebooks/15_neural_operators_ode_pde.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
