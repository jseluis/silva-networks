# Neural Operators, ODEs, PDEs, and SILVA

This guide connects four objects that are often introduced separately:

1. an ODE evolves a state through continuous time, including the learned
   dynamics formulation of Neural ODEs
   [[7]](../paper/references.md#ref-7){ .silva-cite };
2. a PDE evolves or constrains a field over space and possibly time;
3. a neural operator learns a map between functions
   [[32]](../paper/references.md#ref-32){ .silva-cite }, with the Fourier Neural
   Operator as a spectral construction
   [[31]](../paper/references.md#ref-31){ .silva-cite };
4. SILVA builds a structured transition from named fields and solves its fixed
   point [[1]](../paper/references.md#ref-1){ .silva-cite }.

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/15_neural_operators_ode_pde/">Open executable notebook</a>
  <a class="md-button" href="../../package-notebooks/15_neural_operators_ode_pde/15_neural_operators_ode_pde.ipynb" download>Download notebook</a>
</div>

## State Maps and Operators

A finite-dimensional layer maps one vector to another:

$$
f_\theta:\mathbb R^D\rightarrow\mathbb R^D.
$$

An operator maps a function to another function:

$$
\mathcal G:\mathcal A(\Omega)\rightarrow\mathcal U(\Omega).
$$

For a PDE, $a\in\mathcal A(\Omega)$ may be a source, coefficient field, initial
condition, or boundary condition. The output $u=\mathcal G(a)$ is a solution
field. A grid converts the functions into tensors, but the modeling object is
still the function-to-function map.

| Object | Input | Output | Typical tensor |
| --- | --- | --- | --- |
| vector layer | feature vector | feature vector | `(batch, channels)` |
| sequence layer | token function | token function | `(batch, tokens, channels)` |
| spatial operator | sampled field | sampled field | `(batch, channels, height, width)` |
| PDE solution operator | coefficient/source field | solution field | grid tensors with physical coordinates |

## ODE to Repeated Transition

Consider a state governed by

$$
\frac{dh(t)}{dt}=v_\theta(h(t),t,x),
\qquad h(0)=h_0.
$$

Explicit Euler with step size $\Delta t$ gives

$$
h_{k+1}=h_k+\Delta t\,v_\theta(h_k,t_k,x).
$$

The finite trajectory after $K$ steps is

$$
h_K=h_0+\Delta t\sum_{k=0}^{K-1}v_\theta(h_k,t_k,x).
$$

This trajectory is not automatically an equilibrium. A steady state satisfies

$$
v_\theta(h^\star,t,x)=0.
$$

For the relaxation ODE

$$
\frac{dh}{dt}=-\lambda(h-u),
$$

the analytic solution and steady state are

$$
h(t)=u+(h_0-u)e^{-\lambda t},
\qquad
h^\star=u.
$$

One Euler step can be written as the fixed-point map

$$
T(h)=(1-\beta)h+\beta u,
\qquad \beta=\lambda\Delta t.
$$

SILVA can represent this map with the target $u$ as the stimulus and the
state-dependent term $(1-\beta)h$ as an internal or self field. The explicit ODE
block follows a finite trajectory; `SILVACortexLayer` solves the corresponding
self-consistency equation and records its residual.

```python
import torch
from torch import nn
from silva_networks import SILVACortexLayer, SILVAEulerFlowBlock, SolverConfig


class Scale(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, z):
        return self.value * z


beta = 0.25
equilibrium = SILVACortexLayer(
    input_encoder=Scale(beta),
    state_network=Scale(1.0 - beta),
    activation=lambda z: z,
    output_activation=lambda z: z,
    normalize=False,
    config=SolverConfig(solver="picard", max_iter=30, alpha=1.0),
)

target = torch.tensor([[1.0, -0.5, 0.25]])
result = equilibrium(target, return_result=True)
assert torch.allclose(result.z, target, atol=1e-3)
```

## PDE to Discrete Residual

For a concrete elliptic example, consider the Poisson equation on the unit
square with homogeneous Dirichlet boundary conditions:

$$
-\Delta u(x,y)=q(x,y),
\qquad
u|_{\partial\Omega}=0.
$$

On a regular grid with spacing $h$, the five-point Laplacian is

$$
(\Delta_hu)_{i,j}
=
\frac{
u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}
}{h^2}.
$$

The discrete PDE residual is

$$
r_{\mathrm{PDE}}(u,q)=-\Delta_hu-q.
$$

A classical residual-correction iteration has the form

$$
u_{k+1}=u_k+\eta M^{-1}\left(q-A_hu_k\right),
$$

where $A_h$ discretizes $-\Delta$ and $M$ is a preconditioner. This is already
a fixed-point equation:

$$
u^\star
=u^\star+\eta M^{-1}\left(q-A_hu^\star\right).
$$

The local stencil belongs naturally in a SILVA local field. Boundary forcing,
material coefficients, or observations belong in the stimulus. A global or
spectral field can communicate across the full domain to accelerate long-range
corrections.

## Implicit Time Stepping Is a SILVA Point

The connection is especially direct for a time-dependent reaction-diffusion
equation:

$$
\frac{\partial u}{\partial t}
=D\Delta u+r_\theta(u)+s(x,t).
$$

A backward-Euler step evaluates the right-hand side at the unknown next state:

$$
u^{n+1}
=u^n+\Delta t\left[
D\Delta_hu^{n+1}+r_\theta(u^{n+1})+s^{n+1}
\right].
$$

This is a fixed-point equation with an exact SILVA decomposition:

$$
\underbrace{u^n+\Delta t\,s^{n+1}}_{S(x)}
+\underbrace{\Delta t\,r_\theta(u^{n+1})}_{H(u^{n+1})}
+\underbrace{\Delta t\,D\Delta_hu^{n+1}}_{L(u^{n+1})}.
$$

The previous time slice and forcing form the stimulus, the reaction law is a
self field, and the discrete Laplacian is a local field. A nonlocal closure,
integral constraint, or learned spectral correction can be added as a global
field. The solver computes the implicit next time slice. Repeating the point
advances a trajectory; solving it once computes one implicit time step.

For a linear diffusion step, the notebook compares the SILVA solution with the
direct linear solve

$$
\left(I-\Delta t\,D\Delta_h\right)u^{n+1}=u^n.
$$

That comparison checks the numerical construction independently of training.

### General Semidiscrete Form

After spatial discretization, many PDEs become an ODE system

$$
\frac{d\mathbf u}{dt}=R_h(\mathbf u,\mathbf c),
$$

where $\mathbf u$ contains sampled field values and $\mathbf c$ contains
coefficients, forcing, geometry, or boundary data. Backward Euler defines

$$
T(\mathbf z;\mathbf u^n,\mathbf c)
=
\mathbf u^n+\Delta t\,R_h(\mathbf z,\mathbf c),
\qquad
\mathbf u^{n+1}=T(\mathbf u^{n+1};\mathbf u^n,\mathbf c).
$$

The transition Jacobian is

$$
J_T(\mathbf z)=\Delta t\,J_{R_h}(\mathbf z).
$$

A sufficient local contraction condition is

$$
\Delta t\,\lVert J_{R_h}(\mathbf u^{n+1})\rVert<1.
$$

This is a condition on the chosen fixed-point iteration, not on the
backward-Euler method's classical stability region. If the undamped map is not
contractive enough, damping changes the executed transition to

$$
T_\alpha(\mathbf z)
=(1-\alpha)\mathbf z+\alpha T(\mathbf z),
\qquad
J_{T_\alpha}=(1-\alpha)I+\alpha J_T.
$$

Anderson or Broyden acceleration can improve a difficult solve, but the
residual and convergence flag must still be checked.

### Public Implicit-Step API

The package keeps the right-hand side independent from the solver:

```python
from torch import nn
from silva_networks import (
    SILVADirichletBoundary2D,
    SILVAImplicitTimeStep,
    SILVAReactionDiffusionRHS2D,
    SolverConfig,
)


class LogisticReaction(nn.Module):
    def forward(self, u):
        return 0.2 * u * (1.0 - u)


rhs = SILVAReactionDiffusionRHS2D(
    diffusion=0.01,
    reaction=LogisticReaction(),
    spacing=1.0 / 31.0,
    boundary="dirichlet",
)
step = SILVAImplicitTimeStep(
    rhs,
    step_size=0.005,
    projector=SILVADirichletBoundary2D(0.0),
    config=SolverConfig(max_iter=40, tol=1e-6, alpha=0.8),
)

result = step(previous_field, context=forcing, return_result=True)
next_field = result.z
```

The shape contract is

$$
R_h:\mathbb R^{B\times C\times H\times W}
\times\mathbb R^{B\times C\times H\times W}
\rightarrow\mathbb R^{B\times C\times H\times W}.
$$

`context` is optional, but when supplied it must have the state shape for the
built-in reaction-diffusion and Burgers fields. A custom right-hand side may
encode parameters in an `nn.Module` or use a state-shaped context field.
When damping is active, initialize the state on the constraint set as well as
projecting the transition; the damped update retains a fraction of the current
iterate.

## Neural Solution Operators

Instead of solving one right-hand side $q$, operator learning fits a map

$$
\mathcal G_\theta:q\mapsto u
$$

over a distribution of source fields. A common operator layer is

$$
v_{\ell+1}(x)
=
\sigma\!\left(
W_\ell v_\ell(x)
+\int_\Omega
\kappa_\ell(x,y)v_\ell(y)\,dy
\right).
$$

The pointwise map $W_\ell$ mixes channels at one coordinate. The integral
kernel communicates between coordinates. A discretized dense kernel is costly;
Fourier convolution parameterizes a translation-invariant kernel efficiently.

## Fourier Neural Operator Derivation

For a periodic convolution kernel, the convolution theorem gives

$$
\mathcal F(\kappa*v)(k)
=
\widehat\kappa(k)\widehat v(k).
$$

With $C$ input and output channels, each retained frequency has a learned
complex channel matrix:

$$
\widehat w_o(k)
=
\sum_{i=1}^{C}R_{oi}(k)\widehat v_i(k),
\qquad k\in\mathcal K.
$$

The compact SILVA Fourier architecture implements

$$
B_\theta(v)
=s\left[
\mathcal F_h^{-1}\!\left(R_\theta\cdot\mathcal F_hv\right)
+Wv
\right].
$$

The implementation stores separate complex tensors for the retained positive
and negative vertical modes, truncates the horizontal real-FFT frequencies,
zeros all unretained modes, applies the inverse transform, and adds a learned
$1\times1$ local projection.

The constructor arguments control different quantities:

| Argument | Mathematical role |
| --- | --- |
| `channels` | number of field channels $C$ |
| `modes_height` | retained positive and negative vertical frequencies |
| `modes_width` | retained nonnegative horizontal real-FFT frequencies |
| `scale` | multiplier $s$ on the complete local-plus-spectral field |

The number of learned real spectral coefficients is

$$
4C^2m_hm_w,
$$

because there are top and bottom mode tensors and each complex value stores real
and imaginary parts. The local projection adds $C^2+C$ parameters.

## How the Fourier Operator Connects to SILVA

For a source field $q$, a spatial SILVA point can use

$$
u=R_\phi(q)
$$

as a lifted stimulus and solve

$$
z^\star
=
\mathcal N\!\left[
\Psi\!\left(
u
+B_{\mathrm{FNO},\theta}(a(z^\star))
+L_\theta(a(z^\star))
+G_\theta(a(z^\star))
\right)
\right].
$$

The roles are now explicit:

| SILVA component | ODE/PDE/operator interpretation |
| --- | --- |
| `input_encoder` | lift source, coefficients, initial data, or boundary data into state channels |
| `state_network` | learned function-space update, including Fourier, U-Net, or convolutional fields |
| `self_terms` | reaction, persistence, or learned pointwise state contribution |
| `local_terms` | finite-difference stencil, flux, graph neighborhood, or local convolution |
| `global_terms` | mean field, long-range attention, conservation statistic, or global correction |
| `interaction_terms` | coefficients, coordinates, forcing, masks, or problem-specific constraints |
| `output_network` | combine or project the complete transition field |
| `normalizer` | state-layout-compatible normalization |
| solver | compute the self-consistent field and record numerical residuals |
| network head | project the equilibrium state to the requested physical output |

The point is not merely a Fourier operator followed by a solver. The Fourier field is one
named contribution inside a structured equilibrium transition, and it can be
combined with local physics and global context without changing the solver API.

## Complete Spatial Construction

```python
import torch
from silva_networks import SILVACortexLayer, SolverConfig, silva_point_architecture

channels = 8
point = SILVACortexLayer(
    input_encoder=torch.nn.Conv2d(1, channels, kernel_size=1),
    state_network=silva_point_architecture(
        "fourier_operator",
        channels=channels,
        modes_height=6,
        modes_width=6,
        scale=0.05,
    ),
    local_terms=torch.nn.Conv2d(
        channels,
        channels,
        kernel_size=3,
        padding=1,
        groups=channels,
    ),
    output_network=torch.nn.Conv2d(channels, channels, kernel_size=1),
    normalizer=torch.nn.GroupNorm(2, channels),
    config=SolverConfig(
        solver="anderson",
        max_iter=20,
        tol=1e-5,
        alpha=0.2,
        history=4,
    ),
)

source = torch.randn(4, 1, 32, 32)
result = point(source, return_result=True)
assert result.z.shape == (4, channels, 32, 32)
print(result.residuals)
```

The executable notebook trains a small version on analytic Poisson fields and
then reports two distinct diagnostics:

1. the fixed-point residual $\|F_\theta(z_k,q)-z_k\|_2$ measures the numerical solve;
2. the PDE residual $\|-\Delta_h\widehat u-q\|_2$ measures physical equation error.

Neither residual replaces supervised field error. They answer different
questions and should be reported separately.

## Worked Scientific Constructions

### Reaction-Diffusion

For

$$
\frac{\partial u}{\partial t}
=D\Delta u+\rho u(1-u)+s,
$$

the implicit update is

$$
u^{n+1}
=u^n
+\Delta t\,D\Delta_hu^{n+1}
+\Delta t\,\rho u^{n+1}(1-u^{n+1})
+\Delta t\,s^{n+1}.
$$

The SILVA roles are therefore

$$
S=u^n+\Delta t\,s^{n+1},
\qquad
H(z)=\Delta t\,\rho z(1-z),
\qquad
L(z)=\Delta t\,D\Delta_hz.
$$

The built-in `SILVAReactionDiffusionRHS2D` combines the physical right-hand
side, while `SILVAImplicitTimeStep` supplies $u^n$, multiplies by $\Delta t$,
projects boundaries, and solves the equilibrium. A learned reaction module can
replace the analytic law without changing the time-step abstraction.

### Viscous Burgers Equation

For a periodic one-dimensional field,

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
=\nu\frac{\partial^2u}{\partial x^2}+s,
$$

the centered differences are

$$
(D_hu)_i=\frac{u_{i+1}-u_{i-1}}{2h},
\qquad
(\Delta_hu)_i=\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}.
$$

Backward Euler gives

$$
u_i^{n+1}
=u_i^n+\Delta t\left[
-u_i^{n+1}(D_hu^{n+1})_i
+\nu(\Delta_hu^{n+1})_i+s_i^{n+1}
\right].
$$

The nonlinear advection and local diffusion are both state-shaped interaction
fields. The package implementation is:

```python
from silva_networks import SILVABurgersRHS1D, SILVAImplicitTimeStep

rhs = SILVABurgersRHS1D(
    viscosity=0.01,
    spacing=1.0 / number_of_points,
    boundary="periodic",
)
step = SILVAImplicitTimeStep(rhs, step_size=0.001, config=solver_config)
next_line = step(previous_line)
```

Centered advection is intentionally transparent for teaching and compact
checks. High-Reynolds-number simulations generally need a flux formulation,
stabilization, adaptive stepping, or a problem-specific numerical method. Such
a discretization can be supplied as a custom right-hand-side module.

### Variable-Coefficient Elliptic Operator

Consider

$$
-\nabla\cdot\left(a(x)\nabla u(x)\right)=q(x),
\qquad
u|_{\partial\Omega}=g.
$$

The learned operator has the form

$$
\mathcal G_\theta:(a,q,g,\Omega)\mapsto u.
$$

On a fixed rectangular domain, one practical tensor representation is

```text
channel 0: coefficient a(x)
channel 1: source q(x)
channel 2: boundary values g(x), zero away from the boundary
channel 3: boundary or domain mask
channel 4: x coordinate
channel 5: y coordinate
```

Not every channel is required for every problem. The contract is that all
channels share the sampled spatial dimensions and the model's `in_channels`
matches their count. `SILVAOperatorModel` lifts these fields into the recurrent
state. Its internal architecture may be Fourier, U-Net, residual convolution,
or another shape-preserving field.

```python
from silva_networks import SILVAOperatorModel

model = SILVAOperatorModel(
    in_channels=6,
    state_channels=16,
    out_channels=1,
    architecture="unet",
    architecture_kwargs={"base_channels": 24},
    config=solver_config,
    output_transform=boundary_transform,
)
prediction = model(problem_channels)
```

A physical residual for this problem must discretize the flux
$a\nabla u$ before its divergence. The constant-coefficient Poisson helper is
not a substitute for that variable-coefficient residual.

### Public Fourier Equilibrium Operator

The dedicated constructor provides the complete lift, equilibrium point, and
readout:

```python
from silva_networks import SILVAFourierNeuralOperator, SolverConfig

model = SILVAFourierNeuralOperator(
    in_channels=2,
    state_channels=12,
    out_channels=1,
    modes_height=6,
    modes_width=6,
    field_scale=0.05,
    config=SolverConfig(
        solver="anderson",
        max_iter=24,
        tol=1e-5,
        alpha=0.35,
        history=4,
    ),
)

result = model(coefficient_and_source, return_result=True)
solution = result.output
state = result.state
solver_result = result.solver_result
```

The architecture reuses its spectral parameters at another grid resolution:

```python
coarse = model(coarse_problem)
fine = model(fine_problem)
```

This ability to evaluate does not establish resolution generalization. That
claim requires held-out fine-grid targets, physical residuals computed at the
fine spacing, and a comparison against an appropriate baseline.

## Irregular Domains and Graph PDEs

For nodes with edge set $E$, one unweighted graph Laplacian is

$$
(\Delta_Gz)_i
=
\sum_{j:(j,i)\in E}(z_j-z_i).
$$

An implicit graph-diffusion step is

$$
z_i^{n+1}
=z_i^n+\Delta t\,D(\Delta_Gz^{n+1})_i.
$$

This maps directly into a `SILVACortexLayer`: `input_encoder` supplies $z^n$,
and a graph module in `local_terms` computes the edge exchange. The module can
accept `edge_index` and `edge_attr`, so geometric distances, face areas,
conductivities, or learned edge weights can participate in the discretization.

```python
import torch
from torch import nn
from silva_networks import SILVACortexLayer, SolverConfig


class GraphLaplacian(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, z, edge_index):
        source, target = edge_index
        field = torch.zeros_like(z)
        field.index_add_(0, target, z[source] - z[target])
        return self.scale * field


point = SILVACortexLayer(
    input_encoder=nn.Identity(),
    local_terms=GraphLaplacian(scale=delta_t * diffusivity),
    activation=lambda z: z,
    output_activation=lambda z: z,
    normalize=False,
    config=SolverConfig(max_iter=40, tol=1e-6, alpha=0.8),
)
next_nodes = point(previous_nodes, edge_index=edge_index)
```

For a finite-element or finite-volume study, the local module should implement
the chosen mass, stiffness, or flux discretization rather than treating the
unweighted graph Laplacian as a universal physical model.

## How to Make a Scientific Model Work

1. **Write the mathematical problem.** State the domain, unknown field,
   governing equation, initial data, boundary conditions, and desired output.
2. **Choose the state.** Record the exact tensor layout and units of every
   channel. For an irregular domain, record node and edge semantics.
3. **Choose the construction.** Use explicit flow for a finite trajectory,
   `SILVAImplicitTimeStep` for an implicit numerical step, or
   `SILVAOperatorModel` for a learned map over a family of problems.
4. **Discretize independently.** Check derivatives and physical residuals on an
   analytic field before adding a learned model.
5. **Assign SILVA roles.** Put known or encoded inputs in the stimulus, local
   stencils or graph fluxes in local fields, pointwise laws in self fields, and
   nonlocal corrections in global or state-network fields.
6. **Check one forward solve.** Verify shape, dtype, device, finite values,
   residual decrease, iteration count, and boundary behavior.
7. **Check gradients.** Differentiate a scalar task loss with respect to input
   fields and trainable parameters using the selected backward mode.
8. **Train on a small deterministic task.** Keep train and held-out problems
   separate and compare against an analytic or direct numerical solution.
9. **Evaluate the scientific claim.** Report task error, fixed-point residual,
   physical residual, boundary error, solver cost, and transfer to another
   grid, mesh, coefficient range, or time horizon as applicable.
10. **Scale only after the checks pass.** Larger data and architectures cannot
    repair a mismatched equation, tensor contract, or boundary condition.

## What Changes Across Problem Types

| Problem | State | Stimulus | Useful internal fields | Output |
| --- | --- | --- | --- | --- |
| autonomous ODE steady state | vector $h$ | parameters or target | MLP, residual MLP, self field | equilibrium vector |
| time-dependent ODE trajectory | vector $h(t)$ | initial state and time | explicit vector field | terminal state or trajectory |
| elliptic PDE | spatial field $u$ | source and coefficients | stencil, U-Net, Fourier operator | stationary solution field |
| parabolic PDE | field at one time | previous state, coefficients, time | local flux plus spectral/global field | next state or time-slice equilibrium |
| graph PDE or irregular mesh | node/edge field | forcing and geometry | graph local plus global context | node or edge solution |
| operator regression | sampled input function | function values and coordinates | Fourier, U-Net, Transformer, custom kernel | sampled output function |

## Boundaries and Coordinates

Fourier convolution naturally represents periodic global interactions. A
nonperiodic PDE still needs its boundary condition represented explicitly.
Common choices are:

1. include boundary masks or boundary values in input channels;
2. enforce the boundary after each transition with `output_network`;
3. multiply the decoded field by a function that vanishes on the boundary;
4. use sine/cosine bases or a geometry-specific local operator;
5. use graph or mesh interactions on irregular domains.

Coordinates may be appended as input channels before `input_encoder`. They are
not added automatically because coordinate conventions depend on the domain.

## Diagnostics Checklist

For ODE, PDE, and operator examples, record all quantities that support the
claim being made:

| Quantity | Question answered |
| --- | --- |
| task or field error | does the prediction match the target data? |
| fixed-point residual | did the numerical equilibrium solve settle? |
| PDE residual | does the predicted field satisfy the discretized equation? |
| boundary error | are boundary conditions satisfied? |
| Jacobian or damped radius | is the local transition stable near the solution? |
| iteration count and runtime | what did the numerical solve cost? |
| resolution test | does the learned operator remain useful on another grid? |

Continue with the
[Neural Operators, ODEs, and PDEs notebook](../package-notebooks/15_neural_operators_ode_pde.ipynb),
the [Point Architecture Catalog](point-architecture-catalog.md), and the
[Solver Derivation Lab](solver-derivation-lab.md).

## Sources

- [Chen et al., Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [Li et al., Fourier Neural Operator for Parametric Partial Differential Equations](https://openreview.net/forum?id=c8P9NQVtmnO)
- [Kovachki et al., Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](https://www.jmlr.org/papers/v24/21-1524.html)
- [Silva, SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields](https://arxiv.org/abs/2607.28989)

BibTeX entries are collected in [`silva-networks.bib`](../assets/bib/silva-networks.bib).

## Where to Go Next

| Question | Page |
| --- | --- |
| Which spatial internal mappings are available? | [Point Architecture Catalog](point-architecture-catalog.md) |
| Can I execute the operator, ODE, and PDE examples? | [Neural Operators, ODEs, and PDEs Notebook](../package-notebooks/15_neural_operators_ode_pde.ipynb) |
| How is the Fourier mapping constructed through the API? | [Point Architectures API](../api/point_architectures.md) |
