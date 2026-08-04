# Neural Operators, ODEs, PDEs, and SILVA

This guide connects four objects that are often introduced separately:

1. an ODE evolves a state through continuous time;
2. a PDE evolves or constrains a field over space and possibly time;
3. a neural operator learns a map between functions;
4. SILVA builds a structured transition from named fields and solves its fixed point.

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
