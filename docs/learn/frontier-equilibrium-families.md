# Recent Equilibrium Families Inside SILVA

SILVA is the general framework on this page. Fourier operators, graph physics,
continuous homotopy paths, and empirical-measure flows define what happens
inside a SILVA point or how that point is solved. They do not replace the SILVA
source, state, local, and global decomposition
[[1]](../paper/references.md#ref-1){ .silva-cite }.

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/16_frontier_equilibrium_families/">Open executable notebook</a>
  <a class="md-button" href="../../package-notebooks/16_frontier_equilibrium_families/16_frontier_equilibrium_families.ipynb" download>Download notebook</a>
</div>

## What Is Implemented

| Literature family | SILVA status | Package surface |
| --- | --- | --- |
| foundational DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } | implemented | compact, sequence, graph, image, and general equilibrium points |
| multiscale DEQ [[5]](../paper/references.md#ref-5){ .silva-cite } | implemented | `SILVAMultiscaleDEQ`, multiscale point architectures |
| implicit graph network [[36]](../paper/references.md#ref-36){ .silva-cite } | implemented | `SILVAImplicitGraphNetwork` |
| FNO-DEQ [[43]](../paper/references.md#ref-43){ .silva-cite } | implemented as a SILVA family | `SILVAFNODEQ`, `SILVAFNODEQBlock` |
| physics-guided graph DEQ [[44]](../paper/references.md#ref-44){ .silva-cite } | implemented as a SILVA family | `SILVAGraphConvectionDiffusion`, `SILVAPhysicsGuidedGraphDEQ` |
| homotopy equilibrium flow [[46]](../paper/references.md#ref-46){ .silva-cite } | implemented as a transparent SILVA specialization | `SILVAHomotopyEquilibrium` |
| distributional DEQ [[45]](../paper/references.md#ref-45){ .silva-cite } | implemented as a SILVA family | `SILVADistributionalTransition`, `SILVADistributionalDEQ` |
| joint diffusion restoration [[49]](../paper/references.md#ref-49){ .silva-cite } | mechanism already represented | `SILVADiffusionEquilibrium` solves a joint triangular trajectory |
| one-step equilibrium transformer [[48]](../paper/references.md#ref-48){ .silva-cite } | documented port target | requires image patching and an offline teacher-pair training recipe |
| monotone implicit graph network [[47]](../paper/references.md#ref-47){ .silva-cite } | documented port target | requires resolvents and operator-splitting solvers |
| mirror-descent Poisson equilibrium [[50]](../paper/references.md#ref-50){ .silva-cite } | documented port target | requires a positive-domain mirror map and Poisson data fidelity |

“Implemented as a SILVA family” means the mathematical mechanism has a public
class, unit and integration tests, a runnable example, deterministic teaching
data, and executable small-scale reproductions. It does not mean that the large
datasets, model sizes, or paper benchmark tables have been reproduced.

## One Grammar, Four Extensions

The ordinary SILVA equilibrium is

$$
\begin{aligned}
z^\star
&=T_\theta(z^\star;x) \\
&=\sigma\!\left(
S_\theta(x)+H_\theta(z^\star)\right. \\
&\qquad\left.
+L_\theta(z^\star)+G_\theta(z^\star)
\right).
\end{aligned}
$$

The four new families alter a different part of this equation:

| SILVA family | What changes | What remains fixed |
| --- | --- | --- |
| Fourier equilibrium | $H_\theta$ contains a low-mode spectral convolution and local channel map | source injection, fixed-point solve, readout |
| physics graph equilibrium | $L_\theta$ is split into graph diffusion and directed transport | source, state shape, solver, node/graph readout |
| homotopy equilibrium | the path to $z^\star$ is a continuous residual flow | the stationary equation $z^\star=T(z^\star;x)$ |
| distributional equilibrium | the state is an empirical measure and the residual is a measure discrepancy | source conditioning and a repeated SILVA transition |

This separation is useful when constructing heterogeneous models. A Fourier
point can feed a graph-physics point; a distributional point can receive a
global condition from a vector point; every point can have its own solver and
diagnostics.

## SILVA Fourier Equilibrium

### From a steady PDE to a fixed point

Let a steady PDE be written abstractly as

$$
\mathcal N_a(u)=0,
$$

where $a$ contains coefficient, forcing, geometry, or boundary data. If a
nonlinear solution operator $\mathcal K_a$ is available, the same solution can
be characterized by

$$
u^\star=\mathcal K_a(u^\star).
$$

FNO-DEQ uses an input-injected Fourier block as the learned fixed-point map
[[43]](../paper/references.md#ref-43){ .silva-cite }. For a sampled field $v_j$
and lifted forcing $g=P(a)$, one layer is

$$
\begin{aligned}
v_{j+1}
&=g+\sigma\!\left(
W_jv_j+b_j\right. \\
&\qquad\left.
+\mathcal F^{-1}\!\left(R_j\odot\mathcal F(v_j)\right)
\right).
\end{aligned}
$$

$W_j$ is a pointwise channel map, $R_j$ acts on retained Fourier modes, and
$\mathcal F$ is the discrete Fourier transform. If a block contains $J$
internal layers, define

$$
B_\theta(v,g)=\mathcal L_{J-1}(\cdots\mathcal L_0(v,g),g).
$$

The complete SILVA operator solves

$$
v^\star=B_\theta(v^\star,P(a)),
\qquad
\widehat u=Q(v^\star).
$$

### SILVA branch interpretation

The forcing $g$ is $S_\theta(a)$ and is injected at every internal layer. The
spectral convolution is a global state interaction because every retained
frequency can affect the complete spatial field. The $1\times1$ channel map is
a local-in-space self interaction. Additional boundary, graph, or mean-field
branches can still be added around this point.

### Shape contract

```text
forcing field:      (batch, in_channels, height, width)
lifted forcing:     (batch, state_channels, height, width)
equilibrium state:  (batch, state_channels, height, width)
decoded field:      (batch, out_channels, height, width)
```

The learned spectral weights do not depend on `height` or `width`. At runtime,
the implementation retains at most the requested number of modes available on
the current grid. This is resolution compatibility, not by itself a guarantee
of discretization-invariant error.

### Minimal run

```python
import torch
from silva_networks import SILVAFNODEQ, SolverConfig

model = SILVAFNODEQ(
    in_channels=1,
    state_channels=8,
    out_channels=1,
    modes_height=4,
    modes_width=4,
    block_depth=2,
    state_scale=0.05,
    config=SolverConfig(
        solver="anderson",
        max_iter=30,
        tol=1e-6,
        backward_mode="implicit",
    ),
)

forcing = torch.randn(4, 1, 32, 32)
result = model(forcing, return_result=True)
print(result.output.shape)
print(result.solver_result.residual)
```

Use a PDE residual in addition to the solver residual. The first checks
$\|B(v^\star,g)-v^\star\|$; the second checks whether the decoded field obeys
the intended physical equation. They answer different questions.

## SILVA Physics-Guided Graph Equilibrium

### Continuous equation

A convection-diffusion field $c$ can be written

$$
\frac{\partial c}{\partial t}
=-\mathbf v\cdot\nabla c
+d\,\Delta c
+\phi.
$$

The physics-guided graph equilibrium literature moves these transport terms
inside the graph transition rather than using only a loss penalty
[[44]](../paper/references.md#ref-44){ .silva-cite }.

### Graph discretization

For directed edge $i\rightarrow j$, SILVA defines the incoming diffusion and
directed-gradient fields

$$
(\mathcal L_G Z)_j
=\frac{1}{d_j}\sum_{i\rightarrow j}w_{ij}(Z_i-Z_j),
$$

$$
(\nabla_V Z)_j
=\frac{1}{d_j}\sum_{i\rightarrow j}v_{ij}(Z_j-Z_i),
$$

where $w_{ij}$ is a nonnegative diffusion weight, $v_{ij}$ is a signed
directed velocity, and $d_j$ is the number of incoming edges. The SILVA
transition is

$$
\begin{aligned}
T(Z;X)
&=\phi\!\left[
S(X)+\gamma_rR(Z)\right. \\
&\qquad+\gamma_dD(\mathcal L_GZ) \\
&\qquad\left.
-\gamma_aA(\nabla_VZ)
\right].
\end{aligned}
$$

The learned channel maps $R$, $D$, and $A$ let reaction, diffusion, and
advection act differently on each latent feature. The equilibrium is

$$
Z^\star=T(Z^\star;X).
$$

### Minimal run

```python
import torch
from silva_networks import SILVAPhysicsGuidedGraphDEQ

x = torch.randn(6, 3)
edge_index = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
     [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
    dtype=torch.long,
)
edge_velocity = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1.0])

model = SILVAPhysicsGuidedGraphDEQ(
    in_dim=3,
    state_dim=12,
    out_dim=1,
)
result = model(
    x,
    edge_index,
    edge_velocity=edge_velocity,
    return_result=True,
)
print(result.output.shape, result.solver_result.residual)
```

Relabeling nodes and relabeling `edge_index` in the same way relabels the node
outputs without changing their values. `tests/test_frontier.py` verifies that
property for nonuniform diffusion and velocity fields.

### Which graph quantity goes where?

| Data | Argument | Meaning |
| --- | --- | --- |
| node observations, coordinates, sources | `x` | SILVA source branch |
| connectivity | `edge_index` | spatial domain discretization |
| conductance, inverse distance, area factor | `edge_weight` | diffusion coefficient per edge |
| wind, flow, signed directional speed | `edge_velocity` | convection coefficient per edge |
| graph membership for a batch | `batch` | graph-level pooling only |

The package does not infer physical units. Scale all quantities consistently,
record the convention for edge direction, and validate against a numerical or
measured reference.

## SILVA Homotopy Equilibrium

### Fixed-point homotopy

Let

$$
r(z;x)=z-T(z;x).
$$

The fixed point solves $r(z^\star;x)=0$. A classical fixed-point homotopy from
an easy initial equation to the desired residual is

$$
\begin{aligned}
H(z,\lambda;x)
&=\lambda r(z;x) \\
&\quad +(1-\lambda)(z-z_0), \\
&\qquad \lambda\in[0,1].
\end{aligned}
$$

Along the zero path $H(z(s),\lambda(s);x)=0$, differentiation gives

$$
\frac{\partial H}{\partial z}\frac{dz}{ds}
+\frac{\partial H}{\partial\lambda}\frac{d\lambda}{ds}=0.
$$

This equation connects a root problem to a continuous path. HomoODE learns a
conditioned continuous dynamic and uses a shared initial point
[[46]](../paper/references.md#ref-46){ .silva-cite }.

### Transparent SILVA specialization

`SILVAHomotopyEquilibrium` chooses the directly interpretable residual flow

$$
\frac{dz}{dt}=T(z;x)-z=-r(z;x).
$$

Any stationary state obeys the original SILVA equation. The class integrates
the flow with fixed-step Euler or fourth-order Runge-Kutta and reports

$$
\epsilon_T=\max_b\|T(z_b(T);x_b)-z_b(T)\|_2.
$$

This specialization exposes the fixed-point residual explicitly. It is not a
numerically identical reproduction of the learned continuous dynamic in the
HomoODE experiments.

### Analytic example

For

$$
T(z;x)=az+x,
\qquad |a|<1,
$$

the equilibrium and residual flow are

$$
z^\star=\frac{x}{1-a},
\qquad
\frac{dz}{dt}=x-(1-a)z.
$$

Therefore

$$
z(t)=z^\star+(z_0-z^\star)e^{-(1-a)t}.
$$

The notebook uses $a=1/2$, checks $z^\star=2x$, and compares the numerical
terminal state with this closed form.

```python
import torch
from torch import nn
from silva_networks import SILVAHomotopyEquilibrium

class AffineTransition(nn.Module):
    def forward(self, state, condition):
        return 0.5 * state + condition

model = SILVAHomotopyEquilibrium(
    in_dim=1,
    state_dim=1,
    out_dim=1,
    transition=AffineTransition(),
    readout=nn.Identity(),
    steps=64,
    horizon=12.0,
    integrator="rk4",
    learnable_initial=False,
)

x = torch.tensor([[0.4], [-0.7]])
result = model(x, return_result=True)
print(torch.max(torch.abs(result.state - 2.0 * x)))
```

The fixed-step path is differentiated through directly. Its activation memory
grows with `steps`; it does not claim an adjoint-memory result. Use the
fixed-point solvers when an equilibrium is required more directly, and use the
homotopy flow when the continuous path itself is part of the model or analysis.

## SILVA Distributional Equilibrium

### Why an ordinary residual is insufficient

Let input particles $X=(x_1,\ldots,x_M)$ represent an empirical measure

$$
\rho_X=\frac1M\sum_{i=1}^{M}\delta_{x_i},
$$

and let latent particles $Z=(z_1,\ldots,z_N)$ represent

$$
\mu_Z=\frac1N\sum_{j=1}^{N}\delta_{z_j}.
$$

Two matrices that differ only by row order represent the same measure. An
ordinary Euclidean residual $\|F(Z,X)-Z\|$ compares rows by position and does
not respect this equivalence. Distributional DEQs instead define

$$
G_{\theta,X}(\mu)
=\frac12D^2\!\left(\mu,F_\theta(\mu,\rho_X)\right)
$$

and minimize $G$ over measures
[[45]](../paper/references.md#ref-45){ .silva-cite }.

### MMD and energy distance

For a kernel $k$, the biased squared maximum mean discrepancy is

$$
\begin{aligned}
\operatorname{MMD}^2(\mu,\nu)
&=\mathbb E_{x,x'\sim\mu}k(x,x') \\
&\quad+\mathbb E_{y,y'\sim\nu}k(y,y') \\
&\quad-2\mathbb E_{x\sim\mu,y\sim\nu}k(x,y).
\end{aligned}
$$

The Gaussian choice is

$$
k_\ell(x,y)=\exp\!\left(-\frac{\|x-y\|^2}{2\ell^2}\right).
$$

The package also provides the energy distance

$$
\begin{aligned}
D_E^2(\mu,\nu)
&=2\mathbb E\|x-y\| \\
&\quad-\mathbb E\|x-x'\| \\
&\quad-\mathbb E\|y-y'\|,
\end{aligned}
$$

which is the MMD form induced by the negative-distance kernel used in the DDEQ
experiments.

### Wasserstein particle descent

At the measure level, the inner optimization follows the Wasserstein gradient
flow

$$
\partial_t\mu_t
=\nabla\cdot\left(\mu_t\nabla_WG(\mu_t)\right).
$$

For empirical particles, forward Euler gives

$$
z_j^{(k+1)}
=z_j^{(k)}-\eta\,\nabla_{z_j}G(Z^{(k)}).
$$

`SILVADistributionalDEQ` differentiates the discrepancy with respect to the
latent particles and applies this update. `context_mask` and `latent_mask`
support padded batches. `fixed_mask` prevents selected latent particles from
moving, which is useful when observed particles must remain exact.

### EI transition contract

For latent permutation $P$ and input permutation $Q$, the built-in transition
satisfies

$$
F_\theta(PZ,QX)=P F_\theta(Z,X).
$$

It is equivariant in latent ordering and invariant in input ordering. The
implementation obtains this property with self-attention on each measure,
cross-attention from latent to input particles, pooled bilinear context, and
pointwise feed-forward maps, all without positional row encodings.

### Minimal run

```python
import torch
from silva_networks import SILVADistributionalDEQ

context = torch.randn(3, 20, 2)
context_mask = torch.ones(3, 20, dtype=torch.bool)

model = SILVADistributionalDEQ(
    input_dim=2,
    latent_dim=16,
    particles=10,
    heads=4,
    kernel="energy",
    step_size=1.0,
    max_iter=40,
)
result = model(
    context,
    context_mask=context_mask,
    return_result=True,
)
print(result.state.shape)
print(result.discrepancies[0], result.discrepancies[-1])
```

The transition is architecture-pluggable. A replacement must preserve the
shape of `latent` and accept both masks. If the task is order-independent, it
should also satisfy the EI equation above.

## Selecting These Families

The canonical family keys keep every construction under SILVA:

```python
from silva_networks import silva_equilibrium_model

operator = silva_equilibrium_model(
    "silva_fno_deq",
    in_channels=1,
    state_channels=8,
    out_channels=1,
)

particles = silva_equilibrium_model(
    "silva_distributional_deq",
    input_dim=3,
    latent_dim=16,
)
```

The searchable literature aliases `fno_deq`, `pgcn_deq`, `homoode`, and
`ddeq` resolve to these SILVA constructors. New configuration files should use
the canonical `silva_*` keys.

## Dataset-Backed Reproductions

The combined notebook, four focused labs, and `examples/frontier_equilibria.py`
now use deterministic package builders matched to each state geometry.

| SILVA family | Dataset builder | Quantity checked | Focused notebook |
| --- | --- | --- | --- |
| Fourier equilibrium | `make_periodic_elliptic_dataset` | field shape, fixed-point residual, elliptic residual, gradients, resolution change | [Fourier equilibrium lab](../package-notebooks/17_silva_fno_equilibrium_lab.ipynb) |
| physics graph equilibrium | `make_graph_transport_dataset` | discrete transport residual, batched edges, node relabeling, gradients | [Graph transport lab](../package-notebooks/18_silva_graph_transport_lab.ipynb) |
| homotopy equilibrium | `make_affine_homotopy_dataset` | analytic endpoint, complete decay law, Euler/RK4, gradients | [Homotopy equilibrium lab](../package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb) |
| distributional equilibrium | `make_variable_measure_dataset` | masks, counts, moments, permutation behavior, particle descent, gradients | [Distributional equilibrium lab](../package-notebooks/20_silva_distributional_equilibrium_lab.ipynb) |

These experiments validate equations, tensor contracts, solver wiring,
training paths, plots, and gradients. The [dataset-backed lab guide](frontier-dataset-labs.md)
derives every builder and explains the handoff to the paper datasets. The
paper-reported Darcy, Navier-Stokes, environmental, image, and point-cloud
results still require their complete datasets and protocols.

## Extension Requirements

The remaining ports have clear implementation boundaries.

### Monotone graph equilibrium

A monotone graph port needs a resolvent or proximal operator and an
operator-splitting update, such as forward-backward, Peaceman-Rachford, or
Douglas-Rachford. Merely projecting the recurrent weight norm would not
reproduce the monotone formulation
[[47]](../paper/references.md#ref-47){ .silva-cite }.

### One-step equilibrium transformer

A complete port needs image patch embedding, a noncausal equilibrium
transformer, conditioning from the input noise, image reconstruction, and the
offline noise/image-pair distillation objective
[[48]](../paper/references.md#ref-48){ .silva-cite }. The existing SILVA
sequence equilibrium supplies part of the internal transition, but not the
complete training protocol.

### Parallel diffusion restoration

`SILVADiffusionEquilibrium` already represents a selected reverse diffusion
trajectory as one joint triangular fixed point, the central parallel-state
mechanism used in equilibrium restoration
[[49]](../paper/references.md#ref-49){ .silva-cite }. A benchmark reproduction
still needs a compatible pretrained denoiser, degradation operator, schedules,
and restoration datasets.

### Poisson mirror descent

A faithful port needs the Poisson negative log-likelihood, a positive-domain
mirror map, the associated Bregman divergence, and convergence-aware learned
regularization [[50]](../paper/references.md#ref-50){ .silva-cite }. A Euclidean
projected gradient step is not an equivalent substitute.

## Validation in the Repository

| Property | Test or artifact |
| --- | --- |
| Fourier input injection, resolution changes, gradients | `tests/test_frontier.py` |
| graph branch separation, relabeling equivariance, gradients | `tests/test_frontier.py` |
| analytic homotopy endpoint and both integrators | `tests/test_frontier.py` |
| MMD/energy permutation invariance and masks | `tests/test_frontier.py` |
| EI transition property and fixed particles | `tests/test_frontier.py` |
| generated equation datasets, masks, batching, and integrated gradients | `tests/test_frontier_data.py` |
| all four small runs | `examples/frontier_equilibria.py` |
| derivations and progressive experiments | `16_frontier_equilibrium_families.ipynb` |
| dataset-backed derivation and training labs | notebooks `17` through `20` |

## Where to Go Next

| Question | Page |
| --- | --- |
| How do ordinary ODE, PDE, and neural-operator cases enter SILVA? | [Neural Operators, ODEs, PDEs, and SILVA](neural-operators-ode-pde.md) |
| Where is the public API for these four families? | [Recent Equilibrium API](../api/frontier.md) |
| Where are the datasets and focused training labs derived? | [Dataset-Backed Equilibrium Labs](frontier-dataset-labs.md) |
| Where can I run the compact reproductions? | [Recent Equilibrium Examples](../examples/frontier-equilibria.md) |
