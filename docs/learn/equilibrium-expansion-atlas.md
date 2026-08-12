# Equilibrium Expansion Atlas

The families on this page do not compete for one generic label. They change
different parts of a SILVA experiment: the transition, the solver, the backward
rule, the physical objective, or the execution substrate. Keeping those axes
separate makes it possible to combine mechanisms without hiding what each
article contributes.

## One Common SILVA Contract

Every equilibrium family begins with

$$
z^\star=T_\theta(z^\star,x),
\qquad
\widehat y=Q_\psi(z^\star).
$$

The full experiment is the tuple

$$
\mathcal E
=\left(
\mathcal D,
S_\theta,
H_\theta,
L_\theta,
G_\theta,
\mathcal S_f,
\mathcal S_b,
\mathcal L,
\mathcal M
\right),
$$

where \(\mathcal D\) is the data route, \(\mathcal S_f\) and
\(\mathcal S_b\) are forward and backward solvers, \(\mathcal L\) is the
training objective, and \(\mathcal M\) is the metric protocol. Reproducing an
article requires matching the complete tuple, not only constructing a class
with the same family name.

The tensor contract is explicit: \(T_\theta\) accepts and returns exactly the
same state shape. The batch dimension, entity or token axes, channels, spatial
resolution, dtype, and device therefore remain stable across every solver
evaluation. Source and readout modules may change dimensions only outside that
repeated transition.

## What Each Requested Mechanism Changes

| Mechanism | SILVA surface | Defining change | Primary source |
| --- | --- | --- | --- |
| HyperDEQ | `SILVAHyperDEQ` | learns initialization and Anderson updates | [[87]](../paper/references.md#ref-87){ .silva-cite } |
| JFB | `backward_mode="jfb"` | replaces the inverse adjoint factor by identity | [[88]](../paper/references.md#ref-88){ .silva-cite } |
| SHINE | `backward_mode="shine"` | shares the forward Broyden inverse with backward | [[89]](../paper/references.md#ref-89){ .silva-cite } |
| monotone splitting | `SILVAMonotoneOperatorEquilibrium` | constrains the transition and uses operator splitting | [[75]](../paper/references.md#ref-75){ .silva-cite } |
| C-DEQ | `SILVAConsistencyDEQ` | distills the solver trajectory to one/few-step refinement | [[59]](../paper/references.md#ref-59){ .silva-cite } |
| joint diffusion equilibrium | `SILVADiffusionEquilibrium` | solves an entire deterministic diffusion trajectory jointly | [[38]](../paper/references.md#ref-38){ .silva-cite } |
| generative equilibrium transformer | `SILVAGenerativeEquilibriumTransformer` | distills generation into a one-time-injected token equilibrium | [[48]](../paper/references.md#ref-48){ .silva-cite } |
| fixed-point diffusion | `SILVAFixedPointDiffusionModel` | solves a denoiser root at each timestep with variable compute | [[74]](../paper/references.md#ref-74){ .silva-cite } |
| physics-guided diffusion PDE | `SILVAPhysicsGuidedDiffusionPDE` | guides reverse steps with PDE residual energy and boundaries | [[64]](../paper/references.md#ref-64){ .silva-cite } |
| PIDEQ | `SILVAPhysicsInformedEquilibrium` | applies ODE/PDE residual losses at an implicit state | [[51]](../paper/references.md#ref-51){ .silva-cite } |
| QDEQ | `SILVAQuantumDEQ` | uses measured circuit outputs as the tied transition | [[90]](../paper/references.md#ref-90){ .silva-cite } |

## Forward-Solver Axis

The same transition may use Picard, Anderson, or Broyden:

```python
from silva_networks import SolverConfig

picard = SolverConfig(solver="picard", max_iter=50, tol=1e-6)
anderson = SolverConfig(
    solver="anderson",
    max_iter=35,
    tol=1e-6,
    history=6,
    anderson_batch_dims=1,
)
broyden = SolverConfig(
    solver="broyden",
    max_iter=35,
    tol=1e-6,
    history=10,
)
```

HyperDEQ is a fourth route: it learns a fixed number of task-specific Anderson
updates. C-DEQ is another: it learns a map from solver-trajectory states toward
the terminal root. Neither changes the underlying equation.

## Backward Axis

Given

$$
(I-J_T^\top)u=g,
$$

SILVA exposes five backward paths:

```python
exact = SolverConfig(backward_mode="implicit", backward_solver="gmres")
jfb = SolverConfig(backward_mode="jfb")
shine = SolverConfig(
    solver="broyden",
    backward_mode="shine",
    shine_refine_steps=2,
)
phantom = SolverConfig(backward_mode="phantom", phantom_steps=3, phantom_tau=0.5)
unrolled = SolverConfig(backward_mode="unrolled")
```

These are experiment variables. A benchmark table should never report only
"implicit training" when the backward approximation, tolerance, and iteration
budget differ.

## Monotone Operator Splitting

The monotone family parameterizes

$$
W=(1-m)I-A^\top A+B-B^\top,
$$

so

$$
\frac{W+W^\top}{2}=(1-m)I-A^\top A\preceq(1-m)I.
$$

The equilibrium inclusion is

$$
0\in(I-W)z-Ux-b+\partial f(z).
$$

Forward-backward splitting evaluates

$$
z_{k+1}
=\operatorname{prox}_{\alpha f}
\left(z_k-\alpha((I-W)z_k-Ux-b)\right).
$$

Peaceman-Rachford splitting introduces a reflected proximal state and is exposed
through `splitting="peaceman_rachford"`. The operator, source, proximal map,
readout, margin, step size, and solver remain replaceable. See the existing
[monotone operator lab](../package-notebooks/36_silva_monotone_operator_equilibrium.ipynb).

## Diffusion Architectures Are Distinct

### Joint DDIM Equilibrium

Stack the entire trajectory

$$
Z=(x_T,x_{T-1},\ldots,x_0).
$$

The map updates all coordinates from the deterministic DDIM relation, and SILVA
solves

$$
Z^\star=\mathcal T_{\mathrm{DDIM}}(Z^\star;x_T,c).
$$

Use `SILVADiffusionEquilibrium` for this DEQ-DDIM construction
[[38]](../paper/references.md#ref-38){ .silva-cite }.

### One-Time-Injected Generative Equilibrium

GET injects noise and optional class information once, solves a token
equilibrium, and decodes the result. Use
`SILVAGenerativeEquilibriumTransformer` [[48]](../paper/references.md#ref-48){ .silva-cite }.

### Per-Timestep Denoiser Root

FPDM defines a separate equilibrium at each diffusion time:

$$
z_t^\star=F_\theta(z_t^\star,P(x_t),t),
\qquad
\widehat\epsilon_t=Q_\theta(z_t^\star,x_t,t).
$$

Use `SILVAFixedPointDiffusionModel` for variable compute, warm starts, solution
reuse, and per-timestep residuals [[74]](../paper/references.md#ref-74){ .silva-cite }.

### Physics-Guided Reverse Process

The physics-guided field family modifies each reverse step:

$$
u_{t-1}
=\Pi_{\partial\Omega}
\left(\mathcal S(D_\theta(u_t,t))-\eta_t\nabla E_{\mathrm{PDE}}(u_t)+\xi_t\right).
$$

Use `SILVAPhysicsGuidedDiffusionPDE` when the defining mechanism is residual
energy guidance, smoothing, and hard boundary projection
[[64]](../paper/references.md#ref-64){ .silva-cite }.

## PIDEQ: Physics at the Equilibrium

For an implicit prediction \(D_\theta(t)=Q(z^\star(t))\), differentiate

$$
z^\star=T_\theta(z^\star,t)
$$

to obtain

$$
(I-J_zT_\theta)\frac{dz^\star}{dt}=\frac{\partial T_\theta}{\partial t}.
$$

The physics residual is

$$
r_{\mathrm{phys}}(t)
=\frac{dD_\theta}{dt}-\mathcal N(t,D_\theta(t)).
$$

The training objective combines initial/boundary conditions, physics residual,
data when available, and Jacobian regularization. The dynamics callable belongs
to the physical law; the SILVA transition defines the implicit representation.
They are separate extension points.

```python
from silva_networks import SILVAPhysicsInformedEquilibrium

model = SILVAPhysicsInformedEquilibrium(
    state_dim=64,
    output_dim=2,
    transition=my_time_conditioned_transition,
    readout=my_state_readout,
    config=exact,
)

result = model(times, return_result=True)
physics = model.physics_loss(times, result.state, dynamics=my_ode_rhs)
```

The full derivation and Van der Pol route remain in
[Physics-Informed Equilibria and DAEs](physics-informed-equilibria.md).

## QDEQ: Change the Transition Substrate

QDEQ keeps the fixed-point contract but replaces the tied mapping by circuit
encoding, unitary evolution, and measurement:

$$
z^\star=\mathcal M(U_\theta\mathcal E(z^\star+S(x))).
$$

The input adapter, circuit, measurement width, readout, forward solver,
backward mode, and Jacobian penalty are independently configurable. The full
derivation is in [Quantum Equilibria](quantum-equilibria.md).

## Combinations That Are Well Defined

| Combination | Meaning |
| --- | --- |
| HyperDEQ + FNO-DEQ | learn Anderson updates around a Fourier field transition |
| HyperDEQ + PIDEQ | accelerate the latent root while retaining physics losses |
| JFB + QDEQ | keep the circuit root, omit the backward adjoint solve |
| SHINE + PIDEQ | share Broyden inverse factors with physics-informed backward gradients |
| C-DEQ + multiscale transition | distill multiresolution solver trajectories |
| monotone transition + SHINE | retain structural well-posedness and reuse Broyden information |
| fixed-point diffusion + JFB | train each timestep root through one final denoiser transition |

Each combination still needs a compact gradient test, convergence diagnostics,
and a source-scale ablation against the uncombined baseline.

## Minimum Experiment Record

For every run, record:

1. canonical SILVA family and complete constructor;
2. transition, initializer, source, readout, and any custom compressor;
3. forward solver, tolerance, stop mode, history, and iteration cap;
4. backward mode, solver or approximation, tolerance, and refinement count;
5. dataset source, version, split, preprocessing, and class subset;
6. optimizer, schedule, precision, batch size, accumulation, and seeds;
7. task metric, normalized residual, iterations, runtime, memory, and failure count;
8. source article and research repository.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are HyperDEQ, JFB, and SHINE derived? | [Learned Solvers and Backward Approximations](solver-learning-and-gradients.md) |
| How is QDEQ constructed? | [Quantum Equilibria](quantum-equilibria.md) |
| How are all mechanisms executed together? | [Expansion Atlas Lab](../package-notebooks/51_equilibrium_expansion_atlas.ipynb) |
| How do I plan a complete benchmark? | [Reconstructing Paper Experiments](reconstructing-paper-experiments.md) |
