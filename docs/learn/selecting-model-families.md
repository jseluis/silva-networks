# Selecting Model Families

The package exposes SILVA as a general equilibrium grammar. A model family is
chosen by selecting the state, the transition terms, and the solver:

$$
z^\star=f_\theta(z^\star,x).
$$

For branch-structured SILVA layers,

$$
f_\theta(z,x)
=
\Psi_\theta
\left(
S_\theta(x)
+H_\theta(a(z))
+L_\theta(a(z),E)
+G_\theta(a(z),b)
\right).
$$

For broader implicit systems, the state can be a tuple of tensors, a multiscale
state, a flow field, an optimizer variable, a continuous-flow endpoint, or an
empirical measure represented by particles. The numerical interface follows
the state: fixed-point, continuous integration, or measure-discrepancy descent.

The selector covers SILVA [[1]](../paper/references.md#ref-1){ .silva-cite }, DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, MDEQ
[[5]](../paper/references.md#ref-5){ .silva-cite }, Neural ODEs
[[7]](../paper/references.md#ref-7){ .silva-cite }, optimization layers
[[8]](../paper/references.md#ref-8){ .silva-cite }
[[9]](../paper/references.md#ref-9){ .silva-cite }, and optical flow
[[22]](../paper/references.md#ref-22){ .silva-cite }
[[23]](../paper/references.md#ref-23){ .silva-cite }. Recent choices add
input-injected Fourier equilibria [[43]](../paper/references.md#ref-43){ .silva-cite },
physics graph equilibria [[44]](../paper/references.md#ref-44){ .silva-cite },
homotopy flows [[46]](../paper/references.md#ref-46){ .silva-cite }, and
distributional equilibria [[45]](../paper/references.md#ref-45){ .silva-cite }.
The extended selector also covers consistency acceleration
[[59]](../paper/references.md#ref-59){ .silva-cite }, monotone operator
splitting [[75]](../paper/references.md#ref-75){ .silva-cite }, learned
equilibrium solvers [[87]](../paper/references.md#ref-87){ .silva-cite }, and
measured circuit equilibria [[90]](../paper/references.md#ref-90){ .silva-cite }.
JFB [[88]](../paper/references.md#ref-88){ .silva-cite } and SHINE
[[89]](../paper/references.md#ref-89){ .silva-cite } are backward policies that
can be selected independently of the family.

## Family Selector

```python
from silva_networks import available_silva_families, silva_equilibrium_model

print(available_silva_families())
```

| Family | What it builds |
| --- | --- |
| `"silva_layer"` | one generalized SILVA layer |
| `"silva_graph"` | stacked SILVA graph model |
| `"silva_graph_preset"` | reference graph preset with configurable interaction modes |
| `"silva_cortex"` | one flexible cortex-style equilibrium point |
| `"silva_cortex_network"` | linked SILVA points with independent internal architectures |
| `"silva_image_cortex"` | convolutional retina plus linked cortex equilibrium points |
| `"compact_deq"` | affine-tanh DEQ reduction |
| `"message_passing_deq"` | local graph/message-passing DEQ reduction |
| `"mdeq"` | compact multiscale DEQ bridge block |
| `"multiscale_vision_deq"` | simultaneous multiresolution vision equilibrium |
| `"sequence_deq"` | relative-attention or trellis sequence equilibrium |
| `"implicit_graph"` | graph equilibrium with configurable adjacency normalization |
| `"implicit_neural_representation"` | coordinate-based SIREN, Fourier, or Gabor equilibrium |
| `"diffusion_equilibrium"` | joint selected denoising trajectory equilibrium |
| `"scientific_operator"` | source-to-field SILVA point with a selectable internal architecture |
| `"fourier_operator_equilibrium"` | Fourier neural operator field inside a SILVA point |
| `"implicit_time_step"` | backward-Euler ODE or PDE step solved as an equilibrium |
| `"silva_deq_flow"` | compact package-native optical-flow equilibrium |
| `"raft_deq_flow"` | coupled hidden-state and flow RAFT/DEQ-Flow equilibrium |
| `"quadratic_optimization"` | unconstrained quadratic optimization layer |
| `"silva_projected_qp"` | SILVA-named projected constrained quadratic layer |
| `"silva_fno_deq"` | input-injected Fourier block solved inside SILVA |
| `"silva_physics_graph_deq"` | reaction, diffusion, and directed transport graph equilibrium |
| `"silva_homotopy_equilibrium"` | conditioned SILVA residual flow |
| `"silva_distributional_deq"` | permutation-compatible empirical-measure equilibrium |
| `"silva_monotone_graph_equilibrium"` | graph equilibrium with a constrained channel operator |
| `"silva_generative_equilibrium_transformer"` | one-time-injected generative token equilibrium |
| `"silva_poisson_mirror_equilibrium"` | positive Poisson inverse problem with Burg mirror geometry |
| `"silva_physics_informed_equilibrium"` | implicit representation trained with ODE or PDE residuals |
| `"silva_implicit_dae_step"` | implicit Runge-Kutta stage equilibrium for differential-algebraic systems |
| `"silva_consistency_deq"` | consistency-distilled one- or few-step equilibrium refinement |
| `"silva_psi_gnn"` | mixed-boundary Poisson graph solver with typed message channels |
| `"silva_ifno"` | tied Fourier material-response operator |
| `"silva_snarf"` | canonical correspondence roots inside forward skinning |
| `"silva_mesh_inference"` | distributed Gaussian information equilibrium over a mesh |
| `"silva_physics_guided_diffusion_pde"` | reverse diffusion guided by PDE energy and hard boundaries |
| `"silva_therino"` | thermodynamic iteration in the physical solution space |
| `"silva_fixed_point_diffusion"` | per-timestep denoiser equilibrium with variable compute |
| `"silva_monotone_operator_equilibrium"` | monotone inclusion solved by operator splitting |
| `"silva_positive_concave_equilibrium"` | positive concave fixed point with certificate controls |
| `"silva_non_euclidean_equilibrium"` | equilibrium on a declared non-Euclidean state geometry |
| `"silva_efficient_infinite_graph"` | factorized infinite-depth graph equilibrium |
| `"silva_multiscale_graph_implicit"` | multiple graph-power equilibria with learned fusion |
| `"silva_delta_equilibrium"` | thresholded incremental updates with exact residual checks |
| `"silva_hyper_deq"` | learned initializer and learned Anderson controller |
| `"silva_quantum_deq"` | measured parameterized circuit inside a fixed point |

## Reductions

The compact DEQ reduction is obtained by setting

$$
L_\theta=0,\qquad G_\theta=0,\qquad a(z)=z,\qquad H_\theta(z)=W_z z,
$$

so

$$
z^\star
=
\tanh(W_xx+b+W_z z^\star).
$$

```python
from silva_networks import SolverConfig, silva_equilibrium_model

deq = silva_equilibrium_model(
    "compact_deq",
    in_dim=16,
    hidden_dim=64,
    config=SolverConfig(solver="anderson", max_iter=25, alpha=0.6),
)
```

The message-passing DEQ keeps the local field:

$$
z^\star=\Psi_\theta(S_\theta(x)+L_\theta(a(z^\star),E)).
$$

```python
mp_deq = silva_equilibrium_model(
    "message_passing_deq",
    in_dim=features.shape[1],
    hidden_dim=64,
    local="gat",
    local_kwargs={"heads": 4},
)
```

The full SILVA graph model turns local and global fields on, with optional
learned self interaction:

```python
model = silva_equilibrium_model(
    "silva_graph",
    in_dim=features.shape[1],
    hidden_dims=[64, 64, 32],
    out_dim=num_classes,
    local=["graph", "gat", "topk"],
    global_term=["mean", "simple", "topk"],
    self_term=["none", "linear", "none"],
)
```

## Cortex Families

The cortex selector exposes one equilibrium point whose internal transition can
be a deep PyTorch module:

$$
z^\star
=
\Psi\!\left[
R_\phi(x)+B_\theta(\tanh z^\star)+H_\theta(\tanh z^\star)
+L_\theta(\tanh z^\star)+G_\theta(\tanh z^\star)
\right].
$$

```python
cortex = silva_equilibrium_model(
    "silva_cortex",
    input_dim=5,
    state_dim=14,
    state_network=deep_state_network(14, depth=10),
    config=SolverConfig(solver="picard", max_iter=10, alpha=0.5),
)
```

The generic network family links independently constructed points. This is the
family for different internal architectures, state shapes, solvers, and damping
values at each SILVA point:

```python
network = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_unet_point, vector_attention_point],
    links=[spatial_to_vector],
    head=classification_head,
)
```

Each point still solves its own equation

$$
z_\ell^\star=F_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
$$

while the link maps the solved state into the next point's input space.

The image preset adds the convolutional-retina hierarchy:

```python
image_cortex = silva_equilibrium_model(
    "silva_image_cortex",
    in_channels=3,
    hidden_dim=[128, 128],
    num_classes=10,
    image_size=32,
    attention_mode="simple",
    graph_mode="GAT",
    alphas=(0.5, 0.2),
    internal_depth=2,
)
```

## Generalized Sequence, Vision, Graph, Coordinate, and Diffusion Families

The generalized cases keep the fixed-point contract while changing the state
space and transition. Their primary derivations and citations are collected in
[Paper Family Adaptations](paper-family-adaptations.md).

| Family | Equilibrium state | Required constructor information | Principal diagnostic |
| --- | --- | --- | --- |
| `multiscale_vision_deq` | tuple of feature maps at several resolutions | input channels, per-scale channels, multiscale transition settings | packed residual plus per-scale shapes |
| `sequence_deq` | `(batch, tokens, dim)` | state width, vocabulary or features, attention/trellis settings | token-state residual and masking behavior |
| `implicit_graph` | `(nodes, state_dim)` | feature dimensions, edges at call time, adjacency normalization | graph-state residual and edge normalization |
| `implicit_neural_representation` | `(batch, queries, state_dim)` | coordinate width, state width, output width, coordinate injection | field error and coordinate derivatives |
| `diffusion_equilibrium` | stacked selected denoising states | denoiser, cumulative noise schedule, selected timesteps | trajectory residual and final output |
| `raft_deq_flow` | coupled hidden representation and flow | encoder, correlation, update, solver, upsampling settings | endpoint error, flow residual, correction loss |

For example, an implicit coordinate representation solves

$$
z^\star(q)=F_\theta(z^\star(q),\gamma(q)),
\qquad
\widehat u(q)=R_\omega(z^\star(q)),
$$

where $q$ contains coordinates and $\gamma$ is a sine, Fourier, or Gabor
injection. This differs from a grid operator: the query coordinates are inputs,
and spatial derivatives are obtained by differentiating with respect to those
coordinates.

## Scientific Operator Families

Three canonical families expose ODE/PDE and function-space constructions:

```python
from silva_networks import SolverConfig, silva_equilibrium_model

operator = silva_equilibrium_model(
    "scientific_operator",
    in_channels=2,
    state_channels=8,
    out_channels=1,
    architecture="unet",
    architecture_kwargs={"base_channels": 12},
    config=SolverConfig(max_iter=16, alpha=0.35),
)

fno = silva_equilibrium_model(
    "fourier_operator_equilibrium",
    in_channels=2,
    state_channels=8,
    out_channels=1,
    modes_height=4,
    modes_width=4,
)
```

Both models implement a sampled function map

$$
(a,q)\mapsto \widehat u,
$$

but their recurrent fields differ. The generic model may use U-Net,
convolutional, or another shape-preserving spatial module. The Fourier family
uses retained spectral modes plus a local projection. FNO
[[31]](../paper/references.md#ref-31){ .silva-cite } and neural-operator theory
[[32]](../paper/references.md#ref-32){ .silva-cite } motivate the function-space
architecture; SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } supplies
the structured equilibrium composition.

The implicit-time-step family instead requires a right-hand side and step size:

$$
u^{n+1}=u^n+\Delta t\,R(u^{n+1},c).
$$

```python
step = silva_equilibrium_model(
    "implicit_time_step",
    rhs=physical_or_learned_rhs,
    step_size=0.005,
    config=SolverConfig(max_iter=40, tol=1e-6, alpha=0.8),
)
next_state = step(previous_state, context=forcing)
```

The right-hand side must accept `(state, context)` and return the state shape.
Use a projector when every solver evaluation must obey a boundary or state
constraint. The [scientific tutorial](neural-operators-ode-pde.md) derives the
reaction-diffusion, Burgers, Poisson, Fourier, and graph cases in full.

## Recent SILVA Families

These four families remain inside the same SILVA grammar while changing the
internal operator or numerical path:

```python
steady_operator = silva_equilibrium_model(
    "silva_fno_deq",
    in_channels=1,
    state_channels=8,
    out_channels=1,
)

particle_model = silva_equilibrium_model(
    "silva_distributional_deq",
    input_dim=3,
    latent_dim=16,
    particles=10,
)
```

| Family | Internal SILVA mechanism | Main diagnostic |
| --- | --- | --- |
| `silva_fno_deq` | lifted source injected into every tied Fourier layer | fixed-point residual plus PDE/task residual |
| `silva_physics_graph_deq` | reaction, graph diffusion, directed-gradient branches | graph residual plus physical/task error |
| `silva_homotopy_equilibrium` | $\dot z=T(z;x)-z$ from one shared initial state | terminal fixed-point residual and velocity history |
| `silva_distributional_deq` | EI transition and Wasserstein particle descent | MMD or energy discrepancy history |

The full mechanism derivations are in
[Recent Equilibrium Families Inside SILVA](frontier-equilibrium-families.md).
For equation-checked datasets, complete training loops, solver diagnostics, and
four focused notebooks, continue with
[Dataset-Backed Equilibrium Labs](frontier-dataset-labs.md).

## Advanced Equilibrium and Physics Families

Five additional canonical constructors cover monotone graph operators,
one-time-injected transformers, Poisson mirror geometry, physics-informed ODE
equilibria, and implicit DAE stages:

```python
monotone_graph = silva_equilibrium_model(
    "silva_monotone_graph_equilibrium",
    in_dim=3,
    state_dim=16,
    out_dim=2,
)

physics_ode = silva_equilibrium_model(
    "silva_physics_informed_equilibrium",
    state_dim=8,
    output_dim=2,
)
```

| Family | Defining mechanism | Main diagnostic |
| --- | --- | --- |
| `silva_monotone_graph_equilibrium` | constrained channel matrix and forward-backward graph step | certificate, equivariance, residual |
| `silva_generative_equilibrium_transformer` | one-time QKV source injection into tied token blocks | residual and teacher metric |
| `silva_poisson_mirror_equilibrium` | Burg mirror update in the positive orthant | positivity, KL, residual |
| `silva_physics_informed_equilibrium` | latent fixed point and implicit time derivative | boundary, ODE, Jacobian terms |
| `silva_implicit_dae_step` | implicit Runge-Kutta stage root | stage and endpoint constraint residuals |

The adversarial differential-equation residual objective is not a family. It is
combined with a chosen physical model through
`silva_adversarial_residual_loss`. See [Advanced Equilibrium Families](advanced-equilibrium-families.md)
and [Physics-Informed Equilibria](physics-informed-equilibria.md).

## Emerging and Structured Families

The next fourteen constructors preserve the same public selection surface while
placing the equilibrium in a more specialized state space. Their compact
defaults are mechanism checks; each linked family dossier records the data,
architecture, solver, metric, and scale controls needed for a source-level run.

| Family | What is implicit | Replaceable task architecture | Required evidence |
| --- | --- | --- | --- |
| `silva_consistency_deq` | a consistency map from solver time to the terminal root | teacher transition, trajectory parameterization, consistency refiner, readout | endpoint error, residual, one/few-step latency |
| `silva_psi_gnn` | boundary-aware Poisson graph messages | typed graph processor, boundary encoder, decoder | PDE residual, boundary error, convergence certificate |
| `silva_ifno` | tied Fourier material increment | spectral block, local path, displacement/damage head | field error, physical residual, resolution transfer |
| `silva_snarf` | canonical correspondence under forward skinning | deformation field, skinning weights, occupancy network | root success, correspondence error, mesh metric |
| `silva_mesh_inference` | distributed information state | topology, observation policy, local update | centralized agreement, identifiability, message count |
| `silva_physics_guided_diffusion_pde` | guided reverse field trajectory | denoiser, PDE energy, smoother, boundary projector | PDE residual, boundary error, task field error |
| `silva_therino` | thermodynamic physical solution | constitutive operator, Fourier block, loading and projection | stress/energy error, homogenization metric, residual |
| `silva_fixed_point_diffusion` | denoiser state at each diffusion time | latent encoder, timestep block, decoder | generation metric, evaluations, reuse, residual |
| `silva_monotone_operator_equilibrium` | monotone inclusion | structured operator, source, proximal map, readout | monotonicity certificate, residual, task metric |
| `silva_positive_concave_equilibrium` | positive concave state | dense or convolutional positive map, pooling, head | positivity, contraction certificate, residual |
| `silva_non_euclidean_equilibrium` | state in a declared geometry | metric, retraction, tangent update, readout | feasibility, geometric residual, perturbation response |
| `silva_efficient_infinite_graph` | normalized graph spectral system | feature encoder, Gram map, sparse or dense solve, head | denominator margin, graph metric, runtime |
| `silva_multiscale_graph_implicit` | one state per graph power | per-scale operator, source, attention or mean fusion | per-scale residuals, fusion statistics, task metric |
| `silva_delta_equilibrium` | cached incremental update state | eligible linear modules, threshold policy, base transition | activity, full-map disagreement, exact residual, speed |

The derivations, compact simulations, and complete-data routes are in
[Emerging Equilibrium Methods](emerging-equilibrium-methods.md),
[Structured Equilibrium Families](structured-equilibrium-families.md), and
notebooks 28 through 47 in the [Notebook Library](../notebooks.md).

## Learned Solver and Circuit Families

These final two constructors change the numerical strategy or execution
substrate without narrowing the SILVA transition grammar.

```python
learned_solver = silva_equilibrium_model(
    "silva_hyper_deq",
    state_shape=64,
    condition_dim=16,
    learned_steps=6,
    history=5,
)

quantum_equilibrium = silva_equilibrium_model(
    "silva_quantum_deq",
    input_dim=16,
    output_dim=4,
    n_qubits=4,
)
```

| Family | Defining mechanism | Independent controls | Main diagnostic |
| --- | --- | --- | --- |
| `silva_hyper_deq` | input-conditioned initialization and learned Anderson coefficients [[87]](../paper/references.md#ref-87){ .silva-cite } | transition, initializer, residual and condition compressors, controller, readout, teacher solver | teacher distance, residual trajectory, coefficients, latency |
| `silva_quantum_deq` | encoded features, parameterized circuit, measurement, and equilibrium solving [[90]](../paper/references.md#ref-90){ .silva-cite } | input adapter, circuit backend, gate depth, measurement width, readout, solver, backward mode | measurement range, residual, circuit gradient, task metric |

The forward transition of either family can use exact implicit differentiation,
JFB, SHINE, phantom gradients, or finite unrolling when the corresponding
mathematical assumptions and resource tradeoffs are recorded. See
[Learned Solvers and Backward Approximations](solver-learning-and-gradients.md),
[Quantum Equilibria](quantum-equilibria.md), and the
[Equilibrium Expansion Atlas](equilibrium-expansion-atlas.md).

## Optimization Families

The unconstrained quadratic bridge solves

$$
Az^\star=b_\theta(x)
$$

through a fixed-point gradient step. The projected-QP family solves

$$
z^\star
=
\Pi_C[z^\star-\eta(Az^\star-b_\theta(x))]
$$

for package-native constraints such as boxes, simplexes, and affine equalities.
For a full CVXPYlayers-style disciplined convex optimization layer, install
`silva-networks[optimization]` and use `silva_cvxpy_layer`.

Compatibility aliases such as `"optical_flow_deq"` and
`"constrained_quadratic_optimization"` remain accepted by
`silva_equilibrium_model`, but the SILVA-style names above are preferred in new
examples and notebooks.

## Boundary

The package provides SILVA-native implementations and wrappers for the method
families below:

| Source | Package implementation |
| --- | --- |
| DEQ | `compact_deq`, `SILVADEQEngine`, `silva_deq` |
| MDEQ | `mdeq`, `SILVAMultiscaleDEQBlock` |
| TorchDEQ | `SILVADEQEngine`, variational dropout, multi-state packing |
| FNO / neural operators | `scientific_operator`, `fourier_operator_equilibrium`, `SILVAOperatorModel` |
| FNO-DEQ | `silva_fno_deq`, `SILVAFNODEQ`, `SILVAFNODEQBlock` |
| physics-guided graph DEQ | `silva_physics_graph_deq`, `SILVAGraphConvectionDiffusion` |
| homotopy equilibrium | `silva_homotopy_equilibrium`, `SILVAHomotopyEquilibrium` |
| distributional equilibrium | `silva_distributional_deq`, `SILVADistributionalDEQ` |
| ODE / PDE implicit stepping | `implicit_time_step`, numerical derivative and residual helpers |
| RAFT / DEQ-Flow | `silva_deq_flow`, correlation, warping, fixed-point flow |
| OptNet / CVXPYlayers | `silva_projected_qp` plus optional `silva_cvxpy_layer` bridge |
| monotone operator networks | `silva_monotone_operator_equilibrium` with forward-backward or Peaceman-Rachford splitting |
| C-DEQ | `silva_consistency_deq` with replaceable teacher path and refiner |
| HyperDEQ | `silva_hyper_deq` with learned initializer, compressors, and Anderson controller |
| JFB | `SolverConfig(backward_mode="jfb")` for any compatible equilibrium transition |
| SHINE | `SolverConfig(backward_mode="shine")` with a Broyden forward inverse estimate |
| QDEQ | `silva_quantum_deq`, compact statevector backend, and external measured-circuit adapter |
| PIDEQ | `silva_physics_informed_equilibrium`, implicit derivatives, and physics loss terms |
| diffusion equilibria | joint trajectory, generative token, per-timestep root, and physics-guided reverse-process families |

The selector keeps one explicit construction surface while preserving the
state, transition, and solver controls needed for reduced baselines and full
SILVA architectures.

## Validate the Choice

After constructing a family, record the state layout, transition equation,
forward residual, convergence flag, gradient mode, backward residual when
implicit differentiation is used, and task metric. A shared family name does
not imply shared tensor shapes or identical solver conditioning.

The complete method map and primary sources are in
[Paper Families as SILVA Configurations](paper-family-adaptations.md) and
[Paper and References](../paper/references.md). Runnable family instances are
provided in [Paper Family Cases](../examples/paper-family-cases.md).

## Where to Go Next

| Question | Page |
| --- | --- |
| Where can I compare every supported case? | [Case Atlas](case-atlas.md) |
| Which selector objects are public? | [Family Selection API](../api/families.md) |
| Where are several selected families executed together? | [Paper Family Cases](../examples/paper-family-cases.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
