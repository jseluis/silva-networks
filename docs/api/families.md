# Family Selection API

`silva_networks.families` is the high-level factory surface for choosing a
SILVA, DEQ, scientific operator, flow, diffusion, or optimization family by name. Use it when a
notebook, experiment config, or teaching example should select the model family
without importing every concrete class directly.

The named choices connect to DEQ
[[4]](../paper/references.md#ref-4){ .silva-cite }, MDEQ
[[5]](../paper/references.md#ref-5){ .silva-cite }, Neural
ODEs [[7]](../paper/references.md#ref-7){ .silva-cite }, differentiable
optimization [[8]](../paper/references.md#ref-8){ .silva-cite }
[[9]](../paper/references.md#ref-9){ .silva-cite }, and optical-flow equilibria
[[22]](../paper/references.md#ref-22){ .silva-cite }
[[23]](../paper/references.md#ref-23){ .silva-cite }. Scientific operator
families connect to FNO [[31]](../paper/references.md#ref-31){ .silva-cite } and
neural operators [[32]](../paper/references.md#ref-32){ .silva-cite }.
Recent SILVA families also connect to Fourier equilibria
[[43]](../paper/references.md#ref-43){ .silva-cite }, physics-guided graph
equilibria [[44]](../paper/references.md#ref-44){ .silva-cite }, homotopy
continuation [[46]](../paper/references.md#ref-46){ .silva-cite }, and
distributional equilibria [[45]](../paper/references.md#ref-45){ .silva-cite }.
The registry also exposes a learned equilibrium solver
[[87]](../paper/references.md#ref-87){ .silva-cite } and a quantum-circuit
equilibrium [[90]](../paper/references.md#ref-90){ .silva-cite }. JFB
[[88]](../paper/references.md#ref-88){ .silva-cite } and SHINE
[[89]](../paper/references.md#ref-89){ .silva-cite } remain solver-level
backward choices that can be paired with compatible families.

The factory normalizes hyphenated names and compatibility aliases before
dispatching to the package-native constructors. It does not choose dataset
splits, optimizer schedules, checkpoint recipes, or paper-specific metric
claims.

Every equilibrium family still defines

$$
z^\star=f_\theta(z^\star,x),
$$

but the state may be a matrix, image field, multiscale tuple, flow pair,
diffusion trajectory, sampled physical field, or constrained optimization
variable. It may also be a continuous-flow endpoint or an empirical measure
represented by variable-size particles.

## Canonical Families

| Family | Constructor target |
| --- | --- |
| `silva_layer` | generalized SILVA layer |
| `silva_graph` | stacked graph SILVA network |
| `silva_graph_preset` | reference graph SILVA preset |
| `silva_cortex` | single cortex-style equilibrium point |
| `silva_cortex_network` | linked SILVA points with independent internal architectures |
| `silva_image_cortex` | convolutional retina plus linked cortex points |
| `compact_deq` | affine-tanh DEQ reduction |
| `message_passing_deq` | message-passing DEQ reduction |
| `mdeq` | compact multiscale DEQ bridge block |
| `multiscale_vision_deq` | full multiresolution MDEQ-style vision core |
| `sequence_deq` | sequence DEQ with relative attention |
| `implicit_graph` | IGNN-style graph equilibrium |
| `implicit_neural_representation` | coordinate-based implicit representation |
| `diffusion_equilibrium` | joint DDIM trajectory equilibrium |
| `scientific_operator` | selectable source-to-field SILVA operator |
| `fourier_operator_equilibrium` | Fourier neural operator inside a SILVA equilibrium |
| `implicit_time_step` | backward-Euler ODE or PDE step |
| `silva_deq_flow` | SILVA-named optical-flow equilibrium |
| `raft_deq_flow` | coupled RAFT/DEQ-Flow architecture |
| `quadratic_optimization` | unconstrained quadratic optimization layer |
| `silva_projected_qp` | projected quadratic-program layer |
| `silva_fno_deq` | input-injected Fourier block inside a SILVA equilibrium |
| `silva_physics_graph_deq` | SILVA graph equilibrium with reaction, diffusion, and directed transport |
| `silva_homotopy_equilibrium` | conditioned SILVA residual flow with a fixed-point stationary state |
| `silva_distributional_deq` | empirical-measure SILVA equilibrium using discrepancy descent |
| `silva_monotone_graph_equilibrium` | monotone forward-backward graph equilibrium |
| `silva_generative_equilibrium_transformer` | one-time-injected token equilibrium |
| `silva_poisson_mirror_equilibrium` | positive Poisson mirror-descent equilibrium |
| `silva_physics_informed_equilibrium` | physics-informed ODE solution equilibrium |
| `silva_implicit_dae_step` | implicit Runge-Kutta DAE root layer |
| `silva_consistency_deq` | trajectory-consistency refiner for few-step equilibrium inference |
| `silva_psi_gnn` | mixed-boundary Poisson graph equilibrium |
| `silva_ifno` | tied implicit Fourier material-response operator |
| `silva_snarf` | differentiable multi-start forward-skinning roots |
| `silva_mesh_inference` | typed distributed mesh relaxation |
| `silva_physics_guided_diffusion_pde` | reverse diffusion with PDE-energy guidance and boundary projection |
| `silva_therino` | thermodynamically informed physical-state equilibrium |
| `silva_fixed_point_diffusion` | timestep-conditioned fixed-point denoiser |
| `silva_monotone_operator_equilibrium` | strongly monotone equilibrium with selectable operator splitting |
| `silva_positive_concave_equilibrium` | positive-concave dense or convolutional equilibrium |
| `silva_non_euclidean_equilibrium` | weighted-infinity well-posed equilibrium |
| `silva_efficient_infinite_graph` | spectral or iterative infinite-depth graph equilibrium |
| `silva_multiscale_graph_implicit` | graph-power equilibria with nodewise scale attention |
| `silva_delta_equilibrium` | thresholded cached equilibrium updates |
| `silva_hyper_deq` | learned initializer and Anderson controller for a replaceable transition |
| `silva_quantum_deq` | measured quantum-circuit equilibrium with direct and implicit routes |

## Minimal Use

```python
from silva_networks import SolverConfig, silva_equilibrium_model

model = silva_equilibrium_model(
    "silva_graph_preset",
    in_dim=16,
    hidden_dim=32,
    out_dim=3,
    num_layers=2,
    task="graph",
    solver_configs=SolverConfig(solver="anderson", max_iter=20),
)
```

Use `return_result=True` when the selected family supports structured results,
then inspect the state shape, solver residual, convergence flag, and gradient
mode before comparing task metrics. Constructor signatures remain family
specific; `silva_family_description(name)` summarizes the intended state and
use before dispatch.

Full reductions and source links are in
[Selecting Model Families](../learn/selecting-model-families.md).

`canonical_silva_family` resolves aliases without constructing a model.
`build_scaled_silva` then adds scalable numerical defaults while leaving all
task dimensions and modules explicit. See [Full-Scale SILVA](../learn/full-scale-silva.md)
for the all-family data, benchmark, and extension matrix.

For heterogeneous SILVA equilibrium points:

```python
model = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_point, vector_point],
    links=[spatial_to_vector],
    head=classification_head,
)
```

## API

::: silva_networks.families

## Where to Go Next

| Question | Page |
| --- | --- |
| How should I choose among these families? | [Selecting Model Families](../learn/selecting-model-families.md) |
| Which classes implement ODE, PDE, and learned operators? | [Scientific Operators API](scientific.md) |
| Which classes implement the recent operator, graph, flow, and measure families? | [Recent Equilibrium API](frontier.md) |
| Which classes implement monotone, transformer, mirror, physics, and DAE mechanisms? | [Advanced Equilibria API](advanced_equilibria.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
