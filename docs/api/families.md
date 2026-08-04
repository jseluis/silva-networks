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
variable.

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
| Where are several families executed? | [Paper Family Cases](../examples/paper-family-cases.md) |
| Which classes implement the generalized cases? | [Generalized Cases API](cases.md) |
| Which classes implement ODE, PDE, and learned operators? | [Scientific Operators API](scientific.md) |
