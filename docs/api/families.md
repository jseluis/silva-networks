# Family Selection API

`silva_networks.families` is the high-level factory surface for choosing a
SILVA, DEQ, flow, diffusion, or optimization family by name. Use it when a
notebook, experiment config, or teaching example should select the model family
without importing every concrete class directly.

The factory normalizes hyphenated names and compatibility aliases before
dispatching to the package-native constructors. It does not choose dataset
splits, optimizer schedules, checkpoint recipes, or paper-specific metric
claims.

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
