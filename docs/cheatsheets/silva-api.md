# SILVA API Cheatsheet

This page is a compact map of the main objects. The detailed derivations live in
the learning pages; the generated signatures live in the API reference. For the
full case-by-case equation map, see [Case Atlas](../learn/case-atlas.md).

## Core Imports

```python
import torch
from silva_networks import (
    SolverConfig,
    SILVADEQFlow,
    SILVALayer,
    SILVAGraphNetwork,
    SILVAGraphPresetNetwork,
    SILVAProjectedQPLayer,
    available_silva_families,
    silva_projected_qp_layer,
    silva_deq_reduction_layer,
    silva_equilibrium_model,
    silva_generalized_layer,
    silva_message_passing_reduction_layer,
    tabular_to_silva_graph,
    stability_report,
)
```

## Solvers

| Object | Use |
| --- | --- |
| `SolverConfig` | solver name and parameters |
| `fixed_point` | dispatch to Picard, Anderson, or Broyden |
| `picard` | damped residual iteration |
| `anderson` | accelerated fixed-point iteration |
| `broyden` | compact quasi-Newton solve |
| `gmres` | matrix-free linear solve |
| `implicit_adjoint_solve` | DEQ-style adjoint diagnostic |

Common parameters:

| Parameter | Meaning |
| --- | --- |
| `solver` | `"picard"`, `"anderson"`, or `"broyden"` |
| `max_iter` | maximum iterations |
| `tol` | residual tolerance |
| `alpha` | damping |
| `history` | Anderson memory |
| `ridge` | Anderson regularization |
| `beta` | Anderson mixing |

## Layers

| Object | Domain |
| --- | --- |
| `SILVALayer` | generic entity/set equilibrium |
| `SILVAGraphLayer` | graph node states |
| `SILVAImageLayer` | image feature maps |
| `DEQLayer` | arbitrary transition wrapped as a fixed-point layer |

Reduction factories:

| Object | Active terms |
| --- | --- |
| `silva_generalized_layer` | chosen stimulus/self/local/global branches |
| `silva_deq_reduction_layer` | stimulus plus linear self, no local/global |
| `silva_message_passing_reduction_layer` | stimulus plus local graph operator |

Built-in local terms:

| Name | Class |
| --- | --- |
| `"graph"` | `GraphLocal` |
| `"gat"`, `"graph_attention"` | `GraphAttentionLocal` |
| `"topk"` | `TopKLocal` |
| `"channel_knn"`, `"vision_knn"` | `DynamicChannelLocal` |
| `"none"` | `ZeroTerm` |

Built-in global terms:

| Name | Class |
| --- | --- |
| `"mean"` | `MeanFieldGlobal` |
| `"simple"`, `"gated_mean"` | `GatedMeanFieldGlobal` |
| `"static"` | `StaticMeanFieldGlobal` |
| `"topk"`, `"topk_attention"` | `TopKGlobalAttention` |
| `"channel_attention"` | `ChannelSelfAttentionGlobal` |
| `"multi_head_channel_attention"` | `MultiHeadChannelAttentionGlobal` |
| `"static_channel"` | `StaticChannelGlobal` |
| `"none"` | `ZeroTerm` |

Optional self terms:

| Name | Meaning |
| --- | --- |
| `None` or `"none"` | no learned self branch; solver damping supplies self-persistence |
| `"linear"` | learned state-wise self map |
| `"identity"` | add the recurrent signal directly |

## Reference Presets

| Object | Use |
| --- | --- |
| `SILVAGraphPresetLayer` | graph/node equation with `LayerNorm(ReLU(...))` |
| `SILVAGraphPresetNetwork` | stacked graph/node model with SILVA-style modes |
| `SILVAVisionVectorLayer` | hidden-channel vector equilibrium |
| `SILVAVisionVectorClassifier` | flattened/vector image classifier |
| `SILVAConvVisionClassifier` | convolutional stem plus vector SILVA stack |
| `SILVACortexLayer` | one SILVA point with an arbitrary internal PyTorch architecture |
| `SILVACortexNetwork` | linked SILVA points with independent architectures and solvers |
| `silva_point_architecture` | factory for ten shape-preserving internal point architectures |
| `available_silva_point_architectures` | stable point-architecture registry |
| `SILVAMolecularRegressor` | atom/bond graph SILVA regressor |

## Case Picker

| Need | Start with |
| --- | --- |
| one callable fixed point | `DEQLayer` or `fixed_point` |
| one graph node layer | `SILVAGraphLayer` |
| stacked node or graph model | `SILVAGraphNetwork` |
| graph modes and alphas | `SILVAGraphPresetNetwork` |
| flattened image vectors | `SILVAVisionVectorClassifier` |
| image tensor plus conv stem | `SILVAConvVisionClassifier` |
| deep architecture inside one point | `SILVACortexLayer(state_network=...)` |
| heterogeneous linked SILVA points | `SILVACortexNetwork` or family `silva_cortex_network` |
| built-in vector, token, or spatial field | `silva_point_architecture(name, **kwargs)` |
| molecule regression | `SILVAMolecularRegressor` |
| custom local/global physics | `SILVALayer` with custom modules |
| hand-checking math | `np_picard`, `np_exact_tanh_affine_jacobian`, `np_implicit_gradient` |
| TorchDEQ-style state engine | `SILVADEQEngine`, `SILVADEQConfig`, `silva_deq` |
| SILVA DEQ flow | `SILVADEQFlow`, `silva_deq_flow` |
| RAFT/DEQ-Flow compatibility names | `SILVAOpticalFlowDEQ`, `silva_optical_flow_deq` |
| projected quadratic-program layer | `SILVAProjectedQPLayer`, `silva_projected_qp_layer` |
| constrained quadratic compatibility names | `SILVAConstrainedQuadraticLayer`, `silva_constrained_quadratic_layer` |
| choose by family name | `available_silva_families`, `silva_equilibrium_model` |

## Dataset Adapters

| Function | Converts |
| --- | --- |
| `load_tabular_dataset` | registered public tabular data to NumPy arrays |
| `tabular_to_silva_graph` | table rows to kNN graph tensors |
| `images_to_silva_vectors` | image batches to vector rows |
| `images_to_silva_pixel_graph` | image batches to pixel graph tensors |
| `molecular_to_silva_graph` | atom/bond tensors to graph batch |
| `pyg_data_to_silva_graph` | PyG-like object to graph batch |
| `make_knn_edge_index` | feature matrix to COO kNN edges |
| `validate_graph_tensor_batch` | tensor contract validation |

## Tensor Shapes

| Symbol | Shape |
| --- | --- |
| `x` | `(entities, features)` |
| `edge_index` | `(2, edges)` |
| `edge_attr` | `(edges, edge_features)` or `(edges,)` |
| `batch` | `(entities,)` |
| state `z` | `(entities, hidden_dim)` |
| image state | `(batch, channels, height, width)` |

## Common Patterns

SILVA-style graph model:

```python
model = SILVAGraphPresetNetwork(
    in_dim=features,
    hidden_dim=[64, 48],
    out_dim=classes,
    graph_mode="GAT",
    attention_mode="simple",
    stack_alphas=[0.5, 0.2],
    max_iter=15,
)
```

Generic custom stack:

```python
model = SILVAGraphNetwork(
    in_dim=features,
    hidden_dims=[64, 64, 32],
    out_dim=classes,
    local=["graph", "topk", "gat"],
    global_term=["mean", "simple", "topk_attention"],
    config=[
        SolverConfig(solver="picard", alpha=0.5, max_iter=12),
        SolverConfig(solver="anderson", alpha=0.35, max_iter=12, history=4),
        SolverConfig(solver="broyden", alpha=0.25, max_iter=8),
    ],
)
```

GPU/MPS/CPU:

```python
from silva_networks import move_to_device, resolve_device

device = resolve_device("auto")
model = model.to(device)
batch = batch.to(device)
```

Family selector:

```python
model = silva_equilibrium_model(
    "silva_graph",
    in_dim=features,
    hidden_dims=[64, 64],
    out_dim=classes,
    local=["graph", "gat"],
    global_term=["mean", "topk"],
)

optimizer_layer = silva_equilibrium_model(
    "silva_projected_qp",
    in_dim=features,
    state_dim=8,
    constraint="simplex",
)

cortex = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_point, vector_point],
    links=[spatial_to_vector],
    head=classification_head,
)
```

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are complete signatures and object families listed? | [API Reference](../api/reference.md) |
| How are these calls introduced in a small example? | [Introduction by Example](../get-started/introduction-by-example.md) |
| Where are complete runnable programs? | [Examples](../examples/index.md) |
