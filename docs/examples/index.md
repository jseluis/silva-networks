# Examples

The examples are compact CPU-first validations of the public package API. They are not
benchmark scripts; they are compact checks that show how to assemble a SILVA
equilibrium, train it, inspect residuals, and move the same pattern to a larger
experiment.

## Run All Examples

```bash
python examples/scalar_deq.py
python examples/graph_silva.py
python examples/vision_channels.py
python examples/molecules.py
python examples/custom_layers.py
python examples/deq_engine_bridge.py
python examples/add_layers_on_top.py
python examples/cortex_hierarchy.py
python examples/spatial_cortex.py
python examples/point_architecture_catalog.py
python examples/scientific_operators.py
python examples/optical_flow_silva.py
python examples/constrained_optimization.py
python examples/stacked_architecture.py
python examples/datasets_quickstart.py
python examples/paper_family_cases.py
python examples/raft_deq_flow.py
```

## What Each Example Covers

| Example | Package objects | Main check |
| --- | --- | --- |
| [Scalar DEQ](scalar-deq.md) | `fixed_point`, `full_jacobian`, `stability_report` | numerical solution equals a closed-form fixed point |
| [Graph SILVA](graph-silva.md) | `SILVAGraphLayer`, `SolverConfig` | graph state shape, gradients, stability report |
| [Vision Channels](vision-channels.md) | `SILVAImageLayer` | image tensor state and residual solve |
| [Molecules](molecules.md) | `SILVAGraphLayer` | atoms as entities, bonds as edges, molecule pooling |
| [Custom Layers](custom-layers.md) | `SILVALayer`, custom `nn.Module` branches | replacing interaction branches |
| [DEQ Engine Bridge](deq-engine-bridge.md) | `SILVADEQEngine`, `SILVADEQConfig`, `silva_deq` | solving arbitrary single-state and multi-state systems |
| [Cortex Hierarchy](cortex-hierarchy.md) | `SILVACortexLayer`, `SILVACortexNetwork` | deep internal modules inside linked equilibrium points |
| [Spatial SILVA Cortex](spatial-cortex.md) | `SILVACortexLayer`, `silva_equilibrium_model` | residual CNN and U-Net inside one point linked to a different vector point |
| [Point Architecture Catalog](point-architecture-catalog.md) | point architecture registry and `SILVACortexLayer` | ten shape-preserving vector, token, spatial, gradient, and tiny-data checks |
| [Full Cortex Operators](full-cortex-operators.md) | every `SILVACortexLayer` slot and all 25 branch factory names | internal sequence, self, local, global, custom, output, normalization, solver, shapes, and gradients |
| [Scientific Operators](scientific-operators.md) | ODE flow, implicit PDE steps, reaction-diffusion, Burgers, Fourier operators, graph PDEs | analytic errors, fixed-point residuals, boundaries, gradients, and resolution changes |
| [Optical Flow SILVA](optical-flow-silva.md) | `SILVADEQFlow`, RAFT-style correlation helpers | synthetic flow fixed point, EPE, smoothness, gradients |
| [Constrained Optimization](constrained-optimization.md) | `SILVAProjectedQPLayer`, `silva_projected_qp_layer` | projected-QP fixed point, simplex constraints, gradients |
| [Stacked Architecture](stacked-architecture.md) | `SILVAGraphNetwork`, mixed solvers | multi-layer equilibrium stack on a selected device |
| [Dataset Quickstart](datasets-quickstart.md) | dataset loaders and adapters | public data to `GraphTensorBatch` to model |
| [Citation-Aware Reporting](citation-aware-reporting.md) | presets, solvers, diagnostics, citation audit | methods sentence and citation checklist for a concrete configuration |
| [Paper Family Cases](paper-family-cases.md) | sequence DEQ, MDEQ, IGNN, INR, DEQ-DDIM | compact architecture and gradient smokes across generalized cases |
| [RAFT and DEQ-Flow](raft-deq-flow.md) | `SILVARAFTDEQ`, correction loss, cached state | coupled flow solve, sparse correction predictions, backward pass, and reuse |

## Shared Pattern

Every example follows the same engineering path:

$$
\text{raw object}
\to
(x,E,b,y)
\to
z^\star=f_\theta(z^\star,x)
\to
\hat y=R_\phi(z^\star)
\to
\text{diagnostics}.
$$

The same pattern supports user datasets after preprocessing into the package
tensor contract.

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I choose an example for my data and state layout? | [Case Atlas](../learn/case-atlas.md) |
| Where should a first-time reader begin? | [Introduction by Example](../get-started/introduction-by-example.md) |
| How can I run the examples and validation suite? | [Run Everything](../run-everything.md) |
