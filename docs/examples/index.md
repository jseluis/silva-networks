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
python examples/frontier_equilibria.py
python examples/advanced_equilibria.py
python examples/optical_flow_silva.py
python examples/constrained_optimization.py
python examples/stacked_architecture.py
python examples/datasets_quickstart.py
python examples/paper_family_cases.py
python examples/raft_deq_flow.py
python examples/source_data_families.py
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
| [Recent Equilibrium Families](frontier-equilibria.md) | Fourier equilibrium, physics graph, homotopy, and empirical-measure SILVA models | four bounded reproductions with solver or discrepancy diagnostics |
| [Advanced Equilibria](advanced-equilibria.md) | monotone graph, injected transformer, Poisson mirror, physics-informed ODE, DAE root, residual objective | six equation-checked mechanisms with distinct numerical and task diagnostics |
| [Optical Flow SILVA](optical-flow-silva.md) | `SILVADEQFlow`, RAFT-style correlation helpers | synthetic flow fixed point, EPE, smoothness, gradients |
| [Constrained Optimization](constrained-optimization.md) | `SILVAProjectedQPLayer`, `silva_projected_qp_layer` | projected-QP fixed point, simplex constraints, gradients |
| [Stacked Architecture](stacked-architecture.md) | `SILVAGraphNetwork`, mixed solvers | multi-layer equilibrium stack on a selected device |
| [Dataset Quickstart](datasets-quickstart.md) | dataset loaders and adapters | public data to `GraphTensorBatch` to model |
| [Citation-Aware Reporting](citation-aware-reporting.md) | presets, solvers, diagnostics, citation audit | methods sentence and citation checklist for a concrete configuration |
| [Paper Family Cases](paper-family-cases.md) | sequence DEQ, MDEQ, IGNN, INR, DEQ-DDIM | compact architecture and gradient smokes across generalized cases |
| [RAFT and DEQ-Flow](raft-deq-flow.md) | `SILVARAFTDEQ`, correction loss, cached state | coupled flow solve, sparse correction predictions, backward pass, and reuse |
| [Source-Data Families](source-data.md) | source receipts, verified snapshots, monDEQ, pcDEQ, NEMON, EIGNN, MGNNI, DeltaDEQ | six real-tensor mechanism and gradient checks with explicit non-benchmark scope |

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

<!-- silva-worked-example:start -->
## Executable Evidence Map

Every worked page now retains its introductory route and adds the complete
program, measured compact output, mathematical result contract, interpretation,
and full-scale transfer record.

| Worked page | Executable program | Compact evidence |
| --- | --- | --- |
| [Advanced Equilibria](advanced-equilibria.md) | `examples/advanced_equilibria.py` | One result for each advanced equilibrium family, with family-specific residuals |
| [Citation Aware Reporting](citation-aware-reporting.md) | `examples/reproduction_registry.py` | A complete machine-readable source and verification record |
| [Constrained Optimization](constrained-optimization.md) | `examples/constrained_optimization.py` | Simplex feasibility, energy, solver residual, and parameter gradients |
| [Cortex Hierarchy](cortex-hierarchy.md) | `examples/cortex_hierarchy.py` | Per-point state shapes, solver choices, logits, loss, and gradients |
| [Custom Layers](custom-layers.md) | `examples/custom_layers.py` | The custom state shape, final residual, and differentiable loss path |
| [Datasets Quickstart](datasets-quickstart.md) | `examples/datasets_quickstart.py` | Dataset identity, tensor shape, and measured classification accuracy |
| [Deq Engine Bridge](deq-engine-bridge.md) | `examples/deq_engine_bridge.py` | State shape, iterations, residual ratio, and gradient availability |
| [Emerging Equilibria](emerging-equilibria.md) | `examples/emerging_equilibria.py` | Family-specific exact-solution, boundary, reconstruction, or trajectory checks |
| [Frontier Equilibria](frontier-equilibria.md) | `examples/frontier_equilibria.py` | Task, equation, invariance, and fixed-point residuals for four operator classes |
| [Full Cortex Operators](full-cortex-operators.md) | `examples/full_cortex_operators.py` | Branch activations, solver history, state shape, loss, and gradients |
| [Full Scale Training](full-scale-training.md) | `examples/add_layers_on_top.py` | A measured training loss from the complete optimization path |
| [Graph Silva](graph-silva.md) | `examples/graph_silva.py` | Node-state shape, task loss, equilibrium residual, and gradients |
| [Molecules](molecules.md) | `examples/molecules.py` | Atom and molecule tensor shapes, residual, prediction loss, and gradients |
| [Optical Flow Silva](optical-flow-silva.md) | `examples/optical_flow_silva.py` | Flow shape, endpoint error, iterations, residual, and gradients |
| [Paper Family Cases](paper-family-cases.md) | `examples/paper_family_cases.py` | Shape and residual checks across sequence, vision, graph, and diffusion cases |
| [Point Architecture Catalog](point-architecture-catalog.md) | `examples/point_architecture_catalog.py` | Parameters, loss, residual trajectory, and gradient norm for every architecture |
| [Raft Deq Flow](raft-deq-flow.md) | `examples/raft_deq_flow.py` | Flow shape, correction trajectory, solver residual, loss, and gradients |
| [Reproduction Registry](reproduction-registry.md) | `examples/reproduction_registry.py` | Verification levels, preserved mechanisms, scale tiers, and source obligations |
| [Scalar Deq](scalar-deq.md) | `examples/scalar_deq.py` | Closed-form agreement, final residual, iteration count, and implicit gradient |
| [Scientific Operators](scientific-operators.md) | `examples/scientific_operators.py` | Ode error plus pde, boundary, and equilibrium residuals |
| [Source Data](source-data.md) | `examples/source_data_families.py` | Losses, certificates, residuals, scale allocation, and cache activity on source data |
| [Spatial Cortex](spatial-cortex.md) | `examples/spatial_cortex.py` | Spatial and vector state shapes, per-point solvers, loss, and gradients |
| [Stacked Architecture](stacked-architecture.md) | `examples/stacked_architecture.py` | Logit shape, pointwise solvers, loss, and gradient flow across the stack |
| [Structured Equilibria](structured-equilibria.md) | `examples/structured_equilibria.py` | Certificates, positivity, one-sided bounds, scale weights, and cache activity |
| [Vision Channels](vision-channels.md) | `examples/vision_channels.py` | Image-state shape, iteration count, residual, loss, and gradients |

Across the collection, a reported result is treated as the tuple

$$
\mathcal E =
(\text{task metric},\text{fixed-point residual},\text{iterations},
\text{gradient evidence},\text{domain invariants}).
$$

Keeping these entries separate prevents task quality, solver convergence, and
structural validity from being collapsed into one number.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I choose an example for my data and state layout? | [Case Atlas](../learn/case-atlas.md) |
| Where should a first-time reader begin? | [Introduction by Example](../get-started/introduction-by-example.md) |
| How can I run the examples and validation suite? | [Run Everything](../run-everything.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
