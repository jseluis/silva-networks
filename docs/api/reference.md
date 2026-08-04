# API Reference

Pair this API map with [Implementation Derivations](../learn/implementation-derivations.md)
when you need the equations, shape contracts, and solver assumptions behind the
public classes.
Pair it with [Research Citation Audit](../research-citation-audit.md) when you
need to cite the methods behind a class, solver, diagnostic, or preset.
The global registry starts with the SILVA article
[[1]](../paper/references.md#ref-1){ .silva-cite } and archived package
[[2]](../paper/references.md#ref-2){ .silva-cite }.

The API is organized by role. The generated pages show signatures, docstrings,
inputs, outputs, and source links.

| Module | Main contents |
| --- | --- |
| [Public API](public-api.md) | top-level `silva_networks` import contract and exported names |
| [Solvers](solvers.md) | `SolverConfig`, `SolverResult`, `fixed_point`, `solve_equilibrium`, `picard`, `anderson`, `broyden`, `gmres`, implicit adjoints |
| [Jacobians](jacobians.md) | `full_jacobian`, `vjp`, `jvp`, spectral radius, stability reports |
| [Diagnostics](diagnostics.md) | residual curves, damped updates, Lyapunov-style energy traces |
| [Layers](layers.md) | `SILVALayer`, `SILVAGraphLayer`, `silva_generalized_layer`, `silva_deq_reduction_layer`, global/local/self operators |
| [Architectures](architectures.md) | stacks, cortex hierarchies, graph networks, image classifiers, pooling, readout heads |
| [Point Architectures](point_architectures.md) | ten shape-preserving vector, token, convolutional, U-Net, attention, and spectral fields |
| [Scientific Operators](scientific.md) | finite differences, PDE residuals, boundaries, implicit time steps, reaction-diffusion, Burgers, and Fourier equilibrium operators |
| [Recent Equilibrium Families](frontier.md) | input-injected Fourier, physics-guided graph, homotopy-flow, and distributional SILVA equilibria |
| [Recent Equilibrium Datasets](frontier_data.md) | equation-checked fields, transport graphs, homotopy pairs, and variable-size empirical measures |
| [Implicit Bridge](implicit.md) | SILVA-named DEQ transition, fixed-point classifier, Euler flow, quadratic optimization, MDEQ bridge |
| [DEQ Engine](deq-engine.md) | general single-state and multi-state SILVA DEQ engine, variational dropout, state packing |
| [Optical Flow](flow.md) | RAFT-style correlation, warping, DEQ-flow fixed point, synthetic flow data |
| [Generalized Cases](cases.md) | sequence DEQ, multiscale vision DEQ, IGNN, implicit representations, and diffusion equilibria |
| [Family Selection](families.md) | `available_silva_families`, `silva_family_description`, `silva_equilibrium_model`, family aliases |
| [Optimization](optimization.md) | constrained quadratic projections and optional CVXPYlayers bridge |
| [SILVA Presets](presets.md) | SILVA-style graph, vision, convolutional, and molecular presets |
| [Datasets](datasets.md) | download helpers, tensor adapters, `GraphTensorBatch`, validation |
| [Training](training.md) | optional supervised fit/evaluate helpers, seeding, checkpoint/resume |
| [Devices](devices.md) | CPU/CUDA/MPS selection and nested tensor movement |
| [Coverage Registry](coverage.md) | implementation families mapped to tutorials, notebooks, examples, and tests |
| [Educational NumPy](educational.md) | hand-sized fixed-point, Jacobian, power-iteration, and adjoint helpers |

## Case-to-API Map

| Case | Classes/functions |
| --- | --- |
| Scalar DEQ | `fixed_point`, `DEQLayer`, `np_picard` |
| Generic SILVA layer | `SILVALayer`, `silva_generalized_layer`, `make_local_operator`, `make_global_operator`, `make_self_operator` |
| Cortex hierarchy | `SILVACortexLayer`, `SILVACortexNetwork`, `SILVAImageCortexClassifier`, `silva_cortex_layer`, `silva_cortex_network` |
| Internal point architectures | `available_silva_point_architectures`, `silva_point_architecture`, and ten `SILVA...PointArchitecture` modules |
| ODE, PDE, and learned operators | `SILVAImplicitTimeStep`, `SILVAOperatorModel`, `SILVAFourierNeuralOperator`, numerical derivatives, PDE and boundary residuals |
| Recent equilibrium mechanisms | `SILVAFNODEQ`, `SILVAPhysicsGuidedGraphDEQ`, `SILVAHomotopyEquilibrium`, `SILVADistributionalDEQ` |
| SILVA reductions to baseline implicit models | `silva_deq_reduction_layer`, `silva_message_passing_reduction_layer`, `SILVAFixedPointBlock`, `SILVADEQEngine` |
| Graph node or graph prediction | `SILVAGraphLayer`, `SILVAGraphNetwork`, `SILVAGraphPresetNetwork`, `pool_entities` |
| Vision vectors | `SILVAVisionVectorLayer`, `SILVAVisionVectorClassifier`, `DynamicChannelLocal`, `ChannelSelfAttentionGlobal` |
| Convolutional vision | `SILVAConvStem`, `SILVAConvVisionClassifier`, `SILVAImageLayer`, `SILVAImageClassifier` |
| Molecules | `SILVAMolecularLayer`, `SILVAMolecularRegressor`, `molecular_to_silva_graph` |
| Implicit-layer tutorials | `silva_fixed_point_block`, `silva_fixed_point_classifier`, `silva_euler_flow_block`, `silva_quadratic_optimization_layer`, `silva_multiscale_deq_block` |
| General DEQ systems | `SILVADEQEngine`, `SILVADEQConfig`, `silva_deq`, `SILVAVariationalDropout` |
| SILVA DEQ flow | `SILVADEQFlow`, `silva_deq_flow`, `silva_all_pairs_correlation`, `silva_flow_warp`, `silva_endpoint_error` |
| Sequence DEQ | `SILVASequenceDEQ`, `SILVASequenceTransition`, `SILVARelativeSelfAttention` |
| Multiscale vision DEQ | `SILVAMultiscaleDEQ`, `SILVAMultiscaleClassifier`, `SILVAMultiscaleSegmenter` |
| Implicit graph and coordinate fields | `SILVAImplicitGraphNetwork`, `SILVAImplicitNeuralRepresentation`, `SILVACoordinateInjection` |
| Diffusion trajectory equilibrium | `SILVADiffusionEquilibrium` with a user denoiser and schedule |
| Sequence DEQ | `SILVASequenceDEQ`, relative attention, adaptive embedding/projected softmax, custom module hooks |
| Multiscale DEQ | `SILVAMultiscaleDEQ`, learned fusion, weight norm, material classification and segmentation heads |
| Coupled RAFT/DEQ-Flow | `SILVARAFTDEQ`, residual encoders, `SILVACorrelationPyramid`, `SILVARAFTUpdateBlock`, correction loss and cached state |
| Optical-flow compatibility names | `SILVAOpticalFlowDEQ`, `silva_optical_flow_deq` |
| Projected QP | `SILVAProjectedQPLayer`, `silva_projected_qp_layer`, projection helpers |
| Constrained optimization compatibility | `SILVAConstrainedQuadraticLayer`, `silva_constrained_quadratic_layer`, `silva_cvxpy_layer` |
| Family selection | `available_silva_families`, `silva_equilibrium_model`, `silva_family_description`; see [Family Selection](families.md) |
| Dataset conversion | `GraphTensorBatch`, `tabular_to_silva_graph`, `images_to_silva_vectors`, `images_to_silva_pixel_graph`, `pyg_data_to_silva_graph` |
| Custom training objectives | `BatchStep`, `fit_supervised(..., step_fn=...)`, `evaluate(..., step_fn=...)` |
| Optional training loop | `TrainConfig`, `fit_supervised`, `evaluate`, `seed_everything` |
| Diagnostics | `residual_curve`, `stability_report`, `damped_spectral_radius`, `solve_with_energy` |

## Citation Shortcuts

| API family | Citation shortcut |
| --- | --- |
| solvers | Anderson, Broyden, or GMRES depending on `SolverConfig.solver` and adjoint method |
| Jacobian diagnostics | Deep Implicit Layers, Hutchinson, and Jacobian-regularized DEQ when applicable |
| graph attention | GAT and attention literature |
| global pooling/context | Deep Sets plus SILVA for the gated/context field |
| implicit bridge | DEQ, Neural ODEs, OptNet/differentiable optimization, or MDEQ depending on the object |
| DEQ engine | TorchDEQ, DEQ, and SILVA package |
| optical flow | RAFT and DEQ-Flow, plus the optical-flow dataset or benchmark |
| optimization | projected-gradient methods, OptNet, CVXPYlayers depending on the selected layer |
| presets | SILVA paper/package plus the branch-level citations used by the chosen preset |
| point architectures | SILVA plus the primary architecture paper when the selected field derives from a named architecture |
| recent equilibrium families | SILVA plus FNO-DEQ, physics-guided graph DEQ, HomoODE, or DDEQ according to the selected mechanism |

## Common Imports

```python
from silva_networks import (
    SolverConfig,
    SILVACortexLayer,
    SILVACortexNetwork,
    SILVADEQFlow,
    SILVADiffusionEquilibrium,
    SILVAGraphNetwork,
    SILVAGraphPresetNetwork,
    SILVAProjectedQPLayer,
    SILVARAFTDEQ,
    SILVASequenceDEQ,
    SILVAMultiscaleDEQ,
    SILVAImplicitGraphNetwork,
    SILVAImplicitNeuralRepresentation,
    SILVAImageCortexClassifier,
    available_silva_families,
    available_silva_point_architectures,
    silva_deq_reduction_layer,
    silva_projected_qp_layer,
    silva_point_architecture,
    silva_equilibrium_model,
    silva_generalized_layer,
    silva_deq,
    solve_equilibrium,
    silva_deq_flow,
    tabular_to_silva_graph,
    fit_supervised,
    TrainConfig,
    stability_report,
)
```

## Object Families

The package has three layers of abstraction:

| Level | Use when |
| --- | --- |
| `SILVALayer` and operator modules | building a new interaction field directly |
| `SILVACortexLayer` and `SILVACortexNetwork` | putting deep internal modules inside one equilibrium point and linking several points |
| `SILVAStack` and `SILVAGraphNetwork` | stacking equilibrium layers with custom operators |
| reference presets | reproducing or varying the public SILVA configurations |

The tensor adapters are intentionally separate from the model classes. This
keeps the engine stable while allowing new datasets to be preprocessed into the
same `x`, `edge_index`, `edge_attr`, `batch`, `y` structure.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which names form the stable import surface? | [Public API](public-api.md) |
| How are objects organized by scientific case? | [Case Atlas](../learn/case-atlas.md) |
| Where are complete runnable programs? | [Examples](../examples/index.md) |
