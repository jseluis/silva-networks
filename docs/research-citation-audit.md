# Research Citation Audit

This audit maps implemented SILVA Networks package objects to the research
lineage that should be cited when the package is used in papers, reports,
notebooks, or experiment logs.

The SILVA paper and package should be cited for the package-specific structured
interaction field, presets, notebooks, and implementation. Foundational methods
should be cited when a result depends on them.

!!! important "SILVA citation status"
    The SILVA paper citation is:
    Jose Luis Lima de Jesus Silva, *SILVA Networks as Structured Implicit Layers and Vector
    Attractors via Dynamic Interaction Fields*, 2026, arXiv:2607.28989.
    Metadata was verified from arXiv on August 3, 2026. BibTeX is available in
    [Paper and References](paper/references.md) and
    [`docs/assets/bib/silva-networks.bib`](assets/bib/silva-networks.bib).

## Audit Method

The audit was checked against the package source under `src/silva_networks`:

| Source module | Public surface audited |
| --- | --- |
| `solvers.py` | `picard`, `anderson`, `broyden`, `gmres`, `implicit_adjoint_solve` |
| `jacobian.py` | full Jacobian, VJP/JVP, power iteration, Hutchinson estimates, stability reports |
| `layers.py` | stimulus, graph, graph attention, kNN, global attention, channel attention, DEQ wrapper, generic SILVA layer |
| `architectures.py` | stacks, graph/image networks, pooling and readout heads |
| `presets.py` | graph, vector vision, convolutional vision, molecular presets, energy diagnostic |
| `implicit.py` | DEQ/ODE/optimization/MDEQ bridge objects and SILVA-named factories |
| `optimization.py` | projected constrained quadratic layers, projections, and optional CVXPYlayers wrapper |
| `deq_engine.py` | TorchDEQ-style single-state and multi-state package engine, packed state helpers, variational dropout |
| `flow.py` | RAFT/DEQ-Flow-style optical-flow utilities and package-native flow DEQ |
| `datasets.py` | tabular, image, pixel-graph, PyG-like, and molecular adapters |

## Method-to-Citation Matrix

| Package object or family | What it implements | Cite when used |
| --- | --- | --- |
| `fixed_point`, `DEQLayer`, `SILVALayer`, `SILVAStack`, all equilibrium presets | fixed-point layer \(z^\star=f_\theta(z^\star,x)\) and infinite-depth weight-tied view | SILVA paper/package; Bai, Kolter, and Koltun, [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) |
| `implicit_adjoint_solve`, implicit-gradient derivations | adjoint solve \((I-J_f^\top)u=g\) | SILVA paper/package; [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/); Bai et al. DEQ |
| `picard`, damped solver update | damped fixed-point iteration | SILVA paper/package; Banach fixed-point theorem as mathematical background |
| `anderson` | Anderson acceleration for fixed-point residuals | Anderson, [Iterative Procedures for Nonlinear Integral Equations](https://doi.org/10.1145/321296.321305); Walker and Ni, [Anderson Acceleration for Fixed-Point Iterations](https://doi.org/10.1137/10078356X) |
| `broyden` | inverse secant quasi-Newton solve for \(F(z)=0\) | Broyden, [A Class of Methods for Solving Nonlinear Simultaneous Equations](https://doi.org/10.1090/S0025-5718-1965-0198670-6) |
| `gmres` | Krylov minimal-residual linear solve | Saad and Schultz, [GMRES](https://doi.org/10.1137/0907058) |
| `hutchinson_jacobian_norm`, `jacobian_regularization_loss` | stochastic trace/Frobenius estimator for Jacobian penalties | Hutchinson, [A Stochastic Estimator of the Trace](https://doi.org/10.1080/03610918908812806); Bai, Koltun, and Kolter, [Jacobian Regularization](https://arxiv.org/abs/2106.14342) |
| `full_jacobian`, `vjp`, `jvp`, `stability_report` | local linearization and product diagnostics | SILVA paper/package; Deep Implicit Layers tutorial |
| `GraphLocal`, `SILVAGraphLayer`, graph preset mean branch | message passing / graph aggregation over `edge_index` | Kipf and Welling, [GCN](https://arxiv.org/abs/1609.02907); Gilmer et al., [MPNN](https://arxiv.org/abs/1704.01212); Scarselli et al., [GNN model](https://doi.org/10.1109/TNN.2008.2005605) |
| `GraphAttentionLocal`, bond-aware molecular local branch | masked neighbor attention over graph edges | Velickovic et al., [Graph Attention Networks](https://arxiv.org/abs/1710.10903); Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| `MeanFieldGlobal`, `GatedMeanFieldGlobal`, `pool_entities` | permutation-invariant mean/set pooling and broadcast context | Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114); SILVA paper/package for the gated field design |
| `TopKGlobalAttention`, `ChannelSelfAttentionGlobal`, `MultiHeadChannelAttentionGlobal` | scaled dot-product attention variants, restricted to top-k or hidden channels | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762); Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825) for attention on sets |
| `TopKLocal`, `DynamicChannelLocal`, `make_knn_edge_index`, `tabular_to_silva_graph` | dynamic nearest-neighbor local graphs | Wang et al., [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829); SILVA paper/package for the hidden-channel adaptation |
| `SILVAVisionVectorLayer`, `SILVAConvVisionClassifier` | vector/channel SILVA equilibria and convolutional stimulus front-end | SILVA paper/package; Vaswani attention for global channel attention; Wang et al. DGCNN for dynamic kNN analogy |
| `SILVAMolecularLayer`, `SILVAMolecularRegressor`, `molecular_to_silva_graph` | bond-aware graph equilibrium and graph-level molecular readout | Gilmer et al., MPNN; Velickovic et al., GAT; cite molecule benchmark/dataset separately |
| `ExplicitEulerODEBlock`, `SILVAEulerFlowBlock` | finite explicit Euler bridge for neural ODE intuition | Chen et al., [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366); Deep Implicit Layers tutorial |
| `SILVAFourierOperatorPointArchitecture`, `fourier_operator` | low-mode spectral operator with a local channel projection, used as a shape-preserving SILVA transition field | Li et al., [Fourier Neural Operator](https://openreview.net/forum?id=c8P9NQVtmnO); Kovachki et al., [Neural Operator](https://www.jmlr.org/papers/v24/21-1524.html); SILVA paper/package for its placement inside the structured fixed-point transition |
| `QuadraticOptimizationLayer`, `SILVAQuadraticOptimizationLayer` | differentiable quadratic argmin and fixed-point KKT solve | Amos and Kolter, [OptNet](https://arxiv.org/abs/1703.00443); Agrawal et al., [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430); Deep Implicit Layers tutorial |
| `SILVAProjectedQPLayer`, `silva_projected_qp_layer`, `SILVAConstrainedQuadraticLayer`, `silva_cvxpy_layer` | projected constrained quadratic programs and optional CVXPYlayers bridge | OptNet for optimization-layer framing; Agrawal et al. differentiable convex optimization layers; CVXPYlayers when the optional bridge is used |
| `ToyMultiscaleDEQBlock`, `SILVAMultiscaleDEQBlock` | coupled low/high equilibrium state | Bai, Koltun, and Kolter, [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656) |
| `SILVADEQEngine`, `silva_deq`, `pack_state`, `unpack_state`, `SILVAVariationalDropout` | general package-native DEQ engine for one or multiple tensor states | SILVA package; [TorchDEQ](https://github.com/locuslab/torchdeq); DEQ |
| `SILVADEQFlow`, `silva_deq_flow`, `SILVAOpticalFlowDEQ`, `silva_all_pairs_correlation`, `silva_flow_warp`, `silva_local_correlation_lookup` | compact optical-flow fixed-point estimator with RAFT-style correlation and DEQ-Flow framing | Teed and Deng, [RAFT](https://arxiv.org/abs/2003.12039); Bai et al., [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442); [DEQ-Flow repository](https://github.com/locuslab/deq-flow); SILVA package |
| `silva_endpoint_error`, `silva_flow_smoothness_loss` | optical-flow evaluation and regularization utilities | cite the optical-flow benchmark/dataset and RAFT/DEQ-Flow when used with the flow DEQ |
| dataset loaders and adapters | source data, standardization, graph conversion | cite the dataset source, UCI page, torchvision dataset, PyG dataset, or molecular benchmark used in the experiment |

## Claim-to-Citation Guide

Use the narrowest citation set that supports the claim.

| Claim in a report | Minimum citation set |
| --- | --- |
| "SILVA Networks were used." | SILVA paper/package |
| "The model is an equilibrium / infinite-depth layer." | SILVA paper/package and Bai et al. DEQ |
| "Gradients use implicit differentiation / adjoint solve." | Deep Implicit Layers tutorial and Bai et al. DEQ |
| "Anderson acceleration was used." | Anderson 1965 and Walker-Ni 2011 |
| "Broyden acceleration was used." | Broyden 1965 |
| "GMRES was used for the adjoint linear system." | Saad-Schultz 1986 |
| "A Fourier neural operator was used inside a SILVA point." | Li et al. FNO, Kovachki et al. neural operator, and SILVA paper/package |
| "A Jacobian Frobenius penalty was used." | Hutchinson 1989 and Bai-Koltun-Kolter 2021 |
| "The graph local operator is attention-based." | GAT and Attention Is All You Need |
| "The molecular model is message-passing-like." | MPNN and GAT, plus SILVA |
| "The global branch is permutation invariant." | Deep Sets, plus SILVA for gated/top-k branch design |
| "The model uses top-k or channel attention." | Attention Is All You Need; Set Transformer for set-attention framing |
| "The hidden-channel local branch uses dynamic kNN." | Dynamic Graph CNN as related dynamic graph literature, plus SILVA for the hidden-channel adaptation |
| "The optimization layer solves a differentiable argmin." | OptNet or differentiable convex optimization layers |
| "The ODE bridge connects continuous-depth models to equilibria." | Neural ODEs and Deep Implicit Layers tutorial |
| "The engine handles multi-state DEQ systems." | TorchDEQ as related interface lineage; DEQ; SILVA package for this implementation |
| "The optical-flow module uses all-pairs correlation and recurrent refinement." | RAFT; DEQ-Flow if framed as an equilibrium optical-flow solve |

## Package-Specific Contributions

The following are SILVA/package-specific compositions. Cite the SILVA paper and
repository, then cite the relevant building blocks only as background:

| SILVA/package object | Why it is package-specific |
| --- | --- |
| structured \(S+H+L+G\) field | combines stimulus, self, local, and global operators into one equilibrium transition |
| gated mean-field global branch | uses attention-style scoring over a graph/set mean, then broadcasts a gated context |
| hidden-channel dynamic kNN for vector vision | adapts dynamic-neighborhood graph ideas to channels inside a sample |
| quadratic interaction energy | diagnostic proxy for state/field alignment; not a theorem of global Lyapunov stability by itself |
| graph, vision, convolutional, and molecular SILVA presets | package-level reference configurations and tensor contracts |
| package implicit bridge notebooks | tutorials that express DEQ, ODE, optimization, MDEQ, and Jacobian regularization through `silva_networks` APIs |
| method adaptation atlas | source-by-source translation from external implicit-layer, DEQ, ODE, optimization, and optical-flow methods into package equations, APIs, and scope notes |
| package-native DEQ engine | TorchDEQ-style convenience interface implemented through SILVA solvers and state packing |
| package-native optical-flow DEQ | compact RAFT/DEQ-Flow-inspired implementation using SILVA solvers |
| package-native projected QP layer | projected fixed-point QP implementation with selectable nonnegative, box, simplex, and affine constraints |

## Example Citation Checklist

For a graph node model with GAT local branch, gated mean global branch, Anderson
solver, and Jacobian diagnostics, cite:

1. SILVA paper/package.
2. Deep Equilibrium Models.
3. Graph Attention Networks.
4. Attention Is All You Need, if discussing attention scores explicitly.
5. Deep Sets, if discussing permutation-invariant graph/set pooling.
6. Anderson 1965 and Walker-Ni 2011.
7. Hutchinson 1989 and/or Jacobian-regularized DEQs if using Hutchinson
   Jacobian penalties or Frobenius diagnostics.

For a molecular regressor, cite:

1. SILVA paper/package.
2. Deep Equilibrium Models.
3. Neural Message Passing for Quantum Chemistry.
4. Graph Attention Networks.
5. The dataset or benchmark source used for the molecule task.

For an implicit bridge optimization experiment, cite:

1. SILVA package for the tutorial implementation.
2. Deep Implicit Layers tutorial.
3. OptNet or differentiable convex optimization layers.
4. Deep Equilibrium Models if the optimization layer is solved through the
   package fixed-point solver.

For an optical-flow DEQ validation experiment, cite:

1. SILVA package for the package-native implementation.
2. RAFT for all-pairs correlation and recurrent flow refinement lineage.
3. Deep Equilibrium Optical Flow Estimation for the DEQ-flow framing.
4. The optical-flow dataset or benchmark if external data is used.

## Audit Findings

| Finding | Status |
| --- | --- |
| SILVA article metadata is recorded. | Uses [arXiv:2607.28989](https://arxiv.org/abs/2607.28989) with BibTeX. |
| The package already cited several papers in docstrings, but docs did not have a single method-level citation matrix. | Fixed by this audit page. |
| The derivation pages explained equations but did not always name the originating method literature beside the formula. | Cross-links added from foundations and implementation derivations. |
| Some SILVA operators are related to published mechanisms but are not exact reproductions. | Labeled as package-specific compositions above. |
| Flow and DEQ-engine modules were public through `__init__` and needed their own citation/API coverage. | Added RAFT, DEQ-Flow, and TorchDEQ lineage. |
| Dataset adapters cannot be fully cited without knowing the downstream dataset actually used. | Docs instruct users to cite each dataset source separately. |
| External tutorials and repositories needed a clear method lineage. | Added the Method Adaptation Atlas and notebook as SILVA-native translations with upstream citations. |
| The optimization bridge could be misread as one object with one scope. | Scope note added: the implicit tutorial object is an unconstrained quadratic bridge; the optimization module adds projected constrained QP layers and an optional CVXPYlayers wrapper. |
