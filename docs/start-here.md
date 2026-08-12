# Start Here

This path goes from first principles to a usable PyTorch model.

## First Hour

1. Open [Run Everything](run-everything.md) and verify the install.
2. Read [Derivation Workbook](learn/derivation-workbook.md) through the scalar
   and graph-local sections.
3. Run [Equation-to-Code Walkthrough](package-notebooks/08_equation_to_code_walkthrough.ipynb).
4. Open [Selecting Model Families](learn/selecting-model-families.md) to choose
   a SILVA, DEQ, MDEQ, flow, or optimization family from one factory.
5. Open [Method Adaptation Atlas](learn/method-adaptation-atlas.md) to map
   external methods to SILVA APIs and citations.
6. Open [Paper Family Adaptations](learn/paper-family-adaptations.md) for the
   DEQ, MDEQ, Jacobian, TorchDEQ, IGNN, INR, diffusion, RAFT, and DEQ-Flow map.
7. Read [Neural Operators, ODEs, PDEs, and SILVA](learn/neural-operators-ode-pde.md)
   when the state is a dynamical or spatial field.
8. Open [Introduction by Example](get-started/introduction-by-example.md).
9. Read [Data Objects and Batching](get-started/data-and-batching.md).
10. Run `python examples/scalar_deq.py` and `python examples/graph_silva.py`.
11. Open [Full-Scale SILVA](learn/full-scale-silva.md) when the compact path is
    working and you are ready to choose sharding, precision, accumulation,
    matrix-free operators, or distributed execution.
12. Use [Real-Dataset Reproduction](learn/real-dataset-reproduction.md) to move
    from known-solution checks to attributed source subsets and complete local
    dataset protocols.
13. Open the [Family Reproduction Dossiers](families/index.md) to inspect the
    governing equation, replaceable modules, compact evidence, source data,
    acceptance checks, and editable scale plan for each of the 50 families.
14. Use [Learned Solvers and Backward Approximations](learn/solver-learning-and-gradients.md)
    to separate learned forward acceleration from exact implicit, JFB, and
    shared-inverse gradient paths.
15. Use [Quantum Equilibria](learn/quantum-equilibria.md) to construct a
    measured circuit transition, replace its backend, and move from compact
    statevectors to a source-scale experiment.

The main equation is

$$
z^\star=f_\theta(z^\star,x).
$$

The main computational check is

$$
\|f_\theta(z^\star,x)-z^\star\|_2.
$$

## Core Learning Path

| Stage | Page | Skill |
| --- | --- | --- |
| Fixed-point intuition | [Fixed Points](learn/fixed-points.md) | understand residuals and damping |
| Guided derivation | [Derivation Workbook](learn/derivation-workbook.md) | derive scalar, vector, graph, global, data, and diagnostic equations |
| Run path | [Run Everything](run-everything.md) | execute examples, notebooks, tests, docs, and data adapters |
| Mathematical review | [Mathematical Foundations](learn/mathematical-foundations.md) | vectors, matrices, norms, derivatives, graph notation |
| SILVA construction | [SILVA From Scratch](learn/silva-from-scratch.md) | build \(S,L,G,H\) terms and solve the layer |
| Family selection | [Selecting Model Families](learn/selecting-model-families.md) | choose SILVA, DEQ, MDEQ, flow, and optimization modules from one selector |
| Method adaptation | [Method Adaptation Atlas](learn/method-adaptation-atlas.md) | translate external implicit-layer, DEQ, ODE, optimization, and flow sources into SILVA equations |
| SILVA operators | [SILVA Operators](learn/silva-operators.md) | vary Figure 1 branches and ablations |
| Cortex hierarchy | [Cortex Hierarchies](learn/cortex-hierarchy.md) | build linked SILVA points with independently configured MLP, convolutional, U-Net, attention, or graph internals |
| Internal architecture selection | [Point Architecture Catalog](learn/point-architecture-catalog.md) | choose among ten vector, token, and spatial fields and compose them inside or across points |
| Scientific models | [Neural Operators, ODEs, PDEs, and SILVA](learn/neural-operators-ode-pde.md) | derive explicit flow, implicit time stepping, PDE residuals, FNO fields, and graph discretizations |
| Learned forward and backward paths | [Learned Solvers and Backward Approximations](learn/solver-learning-and-gradients.md) | train HyperDEQ controls and compare exact implicit, JFB, and SHINE gradients |
| Quantum equilibrium models | [Quantum Equilibria](learn/quantum-equilibria.md) | derive encodings, gates, measurements, fixed points, gradients, and circuit replacement |
| Cross-family mechanism map | [Equilibrium Expansion Atlas](learn/equilibrium-expansion-atlas.md) | distinguish transition, solver, gradient, objective, and evaluation axes |
| Scale and reproduce | [Full-Scale SILVA](learn/full-scale-silva.md) | move all 64 families from equation checks to benchmark-ready execution |
| Source-data reproduction | [Real-Dataset Reproduction](learn/real-dataset-reproduction.md) | verify source receipts and move from compact real subsets to official full splits |
| Family experiment design | [Family Reproduction Dossiers](families/index.md) | follow six explicit stages from tensor contracts to a complete cited protocol or declared extension |
| New family construction | [Advanced Extension Handbook](learn/advanced-extension-handbook.md) | validate primitive modules, public composition equivalence, gradients, serialization, data, and scale registration |
| Failure analysis | [Failure Diagnostics and Recovery](learn/failure-diagnostics-and-recovery.md) | distinguish slow, oscillatory, expansive, stalled, constrained, and backward-solve failures |
| Stability | [Jacobians and Stability](learn/jacobians.md) | compute Jacobians, products, spectral-radius diagnostics |
| Dataset adaptation | [Datasets and Preprocessing](learn/datasets-and-preprocessing.md) | convert public or private data into the engine |

## Package Path

| Task | Page |
| --- | --- |
| Install package and extras | [Installation](installation.md) |
| Pick a solver | [Solvers API](api/solvers.md) |
| Train a learned solver | [Learned Solver API](api/solver_learning.md) |
| Build a quantum equilibrium | [Quantum Equilibria API](api/quantum_equilibria.md) |
| Pick a layer | [Layers API](api/layers.md) |
| Build cortex hierarchies | [Architectures API](api/architectures.md) |
| Choose an internal point architecture | [Point Architectures API](api/point_architectures.md) |
| Build ODE, PDE, or learned operator models | [Scientific Operators API](api/scientific.md) |
| Pick a model family | [Selecting Model Families](learn/selecting-model-families.md) |
| Use reference presets | [SILVA Presets API](api/presets.md) |
| Use the DEQ engine | [DEQ Engine API](api/deq-engine.md) |
| Use optical-flow utilities | [Optical Flow API](api/flow.md) |
| Use constrained optimization layers | [Optimization API](api/optimization.md) |
| Adapt datasets | [Datasets API](api/datasets.md) |
| Load attributed source data | [Source Data API](api/source_data.md) |
| Check GPU behavior | [Devices API](api/devices.md) |
| Configure full-scale execution | [Scaling API](api/scaling.md) |
| Inspect family experiment contracts | [Research Depth API](api/research_depth.md) |
| Run common-task family comparisons | [Compact Benchmarks API](api/compact_benchmarks.md) |

## Long-Form Study

The [book and solutions manual](book.md)
<span class="silva-coming-soon" title="The companion book and solutions manual are planned public learning assets.">Planned</span>
will carry the long derivations and solved exercises. The
[notebooks](notebooks.md), [Derivation Workbook](learn/derivation-workbook.md),
and [Solver Derivation Lab](learn/solver-derivation-lab.md) connect the
derivations to executable code today.

The loop used throughout the suite is:

$$
\text{derive}
\quad\to\quad
\text{implement}
\quad\to\quad
\text{solve}
\quad\to\quad
\text{diagnose}
\quad\to\quad
\text{extend}.
$$

The attributed source-data route is developed in
[Real-Dataset Reproduction](learn/real-dataset-reproduction.md).

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I install the package and optional components? | [Installation](installation.md) |
| Can I begin with one small executable example? | [Introduction by Example](get-started/introduction-by-example.md) |
| Which SILVA case matches my problem? | [Case Atlas](learn/case-atlas.md) |
| How can I validate the complete repository? | [Run Everything](run-everything.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
