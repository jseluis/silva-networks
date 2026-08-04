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
7. Open [Introduction by Example](get-started/introduction-by-example.md).
8. Read [Data Objects and Batching](get-started/data-and-batching.md).
9. Run `python examples/scalar_deq.py`.
10. Run `python examples/graph_silva.py`.

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
| Stability | [Jacobians and Stability](learn/jacobians.md) | compute Jacobians, products, spectral-radius diagnostics |
| Dataset adaptation | [Datasets and Preprocessing](learn/datasets-and-preprocessing.md) | convert public or private data into the engine |

## Package Path

| Task | Page |
| --- | --- |
| Install package and extras | [Installation](installation.md) |
| Pick a solver | [Solvers API](api/solvers.md) |
| Pick a layer | [Layers API](api/layers.md) |
| Build cortex hierarchies | [Architectures API](api/architectures.md) |
| Choose an internal point architecture | [Point Architectures API](api/point_architectures.md) |
| Pick a model family | [Selecting Model Families](learn/selecting-model-families.md) |
| Use reference presets | [SILVA Presets API](api/presets.md) |
| Use the DEQ engine | [DEQ Engine API](api/deq-engine.md) |
| Use optical-flow utilities | [Optical Flow API](api/flow.md) |
| Use constrained optimization layers | [Optimization API](api/optimization.md) |
| Adapt datasets | [Datasets API](api/datasets.md) |
| Check GPU behavior | [Devices API](api/devices.md) |

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

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I install the package and optional components? | [Installation](installation.md) |
| Can I begin with one small executable example? | [Introduction by Example](get-started/introduction-by-example.md) |
| Which SILVA case matches my problem? | [Case Atlas](learn/case-atlas.md) |
| How can I validate the complete repository? | [Run Everything](run-everything.md) |
