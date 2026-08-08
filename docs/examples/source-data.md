# Source-Data Family Example

The complete runnable program is
[`examples/source_data_families.py`](https://github.com/jseluis/silva-networks/blob/main/examples/source_data_families.py).
It verifies one source-data path for each of six structured equilibrium
families without presenting the compact measurements as benchmark results.

Every example supplies attributed tensors $x_{\mathrm{src}}$ to a SILVA
transition and records the converged state and receipt together:

$$
z^\star = F_\theta\!\left(z^\star, x_{\mathrm{src}}\right),
\qquad
\mathcal{R} = \left(\operatorname{SHA256}(x_{\mathrm{src}}),
\text{source indices},\text{transforms},\text{split}\right).
$$

Run it from the repository root:

```bash
python examples/source_data_families.py
```

The compact records are also available after package installation:

```python
from silva_networks import load_bundled_source_snapshot

cifar10 = load_bundled_source_snapshot("cifar10")
cora = load_bundled_source_snapshot("cora")
motion = load_bundled_source_snapshot("motion")
```

The program performs:

1. a monotone CIFAR-10 forward/backward solve;
2. a positive-concave spatial solve and nonnegative-weight check;
3. a non-Euclidean perturbation and matrix-measure check;
4. an EIGNN masked Cora loss;
5. an MGNNI three-scale Cora loss and fusion check;
6. a DeltaDEQ convolutional cache check on real consecutive frames.

## Build One Family Directly

```python
import torch
from torch.nn import functional as F

from silva_networks import (
    SILVAEfficientInfiniteGraphEquilibrium,
    SolverConfig,
    load_source_snapshot,
    normalized_graph_operator,
)

sample = load_source_snapshot(
    "docs/assets/source-data/cora-induced-96.pt"
)
x = sample.tensors["x"]
edge_index = sample.tensors["edge_index"]
y = sample.tensors["y"].long()
train_mask = sample.tensors["train_mask"].bool()
graph = normalized_graph_operator(edge_index, x.shape[0]).to(x)

model = SILVAEfficientInfiniteGraphEquilibrium(
    in_dim=x.shape[1],
    state_dim=16,
    out_dim=int(y.max()) + 1,
    gamma=0.7,
    solve_mode="iterative",
    config=SolverConfig(
        solver="picard",
        max_iter=80,
        tol=1e-6,
        backward_mode="unrolled",
    ),
)
result = model(x, graph, return_result=True)
loss = F.cross_entropy(result.output[train_mask], y[train_mask])
loss.backward()
print(result.solver_result.residual)
```

The induced snapshot preserves source node ids and mask types, but changes the
transductive graph. Use `load_planetoid_source_subset(..., subset_nodes=None)`
for source-scale Cora, CiteSeer, or PubMed experiments
[[82]](../paper/references.md#ref-82).

## Run Complete Local Data

```python
from silva_networks import load_planetoid_source_subset

cora = load_planetoid_source_subset(
    "Cora",
    root="data/planetoid",
    subset_nodes=None,
    download=False,
)
graph = normalized_graph_operator(
    cora.graph.edge_index,
    cora.graph.num_entities,
    dense=False,
)
```

Use the official full split and evaluate validation and test masks independently.
For source methods, cite EIGNN
[[78]](../paper/references.md#ref-78), MGNNI
[[79]](../paper/references.md#ref-79), and Planetoid
[[82]](../paper/references.md#ref-82).

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**one latent vector per node or entity**, the condition is **node features, edges, edge attributes, and graph batches**, and the
repeated map is **a source-injected graph message or monotone graph transition**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **losses, certificates, residuals, scale allocation, and cache activity on source data**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/source_data_families.py"
```

### Run the Complete Example

```bash
python examples/source_data_families.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
monotone_loss: 2.2958283
monotone_residual: 8.178701e-07
positive_loss: 2.3394742
positive_minimum_weight: 0.0067153582
non_euclidean_logit_shift: 0.048320621
non_euclidean_measure: 0.050000191
eignn_loss: 1.7337496
eignn_residual: 9.6983911e-07
mgnni_loss: 1.9890027
mgnni_mean_scale_entropy: 1.098611
delta_cache_error: 0.037429918
delta_active_fraction: 0.68346354
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: source-data
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: losses, certificates, residuals, scale allocation, and cache activity on source data
  scale_axes: node count, edge count, feature width, and number of graphs
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **the official complete splits with every source receipt retained**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| What exactly does each receipt record? | [Source Data API](../api/source_data.md) |
| How do I move from a snapshot to a paper-scale run? | [Real-Dataset Reproduction](../learn/real-dataset-reproduction.md) |
| Where are the executed plots and derivations? | [Structured Family Notebooks](../notebooks.md#advanced-equilibrium-and-physics-track) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
