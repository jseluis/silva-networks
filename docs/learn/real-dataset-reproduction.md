# Real-Dataset Reproduction

SILVA separates three questions that are easy to conflate:

1. **Is the mathematical mechanism correct?** Known-solution generators test
   fixed points, certificates, gradients, and solver residuals.
2. **Does the implementation accept the source data representation?** Compact,
   attributed source snapshots test real images, graph masks, fields, or video
   frames through the public constructors.
3. **Does a complete experiment reproduce a published result?** That requires
   the full dataset, source split, architecture, training schedule, seeds,
   stopping rules, and metrics.

A compact run answers the first two questions. It is not evidence for the third.

## Source Receipts

Every source adapter returns a `SourceDataReceipt`. If the selected tensors are
$T_1,\ldots,T_m$, the recorded content identifier is

$$
h=\operatorname{SHA256}\left(
  \operatorname{encode}(T_1)\Vert\cdots\Vert\operatorname{encode}(T_m)
\right).
$$

The encoding includes each tensor's dtype, shape, and contiguous bytes. The
receipt also records source indices, split, adapter version, preprocessing,
dataset page, citation, and access statement. A reader can therefore distinguish
two runs that used the same dataset name but different examples or transforms.

```python
from silva_networks import load_source_snapshot

sample = load_source_snapshot(
    "docs/assets/source-data/cifar10-balanced-10.pt"
)
print(sample.receipt.as_dict())
images = sample.tensors["images"]
labels = sample.tensors["labels"]
```

`load_source_snapshot` recomputes the tensor hash by default and rejects altered
content.

## Included Compact Sources

All three records are present both under `docs/assets/source-data/` and inside
the installed package. `load_bundled_source_snapshot("cifar10")`,
`load_bundled_source_snapshot("cora")`, and
`load_bundled_source_snapshot("motion")` therefore reproduce the compact
teaching inputs without a repository checkout.

| Snapshot | Contents | Size in the source collection | Purpose |
| --- | --- | ---: | --- |
| CIFAR-10 | one indexed image per class, resized to $16\times16$ | about 170 MB compressed | vision shape, loss, gradient, and certificate checks |
| Cora | connected 96-node induced graph with source node ids and all split-mask kinds | less than 20 MB | graph propagation, masked loss, and scale-fusion checks |
| public motion | consecutive real-video frames 100 and 101 at $96\times160$ | about 3.5 MB | qualitative cache-activity check |

CIFAR-10 is cited at [[81]](../paper/references.md#ref-81), the Planetoid
protocol at [[82]](../paper/references.md#ref-82), and the public motion source
at [[86]](../paper/references.md#ref-86). The generated
[`manifest.json`](../assets/source-data/manifest.json) records archive and tensor
checksums.

Regenerate the snapshots from locally available source data with:

```bash
python scripts/prepare_source_snapshots.py
```

The script does not enable network access unless `--download` is supplied.

## Live Dataset Adapters

The snapshots keep every notebook executable from a repository checkout. The
same public functions open complete local datasets:

```python
from silva_networks import (
    load_optical_flow_source_subset,
    load_planetoid_source_subset,
    load_vision_source_subset,
)

cifar = load_vision_source_subset(
    "CIFAR10",
    root="data",
    samples_per_class=50,
    seed=0,
    normalization="source",
    download=False,
)

cora = load_planetoid_source_subset(
    "Cora",
    root="data/planetoid",
    subset_nodes=None,  # complete transductive graph
    download=False,
)

sintel = load_optical_flow_source_subset(
    "Sintel",
    root="data/flow",
    split="train",
    pass_name="clean",
    index=0,
)
```

`subset_nodes=None` is essential for a Planetoid benchmark. An induced Cora
snapshot changes the graph and is only a teaching task.

## Six Structured Families

### Monotone Operator Equilibrium

For flattened image $x\in\mathbb{R}^{3HW}$, the equilibrium is the monotone
inclusion

$$
0\in(I-W)z^\star-Ux-b+\partial f(z^\star).
$$

The compact CIFAR-10 section in
[notebook 36](../package-notebooks/36_silva_monotone_operator_equilibrium.ipynb)
trains the public constructor and verifies both residual and monotonicity
certificate. To scale, replace the dense `operator` with a convolutional
monotone module implementing `forward`, `resolvent`, and
`monotonicity_certificate`; preserve the source monDEQ splitting and data
schedule [[75]](../paper/references.md#ref-75).

### Positive-Concave Equilibrium

Unit-valued image tensors enter

$$
z^\star=\phi(W_+z^\star+S_\theta(x)),\qquad W_+\geq0.
$$

[Notebook 37](../package-notebooks/37_silva_positive_concave_equilibrium.ipynb)
keeps the state spatial, pools only the class field, and applies
`project_nonnegative_()` after optimizer steps. Full pcDEQ experiments must
preserve the chosen variant, positive parameterization, and source preprocessing
[[76]](../paper/references.md#ref-76).

### Non-Euclidean Equilibrium

NEMON constrains a weighted matrix measure rather than only an Euclidean norm:

$$
\mu_{\infty,D}(W)<1,\qquad
\lVert z^\star(x+\delta)-z^\star(x)\rVert_{\infty,D}
\leq
\frac{\lVert U\delta\rVert_{\infty,D}}
     {1-\mu_{\infty,D}(W)}.
$$

[Notebook 38](../package-notebooks/38_silva_non_euclidean_equilibrium.ipynb)
reports the certificate, bounded perturbation, observed logit displacement, and
latent bound separately. A complete robustness claim also needs the source
attack or certification protocol [[77]](../paper/references.md#ref-77).

### Efficient Infinite Graph

For normalized graph operator $S$ and normalized channel Gram map $g(F)$,

$$
Z^\star=\gamma S Z^\star g(F)^\mathsf{T}+X_\theta.
$$

[Notebook 39](../package-notebooks/39_silva_efficient_infinite_graph.ipynb)
runs the iterative route on real Cora tensors. Full Cora, CiteSeer, or PubMed
experiments use the complete graph, fixed Planetoid masks, and either the
closed-form eigensystem or iterative route declared before evaluation
[[78]](../paper/references.md#ref-78).

### Multiscale Graph Implicit Network

Each scale solves

$$
Z_m^\star=\gamma S^m Z_m^\star g(F_m)^\mathsf{T}+X_\theta,
$$

then nodewise weights fuse the states:

$$
\bar Z_i=\sum_m\alpha_{im}Z_{m,i}^\star,\qquad
\sum_m\alpha_{im}=1.
$$

[Notebook 40](../package-notebooks/40_silva_multiscale_graph_implicit.ipynb)
plots the full node-by-scale allocation. Paper-level work must preserve the
complete graph, scale set, fusion rule, and split protocol
[[79]](../paper/references.md#ref-79).

### Delta-Cached Equilibrium

For a linear or convolutional recurrent block $L$,

$$
c_{k+1}=c_k+L(\Delta_\tau z_{k+1})-L(0),
\qquad
\Delta_\tau z=\mathbf 1(|\Delta z|>\tau)\odot\Delta z.
$$

[Notebook 41](../package-notebooks/41_silva_delta_equilibrium.ipynb)
measures activity and exact-convolution disagreement on real consecutive frames.
It does not report endpoint error because the public clip has no flow labels.
Use MPI Sintel [[83]](../paper/references.md#ref-83), KITTI Flow
[[84]](../paper/references.md#ref-84), or FlyingChairs
[[85]](../paper/references.md#ref-85) for supervised evaluation, and measure
latency on the target hardware instead of inferring it from activity alone.

## Neural-Operator Fields

`load_darcy_source_subset` accepts local `.pt` or `.npz` archives with
`x/y`, `inputs/targets`, `coeff/solution`, or `a/u` arrays:

```python
from silva_networks import load_darcy_source_subset, silva_fno_deq

darcy = load_darcy_source_subset(
    "data/darcy/darcy_small.npz",
    samples=64,
    seed=0,
)
model = silva_fno_deq(
    in_channels=darcy.inputs.shape[1],
    hidden_channels=64,
    out_channels=darcy.targets.shape[1],
    modes=(16, 16),
)
```

The adapter boundary is intentionally independent of one storage package.
Source-scale FNO-DEQ work must preserve the Darcy or Navier-Stokes resolutions,
normalization, relative-$L^2$ metric, noise protocol, and solver budget
[[31]](../paper/references.md#ref-31)
[[43]](../paper/references.md#ref-43).

## Storage and Access

| Dataset | Approximate local allowance | Access action |
| --- | ---: | --- |
| CIFAR-10 | 170 MB compressed | loader-supported |
| Cora / CiteSeer | below 20 MB each | loader-supported |
| PubMed | below 100 MB | loader-supported |
| MPI Sintel complete | about 5.3 GB | accept and retrieve from the benchmark site |
| KITTI Flow archives | about 2 GB | accept and retrieve from the benchmark site |
| FlyingChairs | about 22 GB | retrieve data and split assignment from the source site |
| Darcy multi-resolution cache | reserve at least 1 GB | use the official operator-data loader or a local mirror |

SILVA does not silently retrieve datasets whose providers require a separate
access step.

## Complete Experiment Checklist

Before calling a run a reproduction:

1. Record the paper, repository revision, dataset receipt, and access page.
2. Preserve official train, validation, and test definitions.
3. Match architecture width, depth or equilibrium transition, normalization,
   augmentation, optimizer, schedule, and regularization.
4. Record forward and backward solver configurations separately.
5. Run every declared seed and report dispersion, convergence failures, wall
   time, memory, and parameter count.
6. Use the source metric implementation and evaluation resolution.
7. Compare against a source-aligned baseline under the same data and compute
   budget.

The packaged profiles
`structured_real_subset_suite.json` and
`structured_source_scale_suite.json` make these choices explicit. They are
starting records, not hidden claims that the complete published runs have
already been executed.

## Extending to Another Paper

A new family does not need a new data system. Implement an adapter that returns
tensors plus `SourceDataReceipt`, then expose the family through the same five
boundaries:

1. source encoder $S_\theta(x)$;
2. state-preserving transition $T_\theta(z,x)$;
3. initializer;
4. readout;
5. `SolverConfig`.

First test one transition step against an independent reference function. Then
test the equilibrium, gradients, compact source subset, and complete-data
configuration. This keeps the new method inside SILVA while leaving every
internal operator replaceable.

The six executed structured-family labs and their retained plots are indexed in
the [Notebook Overview](../notebooks.md).

## Where to Go Next

| Question | Page |
| --- | --- |
| Which functions load and verify source data? | [Source Data API](../api/source_data.md) |
| Where is the complete runnable code? | [Source-Data Family Example](../examples/source-data.md) |
| How are all paper adaptations registered? | [Reproducing SILVA and Source Methods](reproducing-silva-and-papers.md) |
| How are large runs configured? | [Full-Scale SILVA](full-scale-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
