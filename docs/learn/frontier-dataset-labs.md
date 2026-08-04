# Dataset-Backed Equilibrium Labs

The recent SILVA families now have deterministic datasets matched to their
state geometry. These builders make the governing equation or statistical
contract observable before a learned model is introduced.

| Family | Dataset builder | State geometry | Exact check |
| --- | --- | --- | --- |
| Fourier equilibrium | `make_periodic_elliptic_dataset` | regular fields | periodic elliptic residual |
| physics graph equilibrium | `make_graph_transport_dataset` | batched ring graphs | discrete steady transport residual |
| homotopy equilibrium | `make_affine_homotopy_dataset` | condition/root pairs | affine fixed-point residual |
| distributional equilibrium | `make_variable_measure_dataset` | padded empirical measures | masks, counts, and empirical moments |

Each lab pairs a derivation with an executable notebook and retained 300 DPI
plots.

## Periodic Elliptic Fields

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/17_silva_fno_equilibrium_lab/">Open Fourier equilibrium lab</a>
  <a class="md-button" href="../../package-notebooks/17_silva_fno_equilibrium_lab/17_silva_fno_equilibrium_lab.ipynb" download>Download notebook</a>
</div>

The field dataset solves

$$
(-\Delta+m)u=f
$$

on a periodic unit square. For wave vector $k$,

$$
\widehat u(k)
=\frac{\widehat f(k)}{|k|^2+m}.
$$

The forcing is restricted to low modes, and the target is evaluated from this
spectral formula. The returned batch contains

```text
forcing:      (samples, 1, height, width)
target:       (samples, 1, height, width)
coordinates:  (height, width, 2)
```

```python
from silva_networks import make_periodic_elliptic_dataset

data = make_periodic_elliptic_dataset(
    samples=16,
    height=32,
    width=32,
    modes=4,
    mass=1.0,
    seed=17,
)
assert data.equation_residual().abs().max() < 1e-4
```

This builder tests field shapes, Fourier normalization, resolution changes,
and equation-aware evaluation. The full FNO-DEQ study evaluates Darcy flow and
steady Navier-Stokes with its published datasets and protocols
[[43]](../paper/references.md#ref-43){ .silva-cite }.

## Graph Transport Fields

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/18_silva_graph_transport_lab/">Open graph transport lab</a>
  <a class="md-button" href="../../package-notebooks/18_silva_graph_transport_lab/18_silva_graph_transport_lab.ipynb" download>Download notebook</a>
</div>

For every generated graph, the target satisfies

$$
u=s+gamma_r u
+gamma_d\mathcal L_Gu
-\gamma_a\nabla_Vu.
$$

The ring graph has forward and reverse edges. Conductance is stored in
`edge_weight`; direction and speed are stored in `edge_velocity`. Multiple
graphs are packed by offsetting node ids and assigning a graph id in `batch`.

```text
x:              (samples * nodes, 3)
edge_index:     (2, samples * 2 * nodes)
edge_weight:    (samples * 2 * nodes,)
edge_velocity:  (samples * 2 * nodes,)
batch:          (samples * nodes,)
target:         (samples * nodes, 1)
```

```python
from silva_networks import make_graph_transport_dataset

data = make_graph_transport_dataset(samples=8, nodes=24, seed=18)
assert data.equation_residual().abs().max() < 1e-4
assert (data.batch[data.edge_index[0]] == data.batch[data.edge_index[1]]).all()
```

The cited environmental study uses real NO2 and PM2.5 measurements collected
in Antwerp, discretized over spatial locations and hourly intervals
[[44]](../paper/references.md#ref-44){ .silva-cite }. The package ring data is
for equation, batching, and training validation; environmental reporting must
retain the measurement geometry, missing-data rules, split, and physical units
from the study.

## Affine Homotopy Pairs

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/19_silva_homotopy_equilibrium_lab/">Open homotopy equilibrium lab</a>
  <a class="md-button" href="../../package-notebooks/19_silva_homotopy_equilibrium_lab/19_silva_homotopy_equilibrium_lab.ipynb" download>Download notebook</a>
</div>

For transition

$$
T(z;x)=az+x,
\qquad |a|<1,
$$

the exact root is

$$
z^\star=\frac{x}{1-a}.
$$

The residual flow also has the complete analytic trajectory

$$
z(t)=z^\star+(z_0-z^\star)e^{-(1-a)t}.
$$

The builder returns conditions and exact roots. It supports endpoint error,
trajectory error, Euler/RK4 comparison, gradient checks, and horizon studies.

```python
from silva_networks import make_affine_homotopy_dataset

data = make_affine_homotopy_dataset(
    samples=64,
    dimension=4,
    contraction=0.5,
    seed=19,
)
assert data.fixed_point_residual().abs().max() < 1e-6
```

The corresponding research family connects equilibrium models and continuous
paths through homotopy continuation
[[46]](../paper/references.md#ref-46){ .silva-cite }. The exact affine data is a
numerical reference, not a replacement for the vision datasets and training
protocol used in that study.

## Variable-Size Empirical Measures

<div class="silva-document-actions">
  <a class="md-button md-button--primary" href="../../package-notebooks/20_silva_distributional_equilibrium_lab/">Open distributional equilibrium lab</a>
  <a class="md-button" href="../../package-notebooks/20_silva_distributional_equilibrium_lab/20_silva_distributional_equilibrium_lab.ipynb" download>Download notebook</a>
</div>

Each sample contains a variable number of points from a Gaussian mixture. The
batch pads them to one maximum length and supplies a boolean mask:

```text
context:            (samples, max_particles, dimension)
context_mask:       (samples, max_particles)
component_centers:  (samples, components, dimension)
target_mean:        (samples, dimension)
counts:             (samples,)
```

The mask-aware empirical mean is

$$
\overline x_b
=\frac{
\sum_i m_{bi}x_{bi}
}{
\sum_i m_{bi}
}.
$$

```python
import torch
from silva_networks import make_variable_measure_dataset

data = make_variable_measure_dataset(
    samples=32,
    min_particles=16,
    max_particles=40,
    dimension=3,
    components=3,
    seed=20,
)
assert (data.context_mask.sum(dim=1) == data.counts).all()
assert torch.allclose(data.empirical_mean(), data.target_mean)
```

The distributional study evaluates point-cloud classification and completion;
its maintained research materials identify MNIST Point Cloud and ModelNet40
for the reported experiments
[[45]](../paper/references.md#ref-45){ .silva-cite }. Dataset-specific claims
must preserve the official point sampling, splits, augmentation, and metrics.

## Training and Evaluation Contract

The compact datasets support fast integration tests, but the reporting
contract is the same one required for larger experiments.

| Family | Task quantity | Equilibrium quantity | Structural or physical quantity |
| --- | --- | --- | --- |
| Fourier | field MSE or relative error | fixed-point residual | elliptic/PDE residual |
| graph transport | node or graph loss | fixed-point residual | transport residual and relabeling error |
| homotopy | endpoint task loss | terminal fixed-point residual | trajectory error and evaluation count |
| distributional | task-head loss | final measure discrepancy | permutation and mask checks |

Use a training/validation/test split before fitting normalization statistics.
Record the random seed, tensor shapes, solver configuration, equation
coefficients, and acceptance thresholds. A small-scale reproduction should be
described as such; benchmark equivalence requires the benchmark data and full
protocol.

## Validation Map

| Artifact | What it verifies |
| --- | --- |
| `tests/test_frontier_data.py` | equations, deterministic seeds, masks, batching, gradients, and model integration |
| `tests/test_frontier.py` | transition behavior, invariances, solvers, and differentiation |
| notebooks 17-20 | derivations, training loops, diagnostics, plots, and factory construction |
| `scripts/run_notebook_smoke.py` | executable release path for every focused lab |
| `scripts/release_audit.py` | synchronized publication files, navigation, citations, and plot resolution |

## Where to Go Next

| Question | Page |
| --- | --- |
| How do these datasets enter each SILVA transition? | [Recent Equilibrium Families](frontier-equilibrium-families.md) |
| Which classes and builders are public? | [Recent Dataset API](../api/frontier_data.md) |
| How do ordinary dataset adapters work? | [Datasets and Preprocessing](datasets-and-preprocessing.md) |
| Where are the complete citations? | [Paper and References](../paper/references.md) |
