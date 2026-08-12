# Dataset Quickstart

`examples/datasets_quickstart.py` loads a public tabular dataset, adapts it to
SILVA tensors, and trains a small node-level classifier.

```bash
python examples/datasets_quickstart.py
```

## Preprocessing

Rows become entities. Features are standardized:

$$
\tilde X_{ij}
=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

A kNN graph is built in standardized feature space:

$$
j\in\mathcal N_k(i)
\quad\Longleftrightarrow\quad
j\text{ is among the }k\text{ smallest values of }
\|\tilde x_i-\tilde x_j\|_2.
$$

The adapter returns a `GraphTensorBatch`:

```python
dataset = load_tabular_dataset("iris", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, device=device)
graph.validate()
```

## Model

The graph is passed directly into `SILVAGraphNetwork`:

```python
logits = model(graph.x, edge_index=graph.edge_index)
```

The same recipe works for custom tables after replacing the loader with a
tensor, NumPy array, pandas frame, or user-defined feature matrix.

The script reports feature and edge shapes, class balance, losses, and accuracy.
Before interpreting the task metric, verify `graph.validate()`, finite
standardized features, valid edge bounds, and the residual of every SILVA
equilibrium layer. Dataset sources and reporting rules are listed in
[Paper and References](../paper/references.md#citation-rules-for-reports).


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the tensor solved to equilibrium**, the condition is **the observed input or source tensor**, and the
repeated map is **the state-preserving transition evaluated by the root solver**.

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
The reader-facing evidence for this route is **dataset identity, tensor shape, and measured classification accuracy**. The
invariants that must remain true are **shape, device, dtype, finiteness, and differentiability**.


### Run the Complete Example

```bash
python examples/datasets_quickstart.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
dataset iris
shape (150, 4)
accuracy 0.8733333349227905
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
  example: datasets-quickstart
  state: the tensor solved to equilibrium
  condition: the observed input or source tensor
  repeated_transition: the state-preserving transition evaluated by the root solver
  invariant_checks: shape, device, dtype, finiteness, and differentiability
  compact_evidence: dataset identity, tensor shape, and measured classification accuracy
  scale_axes: state width, batch size, and data volume
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

At full scale, move toward **the official split with preprocessing fitted only on training data**. Increase only one of
**state width, batch size, and data volume** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How should datasets be validated before solving? | [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) |
| Which loaders and tensor objects are public? | [Datasets API](../api/datasets.md) |
| Which public dataset experiments are configured? | [Dataset Cases](../experiments/datasets.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
