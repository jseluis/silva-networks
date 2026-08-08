# Molecules

`examples/molecules.py` represents atoms as entities, bonds as graph edges, and
molecule IDs as the `batch` vector used by graph-level pooling. The same SILVA
state equation is solved for all atoms, while the graph and batch tensors keep
local bond interactions separate from molecule-level context.

```bash
python examples/molecules.py
```

The state matrix has one row per atom:

$$
Z\in\mathbb R^{N_{\rm atoms}\times d}.
$$

Bond edges define the local neighborhood:

$$
\mathcal N(i)=\{j:(j,i)\in E_{\rm bonds}\}.
$$

For atom \(i\), the equilibrium has the form

$$
z_i^\star
=
\Phi\left{
S_\theta(x_i)
+L_\theta\left(z_i^\star,\{z_j^\star:j\in\mathcal N(i)\}\right)
+G_\theta\left(z_i^\star,b_i\right)
\right\},
$$

where \(b_i\) is the molecule identifier. `SILVAGraphLayer` supplies the
stimulus, bond-local aggregation, and batch-aware global mean field.

After the equilibrium solve, molecule-level states are obtained by mean pooling:

$$
h_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g} z_i^\star.
$$

The linear head maps each molecule state to a scalar prediction. The printed
shapes confirm the atom state and graph-level output dimensions. The final
residual checks self-consistency, and `stimulus_gradient_norm` confirms that
the graph-level loss differentiates through pooling and the equilibrium solve
to the input projection.

## Tensor Contract

| Tensor | Shape in the example | Meaning |
| --- | --- | --- |
| `atom_features` | `(7, 4)` | four input features for seven atoms |
| `edge_index` | `(2, 8)` | directed bond endpoints |
| `batch` | `(7,)` | atom-to-molecule assignment |
| equilibrium state | `(7, 10)` | ten hidden values per atom |
| prediction | `(2, 1)` | one scalar for each molecule |

## Complete Source

```python
--8<-- "examples/molecules.py"
```

Use [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) for
real molecular records and edge attributes. The graph and molecular sources
are listed in [Graphs, Attention, and Messages](../paper/references.md#graphs-attention-and-messages).

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
The reader-facing evidence for this route is **atom and molecule tensor shapes, residual, prediction loss, and gradients**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Run the Complete Example

```bash
python examples/molecules.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
atom_state_shape (7, 10)
molecule_prediction_shape (2, 1)
final_residual 9.932107786880806e-06
stimulus_gradient_norm 0.052681755274534225
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
  example: molecules
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: atom and molecule tensor shapes, residual, prediction loss, and gradients
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

At full scale, move toward **the official molecular split with the complete feature and metric protocol**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How should molecular tensors be prepared? | [Datasets and Preprocessing](../learn/datasets-and-preprocessing.md) |
| Which molecular adapters are public? | [Datasets API](../api/datasets.md) |
| How does the underlying graph layer work? | [Graph SILVA Example](graph-silva.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
