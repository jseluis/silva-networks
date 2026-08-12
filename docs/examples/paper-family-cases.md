# Paper Family Cases

This example runs five equilibrium families through one SILVA solver contract:
sequence modeling, multiscale vision, implicit graphs, coordinate-based fields,
and diffusion trajectories. Each case changes the state and transition while
retaining

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z^\star)=f_\theta(z^\star,x)-z^\star.
$$

```bash
python examples/paper_family_cases.py
```

## Five State Choices

| Family | Equilibrium state | Input shape | Reported output |
| --- | --- | --- | --- |
| sequence | token features \(Z\in\mathbb R^{B\times T\times d}\) | `(2, 6)` token IDs | `(2, 6, 32)` vocabulary scores |
| multiscale | tuple \((Z_1,Z_2)\) at two resolutions | `(2, 3, 8, 8)` images | `(2, 5)` class scores plus both states |
| graph | node matrix \(Z\in\mathbb R^{N\times d}\) | `(4, 3)` and `(2, 4)` edges | `(4, 2)` node scores |
| implicit representation | coordinate field \(Z(q)\) | `(1, 12, 2)` coordinates | `(1, 12, 3)` field values and coordinate gradients |
| diffusion | complete selected denoising trajectory | `(1, 1, 4, 4)` noise | final image and stacked trajectory |

The SILVA decomposition changes meaning by family. Sequence attention and
causal mixing define interaction branches; multiscale projections connect
resolution-specific states; graph edges define local messages; coordinate
injection supplies a spatial stimulus; and the denoiser couples selected
diffusion steps into a triangular fixed-point system.

## Reading the Results

The sequence, graph, and diffusion cases report solver residuals directly.
The multiscale shapes verify that packing and unpacking preserve every
resolution. The coordinate-gradient shape verifies that derivatives with
respect to query locations remain available. These are architecture checks,
not task-accuracy or convergence comparisons. The shared three-iteration
budget is deliberately small, and a nonzero residual means the state should
not be reported as converged. Full experiments must supply a suitable solver
budget together with the dataset, dimensions, optimization schedule, and
evaluation rules of the selected study.

## Complete Source

```python
--8<-- "examples/paper_family_cases.py"
```

See [Paper Families as SILVA Configurations](../learn/paper-family-adaptations.md)
for the family-by-family derivations and
[Paper and References](../paper/references.md) for the corresponding primary
sources.


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
The reader-facing evidence for this route is **shape and residual checks across sequence, vision, graph, and diffusion cases**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Run the Complete Example

```bash
python examples/paper_family_cases.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
sequence (2, 6, 32) 3.1066622734069824
mdeq (2, 5) [(2, 4, 8, 8), (2, 8, 4, 4)]
ignn (4, 2) 0.37995445728302
inr (1, 12, 3) (1, 12, 2)
ddim (1, 1, 4, 4) (4, 1, 1, 4, 4)
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
  example: paper-family-cases
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: shape and residual checks across sequence, vision, graph, and diffusion cases
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

At full scale, move toward **the complete source task for one selected generalized family**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How does each research family connect to SILVA? | [Paper Family Adaptations](../learn/paper-family-adaptations.md) |
| Which generalized case classes are public? | [Generalized Cases API](../api/cases.md) |
| Can I execute every family in one notebook? | [Paper Family Architectures Notebook](../package-notebooks/12_paper_family_architectures.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
