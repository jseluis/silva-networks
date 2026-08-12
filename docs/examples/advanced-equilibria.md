# Advanced Equilibria Example

Run the six adjacent mechanisms from public package imports:

```bash
python examples/advanced_equilibria.py
```

The script constructs a monotone graph equilibrium, injected equilibrium
transformer, positive Poisson mirror equilibrium, physics-informed ODE
equilibrium, implicit DAE stage root, and adversarial residual objective. Every
case uses a small deterministic generated problem, so the command needs no
external data download.

The printed quantities answer different questions. Fixed-point residuals
diagnose numerical solving; Poisson KL measures data fidelity; the
physics-informed loss separates boundary, ODE, and Jacobian terms; the DAE
residual checks the stage root; and the adversarial objective reports generator
and discriminator terms separately.

## Shared SILVA Contract

The equilibrium families all preserve

$$
z^\star=T_\theta(z^\star;x),
\qquad
\widehat y=Q_\psi(z^\star).
$$

Their tensor contracts differ by domain:

| Mechanism | Source shape | State shape | Output |
| --- | --- | --- | --- |
| monotone graph | `(nodes, in_dim)` plus `(2, edges)` | `(nodes, state_dim)` | node field |
| equilibrium transformer | `(batch, channels, height, width)` | `(batch, patches, hidden_dim)` | decoded image |
| Poisson mirror | nonnegative image tensor | same positive image shape | reconstruction and intensity |
| physics-informed ODE | `(samples, time_dim)` | `(samples, state_dim)` | physical trajectory |
| implicit DAE | differential and algebraic rank-two tensors | packed Runge-Kutta stages | next differential/algebraic state |

The adversarial residual utility accepts a final residual dimension and returns
two losses; it has no equilibrium state by itself.

## Compact Code Path

The script uses generated batches whose equations are checked before the model
calls. A representative graph path is:

```python
from silva_networks import (
    SILVAMonotoneGraphEquilibrium,
    make_monotone_chain_dataset,
)

data = make_monotone_chain_dataset(nodes=8, seed=25)
model = SILVAMonotoneGraphEquilibrium(1, 4, 1)
result = model(data.source, data.edge_index, return_result=True)

assert result.output.shape == data.target.shape
print(result.monotonicity_certificate)
print(result.solver_result.residual)
```

The remaining cases follow the same source, implicit-state, readout, and
diagnostic sequence. The complete script is deliberately CPU-sized.

## Interpretation and Citations

The monotone operator follows Baker et al.
[[47]](../paper/references.md#ref-47){ .silva-cite }; one-time QKV injection
follows Geng, Pokle, and Kolter
[[48]](../paper/references.md#ref-48){ .silva-cite }; Burg mirror equilibrium
follows Daniele et al. [[50]](../paper/references.md#ref-50){ .silva-cite }; and
the physics-informed equilibrium follows Pacheco and Camponogara
[[51]](../paper/references.md#ref-51){ .silva-cite }. The DAE stage mechanism is
connected to DAE-PINN [[52]](../paper/references.md#ref-52){ .silva-cite }. The
adversarial residual objective follows the differential-equation GAN work
[[53]](../paper/references.md#ref-53){ .silva-cite } and is not a
deep-equilibrium family.

For each run, inspect the fixed-point or root residual before interpreting task
quality. The generated data validate equations and gradients; they do not
reproduce the large experiments from the cited papers.


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
The reader-facing evidence for this route is **one result for each advanced equilibrium family, with family-specific residuals**. The
invariants that must remain true are **node relabeling equivariance, graph boundaries, and state shape**.


### Run the Complete Example

```bash
python examples/advanced_equilibria.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
monotone graph: (8, 1) 0.023554455488920212
equilibrium transformer: 0.18536624312400818
Poisson mirror: 0.005979819223284721
physics-informed loss: 0.8003759384155273
implicit DAE step: [0.4761904776096344] 1.862645149230957e-09
adversarial residual objective: 0.7888258695602417 1.3886094093322754
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
  example: advanced-equilibria
  state: one latent vector per node or entity
  condition: node features, edges, edge attributes, and graph batches
  repeated_transition: a source-injected graph message or monotone graph transition
  invariant_checks: node relabeling equivariance, graph boundaries, and state shape
  compact_evidence: one result for each advanced equilibrium family, with family-specific residuals
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

At full scale, move toward **the selected graph, inverse-problem, transformer, ODE, or DAE benchmark**. Increase only one of
**node count, edge count, feature width, and number of graphs** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the mechanisms derived? | [Advanced Equilibrium Families](../learn/advanced-equilibrium-families.md) |
| How are physics-informed and DAE equations constructed? | [Physics-Informed Equilibria](../learn/physics-informed-equilibria.md) |
| Which generated relations are checked? | [Advanced Equilibrium Datasets](../learn/advanced-equilibrium-datasets.md) |
| Where are the full executable labs? | [Notebooks](../notebooks.md#advanced-equilibrium-and-physics-track) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
