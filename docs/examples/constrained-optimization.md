# Constrained Optimization

Run:

```bash
python examples/constrained_optimization.py
```

This example uses the package-native `silva_projected_qp_layer` factory. The
same layer can also be selected with `silva_equilibrium_model("silva_projected_qp", ...)`.
The state is an optimizer variable rather than a node, channel, or flow field.

## Equation

For each input row \(x_i\), the layer forms

$$
b_i=B_\theta x_i+c
$$

and solves

$$
z_i^\star
=
\arg\min_{z\in C}
\frac12 z^\top A z-b_i^\top z,
\qquad
A=L L^\top+\lambda I.
$$

The example chooses the simplex constraint

$$
C=\Delta_1
=
\left\{
z\in\mathbb R^4:
z_j\ge 0,\ \sum_{j=1}^4z_j=1
\right\}.
$$

The fixed-point map is projected gradient descent:

$$
T(z)
=
\Pi_{\Delta_1}
\left[
z-\eta(Az-b_i)
\right],
\qquad
z^\star=T(z^\star).
$$

In code:

```python
from silva_networks import SolverConfig, silva_projected_qp_layer

layer = silva_projected_qp_layer(
    in_dim=3,
    state_dim=4,
    constraint="simplex",
    simplex_mass=1.0,
    step_size=0.08,
    config=SolverConfig(solver="picard", max_iter=25, alpha=1.0),
)
```

## What to Inspect

The printed dictionary reports:

| Field | Meaning |
| --- | --- |
| `device` | resolved CPU, CUDA, or MPS device |
| `state_shape` | optimizer state shape |
| `iterations` | fixed-point iterations used |
| `residual` | final projected fixed-point residual |
| `simplex_sums` | row sums, which should be close to `1.0` |
| `min_entry` | smallest optimizer coordinate, which should be nonnegative |
| `energy` | mean quadratic objective value |
| `has_grad` | whether gradients reached \(B_\theta\) |

## Constraint Choices

The same layer supports:

| `constraint` | Constraint set |
| --- | --- |
| `"none"` | unconstrained quadratic |
| `"nonnegative"` | \(z_j\ge 0\) |
| `"box"` | \(\ell_j\le z_j\le u_j\) |
| `"simplex"` | \(z_j\ge 0,\ \sum_jz_j=m\) |
| `"affine"` | \(A_{\rm eq}z=b_{\rm eq}\) |

Use [Optimization API](../api/optimization.md) for the full signature and
[Selecting Model Families](../learn/selecting-model-families.md) for the
selector table.

## Citations

Cite the SILVA package [[2]](../paper/references.md#ref-2){ .silva-cite } for
this implementation. Cite [OptNet](https://arxiv.org/abs/1703.00443)
[[8]](../paper/references.md#ref-8){ .silva-cite } when discussing
differentiable quadratic-program layers and
[CVXPYlayers](https://github.com/cvxpy/cvxpylayers)
[[40]](../paper/references.md#ref-40){ .silva-cite } when using the optional
general disciplined convex-program bridge.


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the primal variable and any dual or auxiliary state**, the condition is **objective coefficients and constraints**, and the
repeated map is **a projected, proximal, or primal-dual update**.

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
The reader-facing evidence for this route is **simplex feasibility, energy, solver residual, and parameter gradients**. The
invariants that must remain true are **feasibility, domain projection, state shape, and optimality conditions**.


### Run the Complete Example

```bash
python examples/constrained_optimization.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
{'device': 'cpu', 'state_shape': (6, 4), 'iterations': 25, 'residual': 0.00254080886952579, 'simplex_sums': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'min_entry': 0.039561957120895386, 'energy': 0.49771085381507874, 'has_grad': True}
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
  example: constrained-optimization
  state: the primal variable and any dual or auxiliary state
  condition: objective coefficients and constraints
  repeated_transition: a projected, proximal, or primal-dual update
  invariant_checks: feasibility, domain projection, state shape, and optimality conditions
  compact_evidence: simplex feasibility, energy, solver residual, and parameter gradients
  scale_axes: variable count, constraint count, conditioning, and linear-solver budget
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

At full scale, move toward **the complete constrained task at its original variable and constraint count**. Increase only one of
**variable count, constraint count, conditioning, and linear-solver budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Which projected and differentiable layers are public? | [Optimization API](../api/optimization.md) |
| How do optimization layers relate to implicit layers? | [Implicit Layers Bridge](../learn/implicit-bridge.md) |
| Can I execute the quadratic-layer derivation? | [Optimization Layers Notebook](../implicit-bridge-notebooks/05_differentiable_optimization.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
