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

## Where to Go Next

| Question | Page |
| --- | --- |
| Which projected and differentiable layers are public? | [Optimization API](../api/optimization.md) |
| How do optimization layers relate to implicit layers? | [Implicit Layers Bridge](../learn/implicit-bridge.md) |
| Can I execute the quadratic-layer derivation? | [Optimization Layers Notebook](../implicit-bridge-notebooks/05_differentiable_optimization.ipynb) |
