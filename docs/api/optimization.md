# Optimization API

This module contains package-native projected quadratic-program layers and an
optional CVXPYlayers bridge. Its method lineage is OptNet
[[8]](../paper/references.md#ref-8){ .silva-cite }, differentiable convex
optimization layers [[9]](../paper/references.md#ref-9){ .silva-cite }, and the
CVXPYlayers implementation [[40]](../paper/references.md#ref-40){ .silva-cite }.

For the source-to-package derivation and citation scope, see
[Method Adaptation Atlas](../learn/method-adaptation-atlas.md).

## Package-Native Projected QP

The core layer solves

$$
z_i^\star
=
\arg\min_{z\in C}
\frac12 z^\top A z-b_i^\top z,
\qquad
A=L L^\top+\lambda I,
\qquad
b_i=B_\theta x_i+c.
$$

The implemented fixed-point map is projected gradient descent:

$$
T(z)
=
\Pi_C\left[z-\eta(Az-b_i)\right].
$$

Supported package-native constraint choices are:

| `constraint` | Constraint set |
| --- | --- |
| `"none"` | unconstrained quadratic |
| `"nonnegative"` | \(z_j\ge 0\) |
| `"box"` | \(\ell_j\le z_j\le u_j\) |
| `"simplex"` | \(z_j\ge 0,\ \sum_j z_j=m\) |
| `"affine"` | \(A_{\rm eq}z=b_{\rm eq}\) |

```python
from silva_networks import SolverConfig, silva_projected_qp_layer

layer = silva_projected_qp_layer(
    in_dim=8,
    state_dim=4,
    constraint="simplex",
    simplex_mass=1.0,
    config=SolverConfig(solver="picard", max_iter=50, alpha=1.0),
)
z_star = layer(x)
```

## CVXPYlayers Bridge

For general disciplined parametrized convex programs, install the optional
optimization extra on Python 3.11+:

```bash
python -m pip install "silva-networks[optimization]"
```

Then use `silva_cvxpy_layer(...)` to wrap a DPP-compliant CVXPY problem. This
path follows CVXPYlayers and is separate from the core projected-QP layer.

## Public Names

| Preferred name | Compatibility name |
| --- | --- |
| `SILVAProjectedQPLayer` | `SILVAConstrainedQuadraticLayer` |
| `silva_projected_qp_layer` | `silva_constrained_quadratic_layer` |

## Citation Map

| Feature | Cite |
| --- | --- |
| projected quadratic SILVA layer | SILVA package; projected-gradient methods |
| differentiable QP/optimization layer framing | [OptNet](https://arxiv.org/abs/1703.00443) |
| general CVXPYlayers bridge | [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430), [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) |

## API Docs

::: silva_networks.optimization

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is a constrained layer executed? | [Constrained Optimization Example](../examples/constrained-optimization.md) |
| How do optimization layers fit the implicit-layer viewpoint? | [Implicit Layers Bridge](../learn/implicit-bridge.md) |
| Can I run the quadratic derivation? | [Optimization Layers Notebook](../implicit-bridge-notebooks/05_differentiable_optimization.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
