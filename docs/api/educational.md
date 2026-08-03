# Educational NumPy API

The `educational` module mirrors the PyTorch API with small NumPy functions.
Use it when you want to see the algebra without autograd, modules, batching, or
GPU concerns.

## Why This Module Exists

The PyTorch package is the production path. The NumPy helpers are for hand-sized
derivations:

| Helper | Mathematical object |
| --- | --- |
| `np_picard` | Damped fixed-point iteration |
| `np_finite_difference_jacobian` | Central-difference Jacobian |
| `np_exact_tanh_affine_jacobian` | Closed-form Jacobian for `tanh(Wz + s)` |
| `np_power_iteration` | Dominant mode estimate for a materialized matrix |
| `np_implicit_gradient` | Explicit adjoint solve for a small DEQ |

## Minimal Fixed Point

For

$$
f(z)=\tanh(Wz+s),
$$

Picard iteration computes

$$
z_{k+1}=(1-\alpha)z_k+\alpha f(z_k).
$$

```python
import numpy as np
from silva_networks import np_picard

W = np.array([[0.2, 0.1], [-0.1, 0.25]])
s = np.array([0.5, -0.2])
trace = np_picard(lambda z: np.tanh(W @ z + s), np.zeros(2), alpha=0.8)
```

The result stores the final `z` and the residual curve.

## Exact Jacobian for `tanh(Wz + s)`

Let

$$
u=Wz+s,
\qquad
f(z)=\tanh(u).
$$

Because

$$
\frac{d}{du}\tanh(u)=1-\tanh^2(u),
$$

the state Jacobian is

$$
J_f(z)
=
\operatorname{diag}\left(1-\tanh^2(Wz+s)\right)W.
$$

```python
from silva_networks import np_exact_tanh_affine_jacobian

J = np_exact_tanh_affine_jacobian(W, trace.z, s)
```

## Small Implicit Gradient

At a solved equilibrium, the total derivative obeys

$$
(I-J_f)\frac{dz^\star}{d\theta}
=
\frac{\partial f}{\partial \theta}.
$$

For a loss gradient \(g=\partial \mathcal L/\partial z^\star\), solve

$$
(I-J_f^\top)\lambda=g,
$$

then compute

$$
\frac{\partial \mathcal L}{\partial \theta}
=
\lambda^\top
\frac{\partial f}{\partial \theta}.
$$

```python
from silva_networks import np_implicit_gradient

grad_theta = np_implicit_gradient(J, grad_z, df_dtheta)
```

This is the explicit small-matrix version of the adjoint system used by the
PyTorch diagnostics.

::: silva_networks.educational
