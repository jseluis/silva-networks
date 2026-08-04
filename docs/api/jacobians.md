# Jacobians

At an equilibrium

$$
z^\star=f_\theta(z^\star,x),
$$

the local Jacobian with respect to the state is

$$
J_z = \frac{\partial f_\theta}{\partial z}(z^\star,x).
$$

This matrix controls stability, implicit gradients, and solver behavior.

## Full Jacobian

For small states, materialize

$$
J_z[i,j]
=
\frac{\partial f_i}{\partial z_j}.
$$

```python
from silva_networks import full_jacobian

J = full_jacobian(lambda z: transition(z, x), z_star)
```

This is exact up to autograd precision, but memory scales like

$$
O(n^2)
$$

for a flattened state dimension \(n\).

## Vector-Jacobian Product

For large states, a vector-Jacobian product computes

$$
J_z^\top v
$$

without constructing \(J_z\). This is the primitive used by implicit backward
passes and power iteration.

```python
from silva_networks import vjp

jtv = vjp(lambda z: transition(z, x), z_star, v)
```

## Jacobian-Vector Product

The companion product is

$$
J_z v.
$$

```python
from silva_networks import jvp

value, jv = jvp(lambda z: transition(z, x), z_star, v)
```

## Spectral Radius

Local contraction is governed by

$$
\rho(J_z)=\max_i |\lambda_i(J_z)|.
$$

When

$$
\rho(J_z)<1,
$$

the fixed point is locally stable for the linearized update. The package
estimates this quantity using VJP-based power iteration.

## Hutchinson Norm Estimate

For a Rademacher vector \(v\), with entries sampled from \(\{-1,+1\}\),

$$
\mathbb E_v\|J_z^\top v\|_2^2=\|J_z\|_F^2.
$$

The helper `hutchinson_jacobian_norm` averages this quantity over several
probes.

::: silva_networks.jacobian

## Where to Go Next

| Question | Page |
| --- | --- |
| How do these estimates support stability claims? | [Jacobians and Stability](../learn/jacobians.md) |
| Where can I compare diagnostics interactively? | [Interactive Diagnostics Lab](../learn/interactive-diagnostics-lab.md) |
| Which higher-level diagnostics use these functions? | [Diagnostics API](diagnostics.md) |
