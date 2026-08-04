# Derivations to Code

The package is organized so every major equation has a nearby implementation
object. This page shows the main derivation path and the matching code.

For the full traceability manual, including solver algorithms, SILVA
families, tensor contracts, Jacobian estimators, and reporting best practices,
see [Implementation Derivations](../learn/implementation-derivations.md).

## Fixed-Point Residual

Start with a transition map \(f_\theta\). The equilibrium is the state where
applying the map changes nothing:

$$
z^\star=f_\theta(z^\star,x).
$$

Move all terms to one side:

$$
r_\theta(z,x)=f_\theta(z,x)-z.
$$

Solving the layer means finding a state whose residual norm is small:

$$
\|r_\theta(z^\star,x)\|_2
=
\|f_\theta(z^\star,x)-z^\star\|_2
\le \varepsilon.
$$

```python
from silva_networks import SolverConfig, fixed_point

result = fixed_point(f, z0, SolverConfig(solver="picard", max_iter=25, alpha=0.5))
z_star = result.z
residual = result.residual
```

## Damped Solver Step

The Picard step with damping is

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x).
$$

Subtract \(z_k\) to see the residual direction:

$$
z_{k+1}-z_k
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x)-z_k
=
\alpha(f_\theta(z_k,x)-z_k)
=
\alpha r_\theta(z_k,x).
$$

Thus `alpha` controls how far the iterate moves along the residual direction.

```python
config = SolverConfig(solver="picard", alpha=0.4, max_iter=20, tol=1e-6)
```

## SILVA Interaction Field

The generic SILVA transition decomposes into stimulus, optional learned self
interaction, local interaction, and global interaction:

$$
f_\theta(z,x)
=
\operatorname{Norm}
\left[
\tanh
\left(
S_\theta(x)+H_\theta(\chi(z))+L_\theta(\chi(z),E)+G_\theta(\chi(z),b)
\right)
\right].
$$

The package implementation follows the same sequence:

```python
s = self.stimulus(x)
y = self.activation(z)
self_update = self.self_term(y)
local = self.local(y, edge_index=edge_index, edge_attr=edge_attr)
global_context = self.global_term(y, batch=batch)
z_next = self.norm(self.output_activation(s + self_update + local + global_context))
```

## Local Message Passing

For `GraphLocal`, each source state \(z_j\) is first projected:

$$
m_j=W_\ell z_j.
$$

For destination \(i\), incoming messages are averaged:

$$
L_i(z,E)
=
\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_{j\in\mathcal N(i)}m_j,
\qquad
\mathcal N(i)=\{j:(j,i)\in E\}.
$$

Code:

```python
src, dst = edge_index
messages = proj(z)
out = torch.zeros_like(messages)
out.index_add_(0, dst, messages[src])
```

## Graph Attention Local Term

For `GraphAttentionLocal`, first project each state into attention-head space:

$$
h_i=Wz_i.
$$

For an edge \(j\to i\), compute an unnormalized score:

$$
e_{ij}
=
\operatorname{LeakyReLU}
\left(
a_s^\top h_j+a_t^\top h_i
\right).
$$

Normalize only over incoming edges to the same destination:

$$
a_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell\in\mathcal N(i)}\exp(e_{i\ell})}.
$$

Then aggregate:

$$
L_i(z,E)
=
\sum_{j\in\mathcal N(i)}a_{ij}h_j.
$$

Edge attributes add one more score term,

$$
e_{ij}
\leftarrow
e_{ij}+a_e^\top W_e e_{ij}^{\rm attr},
$$

which is the path used by bond-aware molecular SILVA layers.

## Global Mean Field

For each graph \(g\), compute the mean state:

$$
\bar z_g=\frac1{|\mathcal V_g|}\sum_{j\in\mathcal V_g}z_j.
$$

Broadcast a learned projection:

$$
G_i(z)=W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

The gated SILVA-style variant computes

$$
\beta_g
=
\sigma\left(
\frac{(W_q\bar z_g)^\top(W_k\bar z_g)}{\sqrt d}
\right),
\qquad
G_i(z)=\beta_g W_v\bar z_{\operatorname{batch}(i)}.
$$

Code:

```python
from silva_networks import GatedMeanFieldGlobal

global_term = GatedMeanFieldGlobal(dim=64)
g_update = global_term(z, batch=batch)
```

## Jacobian Diagnostics

At an equilibrium, local behavior is controlled by

$$
J_f(z^\star)=\frac{\partial f_\theta}{\partial z}(z^\star,x).
$$

For small states, materialize the full Jacobian:

```python
from silva_networks import full_jacobian

J = full_jacobian(lambda z: layer.f(z, x, edge_index=edge_index), z_star)
```

For larger states, compute products:

$$
Jv
\quad\text{and}\quad
J^\top v.
$$

```python
from silva_networks import jvp, vjp

_, Jv = jvp(f, z_star, probe)
Jtv = vjp(f, z_star, probe)
```

The same interface supports spectral-radius and Lyapunov-style diagnostics:

```python
from silva_networks import damped_spectral_radius, solve_with_energy

rho = damped_spectral_radius(f, z_star, alpha=0.5)
```

## Custom Branch Rule

A custom branch is mathematically valid for the package when it maps the current
state and optional context back to the state shape:

$$
B_\psi:\mathbb R^{N\times d}\to\mathbb R^{N\times d}.
$$

```python
import torch

class MyGlobal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, z, batch=None):
        context = z.mean(dim=0, keepdim=True)
        return self.proj(context).expand_as(z)
```

The solver, gradients, diagnostics, and device behavior stay the same.

## Verify the Translation

For every equation-to-code step, check four invariants:

1. each active branch returns the declared state shape;
2. the composed transition preserves shape, dtype, and device;
3. the solver residual is computed from the same transition used in training;
4. gradients reach every trainable branch that contributes to the loss.

The complete source lineage is organized in
[Implementation Derivations](../learn/implementation-derivations.md), with
primary method links in [Paper and References](../paper/references.md).
