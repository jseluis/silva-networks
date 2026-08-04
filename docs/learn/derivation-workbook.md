# Derivation Workbook

This workbook is a guided path from the first fixed-point equation to the
package objects that run it. It is written so a reader can derive the equations
line by line, then open the matching notebook or API page and execute the same
idea.

Use it with:

- [Mathematical Foundations](mathematical-foundations.md) for the compact
  theory.
- [Implementation Derivations](implementation-derivations.md) for the complete
  equation-to-source trace.
- [Run Everything](../run-everything.md) for commands that execute the package.
- [Equation-to-Code Walkthrough](../package-notebooks/08_equation_to_code_walkthrough.ipynb)
  for an executable notebook version.

<div class="silva-learning-grid" markdown>
<div class="silva-learning-card" markdown>
<strong>1. State</strong>
<span>Choose the tensor whose self-consistent value you want.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>2. Transition</strong>
<span>Write a shape-preserving map \(f_\theta(z,x)\).</span>
</div>
<div class="silva-learning-card" markdown>
<strong>3. Residual</strong>
<span>Move the equation to \(r=f(z,x)-z\).</span>
</div>
<div class="silva-learning-card" markdown>
<strong>4. Solver</strong>
<span>Pick Picard, Anderson, Broyden, or a package engine.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>5. Diagnose</strong>
<span>Record residuals, Jacobians, energy, and stability evidence.</span>
</div>
<div class="silva-learning-card" markdown>
<strong>6. Report</strong>
<span>State tensors, solver settings, citations, and data source.</span>
</div>
</div>

## The One Equation

A SILVA or DEQ layer is a state that agrees with its own update:

$$
z^\star=f_\theta(z^\star,x).
$$

That is the whole object. The rest of the package answers five practical
questions:

| Question | Equation | Package object |
| --- | --- | --- |
| what is the state? | \(z\in\mathcal Z\) | `z0`, solver state, model state |
| how does it update? | \(f_\theta(z,x)\) | `layer.f`, transition callable |
| how close is close enough? | \(\|f(z)-z\|\le\varepsilon\) | `SolverResult.residuals` |
| how is it solved? | \(z_{k+1}=T(z_k)\) | `fixed_point`, `silva_deq` |
| what is reportable? | residual, Jacobian, metric, citations | diagnostics and audit pages |

## Scalar Fixed Point

Start with one number:

$$
f(z)=az+m.
$$

The fixed point is

$$
z^\star=az^\star+m.
$$

Move all terms involving \(z^\star\) to the left:

$$
z^\star-az^\star=m.
$$

Factor:

$$
(1-a)z^\star=m.
$$

Solve:

$$
z^\star=\frac{m}{1-a},
\qquad
a\ne 1.
$$

The iteration

$$
z_{k+1}=az_k+m
$$

has error

$$
e_{k+1}
=
z_{k+1}-z^\star
=
a(z_k-z^\star)
=
ae_k.
$$

After \(k\) steps:

$$
e_k=a^k e_0.
$$

So the iteration converges when

$$
|a|<1.
$$

In code:

```python
import torch
from silva_networks import SolverConfig, fixed_point

a = 0.6
m = torch.tensor([2.0])
z0 = torch.zeros_like(m)
result = fixed_point(lambda z: a * z + m, z0, SolverConfig(alpha=1.0))
z_star = result.z
```

## Damping

When the raw transition is too aggressive, the package can damp it:

$$
T_\alpha(z)
=
(1-\alpha)z+\alpha f(z),
\qquad
0<\alpha\le 1.
$$

The fixed point is unchanged:

<div class="silva-derivation-step" markdown>

$$
z^\star=T_\alpha(z^\star)
$$

$$
z^\star=(1-\alpha)z^\star+\alpha f(z^\star)
$$

$$
\alpha z^\star=\alpha f(z^\star)
$$

$$
z^\star=f(z^\star).
$$

</div>

For the scalar affine map, the damped error multiplier is

$$
e_{k+1}
=
\left(1-\alpha+\alpha a\right)e_k.
$$

Thus damping converges locally when

$$
|1-\alpha+\alpha a|<1.
$$

This is the scalar version of the package diagnostic

$$
\rho\left((1-\alpha)I+\alpha J_f(z^\star)\right)<1.
$$

## Vector Tanh Fixed Point

Now let the state be a vector:

$$
z\in\mathbb R^d,
\qquad
x\in\mathbb R^p.
$$

The bridge transition is

$$
f_\theta(z,x)
=
\tanh(W_z z+W_x x+b).
$$

The equilibrium is

$$
z^\star
=
\tanh(W_z z^\star+W_xx+b).
$$

The Jacobian with respect to \(z\) is derived by the chain rule. Let

$$
u=W_z z+W_xx+b.
$$

Then

$$
\frac{\partial \tanh(u_i)}{\partial u_i}
=
1-\tanh^2(u_i).
$$

So

$$
J_f(z,x)
=
\operatorname{diag}(1-\tanh^2(u))W_z.
$$

Since \(0\le 1-\tanh^2(u_i)\le 1\),

$$
\|J_f(z,x)\|_2
\le
\|W_z\|_2.
$$

A conservative stability design is therefore

$$
\|W_z\|_2<1.
$$

Package objects:

```python
from silva_networks import SolverConfig, silva_fixed_point_block

block = silva_fixed_point_block(
    in_dim=4,
    state_dim=16,
    config=SolverConfig(solver="anderson", alpha=0.6, max_iter=20),
)
z_star = block(x)
```

## From Vector DEQ to SILVA Field

The vector DEQ hides all structure inside one map:

$$
f_\theta(z,x).
$$

SILVA opens the map into interpretable parts:

$$
f_\theta(z,x)
=
\Phi\left(
S_\theta(x)
+H_\theta(\chi(z))
+L_\theta(\chi(z),E,e)
+G_\theta(\chi(z),b)
\right).
$$

| Term | Derive it as | Think of it as |
| --- | --- | --- |
| \(S_\theta(x)\) | input projected into state width | stimulus |
| \(H_\theta(\chi(z))\) | optional state-only branch | self-interaction |
| \(L_\theta(\chi(z),E,e)\) | neighbor or local exchange | local field |
| \(G_\theta(\chi(z),b)\) | set/graph/sample context | global field |
| \(\Phi\) | activation and normalization | state proposal |

In `SILVALayer.f`, this is exactly:

```python
s = self.stimulus(x)
y = self.activation(z)
self_update = self.self_term(y, ...)
local = self.local(y, ...)
global_context = self.global_term(y, ...)
return self.norm(self.output_activation(s + self_update + local + global_context))
```

## Graph Local Branch

Let \(E\) contain directed edges \(j\to i\), with sources in
`edge_index[0]` and destinations in `edge_index[1]`.

Project each source state:

$$
m_j=W_\ell z_j.
$$

Aggregate incoming messages:

$$
\tilde L_i
=
\sum_{j:(j,i)\in E}m_j.
$$

Normalize by in-degree:

$$
L_i
=
\frac{\tilde L_i}{\max(1,d_i)},
\qquad
d_i=|\{j:(j,i)\in E\}|.
$$

The implementation uses `index_add_`:

```python
src, dst = edge_index
messages = proj(z)
out = torch.zeros_like(messages)
out.index_add_(0, dst, messages[src])
```

## Graph Attention Branch

For each node:

$$
h_i=Wz_i.
$$

For edge \(j\to i\):

$$
e_{ij}
=
\operatorname{LeakyReLU}(a_s^\top h_j+a_t^\top h_i).
$$

If edge attributes exist:

$$
e_{ij}
\leftarrow
e_{ij}+a_e^\top W_e e_{ij}^{attr}.
$$

Normalize over incoming edges:

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell:(\ell,i)\in E}\exp(e_{i\ell})}.
$$

Aggregate:

$$
L_i
=
\sum_{j:(j,i)\in E}\alpha_{ij}h_j.
$$

This is the local branch used by graph and molecular SILVA presets when
`graph_mode="GAT"` or bond-aware attention is selected.

## Global Context Branch

For a graph or set \(g\), compute

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i.
$$

The mean-field branch broadcasts

$$
G_i
=
W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

The gated branch computes a scalar gate:

$$
\beta_g
=
\sigma\left(
\frac{(W_q\bar z_g)^\top(W_k\bar z_g)}{\sqrt d}
\right),
$$

then broadcasts

$$
G_i
=
\beta_g W_v\bar z_{\operatorname{batch}(i)}.
$$

This gives every entity graph-scale information without mixing different
graphs in the same minibatch.

## Data to Equation

All graph-style cases enter the package as

$$
(x,\texttt{edge\_index},\texttt{edge\_attr},\texttt{batch},y).
$$

For tabular data:

$$
\tilde X_{ij}
=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

For a kNN graph:

$$
\mathcal N_k(i)
=
\operatorname*{arg\,topk}_{j\ne i}
\left(-d(\tilde x_i,\tilde x_j)\right),
$$

and edges are

$$
E=\{(j,i):j\in\mathcal N_k(i)\}.
$$

Package path:

```python
from silva_networks import load_tabular_dataset, tabular_to_silva_graph

dataset = load_tabular_dataset("iris", root="data")
graph = tabular_to_silva_graph(dataset, k=6, undirected=True)
graph.validate()
```

For images, the vector adapter computes

$$
x_b=\operatorname{vec}(\operatorname{image}_b).
$$

The pixel-graph adapter instead creates one entity per pixel and grid edges
between neighboring pixels.

## Diagnostics You Can Derive

The residual is

$$
r_k=f(z_k,x)-z_k.
$$

The reported scalar is

$$
\epsilon_k=\|r_k\|_2.
$$

The damped local Jacobian is

$$
J_{T_\alpha}
=
(1-\alpha)I+\alpha J_f.
$$

Local stability evidence is

$$
\rho(J_{T_\alpha})<1.
$$

Hutchinson's estimator avoids materializing \(J_f\):

$$
\mathbb E_v\|J_f^\top v\|_2^2
=
\|J_f\|_F^2.
$$

The package route:

```python
from silva_networks import damped_spectral_radius, hutchinson_jacobian_norm

rho = damped_spectral_radius(f, z_star, alpha=0.5)
penalty = hutchinson_jacobian_norm(f, z_star, samples=4)
```

## What to Report

Use this table when turning a derivation into an experiment.

| Item | Why it matters |
| --- | --- |
| state shape | proves the transition is shape-preserving |
| transition terms | identifies \(S,H,L,G,\Phi\) |
| solver | changes convergence path and citations |
| `alpha`, `tol`, `max_iter` | defines numerical approximation |
| residual curve | shows the fixed point was actually solved |
| Jacobian or spectral-radius evidence | supports local stability claims |
| dataset source | makes the data path reproducible |
| citations | separates SILVA contributions from inherited methods |

## Minimal End-to-End Derivation

1. Choose \(x\in\mathbb R^{N\times d_x}\).
2. Choose \(z\in\mathbb R^{N\times d_h}\).
3. Define branches \(S,H,L,G\) that return \(N\times d_h\).
4. Compose \(f_\theta(z,x)=\Phi(S+H+L+G)\).
5. Solve \(z^\star=f_\theta(z^\star,x)\).
6. Check \(\|f(z^\star,x)-z^\star\|_2\).
7. Read out \(\hat y=R_\phi(z^\star)\).
8. Report tensor shapes, solver settings, diagnostics, data, and citations.

## Sources and Executable Continuation

The fixed-point, solver, graph, attention, and SILVA sources used throughout
the workbook are collected in [Paper and References](../paper/references.md).
Run the same sequence in
[Equation-to-Code Walkthrough](../package-notebooks/08_equation_to_code_walkthrough.ipynb),
where every derived quantity is evaluated in a separate cell.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all transitions derived directly from implementation? | [Implementation Derivations](implementation-derivations.md) |
| Can I execute the derivation cell by cell? | [Equation-to-Code Walkthrough](../package-notebooks/08_equation_to_code_walkthrough.ipynb) |
| Which assumptions make the fixed point meaningful? | [Fixed Points](fixed-points.md) |
