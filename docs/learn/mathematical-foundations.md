# Mathematical Foundations

SILVA Networks are fixed-point models with a structured interaction field. This
page collects the derivations that connect the package API to the math.

For the implementation-level trace from each symbol to the package classes,
solver updates, diagnostics, and reference cases, read
[Implementation Derivations](implementation-derivations.md) after this page.

For method citations and claim-level citation rules, read the
[Research Citation Audit](../research-citation-audit.md).

## Research Lineage

| Topic in this page | Literature to cite |
| --- | --- |
| equilibrium layer \(z^\star=f_\theta(z^\star,x)\) | Bai, Kolter, and Koltun, [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377); Deep Implicit Layers tutorial |
| implicit differentiation / adjoint solve | [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/); DEQ |
| Anderson acceleration | Anderson, [1965](https://doi.org/10.1145/321296.321305); Walker and Ni, [2011](https://doi.org/10.1137/10078356X) |
| Broyden update | Broyden, [1965](https://doi.org/10.1090/S0025-5718-1965-0198670-6) |
| GMRES adjoint linear solve | Saad and Schultz, [1986](https://doi.org/10.1137/0907058) |
| graph local / graph attention terms | Kipf and Welling, [GCN](https://arxiv.org/abs/1609.02907); Velickovic et al., [GAT](https://arxiv.org/abs/1710.10903); Gilmer et al., [MPNN](https://arxiv.org/abs/1704.01212) |
| global set pooling and attention | Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114); Vaswani et al., [Attention](https://arxiv.org/abs/1706.03762); Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825) |

## Symbols

| Symbol | Meaning | Package object |
| --- | --- | --- |
| \(x\) | External input features | `x` |
| \(z_k\) | Recurrent state at solver step \(k\) | internal solver state |
| \(z^\star\) | Equilibrium state | `SolverResult.z` |
| \(S_\theta\) | Stimulus encoder | `StimulusEncoder`, `input_injection` |
| \(H_\theta\) | Optional learned self branch | `self_term` |
| \(L_\theta\) | Local interaction | `local` |
| \(G_\theta\) | Global interaction | `global_term` |
| \(\alpha\) | Damping | `SolverConfig.alpha` |
| \(r(z)\) | Fixed-point residual | `result.residuals` |

## From Infinite Depth to One Equation

An explicit weight-tied network repeats the same transition:

$$
z_{k+1}=f_\theta(z_k,x).
$$

If the sequence converges, its limit satisfies

$$
z^\star
=
\lim_{k\to\infty}z_k
=
\lim_{k\to\infty}f_\theta(z_k,x)
=
f_\theta(z^\star,x),
$$

assuming \(f_\theta\) is continuous near the limit. The layer can therefore be
defined by the residual equation

$$
F_\theta(z,x)=f_\theta(z,x)-z=0.
$$

SILVA chooses a structured \(f_\theta\):

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

The package keeps each term independently replaceable while the solver only
needs the callable \(z\mapsto f_\theta(z,x)\).

This infinite-depth/equilibrium viewpoint follows the DEQ literature, while
the \(S_\theta+H_\theta+L_\theta+G_\theta\) decomposition is the SILVA
structured interaction field.

## Damping and Local Stability

The executed Picard map is often damped:

$$
T_\alpha(z)
=(1-\alpha)z+\alpha f_\theta(z,x).
$$

The equilibrium is unchanged because

$$
z^\star=T_\alpha(z^\star)
\iff
z^\star=(1-\alpha)z^\star+\alpha f_\theta(z^\star,x)
\iff
z^\star=f_\theta(z^\star,x).
$$

Linearize \(T_\alpha\) around \(z^\star\). With an error
\(e_k=z_k-z^\star\),

$$
e_{k+1}
\approx
J_{T_\alpha}(z^\star)e_k,
$$

where

$$
J_{T_\alpha}(z^\star)
=
(1-\alpha)I+\alpha J_f(z^\star).
$$

If

$$
\rho(J_{T_\alpha})<1,
$$

the linearized dynamics contract. This is exactly the quantity estimated by
`damped_spectral_radius`.

## Banach Fixed-Point Lens

If \(f\) is a contraction on a complete metric space,

$$
\|f(z)-f(y)\|\le q\|z-y\|,
\qquad 0\le q<1,
$$

then the fixed point exists, is unique, and Picard iteration converges. In
finite-dimensional differentiable settings, a sufficient local check is

$$
\|J_f(z)\| < 1
$$

in an operator norm on the neighborhood being inspected. SILVA diagnostics do
not claim this condition globally; they provide local evidence around the
computed state.

## Implicit Differentiation

Let the solved state satisfy

$$
F(z^\star,\theta,x)=f_\theta(z^\star,x)-z^\star=0.
$$

Differentiate with respect to a parameter block \(\theta\):

$$
\frac{\partial F}{\partial z}
\frac{dz^\star}{d\theta}
+
\frac{\partial F}{\partial \theta}
=0.
$$

Because

$$
\frac{\partial F}{\partial z}=J_f-I,
\qquad
\frac{\partial F}{\partial \theta}
=
\frac{\partial f_\theta}{\partial\theta},
$$

the derivative equation becomes

$$
(I-J_f)
\frac{dz^\star}{d\theta}
=
\frac{\partial f_\theta}{\partial\theta}.
$$

For a scalar loss \(\mathcal L(z^\star)\), define the adjoint vector \(\lambda\)
by

$$
(I-J_f^\top)\lambda
=
\frac{\partial\mathcal L}{\partial z^\star}.
$$

Then

$$
\frac{\partial\mathcal L}{\partial\theta}
=
\lambda^\top
\frac{\partial f_\theta}{\partial\theta}.
$$

`implicit_adjoint_solve` exposes this linear solve for diagnostics. Public
training still works as ordinary PyTorch code, differentiating through the
finite solver steps.

This derivation is the same implicit-function theorem route used in DEQ and
Deep Implicit Layers tutorials. Cite those sources whenever a result relies on
the adjoint equation rather than ordinary finite unrolling.

## Anderson Acceleration as Constrained Residual Minimization

Let recent residuals be

$$
r_i=f(z_i)-z_i.
$$

Collect them as columns:

$$
G_k=[r_{k-m+1},\dots,r_k].
$$

Anderson acceleration chooses coefficients \(c\) that reduce the mixed residual:

$$
\min_c \|G_kc\|_2^2+\lambda\|c\|_2^2
\quad
\text{subject to}
\quad
\mathbf 1^\top c=1.
$$

The KKT conditions are

$$
\begin{bmatrix}
G_k^\top G_k+\lambda I & \mathbf 1\\
\mathbf 1^\top & 0
\end{bmatrix}
\begin{bmatrix}
c\\
\nu
\end{bmatrix}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}.
$$

The implementation solves this system and mixes recent transition outputs.
`history`, `ridge`, and `beta` control the memory, regularization, and mixing.

Use Anderson's original fixed-point acceleration paper and Walker-Ni's modern
analysis when reporting this solver.

## Broyden as an Inverse Secant Update

Broyden uses the root form \(F(z)=0\). With an inverse-Jacobian estimate
\(B_k\), the step is

$$
s_k=-\alpha B_kF(z_k),
\qquad
z_{k+1}=z_k+s_k.
$$

The secant condition asks the next inverse estimate to satisfy

$$
B_{k+1}y_k=s_k,
\qquad
y_k=F(z_{k+1})-F(z_k).
$$

The compact good-Broyden update used here is

$$
B_{k+1}
=
B_k
+
\frac{(s_k-B_ky_k)(s_k^\top B_k)}
{s_k^\top B_ky_k}.
$$

Because this educational implementation stores a dense inverse estimate, it is
best for small states and controlled experiments.

Use Broyden's 1965 secant-method paper when reporting this solver.

## Graph Local Term

Let \(E\) contain directed edges \(j\to i\). The mean local branch is

$$
L_i(Z,E)
=
\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_{j\in\mathcal N(i)}W_\ell z_j,
\qquad
\mathcal N(i)=\{j:(j,i)\in E\}.
$$

This is permutation-equivariant: relabeling nodes relabels the output in the
same way, provided `edge_index` and `batch` are relabeled consistently.

Use GCN, MPNN, or general GNN references when describing this term as graph
message passing.

## Graph Attention Term

For graph attention, project each state:

$$
h_i=Wz_i.
$$

For an edge \(j\to i\),

$$
e_{ij}
=
\operatorname{LeakyReLU}
\left(
a_s^\top h_j+a_t^\top h_i+a_e^\top W_e e_{ij}^{attr}
\right),
$$

with the edge-attribute term omitted when no `edge_attr` is supplied. Incoming
normalization gives

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell\in\mathcal N(i)}\exp(e_{i\ell})},
$$

and the local update is

$$
L_i(Z,E)
=
\sum_{j\in\mathcal N(i)}\alpha_{ij}h_j.
$$

Use the GAT paper for the graph-local attention mechanism and the Transformer
paper when discussing the scaled dot-product attention pattern.

## Global Mean Field and Gated Context

For graph \(g\),

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i.
$$

The static mean-field branch broadcasts

$$
G_i=W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

The gated branch first computes

$$
\beta_g
=
\sigma\left(
\frac{(W_q\bar z_g)^\top(W_k\bar z_g)}{\sqrt d}
\right),
$$

then broadcasts

$$
G_i=\beta_g W_v\bar z_{\operatorname{batch}(i)}.
$$

This gives every node access to graph-scale context while preserving
per-minibatch graph separation.

Use Deep Sets for the permutation-invariant mean/set pooling lens. Use the
SILVA citation for the specific gated global-context field.

## Complexity Checklist

Let \(N\) be the number of entities, \(E\) the number of edges, \(d\) the state
dimension, \(K\) the solver iteration budget, and \(m\) the Anderson history.

| Component | Typical memory | Typical work per solver step |
| --- | --- | --- |
| `GraphLocal` | \(O(Nd+E)\) | \(O(Ed)\) |
| `GraphAttentionLocal` | \(O(Nd+Eh)\) | \(O(Ed)\) plus segment softmax |
| `MeanFieldGlobal` | \(O(Nd)\) | \(O(Nd)\) |
| `TopKGlobalAttention` | \(O(N^2)\) scores per graph | \(O(N^2d)\) before top-k |
| Picard | \(O(Nd)\) | one transition call |
| Anderson | \(O(mNd+m^2)\) | one transition plus small KKT solve |
| Broyden | \(O((Nd)^2)\) | dense inverse update |
| Full Jacobian | \(O((Nd)^2)\) | many autograd calls |
| VJP/JVP | \(O(Nd)\) activation-dependent | one product call |

## Practical Reading Rule

Use the fixed-point residual to decide whether the solve finished, the damped
spectral radius to inspect local stability, the tensor contract to validate
dataset conversion, and the operator tables to explain what a model is allowed
to claim.

```python
import torch
from silva_networks import SolverConfig, fixed_point, stability_report

W = torch.tensor([[0.2, 0.1], [0.0, 0.3]])
b = torch.tensor([0.1, -0.2])
f = lambda z: torch.tanh(W @ z + b)

result = fixed_point(f, torch.zeros(2), SolverConfig(max_iter=30, tol=1e-7))
report = stability_report(f, result.z, samples=4, iters=12)
print(result.z.shape, result.residual, report.spectral_radius)
```
