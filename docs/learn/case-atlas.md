# SILVA Case Atlas

This atlas maps every SILVA case family in the package and companion book to
the equation, public API, input tensors, and diagnostics. It is the safest place
to decide what to run before opening a notebook or experiment config.

For a deeper equation-to-source audit of each case, see
[Implementation Derivations](implementation-derivations.md).
For a full method-to-paper audit, see
[Research Citation Audit](../research-citation-audit.md).

## Coverage Map

| Case | Status in package | Main API | Core tensors |
| --- | --- | --- | --- |
| Scalar and toy DEQ | Implemented | `fixed_point`, `DEQLayer`, educational NumPy helpers | `z0`, callable `f` |
| Generic entity SILVA | Implemented | `SILVALayer`, `SILVAStack` | `x`, optional `edge_index`, `edge_attr`, `batch` |
| Graph/node SILVA | Implemented | `SILVAGraphLayer`, `SILVAGraphNetwork`, `SILVAGraphPresetNetwork` | `x`, `edge_index`, optional `batch` |
| Graph-level prediction | Implemented | `SILVAGraphNetwork(task="graph")`, `pool_entities` | `x`, `edge_index`, `batch`, `y` |
| Vision vector SILVA | Implemented | `SILVAVisionVectorLayer`, `SILVAVisionVectorClassifier` | image vectors or flattened image tensors |
| Convolutional vision SILVA | Implemented | `SILVAConvStem`, `SILVAConvVisionClassifier` | `(batch, channels, height, width)` |
| Molecular SILVA | Implemented | `SILVAMolecularLayer`, `SILVAMolecularRegressor` | atom features, bond edges, bond features, molecule batch |
| Dataset adaptation | Implemented | `GraphTensorBatch`, `tabular_to_silva_graph`, image and molecular adapters | `x`, `edge_index`, `edge_attr`, `batch`, `y` |
| Diagnostics and failure modes | Implemented | `residual_curve`, `stability_report`, `solve_with_energy` | transition `f`, state `z`, optional energy |
| General DEQ engine | Implemented | `SILVADEQEngine`, `silva_deq`, `pack_state`, `unpack_state` | tensor, tuple, or list state |
| SILVA DEQ flow | Implemented | `SILVADEQFlow`, `silva_deq_flow`, `silva_flow_warp`, `silva_all_pairs_correlation` | image pair, flow field, validity mask |
| Path sums and interaction histories | Documented and notebook-facing | solvers, Jacobian helpers, examples | linearized transitions |
| PDE, homotopy, diffusion, scientific, distributional, algorithmic, quantum cases | Book extension material | notebooks and user-defined `DEQLayer`/`SILVALayer` | custom states and residuals |

## Citation Map by Case

| Case | Cite |
| --- | --- |
| Scalar and toy DEQ | SILVA package; [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377); [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/) |
| Generic entity SILVA | SILVA paper/package; DEQ for the equilibrium-layer framing |
| Graph/node SILVA | SILVA; DEQ; [GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903), or [MPNN](https://arxiv.org/abs/1704.01212) depending on the local branch |
| Graph-level prediction | SILVA; Deep Sets for mean/sum/max set pooling when discussing permutation-invariant graph readouts |
| Vision vector SILVA | SILVA; [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for channel attention; [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) as related dynamic-kNN literature |
| Convolutional vision SILVA | SILVA; cite the convolutional benchmark/dataset used; attention/dynamic-kNN papers if those branches are discussed |
| Molecular SILVA | SILVA; [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212); [Graph Attention Networks](https://arxiv.org/abs/1710.10903); dataset or molecular benchmark source |
| Dataset adaptation | dataset source; SILVA package for the adapter; Dynamic Graph CNN when reporting kNN graph construction as a dynamic graph method |
| Jacobian regularization / stability | Hutchinson trace estimator; [Jacobian-regularized DEQ](https://arxiv.org/abs/2106.14342); DEQ/implicit-layer sources |
| ODE / optimization / MDEQ bridge | [Neural ODEs](https://arxiv.org/abs/1806.07366), [OptNet](https://arxiv.org/abs/1703.00443), [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430), [MDEQ](https://arxiv.org/abs/2006.08656), as applicable |
| General DEQ engine | SILVA package; [TorchDEQ](https://github.com/locuslab/torchdeq); DEQ |
| Optical flow DEQ | [RAFT](https://arxiv.org/abs/2003.12039); [Deep Equilibrium Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_Deep_Equilibrium_Optical_Flow_Estimation_CVPR_2022_paper.html); SILVA package |

## Universal Contract

Every implemented case reduces to

$$
z^\star=f_\theta(z^\star,x),
\qquad
r_\theta(z,x)=f_\theta(z,x)-z.
$$

The executed damped solver step is

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x)
=
z_k+\alpha r_\theta(z_k,x).
$$

The package's structured SILVA field is

$$
f_\theta(z,x)
=
\Phi
\left(
S_\theta(x)
+H_\theta(\chi(z))
+L_\theta(\chi(z),E,e)
+G_\theta(\chi(z),b)
\right).
$$

The branch meanings are:

| Branch | Role | Typical implementation |
| --- | --- | --- |
| \(S_\theta\) | inject external stimulus into the recurrent state dimension | affine map, convolutional stem, atom embedding |
| \(H_\theta\) | optional learned self-interaction inside the transition | `SelfInteraction`, `IdentityTerm`, custom module |
| \(L_\theta\) | local exchange between nearby entities | graph aggregation, GAT, kNN, channel kNN |
| \(G_\theta\) | global context shared inside a set, graph, molecule, or sample | mean field, gated mean, top-k attention, channel attention |
| \(\Phi\) | output block for the next recurrent state | `tanh` plus norm, `LayerNorm(ReLU(...))`, or case-specific block |

## Scalar and Toy DEQ Case

Use this case to verify the mechanics before adding graph structure.

$$
f(z)=\tanh(Wz+s),
\qquad
z^\star=f(z^\star).
$$

The Jacobian is

$$
J_f(z)
=
\operatorname{diag}\left(1-\tanh^2(Wz+s)\right)W.
$$

With a scalar affine map \(f(z)=az+m\), the exact equilibrium is

$$
z^\star=\frac{m}{1-a},
\qquad |a|<1.
$$

Use:

```python
from silva_networks import SolverConfig, fixed_point

result = fixed_point(f, z0, SolverConfig(alpha=0.5, max_iter=25))
```

Evidence to record:

| Diagnostic | Equation | API |
| --- | --- | --- |
| residual | \(\|f(z_k)-z_k\|_2\) | `result.residuals` |
| Jacobian | \(J_f(z^\star)\) | `full_jacobian` |
| local stability | \(\rho((1-\alpha)I+\alpha J_f)\) | `damped_spectral_radius` |

## Generic Entity SILVA Case

`SILVALayer` treats rows as exchangeable entities:

$$
x\in\mathbb R^{N\times d_{in}},
\qquad
z\in\mathbb R^{N\times d_h}.
$$

The default transition is

$$
f_\theta(Z,X)
=
\operatorname{LayerNorm}
\left[
\tanh
\left(
XW_s^\top+b_s
+H_\theta(\tanh Z)
+L_\theta(\tanh Z,E,e)
+G_\theta(\tanh Z,b)
\right)
\right].
$$

Use this case when your entities are neither image pixels nor molecules but can
still be represented as rows.

```python
from silva_networks import SILVALayer, SolverConfig

layer = SILVALayer(
    in_dim=features,
    hidden_dim=64,
    local="topk",
    global_term="simple",
    self_term="linear",
    local_kwargs={"k": 8},
    config=SolverConfig(solver="anderson", alpha=0.4, max_iter=15),
)
```

## Graph and Node Case

Graph SILVA has one row per node:

$$
X\in\mathbb R^{|\mathcal V|\times d_x},
\qquad
E\subseteq \mathcal V\times\mathcal V.
$$

The reference graph layer computes

$$
y_k=\tanh(z_k),
$$

$$
f_\theta(z_k,x)
=
\operatorname{LayerNorm}
\left[
\operatorname{ReLU}
\left(
W_{\rm stim}x
+L_\theta(y_k,E)
+G_\theta(y_k,b)
\right)
\right].
$$

The local mean branch is

$$
L_i(Y,E)
=
\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_{j\in\mathcal N(i)}W_\ell y_j,
\qquad
\mathcal N(i)=\{j:(j,i)\in E\}.
$$

The graph-attention branch is

$$
h_i=W y_i,
\qquad
e_{ij}
=
\operatorname{LeakyReLU}
\left(a_s^\top h_j+a_t^\top h_i\right),
$$

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell\in\mathcal N(i)}\exp(e_{i\ell})},
\qquad
L_i(Y,E)=\sum_{j\in\mathcal N(i)}\alpha_{ij}h_j.
$$

Use:

```python
from silva_networks import SILVAGraphPresetNetwork

model = SILVAGraphPresetNetwork(
    in_dim=num_features,
    hidden_dim=[64, 48],
    out_dim=num_classes,
    task="node",
    graph_mode="GAT",
    attention_mode="simple",
    stack_alphas=[0.5, 0.2],
)
```

## Graph-Level Prediction Case

Graph prediction uses the same equilibrium state, then pools node states:

$$
h_g
=
\operatorname{pool}_{i\in\mathcal V_g}z_i^\star,
\qquad
\hat y_g=R_\phi(h_g).
$$

Mean pooling is

$$
h_g=\frac{1}{|\mathcal V_g|}\sum_{i\in\mathcal V_g}z_i^\star.
$$

Use:

```python
model = SILVAGraphNetwork(
    in_dim=num_features,
    hidden_dims=[64, 64],
    out_dim=num_targets,
    task="graph",
    pooling="mean",
)
```

## Global Context Cases

Mean-field global context is the simplest permutation-invariant case:

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i,
\qquad
G_i(Z)=W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

The gated SILVA-style variant is

$$
\beta_g
=
\sigma
\left(
\frac{(W_q\bar z_g)^\top(W_k\bar z_g)}{\sqrt d}
\right),
\qquad
G_i(Z)=\beta_g W_v\bar z_{\operatorname{batch}(i)}.
$$

Top-k global attention restores receiver-specific global context:

$$
s_{ij}
=
\frac{(W_qz_i)^\top(W_kz_j)}{\sqrt d},
\qquad
\mathcal T_k(i)=\operatorname*{arg\,topk}_j s_{ij},
$$

$$
G_i(Z)
=
\sum_{j\in\mathcal T_k(i)}
\operatorname{softmax}_j(s_{ij})W_vz_j.
$$

## Vision Vector Case

Vector vision treats hidden channels as the interacting entities inside each
sample. For a batch of flattened inputs,

$$
x_b\in\mathbb R^{d_x},
\qquad
z_b\in\mathbb R^{d_h}.
$$

The transition is

$$
f_\theta(z_b,x_b)
=
W_{\rm stim}x_b
+L_{\rm ch}(\tanh z_b)
+G_{\rm ch}(\tanh z_b).
$$

Dynamic channel local interaction builds a kNN graph over channel values:

$$
\mathcal N_k(i,b)
=
\operatorname*{arg\,topk}_{j\ne i}
\left(-|z_{b,i}-z_{b,j}|^2\right),
$$

$$
L_{{\rm ch},i}(z_b)
=
\frac{1}{|\mathcal N_k(i,b)|}
\sum_{j\in\mathcal N_k(i,b)}[W_\ell z_b]_j.
$$

Per-sample channel attention computes

$$
q_b=W_qz_b,
\qquad
k_b=W_kz_b,
\qquad
A_b=\operatorname{softmax}
\left(
\frac{q_bk_b^\top}{\sqrt{d_h}}
\right),
$$

$$
G_{\rm ch}(z_b)=z_bA_b.
$$

No information crosses from one image in the batch to another image.

Use:

```python
from silva_networks import SILVAVisionVectorClassifier

model = SILVAVisionVectorClassifier(
    in_dim=28 * 28,
    hidden_dim=[128, 64],
    num_classes=10,
    attention_mode="simple",
    graph_mode="knn",
    alphas=(0.5, 0.2),
)
```

## Convolutional Vision Case

The convolutional case first extracts a vector stimulus:

$$
u_b=C_\psi(x_b),
$$

then sends it to the same vector SILVA equilibrium stack:

$$
z_\ell^\star
=
f_{\theta_\ell}(z_\ell^\star,\tanh z_{\ell-1}^\star),
\qquad
\ell=1,\dots,K.
$$

Use:

```python
from silva_networks import SILVAConvVisionClassifier

model = SILVAConvVisionClassifier(
    in_channels=3,
    image_size=32,
    hidden_dim=[128, 64],
    num_classes=10,
    alphas=(0.5, 0.2),
)
```

## Molecular Case

Molecular SILVA has atoms as entities, bonds as edges, and molecules as batch
groups:

$$
x_i=E_{\rm atom}(a_i)
\quad\text{or}\quad
x_i=W_{\rm atom}u_i,
$$

$$
e_{ij}=E_{\rm bond}(b_{ij})
\quad\text{or}\quad
e_{ij}=W_{\rm bond}v_{ij}.
$$

Each layer computes

$$
f_\theta(z,x,E,e,b)
=
\operatorname{LayerNorm}
\left[
\operatorname{ReLU}
\left(
W_{\rm stim}x
+L_{\rm bondGAT}(z,E,e)
+W_g\bar z_{\operatorname{batch}(i)}
\right)
\right].
$$

The molecule state is pooled for regression:

$$
h_m=\frac{1}{|\mathcal A_m|}\sum_{i\in\mathcal A_m}z_i^\star,
\qquad
\hat y_m=R_\phi(h_m).
$$

Use:

```python
from silva_networks import SILVAMolecularRegressor

model = SILVAMolecularRegressor(
    hidden_dim=[128, 64],
    atom_feature_dim=atom_features.shape[1],
    bond_feature_dim=bond_features.shape[1],
    alphas=(0.5, 0.2),
)
```

## Dataset Adaptation Cases

The package has one tensor contract:

$$
(x,\texttt{edge\_index},\texttt{edge\_attr},\texttt{batch},y).
$$

Tabular data becomes a sample graph by standardizing features and connecting
nearest neighbors:

$$
\tilde X_{ij}
=
\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)},
$$

$$
E
=
\{(j,i):j\in\operatorname*{arg\,topk}_{\ell\ne i}
(-d(\tilde x_i,\tilde x_\ell))\}.
$$

Pixel graph images use grid neighbors. Molecules preserve bond edges. PyG-like
objects are represented as `GraphTensorBatch` values without requiring PyTorch
Geometric as a package dependency.

## Path-Sum and Linear Response Case

Near an equilibrium, write the damped linearized update as

$$
\delta z_{k+1}=T\delta z_k+\alpha s.
$$

Repeated substitution gives the finite response

$$
\delta z_K
=
\alpha\sum_{t=0}^{K-1}T^t s
+T^K\delta z_0.
$$

When \(\rho(T)<1\),

$$
\lim_{K\to\infty}\alpha\sum_{t=0}^{K-1}T^t s
=
\alpha(I-T)^{-1}s.
$$

This is the "interaction history" view: local and global operators contribute
through repeated powers of the linearized transition.

## Diagnostics and Failure Cases

A result is not trustworthy from task accuracy alone. Record:

| Quantity | Equation | API |
| --- | --- | --- |
| residual | \(\|f(z_K)-z_K\|_2\) | `SolverResult.residual` |
| residual curve | \((\|r_1\|,\dots,\|r_K\|)\) | `residual_curve` |
| damped radius | \(\rho((1-\alpha)I+\alpha J_f)\) | `damped_spectral_radius` |
| Jacobian norm | \(\|J_f\|_F\) estimate | `hutchinson_jacobian_norm` |
| energy trend | \(E_{k+1}-E_k\) | `solve_with_energy`, `energy_deltas` |
| descent share | fraction of nonincreasing energy steps | `descent_fraction` |

Common failure modes:

| Symptom | Likely issue | First check |
| --- | --- | --- |
| Residual plateaus high | solver budget too small or map too expansive | reduce `alpha`, increase `max_iter`, inspect \(\rho\) |
| Residual oscillates | damping too aggressive | lower `alpha` or try Anderson with ridge |
| Full Jacobian is too large | state is not a toy state | switch to VJP/JVP diagnostics |
| Batch leakage | global term ignored `batch` | validate `batch` and use package global operators |
| Edge mismatch | `edge_attr` rows do not match edges | `GraphTensorBatch.validate()` |

## Book Extension Cases

The companion book includes additional research routes. The package can host
them through custom `DEQLayer` or custom SILVA operators, but they are not
advertised as prebuilt classes unless a public API exists.

| Book route | Fixed-point form | Package entry point |
| --- | --- | --- |
| Neural operators and PDEs | \(u^\star=\mathcal T_\theta(u^\star, a)\) | custom `DEQLayer` |
| Homotopy and continuation | \(F(z^\star(t),t)=0\) | continuation loop around `fixed_point` |
| TorchDEQ and DeltaDEQ engineering | solver interface and heterogeneous convergence | compare against `SolverConfig` and solver traces |
| Certified and Lipschitz DEQs | \(\|(I-J_f)^{-1}\|\) sensitivity bounds | `jvp`, `vjp`, spectral diagnostics |
| Score and diffusion equilibria | \(s^\star=f_\theta(s^\star,x,t)\) | custom transition |
| Scientific self-consistency | \(H^\star=\mathcal H_\theta(H^\star,x)\) | custom transition plus diagnostics |
| Recent theory and finite-solve bias | \(z_K-z^\star\) and local claims | residual curves and stability reports |
| Distributional DEQs | empirical measure or particle fixed point | `SILVALayer` with permutation-equivariant branches |
| Algorithmic and quantum reasoning | Bellman or circuit self-consistency | custom `DEQLayer` |

The rule for all extension cases is the same: state the residual, show the
solver trace, inspect the linearization, and make only the claim supported by
those diagnostics.
