# Advanced Equilibrium Families Inside SILVA

SILVA provides one language for source injection, state interaction, operator
structure, fixed-point solving, and readout. This page develops three
mechanisms within that language: monotone graph equilibria, one-time-injected
equilibrium transformers, and positive Poisson mirror equilibria. Each remains
an instance of

$$
z^\star=T_\theta(z^\star;x),
\qquad
\widehat y=Q_\psi(z^\star).
$$

The implementations preserve the defining mechanisms of their primary sources
[[47]](../paper/references.md#ref-47){ .silva-cite }
[[48]](../paper/references.md#ref-48){ .silva-cite }
[[50]](../paper/references.md#ref-50){ .silva-cite } while exposing shared SILVA
solver and diagnostic contracts. Compact defaults support instruction and
small-scale validation; they do not replace the datasets, widths, schedules,
or metrics of a full paper reproduction.

## Architecture Map

| SILVA family | State | Source | Reused transition | Essential constraint |
| --- | --- | --- | --- | --- |
| `silva_monotone_graph_equilibrium` | node matrix | node features | forward-backward graph step | monotone channel parameterization |
| `silva_generative_equilibrium_transformer` | patch tokens | one-time QKV injection | injected attention and FFN blocks | stable token fixed point |
| `silva_poisson_mirror_equilibrium` | positive image | Poisson observation | Burg mirror-descent step | positive Bregman geometry |

The aliases `mignn`, `get`, and `deq_md` resolve to these canonical SILVA
constructors.

## Monotone Graph Equilibrium

### Graph operator

For graph adjacency $A$, degree matrix $D$, and node state $Z$, define

$$
G=\frac12\left(I-D^{-1/2}AD^{-1/2}\right).
$$

The factor $1/2$ places the spectrum of a symmetric normalized Laplacian in
$[0,1]$. The implicit graph equation is

$$
Z^\star=\sigma\left(WGZ^\star+B(X)\right).
$$

$B(X)$ is the SILVA source branch. $GZ$ communicates over graph edges, and $W$
mixes state channels after local propagation.

### Monotone parameterization

An unconstrained $W$ can produce an unstable or ill-posed fixed point. The
monotone construction uses

$$
W=(1-m)I-CC^T+F-F^T,
\qquad m>0.
$$

The symmetric part gives

$$
\begin{aligned}
I-\frac{W+W^T}{2}
&=I-\left[(1-m)I-CC^T\right]\\
&=mI+CC^T\\
&\succeq mI.
\end{aligned}
$$

The skew term $F-F^T$ changes channel dynamics without weakening this
certificate. `monotonicity_certificate()` computes the smallest eigenvalue of
the left side, following the monotone operator viewpoint of Baker et al.
[[47]](../paper/references.md#ref-47){ .silva-cite }.

### Forward-backward step

One operator-splitting step is

$$
Z^{k+1}
=\operatorname{prox}_{\alpha f}
\left((1-\alpha)Z^k+\alpha[WGZ^k+B(X)]\right).
$$

For the nonnegative orthant, the proximal map is ReLU. Other proximal maps can
be supplied through the transition activation.

```python
from silva_networks import (
    SILVAMonotoneGraphEquilibrium,
    SolverConfig,
    make_monotone_chain_dataset,
)

data = make_monotone_chain_dataset(nodes=16, diffusion=0.5, seed=47)
model = SILVAMonotoneGraphEquilibrium(
    in_dim=1,
    state_dim=8,
    out_dim=1,
    margin=0.1,
    step_size=0.8,
    config=SolverConfig(solver="anderson", max_iter=40, tol=1e-6),
)
result = model(data.source, data.edge_index, return_result=True)
print(result.monotonicity_certificate)
print(result.solver_result.residual)
```

For node permutation $P$, relabeling features and graph connectivity together
gives

$$
T(PZ;PX,PEP^T)=PT(Z;X,E).
$$

The test suite verifies this equivariance contract.

## Generative Equilibrium Transformer

### One-time injection

For patch size $p$, an image becomes a token sequence:

$$
x\in\mathbb R^{B\times C\times H\times W}
\longmapsto
X_p\in\mathbb R^{B\times N\times d},
\qquad N=\frac Hp\frac Wp.
$$

The finite injection path computes

$$
U=I_\phi(X_p+P_{2D}),
$$

then projects $U$ into one QKV injection for each of $L$ equilibrium blocks:

$$
(U_1,\ldots,U_L)=W_UI_\phi(X_p+P_{2D}),
\qquad
U_\ell\in\mathbb R^{B\times N\times3d}.
$$

This is the source branch. It is evaluated once per model call, outside the
root solver.

### QKV-injected fixed point

At block $\ell$,

$$
Z W_{qkv}^{(\ell)}+U_\ell+C_y
=\left[Q_\ell,K_\ell,V_\ell\right],
$$

$$
A_\ell
=\operatorname{softmax}
\left(\frac{Q_\ell K_\ell^T}{\sqrt{d_h}}\right)V_\ell.
$$

The compact SILVA block evaluates

$$
\widetilde Z_\ell=Z_\ell+A_\ell,
$$

$$
Z_{\ell+1}
=\tanh\left(s[\widetilde Z_\ell+
\operatorname{FFN}_\ell(\widetilde Z_\ell)]\right).
$$

Composing the blocks gives $T_\theta$, and SILVA solves

$$
Z^\star=T_\theta(Z^\star;U,C_y).
$$

The bounded scaling $s$ is the compact implementation's stability envelope.
The one-time QKV-injection mechanism follows Geng, Pokle, and Kolter
[[48]](../paper/references.md#ref-48){ .silva-cite }.

The decoder reverses patching. Teacher matching uses

$$
\mathcal L_{\mathrm{distill}}
=\frac1{BCHW}
\|Q_\psi(Z^\star)-x_{\mathrm{teacher}}\|_2^2.
$$

The equilibrium residual and distillation loss answer different questions.

```python
from silva_networks import (
    SILVAGenerativeEquilibriumTransformer,
    make_teacher_image_pairs,
    silva_distillation_loss,
)

data = make_teacher_image_pairs(samples=4, height=8, width=8, seed=48)
model = SILVAGenerativeEquilibriumTransformer(
    in_channels=1,
    patch_size=2,
    hidden_dim=16,
    heads=4,
    injection_depth=1,
    equilibrium_depth=2,
)
result = model(data.noise, return_result=True)
loss = silva_distillation_loss(result.output, data.target)
print(loss, result.solver_result.residual)
```

The generated pairs validate the architecture and gradient path. A diffusion
distillation reproduction requires the exact teacher, noise schedule, class
conditioning, preprocessing, optimizer, and evaluation protocol.

## Poisson Mirror Equilibrium

### Data fidelity

For nonnegative image $x$, sensing operator $A$, and observed counts $y$,

$$
y_i\sim\operatorname{Poisson}((Ax)_i).
$$

The generalized KL fidelity and gradient are

$$
D_{\mathrm{KL}}(y,Ax)
=\sum_i y_i\log\frac{y_i}{(Ax)_i}+(Ax)_i-y_i,
$$

$$
\nabla_xD_{\mathrm{KL}}(y,Ax)
=A^T\left(1-\frac{y}{Ax}\right).
$$

The implementation accepts both forward and adjoint callables. Verify

$$
\langle Au,v\rangle=\langle u,A^Tv\rangle
$$

before interpreting a reconstruction.

### Burg update

The Burg entropy is

$$
h(x)=-\sum_i\log x_i,
\qquad
\nabla h(x)=-x^{-1}.
$$

Substituting its mirror map gives

$$
x^+
=\frac{x}{1+\tau x\odot
\left[A^T\left(1-\frac{y}{Ax}\right)+r_\theta(x)\right]}.
$$

`SILVABurgMirrorTransition` applies this expression and a positive box
projection. Reusing it defines

$$
x^\star=T_{\mathrm{Burg},\theta}(x^\star;y).
$$

This translation follows the mirror-descent equilibrium of Daniele et al.
[[50]](../paper/references.md#ref-50){ .silva-cite }.

```python
from silva_networks import (
    SILVABurgMirrorTransition,
    SILVAPoissonMirrorEquilibrium,
    make_poisson_inverse_dataset,
)

data = make_poisson_inverse_dataset(samples=4, height=8, width=8, seed=50)
transition = SILVABurgMirrorTransition(
    forward_operator=data.forward_operator,
    adjoint_operator=data.adjoint_operator,
    step_size=0.05,
    maximum=3.0,
)
result = SILVAPoissonMirrorEquilibrium(transition=transition)(
    data.observation,
    return_result=True,
)
print(result.output.min())
print(data.data_fidelity(result.output))
print(result.solver_result.residual)
```

`regularizer_gradient` can be a convolutional module, U-Net, operator block, or
another shape-preserving SILVA point. If it is described as the gradient of a
scalar potential, that integrability claim needs a separate check.

## Diagnostics

| Family | Numerical diagnostic | Structural diagnostic | Task diagnostic |
| --- | --- | --- | --- |
| monotone graph | fixed-point residual | certificate and node equivariance | node or graph metric |
| equilibrium transformer | fixed-point residual | token shape and injection contract | teacher loss and image metric |
| Poisson mirror | fixed-point residual | positivity and adjoint check | KL and reconstruction metric |

A low fixed-point residual does not establish task quality, and a low task loss
does not establish a well-solved equilibrium.

## Executable Labs

| Question | Notebook |
| --- | --- |
| How is monotonicity derived and tested? | [Monotone Graph Equilibrium](../package-notebooks/21_silva_monotone_graph_equilibrium.ipynb) |
| How does one-time QKV injection work? | [Generative Equilibrium Transformer](../package-notebooks/22_silva_generative_equilibrium_transformer.ipynb) |
| Why does Burg geometry preserve positivity? | [Poisson Mirror Equilibrium](../package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb) |

## Where to Go Next

| Question | Page |
| --- | --- |
| How are physics-informed equilibria and DAEs represented? | [Physics-Informed Equilibria](physics-informed-equilibria.md) |
| Which generated equations define the teaching datasets? | [Advanced Equilibrium Datasets](advanced-equilibrium-datasets.md) |
| Which classes are public? | [Advanced Equilibria API](../api/advanced_equilibria.md) |
| Where are recent mechanisms compared? | [Recent Equilibrium Families](frontier-equilibrium-families.md) |
