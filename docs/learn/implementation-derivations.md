# Implementation Derivations

This page is the traceability layer between the SILVA equations, the SILVA paper
families, and the package implementation. It is written for readers who need to
audit, extend, or cite the implementation rather than only run examples.

Pair this page with the [Research Citation Audit](../research-citation-audit.md)
when preparing a paper, README, model card, or experiment report.

## Citation Trace

| Implementation family | Package surface | Primary references to cite |
| --- | --- | --- |
| fixed-point layer | `fixed_point`, `DEQLayer`, `SILVALayer`, presets | SILVA paper/package [[1]](../paper/references.md#ref-1){ .silva-cite } [[2]](../paper/references.md#ref-2){ .silva-cite }; Bai et al., [DEQ](https://arxiv.org/abs/1909.01377) [[4]](../paper/references.md#ref-4){ .silva-cite } |
| implicit gradients | `implicit_adjoint_solve`, VJP/JVP helpers | [Deep Implicit Layers tutorial](https://implicit-layers-tutorial.org/) [[3]](../paper/references.md#ref-3){ .silva-cite }; DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } |
| Anderson solver | `anderson` | Anderson, [1965](https://doi.org/10.1145/321296.321305) [[10]](../paper/references.md#ref-10){ .silva-cite }; Walker-Ni, [2011](https://doi.org/10.1137/10078356X) [[11]](../paper/references.md#ref-11){ .silva-cite } |
| Broyden solver | `broyden` | Broyden, [1965](https://doi.org/10.1090/S0025-5718-1965-0198670-6) [[12]](../paper/references.md#ref-12){ .silva-cite } |
| GMRES adjoint solve | `gmres`, `LinearSolveResult` | Saad-Schultz, [1986](https://doi.org/10.1137/0907058) [[13]](../paper/references.md#ref-13){ .silva-cite } |
| graph local terms | `GraphLocal`, `GraphAttentionLocal` | Kipf-Welling [GCN](https://arxiv.org/abs/1609.02907) [[15]](../paper/references.md#ref-15){ .silva-cite }; Velickovic et al. [GAT](https://arxiv.org/abs/1710.10903) [[16]](../paper/references.md#ref-16){ .silva-cite }; Gilmer et al. [MPNN](https://arxiv.org/abs/1704.01212) [[17]](../paper/references.md#ref-17){ .silva-cite } |
| global set/attention terms | `MeanFieldGlobal`, `TopKGlobalAttention`, channel attention | Deep Sets [[18]](../paper/references.md#ref-18){ .silva-cite }, [Attention](https://arxiv.org/abs/1706.03762) [[29]](../paper/references.md#ref-29){ .silva-cite }, [Set Transformer](https://arxiv.org/abs/1810.00825) [[19]](../paper/references.md#ref-19){ .silva-cite } |
| dynamic kNN local terms | `TopKLocal`, `DynamicChannelLocal`, dataset kNN graph builders | Wang et al., [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) [[20]](../paper/references.md#ref-20){ .silva-cite }; SILVA for hidden-channel adaptation [[1]](../paper/references.md#ref-1){ .silva-cite } |
| Jacobian penalty | `hutchinson_jacobian_norm`, `jacobian_regularization_loss` | Hutchinson, [1989](https://doi.org/10.1080/03610918908812806) [[14]](../paper/references.md#ref-14){ .silva-cite }; Bai et al., [Jacobian-regularized DEQ](https://arxiv.org/abs/2106.14342) [[6]](../paper/references.md#ref-6){ .silva-cite } |
| implicit bridge | `SILVAFixedPointBlock`, `SILVAEulerFlowBlock`, `SILVAQuadraticOptimizationLayer`, `SILVAMultiscaleDEQBlock` | DEQ [[4]](../paper/references.md#ref-4){ .silva-cite }, Neural ODEs [[7]](../paper/references.md#ref-7){ .silva-cite }, OptNet [[8]](../paper/references.md#ref-8){ .silva-cite }, differentiable convex optimization layers [[9]](../paper/references.md#ref-9){ .silva-cite }, MDEQ [[5]](../paper/references.md#ref-5){ .silva-cite } |
| scientific operators | `SILVAImplicitTimeStep`, `SILVAOperatorModel`, `SILVAFourierNeuralOperator`, finite-difference and residual helpers | Neural ODEs [[7]](../paper/references.md#ref-7){ .silva-cite }, FNO [[31]](../paper/references.md#ref-31){ .silva-cite }, neural operators [[32]](../paper/references.md#ref-32){ .silva-cite }, SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| input-injected Fourier equilibrium | `SILVAFNODEQBlock`, `SILVAFNODEQ` | FNO-DEQ [[43]](../paper/references.md#ref-43){ .silva-cite }, FNO [[31]](../paper/references.md#ref-31){ .silva-cite }, SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| physics-guided graph equilibrium | `graph_convection_diffusion`, `SILVAGraphConvectionDiffusion`, `SILVAPhysicsGuidedGraphDEQ` | physics-guided graph DEQ [[44]](../paper/references.md#ref-44){ .silva-cite }, graph convolution [[15]](../paper/references.md#ref-15){ .silva-cite }, SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| continuous residual path | `SILVAHomotopyTransition`, `SILVAHomotopyEquilibrium` | HomoODE [[46]](../paper/references.md#ref-46){ .silva-cite }, Neural ODEs [[7]](../paper/references.md#ref-7){ .silva-cite }, SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| empirical-measure equilibrium | `distributional_discrepancy`, `SILVADistributionalTransition`, `SILVADistributionalDEQ` | DDEQ [[45]](../paper/references.md#ref-45){ .silva-cite }, SILVA [[1]](../paper/references.md#ref-1){ .silva-cite } |
| DEQ engine | `SILVADEQEngine`, `silva_deq`, `pack_state`, `SILVAVariationalDropout` | SILVA package [[2]](../paper/references.md#ref-2){ .silva-cite }, [TorchDEQ](https://github.com/locuslab/torchdeq) [[35]](../paper/references.md#ref-35){ .silva-cite }, DEQ [[4]](../paper/references.md#ref-4){ .silva-cite } |
| SILVA DEQ flow | `SILVADEQFlow`, flow warp, all-pairs correlation | [RAFT](https://arxiv.org/abs/2003.12039) [[22]](../paper/references.md#ref-22){ .silva-cite }, [DEQ-Flow](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_Deep_Equilibrium_Optical_Flow_Estimation_CVPR_2022_paper.html) [[23]](../paper/references.md#ref-23){ .silva-cite }, SILVA package [[2]](../paper/references.md#ref-2){ .silva-cite } |

## Reading Contract

Every SILVA equilibrium layer exposes three levels of meaning:

| Level | Mathematical object | Package surface |
| --- | --- | --- |
| Transition | \(z \mapsto f_\theta(z,x)\) | `layer.f(...)`, custom transition callable |
| Solver | \(z_{k+1}=T(z_k)\) until \(\|f(z_k)-z_k\|\le\varepsilon\) | `fixed_point`, `SolverConfig`, `SolverResult` |
| Model | equilibrium state plus readout | `SILVAStack`, `SILVAGraphNetwork`, preset classifiers/regressors |

The core contract is shape preservation:

$$
f_\theta:\mathcal Z\to\mathcal Z,
\qquad
z_0,z_1,\ldots,z^\star\in\mathcal Z.
$$

For entity and graph layers,

$$
\mathcal Z=\mathbb R^{N\times d_h},
\qquad
x\in\mathbb R^{N\times d_x}.
$$

For vector vision layers,

$$
\mathcal Z=\mathbb R^{B\times C_h},
\qquad
x\in\mathbb R^{B\times d_x}.
$$

For convolutional image layers,

$$
\mathcal Z=\mathbb R^{B\times C_h\times H\times W}.
$$

## Universal Fixed-Point Form

All implemented families reduce to

$$
z^\star=f_\theta(z^\star,x),
\qquad
r_\theta(z,x)=f_\theta(z,x)-z.
$$

The convergence check recorded in `SolverResult.residuals` is

$$
\|r_\theta(z_k,x)\|_2
=\|f_\theta(z_k,x)-z_k\|_2.
$$

The default zero initialization in the package is not part of the mathematical
definition; it is an implementation convention:

$$
z_0=0
$$

unless `z0` is supplied by the caller. Warm starts, learned starts, or
continuation states are valid as long as they have the same shape as the
equilibrium state.

## Generic `SILVALayer`

The implementation in `SILVALayer.f` computes

$$
s=S_\theta(x),
\qquad
y=\chi(z),
$$

then adds self, local, and global terms:

$$
u
=
s
+H_\theta(y)
+L_\theta(y,E,e)
+G_\theta(y,b).
$$

The returned transition is

$$
f_\theta(z,x)
=
\operatorname{LayerNorm}\left(\tanh(u)\right)
$$

when `normalize=True`, and \(\tanh(u)\) otherwise. In code:

```python
s = self.stimulus(x)
y = self.activation(z)
self_update = self.self_term(y, ...)
local = self.local(y, ...)
global_context = self.global_term(y, ...)
return self.norm(self.output_activation(s + self_update + local + global_context))
```

The keyword dispatcher passes only the arguments a branch accepts. This lets a
custom branch be as small as

```python
def forward(self, z):
    ...
```

or as context-aware as

```python
def forward(self, z, edge_index=None, edge_attr=None, batch=None):
    ...
```

without changing the solver.

### Generic Branch Conditions

A branch is compatible when it is shape-preserving:

$$
B_\psi(y,\mathrm{context})\in\mathbb R^{N\times d_h}
\quad
\text{whenever}
\quad
y\in\mathbb R^{N\times d_h}.
$$

The branch may be zero, identity, linear, graph-local, attention-based, or a
user module. What matters for the fixed-point solver is that the full
transition returns a tensor shaped like `z0`.

## Graph Preset Layer

`SILVAGraphPresetLayer` uses the reference graph form

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

The SILVA layer intentionally omits the generic learned self branch. State
persistence comes from the damped solver:

$$
z_{k+1}
=(1-\alpha)z_k+\alpha f_\theta(z_k,x).
$$

When `local_depth > 1`, the local branch is tied inside one solver step:

$$
h^{(0)}=y_k,
\qquad
h^{(\ell+1)}=L_\theta(\tanh(h^{(\ell)}),E)
$$

for intermediate hops, with the final local term \(h^{(D)}\). This gives a
controlled way to increase graph mixing without adding another equilibrium
layer.

## Mean Graph Local Branch

`GraphLocal` projects every source state and averages incoming messages. For
edges \(j\to i\),

$$
m_j=W_\ell z_j+b_\ell,
$$

$$
L_i(Z,E)
=
\frac{1}{\max(1,|\mathcal N(i)|)}
\sum_{j\in\mathcal N(i)}m_j,
\qquad
\mathcal N(i)=\{j:(j,i)\in E\}.
$$

The implementation uses `index_add_` on the destination indices. This is
permutation-equivariant: if node order is permuted and `edge_index` is permuted
consistently, the output is permuted the same way.

If `edge_index is None`, `GraphLocal` returns the projected state when
`self_loop_when_empty=True`; otherwise it returns zeros. This matters when using
generic entity sets without graph structure.

## Graph Attention Local Branch

`GraphAttentionLocal` implements pure-PyTorch multi-head graph attention over
`edge_index`. With \(H\) heads and head width \(d_h/H\), the projected state is

$$
h_i^{(q)}=W^{(q)}z_i,
\qquad q=1,\ldots,H.
$$

For each directed edge \(j\to i\),

$$
\ell_{ij}^{(q)}
=
a_s^{(q)\top}h_j^{(q)}
+a_t^{(q)\top}h_i^{(q)}.
$$

If edge attributes are present,

$$
\ell_{ij}^{(q)}
\leftarrow
\ell_{ij}^{(q)}
+a_e^{(q)\top}W_e^{(q)}e_{ij}.
$$

After `LeakyReLU`, normalization is destination-segment softmax:

$$
\alpha_{ij}^{(q)}
=
\frac{\exp(\operatorname{LReLU}(\ell_{ij}^{(q)}))}
{\sum_{\ell\in\mathcal N(i)}
\exp(\operatorname{LReLU}(\ell_{i\ell}^{(q)}))}.
$$

The head output is

$$
o_i^{(q)}
=
\sum_{j\in\mathcal N(i)}
\alpha_{ij}^{(q)}h_j^{(q)}.
$$

When `concat=True`, heads are concatenated. When `concat=False`, heads are
averaged and projected back to the state dimension.

## Top-k Local Branch

`TopKLocal` builds a dynamic nearest-neighbor graph from the current state. For
entity \(i\),

$$
\mathcal K_i(z)
=
\operatorname{arg\,topk}_{j\ne i}
\left(-\|z_i-z_j\|_2\right).
$$

The local term is

$$
L_i(z)
=
W_\ell
\left(
\frac{1}{|\mathcal K_i|}
\sum_{j\in\mathcal K_i(z)}z_j
\right).
$$

This branch is state-dependent: changing \(z_k\) can change the neighborhood
used at the next solver step. It is useful for entity sets and hidden-channel
vision cases where no fixed graph is supplied.

## Global Mean and Gated Mean

`MeanFieldGlobal` pools one mean state per graph:

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i.
$$

It broadcasts the projection

$$
G_i(z)=W_g\bar z_{\operatorname{batch}(i)}+b_g.
$$

`GatedMeanFieldGlobal` adds a scalar graph gate:

$$
\beta_g
=
\sigma
\left(
\frac{(W_q\bar z_g)^\top(W_k\bar z_g)}
{\sqrt{d_h}}
\right),
$$

$$
G_i(z)=
\beta_g W_v\bar z_{\operatorname{batch}(i)}.
$$

The batch tensor is therefore not an input feature; it is a segmentation
operator that prevents graph-level context from leaking across examples in a
minibatch.

## Top-k Global Attention

`TopKGlobalAttention` computes bounded dense attention inside each graph. For
each receiver \(i\),

$$
q_i=W_qz_i,\qquad k_j=W_kz_j,\qquad v_j=W_vz_j,
$$

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.
$$

Only the top \(k\) source indices are kept:

$$
\mathcal A_i
=
\operatorname{arg\,topk}_{j}(s_{ij}).
$$

Then

$$
\omega_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell\in\mathcal A_i}\exp(s_{i\ell})},
\qquad
G_i(z)=\sum_{j\in\mathcal A_i}\omega_{ij}v_j.
$$

This gives a global branch with work \(O(Nkd_h)\) after score construction,
rather than using every source in the final weighted sum.

## Vector Vision Layers

`SILVAVisionVectorLayer` treats hidden channels as interacting entities inside
each sample. The state shape is

$$
z\in\mathbb R^{B\times C_h}.
$$

The transition is intentionally raw-sum rather than `LayerNorm(ReLU(...))`:

$$
f_\theta(z,x)
=
W_{\rm stim}x
+L_\theta(\tanh z)
+G_\theta(\tanh z).
$$

The dynamic channel local branch forms k-nearest neighbors among channel values
within each sample:

$$
\mathcal K_{b,c}(z)
=
\operatorname{arg\,topk}_{r\ne c}
\left(-|z_{b,c}-z_{b,r}|^2\right),
$$

symmetrizes the channel adjacency, and averages projected channel states by
degree. This is the implemented hidden-channel interaction graph.

The single-head channel global branch forms

$$
Q=W_qz,\qquad K=W_kz,
$$

$$
A_b
=
\operatorname{softmax}
\left(
\frac{Q_b^\top K_b}{\sqrt{C_h}}
\right),
\qquad
G_b=z_bA_b.
$$

The multi-head variant computes an intermediate attended representation and
then builds a dense channel attention matrix from the mixed vector.

## Convolutional Vision Classifier

`SILVAConvVisionClassifier` factors image processing into

$$
x_{\rm image}
\xrightarrow{\texttt{SILVAConvStem}}
h_0\in\mathbb R^{B\times d}
\xrightarrow{\texttt{SILVAVisionVectorClassifier}}
\hat y.
$$

The convolutional stem is not itself an equilibrium. It is a feature extractor:

$$
h_0
=
P
\left(
\operatorname{Dropout}
\left[
\operatorname{Pool}
\left(
\operatorname{ReLU}(\operatorname{BN}_2(\operatorname{Conv}_2(
\operatorname{Pool}(\operatorname{ReLU}(\operatorname{BN}_1(\operatorname{Conv}_1(x)))))
)))
\right)
\right]
\right).
$$

The equilibrium dynamics happen after the stem in the vector SILVA core.

## Molecular SILVA

`SILVAMolecularLayer` is a bond-aware graph equilibrium. Atom and bond features
are first embedded or projected to the hidden width. Inside each molecular
layer,

$$
s_i=W_{\rm stim}x_i,
$$

$$
l_i=L_\theta(z,E,e)_i
$$

where \(L_\theta\) is edge-aware graph attention using bond attributes. The
global molecule context is

$$
\bar z_g
=
\frac{1}{|\mathcal V_g|}
\sum_{i\in\mathcal V_g}z_i,
\qquad
g_i=W_g\bar z_{\operatorname{batch}(i)}.
$$

The transition is

$$
f_\theta(z,x,E,e,b)_i
=
\operatorname{Dropout}
\left(
\operatorname{LayerNorm}
\left[
\operatorname{ReLU}(s_i+l_i+g_i)
\right]
\right).
$$

When spectral normalization is enabled, the stimulus and global projections are
constrained by `torch.nn.utils.spectral_norm`. This is an implementation-level
stabilizer, not a proof of global contraction.

## Stacked Equilibrium Models

`SILVAStack` composes equilibrium layers:

$$
z^{(1)\star}
=
f_{\theta_1}^{(1)}(z^{(1)\star},x),
$$

$$
z^{(\ell)\star}
=
f_{\theta_\ell}^{(\ell)}
\left(
z^{(\ell)\star},
z^{(\ell-1)\star}
\right),
\qquad \ell=2,\ldots,L.
$$

Graph preset and molecular networks often pass

$$
\tanh(z^{(\ell-1)\star})
$$

as the next layer input. Each layer has its own `SolverConfig`, so a fast/slow
hierarchy is represented by different damping values:

$$
\alpha_1>\alpha_2
$$

or by an arbitrary `stack_alphas` sequence.

## Readouts and Pooling

For node tasks, the readout is applied per entity:

$$
\hat y_i=h_\phi(z_i^\star).
$$

For graph tasks, states are pooled first:

$$
p_g
=
\operatorname{pool}
\{z_i^\star:i\in\mathcal V_g\},
\qquad
\hat y_g=h_\phi(p_g).
$$

Implemented pooling modes are

$$
\operatorname{mean},\qquad
\operatorname{sum},\qquad
\operatorname{max}.
$$

## Picard Solver

The package records residuals before applying the next damped step:

$$
r_k=f(z_k)-z_k,
\qquad
\rho_k=\|r_k\|_2.
$$

The update is

$$
z_{k+1}
=(1-\alpha)z_k+\alpha f(z_k)
=z_k+\alpha r_k.
$$

Convergence is declared when

$$
\rho_k<\texttt{tol}.
$$

Best practice: lower `alpha` when residuals oscillate, increase `max_iter` only
after checking whether the residual curve is still decreasing, and inspect
`result.converged` rather than assuming the last iterate is an equilibrium.

## Anderson Solver

Anderson acceleration stores recent states and transition outputs:

$$
X=[x_{k-m+1},\ldots,x_k],
\qquad
F=[f_{k-m+1},\ldots,f_k].
$$

Residual columns are

$$
G=F-X.
$$

The coefficients solve the ridge-regularized constrained least-squares problem

$$
\min_c \|Gc\|_2^2+\lambda\|c\|_2^2
\quad
\text{subject to}
\quad
\mathbf 1^\top c=1.
$$

The implemented KKT system is

$$
\begin{bmatrix}
G^\top G+\lambda I & \mathbf 1\\
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

The mixed output is

$$
z_{\rm mix}
=
\sum_i c_i f_i,
$$

and the final update is

$$
z_{k+1}
=
\beta z_{\rm mix}+(1-\beta)f(z_k).
$$

Implementation note: stored states and transition outputs are detached. This
makes the solver practical for documentation examples and finite-step training,
while implicit-gradient diagnostics are handled separately.

## Broyden Solver

The package uses the root form

$$
R(z)=f(z)-z.
$$

It stores a dense inverse-Jacobian approximation \(B_k\), initialized as

$$
B_0=-I.
$$

The step is

$$
s_k=-\alpha B_kR(z_k),
\qquad
z_{k+1}=z_k+s_k.
$$

Because \(B_0=-I\), the first Broyden step matches the damped residual
direction:

$$
s_0=\alpha R(z_0).
$$

With

$$
y_k=R(z_{k+1})-R(z_k),
$$

the inverse secant condition is

$$
B_{k+1}y_k=s_k.
$$

The implemented good-Broyden update is

$$
B_{k+1}
=
B_k
+
\frac{(s_k-B_ky_k)(s_k^\top B_k)}
{s_k^\top B_ky_k}.
$$

Because the implementation materializes a dense \(n\times n\) inverse estimate
for \(n=\operatorname{numel}(z)\), it is best for small states and controlled
diagnostic experiments.

## GMRES and Implicit Adjoints

For an equilibrium \(z^\star\), define

$$
F(z,\theta)=f_\theta(z)-z.
$$

Implicit differentiation gives

$$
(I-J_f(z^\star)^\top)\lambda
=
\frac{\partial\mathcal L}{\partial z^\star}.
$$

For the executed damped update

$$
T_\alpha(z)=(1-\alpha)z+\alpha f(z),
$$

the diagnostic adjoint solve in the package uses

$$
(I-J_{T_\alpha}(z^\star)^\top)u=g.
$$

Since

$$
J_{T_\alpha}=(1-\alpha)I+\alpha J_f,
$$

the linear operator is

$$
I-J_{T_\alpha}^\top
=
\alpha(I-J_f^\top).
$$

`implicit_adjoint_solve` applies this operator with VJP calls and solves the
matrix-free system using GMRES.

GMRES builds an Arnoldi basis \(V_m\) and upper Hessenberg matrix \(H_m\):

$$
AV_m\approx V_{m+1}H_m.
$$

At iteration \(m\), it solves

$$
y_m
=
\arg\min_y
\left\|
\beta e_1-H_my
\right\|_2,
\qquad
x_m=V_my_m.
$$

The package exposes the residual history in `LinearSolveResult.residuals`.

## Jacobian Diagnostics

For small states, `full_jacobian` materializes

$$
J_f(z)=\frac{\partial \operatorname{vec}(f(z))}
{\partial \operatorname{vec}(z)}.
$$

For larger states, use products:

$$
\operatorname{jvp}(f,z,v)=(f(z),J_f(z)v),
$$

$$
\operatorname{vjp}(f,z,v)=J_f(z)^\top v.
$$

The spectral-radius estimator runs power iteration on VJP products:

$$
v_{t+1}
=
\frac{J^\top v_t}{\|J^\top v_t\|_2},
\qquad
\rho_t=\|J^\top v_t\|_2.
$$

`damped_spectral_radius(f, z_star, alpha)` estimates

$$
\rho((1-\alpha)I+\alpha J_f(z^\star)).
$$

The Hutchinson Frobenius estimator uses Rademacher probes \(v\):

$$
\mathbb E_v\|J_f^\top v\|_2^2
=
\|J_f\|_F^2.
$$

This is why `hutchinson_jacobian_norm` can estimate a norm without storing the
full Jacobian.

## Energy Diagnostics

The package's quadratic interaction energy is

$$
E_i(z,h)
=
\|z_i\|_2^2-z_i^\top h_i.
$$

For a local plus global interaction \(h_i=L_i+G_i\), lower energy means the
state aligns more strongly with the interaction field under this diagnostic.
It is a monitoring quantity:

$$
E_{k+1}\le E_k
$$

is evidence of descent for the chosen trace, but it is not by itself a
Lyapunov proof unless the user's model assumptions justify it.

`solve_with_energy` evaluates the energy function before each fixed-point
transition and returns:

| Quantity | Meaning |
| --- | --- |
| `result` | final `SolverResult` |
| `energies` | per-iteration diagnostic energy |
| `energy_deltas` | \(E_{k+1}-E_k\) |
| `stability` | optional local Jacobian report |

## Implicit Bridge Formula Map

The `silva_networks.implicit` module keeps tutorial implicit-layer models inside
the package solver and Jacobian APIs. These modules are deliberately compact,
but each one corresponds to a standard implicit-learning case.

### Affine-Tanh DEQ Transition

`DEQMLPTransition` implements

$$
f_\theta(z,x)
=
\phi(W_z z + W_x x + b),
$$

with default \(\phi=\tanh\). The fixed-point block solves

$$
z^\star=f_\theta(z^\star,x),
$$

using the package update

$$
z_{k+1}
=
(1-\alpha)z_k+\alpha f_\theta(z_k,x).
$$

Because

$$
J_f(z,x)
=
\operatorname{diag}\!\left(\phi'(W_z z+W_x x+b)\right)W_z,
$$

a simple sufficient local contraction condition is

$$
\sup_z\|J_f(z,x)\|_2<1.
$$

For tanh, \(|\phi'|\le 1\), so controlling the recurrent spectral norm gives
the conservative check

$$
\|W_z\|_2<1
\quad\Longrightarrow\quad
\|J_f(z,x)\|_2\le \|W_z\|_2<1.
$$

`DEQMLPTransition.project_state_weight(max_norm)` applies this tutorial-scale
initialization guard to the recurrent matrix.

`TanhFixedPointClassifier` adds a readout

$$
\hat y
=
W_o\,D(z^\star)+c,
$$

where \(D\) is dropout or the identity.

Use the DEQ paper and Deep Implicit Layers tutorial when citing this bridge
case. Use the SILVA package citation when using the package-native classes or
SILVA-named factories.

### Explicit Euler ODE Bridge

`ExplicitEulerODEBlock` is not an equilibrium solver. It is included to show how
continuous-depth intuition relates to repeated state updates. Starting from

$$
\frac{dh(t)}{dt}=v_\theta(h(t)),
$$

explicit Euler gives

$$
h_{k+1}
=
h_k+\Delta t\,v_\theta(h_k).
$$

After \(K\) steps,

$$
h_K
=
h_0+\Delta t\sum_{k=0}^{K-1}v_\theta(h_k).
$$

The same state-space thinking appears in equilibrium models, but a DEQ solves
for a stationary state while the Euler block returns the terminal state of a
finite trajectory.

Use Neural ODEs when citing the continuous-depth model and the Deep Implicit
Layers tutorial when citing this bridge in the implicit-layer context.

### Implicit ODE/PDE and Operator Implementation

`SILVAImplicitTimeStep` represents a backward-Euler step for a semidiscrete
right-hand side \(R_h\):

$$
u^{n+1}
=u^n+\Delta t\,R_h(u^{n+1},c).
$$

Its source-level transition is equivalent to

```python
field = rhs(state, context)
proposal = previous + step_size * field
return projector(proposal)
```

The implementation checks that `rhs` preserves the state shape, applies the
projector on every transition evaluation, and calls `solve_equilibrium` with
the module parameters and differentiable context tensors. Therefore
`SolverConfig.backward_mode` has the same meaning as it does for other SILVA
points.

The numerical helpers expose the centered formulas directly. In one dimension,

$$
(D_hu)_i=\frac{u_{i+1}-u_{i-1}}{2h},
\qquad
(\Delta_hu)_i=\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}.
$$

In two dimensions,

$$
(\Delta_hu)_{i,j}
=\frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2}.
$$

`SILVAReactionDiffusionRHS2D` computes

$$
R_h(u,s)=D\Delta_hu+r(u)+s,
$$

and `SILVABurgersRHS1D` computes

$$
R_h(u,s)=-uD_hu+\nu\Delta_hu+s.
$$

Both accept an optional state-shaped context and return the state shape.
Boundary projection is separate through `SILVADirichletBoundary2D`, so the
discretization and constraint can be tested independently.

`SILVAOperatorModel` constructs the learned function map from existing
abstractions:

1. `Conv2d(in_channels, state_channels, 1)` lifts the sampled input;
2. a built-in spatial point architecture or supplied module defines the
   state-dependent field;
3. `SILVACortexLayer` composes optional self, local, global, and interaction
   fields and solves the equilibrium;
4. a readout maps the equilibrium state to `out_channels`;
5. an optional output transform applies a boundary mask or physical projection.

The input and output contract is

$$
\mathbb R^{B\times C_{in}\times H\times W}
\rightarrow
\mathbb R^{B\times C_{out}\times H\times W}.
$$

`SILVAFourierNeuralOperator` selects the Fourier point architecture, whose
state field is

$$
B_\theta(z)
=s\left[\mathcal F_h^{-1}
\left(R_\theta\mathcal F_hz\right)+Wz\right].
$$

The full `SILVAOperatorOutput` contains the decoded output, equilibrium state,
and `SolverResult`. Physical diagnostics remain separate:

$$
r_{\mathrm{Poisson}}=-\Delta_h\widehat u-q,
\qquad
e_{\partial\Omega}
=\operatorname{RMS}_{x\in\partial\Omega}(\widehat u(x)-g(x)).
$$

This separation prevents a low solver residual from being reported as evidence
that the governing equation or boundary condition is satisfied.

### Quadratic Optimization Layer

`QuadraticOptimizationLayer` forms

$$
A=L L^\top+\lambda I,
\qquad
b_\theta(x)=B_\theta x+c,
$$

with \(\lambda>0\), so \(A\) is symmetric positive definite. The layer's
objective is

$$
J(z;x)
=
\frac12 z^\top A z-b_\theta(x)^\top z.
$$

The first-order condition is

$$
\nabla_z J(z;x)
=
Az-b_\theta(x)=0.
$$

Thus the exact optimizer is

$$
z^\star
=
A^{-1}b_\theta(x).
$$

The package also exposes the equivalent gradient-descent fixed-point map

$$
T_\eta(z)
=
z-\eta(Az-b_\theta(x)).
$$

For this quadratic case, the fixed-point iteration converges when

$$
0<\eta<\frac{2}{\lambda_{\max}(A)}.
$$

This is a useful bridge from differentiable optimization layers to SILVA-style
fixed-point solvers: the implicit state is defined by an equation rather than by
a fixed number of explicit neural layers.

Use OptNet or differentiable convex optimization layers when citing the
optimization-layer perspective. Use the SILVA citation for this package's
quadratic tutorial implementation.

### Toy Multiscale DEQ

`ToyMultiscaleDEQBlock` splits the state into low- and high-resolution parts,

$$
z=(z_\ell,z_h).
$$

Its transition is

$$
z_\ell^+
=
\tanh(S_\ell(x)+A_{\ell\ell}z_\ell+A_{h\ell}z_h),
$$

$$
z_h^+
=
\tanh(S_h(x)+A_{hh}z_h+A_{\ell h}z_\ell).
$$

The joint equilibrium is therefore

$$
\begin{bmatrix}
z_\ell^\star\\
z_h^\star
\end{bmatrix}
=
\begin{bmatrix}
\tanh(S_\ell(x)+A_{\ell\ell}z_\ell^\star+A_{h\ell}z_h^\star)\\
\tanh(S_h(x)+A_{hh}z_h^\star+A_{\ell h}z_\ell^\star)
\end{bmatrix}.
$$

This compact module mirrors the multiscale DEQ idea: several feature scales are
coupled and solved together.

Use Multiscale Deep Equilibrium Models when citing the coupled-scale
equilibrium idea.

### Jacobian Regularization and Residual Ratio

`jacobian_regularization_loss` adds a stability penalty of the form

$$
\mathcal L
=
\mathcal L_{\text{task}}
+\gamma\|J_f(z^\star)\|_F^2.
$$

The package estimates the Frobenius norm with Hutchinson probes:

$$
\|J_f(z^\star)\|_F^2
=
\operatorname{tr}(J_f^\top J_f)
=
\mathbb E_v
\left[
\|J_f(z^\star)^\top v\|_2^2
\right].
$$

`residual_ratio` reports

$$
r_{\text{ratio}}
=
\frac{r_{\text{final}}}{\max(r_{\text{initial}},10^{-12})}.
$$

Values below one indicate that the numerical solve reduced the residual; values
near zero indicate a strong solve relative to the initial state.

Use Hutchinson's trace estimator for the stochastic norm estimate and
Jacobian-regularized DEQs for the DEQ stability-regularization objective.

### Bridge Case Summary

| Module | Equation class | Package behavior |
| --- | --- | --- |
| `DEQMLPTransition` | affine nonlinear fixed point | defines \(f_\theta(z,x)\) |
| `TanhFixedPointBlock` | vector DEQ solve | calls `fixed_point` and returns \(z^\star\) |
| `TanhFixedPointClassifier` | DEQ representation plus readout | maps \(z^\star\) to logits |
| `ExplicitEulerODEBlock` | finite explicit trajectory | returns \(h_K\), optionally with the trajectory |
| `SILVAImplicitTimeStep` | backward-Euler ODE/PDE point | returns \(u^{n+1}\) or a `SolverResult` |
| `SILVAOperatorModel` | learned sampled function map | returns a decoded field or `SILVAOperatorOutput` |
| `SILVAFourierNeuralOperator` | Fourier field inside a SILVA point | reuses spectral parameters across compatible grid sizes |
| `QuadraticOptimizationLayer` | implicit argmin / KKT equation | solves \(Az=b_\theta(x)\) by fixed-point iteration or exact solve |
| `ToyMultiscaleDEQBlock` | coupled multiscale fixed point | solves \((z_\ell^\star,z_h^\star)\) jointly |
| `jacobian_regularization_loss` | VJP-based stability penalty | estimates \(\|J_f\|_F^2\) |
| `residual_ratio` | residual trace summary | compresses solver progress into one diagnostic |

## General DEQ Engine Formula Map

The `silva_networks.deq_engine` module exposes a package-native DEQ interface
for one tensor state or a tuple/list of tensor states. This is useful for
models whose equilibrium state has several coupled components.

For a multi-state transition,

$$
s=(z^{(1)},z^{(2)},\dots,z^{(m)}),
\qquad
s^+=F_\theta(s,x),
$$

the equilibrium equation is

$$
s^\star=F_\theta(s^\star,x).
$$

`pack_state` converts the state into one solver vector:

$$
v
=
P(s)
=
\operatorname{concat}
\left[
\operatorname{vec}(z^{(1)}),\dots,\operatorname{vec}(z^{(m)})
\right].
$$

The engine solves the packed fixed point

$$
v^\star
=
\tilde F_\theta(v^\star,x),
\qquad
\tilde F_\theta(v,x)
=
P(F_\theta(P^{-1}(v),x)).
$$

After solving, `unpack_state` returns

$$
s^\star=P^{-1}(v^\star).
$$

When `reengage=True`, the engine evaluates one more differentiable transition:

$$
s_{\rm out}=F_\theta(s^\star,x),
$$

and stores the packed result back into `solver_result.z`. This mirrors the
package's compact bridge modules: acceleration history can be numerical, while
the final returned state remains connected to PyTorch autograd.

`SILVAVariationalDropout` uses one mask during a solve:

$$
\tilde x
=
x\odot \frac{m}{1-p},
\qquad
m_i\sim\operatorname{Bernoulli}(1-p).
$$

The mask is intentionally reusable across solver calls until reset, avoiding a
different stochastic transition at every fixed-point iteration.

## Optical Flow Formula Map

The `silva_networks.flow` module is a compact RAFT/DEQ-Flow-inspired validation
case implemented with SILVA solvers.

For a pixel

$$
p=(x,y),
$$

and a flow field

$$
u(p)=(u_x(p),u_y(p)),
$$

`silva_flow_warp` samples the source tensor at

$$
p+u(p).
$$

Feature maps

$$
F_1,F_2\in\mathbb R^{B\times C\times H\times W}
$$

produce the all-pairs correlation volume

$$
C_{b,i,j,k,\ell}
=
\frac{
\langle F_{1,b,:,i,j},F_{2,b,:,k,\ell}\rangle
}{\sqrt C}.
$$

The local lookup samples the neighborhood

$$
\mathcal N_r(p+u(p))
=
\left\{
p+u(p)+(\Delta x,\Delta y):
|\Delta x|\le r,\ |\Delta y|\le r
\right\}.
$$

The update block predicts an increment from the current flow, image features,
warped features, residual features, and local correlations:

$$
\Delta u
=
\Delta_\theta(u,F_1,\operatorname{warp}(F_2,u),F_1-\operatorname{warp}(F_2,u),C_r).
$$

The transition is

$$
T_\theta(u)
=
u+\gamma\tanh(\Delta u).
$$

`SILVADEQFlow` solves

$$
u^\star=T_\theta(u^\star).
$$

Endpoint error is

$$
\operatorname{EPE}
=
\|u_{\rm pred}-u_{\rm target}\|_2.
$$

With a validity mask \(M\), the mean EPE is

$$
\frac{
\sum_p M(p)\operatorname{EPE}(p)
}{
\max(1,\sum_p M(p))
}.
$$

The first-order smoothness penalty is

$$
\mathcal L_{\rm smooth}
=
\sum_p
\left(
|u(p+\hat x)-u(p)|
+
|u(p+\hat y)-u(p)|
\right).
$$

Use RAFT for the all-pairs correlation and recurrent refinement lineage, and
DEQ-Flow when citing the optical-flow equilibrium framing.

## Tensor Contracts

| Case | Required tensors | Shape contract |
| --- | --- | --- |
| Generic entity | `x`, optional `edge_index`, `edge_attr`, `batch` | `x: (entities, in_dim)` |
| Graph/node | `x`, `edge_index`, optional `batch` | `edge_index: (2, edges)` |
| Graph-level | `x`, `edge_index`, `batch` | pooled by graph id |
| Vision vector | image vectors or flattened tensors | internal state `(batch, hidden_dim)` |
| Conv vision | `x` | `(batch, channels, height, width)` |
| Molecular | `x`, `edge_index`, `edge_attr`, `batch` | atom and bond tensors segmented by molecule |
| Implicit bridge vector | `x`, optional `z0` | `x: (batch, in_dim)`, state `(batch, state_dim)` |
| Implicit bridge multiscale | `x`, optional `z0` | concatenated state `(batch, low_dim + high_dim)` |
| General DEQ engine | tensor, tuple, or list state | each component is packed into one solver vector |
| Optical flow | `image1`, `image2`, optional `flow0` | images `(batch, channels, height, width)`, flow `(batch, 2, height, width)` |

Validation best practices:

1. Confirm the transition returns the same shape as `z0`.
2. Confirm `edge_index[0]` are sources and `edge_index[1]` are destinations.
3. Confirm `batch` is present for multiple graphs or molecules.
4. Confirm `edge_attr.shape[0] == edge_index.shape[1]` when edge attributes are used.
5. Request `return_result=True` or `return_results=True` during development.

## Professional Reporting Checklist

When reporting a SILVA experiment, include:

| Item | Why it matters |
| --- | --- |
| layer family | generic, graph, vector vision, conv vision, molecular |
| local/global/self branches | identifies the implemented \(L,G,H\) terms |
| solver and `alpha` | defines the executed dynamics |
| `max_iter`, `tol`, convergence rate | separates approximation quality from model accuracy |
| residual summary | shows whether the equilibrium was actually reached |
| Jacobian or spectral-radius diagnostic | local stability evidence |
| dataset adapter and tensor shapes | reproducibility and batching correctness |
| readout and pooling | distinguishes node, graph, image, and molecule outputs |

## Common Failure Modes

| Symptom | Likely cause | Check |
| --- | --- | --- |
| residual plateaus | transition is not locally contractive, `alpha` too high, normalization mismatch | lower `alpha`, inspect `result.residuals`, estimate damped spectral radius |
| residual oscillates | damped map has eigenvalues near or below \(-1\) | reduce `alpha`, try Anderson with ridge |
| graph output changes under node reorder | `edge_index` or `batch` not permuted consistently | validate data adapter |
| molecules mix context | missing or wrong `batch` | inspect molecule ids and pooled count |
| Jacobian computation is slow | full materialization on a large state | use `vjp`, `jvp`, Hutchinson, or spectral-radius estimates |
| energy decreases but residual is high | diagnostic alignment improved but fixed point not solved | report both energy and residual |

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is the fixed-point mathematics developed from first principles? | [Mathematical Foundations](mathematical-foundations.md) |
| How are the supported scientific cases organized? | [Case Atlas](case-atlas.md) |
| Where can I execute the implicit-layer derivations? | [Implicit Layers Bridge](implicit-bridge.md) |
| Which package signatures implement these equations? | [API Reference](../api/reference.md) |
