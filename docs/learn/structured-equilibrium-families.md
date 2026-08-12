# Structured Equilibrium Families

This chapter derives six source-grounded equilibrium mechanisms and shows how
they fit inside SILVA without losing their defining structure. The compact
experiments establish equations, shapes, certificates, solver behavior, and
gradients. Reproducing a published benchmark additionally requires the cited
data, preprocessing, architecture scale, training schedule, and evaluation
budget.

## The Shared SILVA Contract

Every family still has the SILVA decomposition

$$
z^\star=T_\theta(z^\star,x),
\qquad
\widehat y=Q_\psi(z^\star),
$$

but each source constrains a different part of $T_\theta$:

| Family | Defining structure | Public SILVA object | Reference |
| --- | --- | --- | --- |
| monDEQ | strongly monotone operator and splitting | `SILVAMonotoneOperatorEquilibrium` | [[75]](../paper/references.md#ref-75){ .silva-cite } |
| pcDEQ | positive-concave transition | `SILVAPositiveConcaveEquilibrium` | [[76]](../paper/references.md#ref-76){ .silva-cite } |
| NEMON | weighted-infinity contraction certificate | `SILVANonEuclideanEquilibrium` | [[77]](../paper/references.md#ref-77){ .silva-cite } |
| EIGNN | normalized channel Gram map and spectral solve | `SILVAEfficientInfiniteGraphEquilibrium` | [[78]](../paper/references.md#ref-78){ .silva-cite } |
| MGNNI | graph-power equilibria and nodewise scale fusion | `SILVAMultiscaleGraphImplicitNetwork` | [[79]](../paper/references.md#ref-79){ .silva-cite } |
| DeltaDEQ | cached linear updates from sparse state changes | `SILVADeltaEquilibrium` | [[80]](../paper/references.md#ref-80){ .silva-cite } |

The package does not force one task head on these mechanisms. A user can supply
the source-aligned head or a new task-specific readout while preserving the
equilibrium construction.

## Monotone Operator Equilibrium

### From a fixed point to an inclusion

Begin with a proximal equilibrium

$$
z^\star=\operatorname{prox}_{f}
\left(Wz^\star+Ux+b\right).
$$

The proximal identity

$$
v=\operatorname{prox}_{f}(q)
\quad\Longleftrightarrow\quad
q-v\in\partial f(v)
$$

turns the state equation into

$$
0\in (I-W)z^\star-Ux-b+\partial f(z^\star).
$$

The source construction parameterizes the recurrent matrix as
[[75]](../paper/references.md#ref-75){ .silva-cite }

$$
W=(1-m)I-A^\mathsf{T}A+B-B^\mathsf{T},
\qquad m>0.
$$

Its symmetric contribution satisfies

$$
\frac{(I-W)+(I-W)^\mathsf{T}}{2}
=mI+A^\mathsf{T}A\succeq mI.
$$

Therefore $I-W$ is strongly monotone with margin at least $m$. The skew term
$B-B^\mathsf{T}$ can increase expressiveness without changing this certificate.
`SILVAMonotoneDenseOperator.monotonicity_certificate()` computes the smallest
eigenvalue of the symmetric part rather than only returning the requested
lower bound.

### Forward-backward splitting

For a step $a>0$, the implemented forward-backward map is

$$
z_{k+1}=\operatorname{prox}_{af}
\left((1-a)z_k+a(Wz_k+Ux+b)\right).
$$

ReLU is the default proximal map because it is projection onto the nonnegative
orthant and is independent of $a$. A custom proximal callable must include any
step dependence required by its regularizer.

### Peaceman-Rachford splitting

Let $u_k$ be the reflected state and $c=Ux+b$. One update is

$$
z_k=\operatorname{prox}_{af}(u_k),
$$

$$
u_{k+\frac12}=2z_k-u_k,
$$

$$
z_{k+\frac12}
=\left((1+a)I-aW\right)^{-1}
\left(u_{k+\frac12}+ac\right),
$$

$$
u_{k+1}=2z_{k+\frac12}-u_{k+\frac12}.
$$

The dense operator exposes the resolvent directly. A custom structured
operator can implement the same `resolvent(values, step_size)` interface with
a convolutional solve, diagonalization, or matrix-free linear method.

```python
model = SILVAMonotoneOperatorEquilibrium(
    in_dim=32,
    state_dim=128,
    out_dim=10,
    splitting="peaceman_rachford",
    step_size=0.5,
    margin=0.2,
    source=custom_source,
    prox=custom_prox,
    readout=classifier,
    config=solver_config,
)
result = model(features, return_result=True)
assert result.monotonicity_certificate > 0
```

The source-faithful core is the monotone parameterization plus operator
splitting. SILVA additionally makes source, proximal map, resolvent, readout,
and numerical configuration independently replaceable.

## Positive-Concave Equilibrium

### Positive order structure

pcDEQ uses a map on the positive orthant
[[76]](../paper/references.md#ref-76){ .silva-cite }:

$$
z^\star=\phi(W_+z^\star+s_+(x)),
\qquad
W_+\geq0,
\qquad
s_+(x)\geq0.
$$

If $z\leq z'$ entrywise, then $W_+z\leq W_+z'$. An increasing activation
preserves this order. Concavity on the positive orthant supplies the nonlinear
Perron-Frobenius structure used by the source analysis for existence,
uniqueness, and geometric convergence.

The default SILVA route obtains recurrent weights through

$$
W_+=\operatorname{softplus}(\widetilde W)+\epsilon_w,
$$

so optimization occurs in unconstrained parameters while the evaluated map is
strictly nonnegative.

The source repository instead applies weight normalization and projects both
its direction and magnitude after every optimizer update:

$$
W_+=g_+\frac{v_+}{\lVert v_+\rVert}+epsilon_w,
\qquad v_+\geq0,
\qquad g_+\geq0.
$$

Select `weight_parameterization="source_weight_norm"` and call
`model.project_nonnegative_()` immediately after `optimizer.step()` to follow
that policy. `weight_parameterization="projected"` provides a direct projected
weight without normalization; it is a SILVA ablation, not the source training
parameterization.

### The two implemented variants

Variant 1 uses a strictly positive injection

$$
s_+(x)=\operatorname{softplus}(S_\theta(x))
$$

with `tanh`, `softsign`, or `relu6`. Variant 2 uses

$$
s_+(x)=\operatorname{ReLU}(S_\theta(x))
$$

with `sigmoid`. Both variants support a dense operator or a shape-preserving
two-dimensional convolution.

```python
vector_model = SILVAPositiveConcaveEquilibrium(
    in_dim=64,
    state_dim=256,
    out_dim=10,
    variant=1,
    activation="softsign",
    operator="linear",
)

image_model = SILVAPositiveConcaveEquilibrium(
    in_dim=32,
    state_dim=64,
    out_dim=10,
    variant=2,
    operator="conv2d",
    kernel_size=3,
)
```

The source-aligned optimizer step is explicit:

```python
model = SILVAPositiveConcaveEquilibrium(
    in_dim=64,
    state_dim=256,
    out_dim=10,
    variant=1,
    weight_parameterization="source_weight_norm",
)
loss = criterion(model(features), labels)
loss.backward()
optimizer.step()
model.project_nonnegative_()
```

The returned `minimum_weight` verifies the parameterized operator. State
positivity is checked directly from `result.state`. Replacing the packaged
transition is allowed, but a positive-concave claim then requires the custom
module to preserve the corresponding mathematical conditions.

## Non-Euclidean Monotone Operator Network

### Weighted infinity norm

For positive metric weights $d_i$, define

$$
D=\operatorname{diag}(d_1,\ldots,d_n),
\qquad
\lVert z\rVert_{\infty,D}=\lVert Dz\rVert_\infty.
$$

The associated matrix measure is

$$
\mu_{\infty,D}(W)
=\mu_\infty(DWD^{-1})
=\max_i\left(
(DWD^{-1})_{ii}
+\sum_{j\neq i}\left|(DWD^{-1})_{ij}\right|
\right).
$$

NEMON uses this non-Euclidean geometry to enlarge the certified parameter
region and derive input-output robustness bounds
[[77]](../paper/references.md#ref-77){ .silva-cite }.

### Certified parameterization

With free matrix $A$, learned positive metric $D$, and target $m<1$, SILVA uses

$$
W=mI+D^{-1}AD-\operatorname{diag}(|A|\mathbf 1).
$$

After similarity transformation,

$$
DWD^{-1}=mI+A-\operatorname{diag}(|A|\mathbf 1),
$$

which gives

$$
\mu_{\infty,D}(W)\leq m<1.
$$

### Averaged iteration and sensitivity

For

$$
F(z,x)=\operatorname{ReLU}(Wz+Ux+b),
$$

the fixed point is unchanged by averaging:

$$
z_{k+1}=(1-\alpha)z_k+\alpha F(z_k,x).
$$

The packaged operator computes the source-style recommended coefficient from
a diagonal derivative lower bound,

$$
\alpha^\star=\frac{1}{1-\operatorname{diagL}(F)},
$$

clipped to $(0,1]$. For a linear source map, a latent input sensitivity bound
has the form

$$
\operatorname{Lip}(x\mapsto z^\star(x))
\leq
\frac{\lVert DU\rVert_\infty}
{1-\mu_{\infty,D}(W)}.
$$

```python
model = SILVANonEuclideanEquilibrium(
    in_dim=32,
    state_dim=128,
    out_dim=10,
    one_sided_bound=0.05,
    averaging=None,
)
result = model(features, return_result=True)
print(result.one_sided_lipschitz)
print(result.latent_input_lipschitz_bound)
```

For a custom source or readout, the equilibrium remains usable, but the
packaged analytic input bound is returned only when the source norm can be
computed from a dense linear map.

## Efficient Infinite-Depth Graph Equilibrium

### Node-major graph equation

EIGNN normalizes a channel factor into a positive-semidefinite map
[[78]](../paper/references.md#ref-78){ .silva-cite }:

$$
C=g(F)=\frac{F^\mathsf{T}F}
{\lVert F^\mathsf{T}F\rVert_F+\epsilon_F}.
$$

For node-major state $Z\in\mathbb R^{N\times d}$ and graph operator
$S\in\mathbb R^{N\times N}$, SILVA solves

$$
Z^\star=\gamma S^\mathsf{T}Z^\star C^\mathsf{T}+X,
\qquad 0\leq\gamma<1.
$$

This is the named SILVA transition

$$
T_\theta(Z,X)=S_\theta(X)+L_\theta(Z;S),
$$

with source injection $S_\theta(X)$ and graph-local equilibrium field
$L_\theta(Z;S)=\gamma S^\mathsf{T}ZC^\mathsf{T}$.

### Closed-form derivation

For symmetric dense operators, write

$$
S=Q\Lambda Q^\mathsf{T},
\qquad
C=V\Sigma V^\mathsf{T}.
$$

Set $\widetilde Z=Q^\mathsf{T}ZV$ and
$\widetilde X=Q^\mathsf{T}XV$. Then each graph-channel mode decouples:

$$
\widetilde Z_{ij}
=\gamma\lambda_i\sigma_j\widetilde Z_{ij}
+\widetilde X_{ij},
$$

so

$$
\widetilde Z_{ij}
=\frac{\widetilde X_{ij}}
{1-\gamma\lambda_i\sigma_j},
$$

and

$$
Z^\star=Q\widetilde ZV^\mathsf{T}.
$$

The minimum absolute denominator is returned as `denominator_margin`. The
closed form is differentiable through the channel factor and source. For a
sparse, directed, or otherwise non-symmetric graph, `solve_mode="iterative"`
uses the same equation through the standard SILVA solver interface.

```python
spectrum = SILVAEfficientInfiniteGraphEquilibrium.precompute_spectrum(S)
model = SILVAEfficientInfiniteGraphEquilibrium(
    in_dim=features.shape[1],
    state_dim=128,
    out_dim=num_classes,
    gamma=0.8,
    solve_mode="auto",
)
result = model(features, S, spectrum=spectrum, return_result=True)
```

Do not materialize a dense eigendecomposition for a graph that does not fit in
memory. Use sparse iterative propagation and preserve the benchmark's graph
normalization convention.

## Multiscale Graph Implicit Network

### Parallel graph scales

MGNNI gives each graph-power scale $m$ its own infinite-depth state
[[79]](../paper/references.md#ref-79){ .silva-cite }:

$$
Z_m^\star
=\gamma (S^m)^\mathsf{T}Z_m^\star C_m^\mathsf{T}+X,
$$

$$
C_m=g(F_m)
=\frac{F_m^\mathsf{T}F_m}
{\lVert F_m^\mathsf{T}F_m\rVert_F+\epsilon_F}.
$$

Scale $m=1$ propagates through immediate neighborhoods at every fixed-point
update. Larger powers change the single-update receptive field while the
equilibrium still aggregates repeatedly.

### Nodewise attention fusion

For node $i$ and scale $m$, compute

$$
e_{im}=q^\mathsf{T}\tanh(W_a z_{im}^\star+b_a),
$$

$$
\beta_{im}
=\frac{\exp(e_{im})}
{\sum_r\exp(e_{ir})},
$$

$$
z_i=\sum_m\beta_{im}z_{im}^\star.
$$

The result object exposes every $Z_m^\star$, every solver trace, and the full
$N\times M$ attention matrix. This makes scale collapse, scale specialization,
and per-scale convergence observable.

```python
model = SILVAMultiscaleGraphImplicitNetwork(
    in_dim=features.shape[1],
    state_dim=128,
    out_dim=num_classes,
    scales=(1, 2, 4, 8),
    gamma=0.8,
    fusion="attention",
    config=per_scale_solver_or_one_shared_config,
)
result = model(features, graph_operator, return_result=True)
assert result.attention_weights.shape == (features.shape[0], 4)
```

For a graph-conditioned source $f(X,G)$, pass a module with
`forward(features, graph_operator)`. Its result must have shape
$(N,d_{\mathrm{state}})$:

```python
class GraphSource(nn.Module):
    def __init__(self, in_dim, state_dim):
        super().__init__()
        self.projection = nn.Linear(2 * in_dim, state_dim)

    def forward(self, features, graph_operator):
        propagated = graph_operator @ features
        return self.projection(torch.cat((features, propagated), dim=-1))

model = SILVAMultiscaleGraphImplicitNetwork(
    in_dim=features.shape[1],
    state_dim=128,
    out_dim=num_classes,
    graph_source=GraphSource(features.shape[1], 128),
)
```

The simple feature projection remains the source-aligned path for the published
node and graph configurations. `graph_source` is an additive SILVA extension
for architectures whose injection itself performs graph processing.

`fusion="mean"` is a controlled ablation. It is useful for separating the
benefit of multiple equilibrium scales from the learned fusion mechanism.

## Delta-Cached Equilibrium Inference

### Heterogeneous convergence

DeltaDEQ observes that state coordinates often stabilize at different rates
[[80]](../paper/references.md#ref-80){ .silva-cite }. For a linear recurrent
operator

$$
L(z)=Wz+b,
$$

cache $c_k=L(z_k)$. The exact linear identity is

$$
L(z_{k+1})
=L(z_k)+W(z_{k+1}-z_k).
$$

Define a thresholded delta

$$
\Delta_\tau z_{k+1}
=\mathbf 1\left(|z_{k+1}-z_k|>\tau\right)
\odot(z_{k+1}-z_k).
$$

The cached approximation is

$$
c_{k+1}=c_k+W\Delta_\tau z_{k+1}.
$$

At $\tau=0$, this is algebraically equivalent to full linear or convolutional
evaluation. Floating-point accumulation can differ by roundoff, so SILVA also
computes the exact full-map residual

$$
r_{\mathrm{exact}}
=\left\lVert z-\sigma(L(z)+S(x))\right\rVert_2.
$$

At $\tau>0$, the state is an approximation whose cost-quality tradeoff must be
measured. `mean_active_fraction` reports the retained fraction of state changes;
it is an activity proxy, not a hardware-independent wall-time guarantee.

```python
model = SILVADeltaEquilibrium(
    in_dim=feature_dim,
    state_dim=hidden_dim,
    out_dim=output_dim,
    recurrent=expensive_linear_or_convolution,
    source=source_module,
    activation=activation,
    readout=task_head,
    delta_threshold=1e-3,
    config=solver_config,
)

model.train()
prediction = model(batch, use_delta=False)

model.eval()
result = model(batch, use_delta=True, return_result=True)
print(result.mean_active_fraction, result.exact_residual)
```

This full-map-training/delta-evaluation route matches the source protocol.
SILVA additionally permits `use_delta=True` during training when
`backward_mode` is `"implicit"` or `"phantom"`. The forward state then uses
the cached approximation, while the backward sensitivity uses the exact full
map. This extension is useful for studying delta-aware training, but it must be
reported separately from source benchmark reproduction. Mutable cache steps
cannot use unrolled differentiation and raise an error rather than silently
producing a misleading gradient.

The source studies implicit image representation and optical flow. The SILVA
wrapper is operator-general: standard dense and one-, two-, or three-dimensional
convolutions can use cached increments. Nonlinear recurrent modules must be
decomposed so only their linear sub-operators are wrapped.

## Compact Known-Solution Data

Each builder isolates one mechanism and returns the parameters used to make its
target:

| Builder | Known structure | Main assertion |
| --- | --- | --- |
| `make_monotone_operator_dataset` | affine source plus monotone ReLU point | solver state and certificate |
| `make_positive_concave_dataset` | bounded positive-concave map | positive state and target shape |
| `make_non_euclidean_robustness_dataset` | clean and entrywise perturbed inputs | certificate and empirical sensitivity |
| `make_eignn_chain_dataset` | normalized chain and exact long-range state | closed-form/iterative agreement |
| `make_mgnni_multiscale_dataset` | exact states for several graph powers | per-scale states and attention normalization |
| `make_delta_heterogeneous_dataset` | diagonal affine map with known fixed point | full/delta agreement and activity reduction |

These generated problems are equation checks, not substitutes for MNIST,
CIFAR-10, citation graphs, implicit image representation, Sintel, or KITTI.

## Building a New Structured Family

An advanced user can retain a source mechanism while changing the architecture
inside the SILVA point:

1. Define the state shape and a source module $S_\theta(x)$.
2. Define a shape-preserving recurrent operator with the required certificate
   or algebraic structure.
3. Supply the proximal map, activation, graph propagation, fusion, or cache
   policy required by the family.
4. Choose a solver and verify both its residual and the family-specific
   diagnostic.
5. Attach any readout $Q_\psi$ that maps the equilibrium to the task target.
6. Validate forward shape, loss, backward gradients, checkpoint reload, and a
   compact known-solution case before scaling.

For example, a convolutional monotone operator is valid only if it provides the
same three methods used by the dense default:

```python
class StructuredMonotoneOperator(nn.Module):
    def forward(self, state):
        ...

    def resolvent(self, values, step_size):
        ...

    def monotonicity_certificate(self):
        ...
```

The equilibrium model then keeps its splitting, source, readout, result object,
and training interface unchanged.

## Source-Scale Reproduction

### What is implemented and what remains task-specific

| Family | Implemented source mechanism | Source-scale assembly still required |
| --- | --- | --- |
| monDEQ | dense monotone parameterization, forward-backward and Peaceman-Rachford splitting, resolvent interface, implicit gradients | choose the source dense, single-convolution, or multiscale convolutional operator; the packaged dense default does not silently stand in for the source vision operator |
| pcDEQ | both activation variants, dense/convolutional points, fixed-point iteration, source-style weight normalization and projection | assemble the source normalization, pooling, downsampling, classifier, augmentation, and training schedule around one or more points |
| NEMON | dense weighted-infinity parameterization, averaging, certificate, and sensitivity bound | provide the source convolutional operator and robustness protocol when reproducing the image experiments |
| EIGNN | normalized Gram map, exact dense spectral solve, iterative sparse/directed solve, differentiable source and readout | preserve the official split, adjacency normalization, preprocessing network, classifier, early stopping, and dense-versus-sparse choice |
| MGNNI | independent graph-power equilibria, per-scale factors and solvers, nodewise attention, mean ablation, optional graph-conditioned source | provide the source feature encoder, residual classifier or graph pooling head, split, normalization, scale list, and optimizer schedule |
| DeltaDEQ | thresholded linear/convolutional cache, exact zero-threshold identity, activity records, full-map residual, full-map and delta-forward training routes | instrument every eligible linear sub-operator in the source update block, restore the base checkpoint, reset all caches per sequence, and reproduce the optical-flow data and timing protocol |

The public objects are therefore sufficient building blocks for source-scale
construction, but the package does not claim that a compact constructor alone
is a benchmark-equivalent source model. The source-scale notebook cells print
the exact remaining obligations so a run cannot be mislabeled accidentally.

| Family | Start from | Source-scale controls | Required evaluation |
| --- | --- | --- | --- |
| monDEQ | source vision preprocessing and structured operator | width, monotone factors, split, step, tolerances | accuracy, certificate, residual, evaluations, memory |
| pcDEQ | source positive architecture and variant | positive parameterization, activation, kernels, FPI budget | accuracy, positivity, residual, runtime |
| NEMON | source clean/perturbed protocol | metric, one-sided bound, averaging, robustness norm | clean/robust metric, certificate, sensitivity |
| EIGNN | official graph split and normalization | gamma, width, dense spectrum or sparse solve | accuracy, denominator margin, runtime, memory |
| MGNNI | official graph tasks | graph powers, per-scale states, attention | node/graph metric, per-scale residual, fusion statistics |
| DeltaDEQ | trained source-compatible base model | threshold, warm start, early stop, precision, hardware | task metric, activity, exact residual, FLOPs and wall time |

Use `silva_reproduction_spec(family)` to obtain the complete data sources,
access notes, storage plan, metrics, notebooks, tests, preserved mechanisms, and
extension points for any of these families.

```python
spec = silva_reproduction_spec("mgnni")
print(spec.datasets)
print(spec.data_sources)
print(spec.source_scale_steps)
print(spec.configurable_parts)
```

### Executed Real-Data Layer

The focused notebooks now retain every known-solution section above and add a
second validation layer based on attributed public data:

| Family notebooks | Real source path | Executed evidence |
| --- | --- | --- |
| 36-38 | balanced source-indexed CIFAR-10 snapshot [[81]](../paper/references.md#ref-81) | loss and gradient paths, residuals, monotonicity or positivity certificates, perturbation response |
| 39-40 | connected source-indexed Cora subset with Planetoid masks [[82]](../paper/references.md#ref-82) | masked losses, graph residuals, predictions, and nodewise multiscale allocation |
| 41 | consecutive public real-video frames [[86]](../paper/references.md#ref-86) | threshold/activity curve and exact-convolution disagreement |

The Cora subset is intentionally labeled as an induced teaching graph. Calling
`load_planetoid_source_subset(..., subset_nodes=None)` restores the complete
transductive graph required for source comparisons. Sintel, KITTI Flow, and
FlyingChairs remain local access-controlled routes for supervised optical-flow
metrics. See [Real-Dataset Reproduction](real-dataset-reproduction.md) for
storage, receipts, loaders, and complete experiment checklists.

The complete-data budgeting route remains available in
[Full-Scale SILVA](full-scale-silva.md).

## Backward Policies Without Losing Structure

Monotonicity, positivity, geometric feasibility, graph normalization, and delta
activity are properties of the forward construction. A backward approximation
must not be described as proving those properties, but it can be evaluated
without replacing them.

| Forward family | Structure retained in every solve | Additional backward study |
| --- | --- | --- |
| monDEQ | monotone parameterization and declared splitting map | compare exact adjoint, JFB, and SHINE while preserving margin and step size |
| pcDEQ | nonnegative parameters and positive-concave activation | compare gradients without relaxing positivity projection |
| NEMON | weighted-infinity certificate and averaged update | report sensitivity bound beside gradient approximation error |
| EIGNN | normalized Gram operator and denominator margin | compare dense exact gradients with iterative implicit/JFB routes on the same graph |
| MGNNI | independent graph-power equilibria and fusion | report per-scale forward and backward residuals |
| DeltaDEQ | full-map fixed point and exact residual | distinguish delta-forward savings from the selected training gradient |

JFB replaces the inverse adjoint factor by identity
[[88]](../paper/references.md#ref-88){ .silva-cite }. SHINE shares the Broyden
inverse estimate and can refine it [[89]](../paper/references.md#ref-89){
.silva-cite }. The packaged modes are selected through `SolverConfig`; notebook
49 provides an analytic comparison before these modes are attached to a larger
structured operator.

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are the public constructor signatures? | [Structured Equilibria API](../api/structured_equilibria.md) |
| Where are the executable derivations and plots? | [Notebook Library](../notebooks.md) |
| How do I run all six families together? | [Structured Equilibria Example](../examples/structured-equilibria.md) |
| How do I run attributed real subsets or complete local data? | [Real-Dataset Reproduction](real-dataset-reproduction.md) |

The [numbered references and source repositories](../paper/references.md#numbered-reference-registry)
remain available beside the implementation and reproduction guides.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
