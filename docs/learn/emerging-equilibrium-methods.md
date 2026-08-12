# Emerging Equilibrium Methods in SILVA

This chapter derives eight additional mechanisms and shows how each becomes a
configurable SILVA construction. The implementations preserve the defining
equations of the cited methods while exposing their components as ordinary
modules and callables. Compact datasets validate the mechanics; source-scale
results require the original data, preprocessing, checkpoints, and budgets.

## One SILVA View

Every family below can be read through the same state equation,

$$
z^\star=T_\theta(z^\star;x,\mathcal C),
$$

but the state and constraints differ. In C-DEQ, $z$ is a teacher solver state.
In Psi-GNN it is a boundary-aware graph field. In IFNO it is a material feature
field. In SNARF it is a canonical correspondence. In mesh inference it is a
typed distributed estimate. In physics-guided diffusion it is a field moving
toward low residual energy.

The public constructors do not hide those differences. Each one exposes the
domain transition, the numerical policy, and the readout separately.

## Consistency DEQ

The teacher is an ordinary SILVA equilibrium,

$$
z^\star=f_\theta(z^\star,x),
\qquad
F_\theta(z;x)=f_\theta(z,x)-z=0.
$$

Fixing the initial state and root solver selects one deterministic sequence
$\mathcal T=\{z_k\}_{k=0}^{K}$. C-DEQ interprets those samples as points on a
solver-induced fixed-point ODE and learns a direct terminal map
[[59]](../paper/references.md#ref-59){ .silva-cite }:

$$
g_\phi(z_{\leq t},t,x)
=c_{\mathrm{skip}}(t)z_t+c_{\mathrm{out}}(t)
P_\phi(z_{\leq t},t,x),
$$

$$
c_{\mathrm{skip}}(t)
=\left(\frac{t-\epsilon}{T-\epsilon}\right)^\gamma,
\qquad
c_{\mathrm{out}}(t)=1-c_{\mathrm{skip}}(t),
$$

$$
t_k=\epsilon+\left(1-e^{-\rho k}\right)(T-\epsilon).
$$

The terminal coefficients enforce $g_\phi(z_T,T,x)=z_T$. Earlier states rely
more strongly on the learned refiner. With two solver states, SILVA forms an
Anderson-structured proposal from the current and previous residuals. The
training objective keeps both the terminal anchor and repeated-application
stability:

$$
\mathcal L_{\mathrm{global}}
=d(g_\phi(z_{t_k},t_k,x),z_K),
$$

$$
\mathcal L_{\mathrm{local}}
=d(g_\phi(z_{t_k},t_k,x),
g_{\phi^-}(z_{t_{k-1}},t_{k-1},x)),
$$

$$
\mathcal L
=\lambda\mathcal L_{\mathrm{global}}
+(1-\lambda)\mathcal L_{\mathrm{local}}
+\lambda_{\mathrm{task}}\mathcal L_{\mathrm{task}}.
$$

```python
model = SILVAConsistencyDEQ(
    state_dim=512,
    condition_dim=512,
    teacher_transition=teacher_transition,
    initializer=task_specific_initializer,
    refiner=task_specific_refiner,
    readout=task_head,
    teacher_config=teacher_solver,
)
trajectory = model.teacher_trajectory(condition)
few_step = model(condition, steps=2, return_result=True)
```

For a new architecture, four contracts matter: the initializer maps the
rank-two condition to a batched vector, sequence, graph, or spatial state; the
teacher transition maps `(state, condition)` to the same state shape; the
refiner maps `(state, time, condition)` to that shape; and the readout may map
the final state to any task output. Supplying an initializer and custom refiner
therefore removes the default vector-state restriction without changing the
distillation loop.

## Psi-GNN for Mixed-Boundary Poisson Problems

For a first-order discretization of

$$
-\Delta u=f\quad\text{in }\Omega,
\qquad
u=g\quad\text{on }\partial\Omega_D,
\qquad
\frac{\partial u}{\partial n}=0
\quad\text{on }\partial\Omega_N,
$$

the finite-element system is $AU=B$. The physics loss is therefore
[[60]](../paper/references.md#ref-60){ .silva-cite }

$$
\mathcal L_{\mathrm{res}}(U,G)
=\frac1N\lVert AU-B\rVert_2^2.
$$

Dirichlet rows are replaced by identity rows. This makes those nodes emit their
known value without receiving updates, while interior and Neumann nodes retain
bidirectional interactions. The processor uses separate messages:

$$
\phi^{I}_{\rightarrow,i}
=\sum_{j\in\mathcal N(i)}
\Phi^{I}_{\rightarrow}(H_i,H_j,d_{ij},\lVert d_{ij}\rVert),
$$

$$
\phi^{I}_{\leftarrow,i}
=\sum_{j\in\mathcal N(i)}
\Phi^{I}_{\leftarrow}(H_i,H_j,d_{ji},\lVert d_{ji}\rVert),
$$

$$
z_i^I=H_i+\Lambda^I(H_i,b_i,\phi^I_{\rightarrow,i},
\phi^I_{\leftarrow,i}),
$$

while Neumann updates also receive the outward normal. The typed processor is
then solved implicitly:

$$
\widehat H=h_\theta(\widehat H,G),
\qquad
\widehat U=D_\theta(\widehat H).
$$

`SILVAPsiGNNProcessor` exposes all three message maps and both update maps.
`SILVAPsiGNN` exposes the encoder, forcing encoder, processor, decoder, and
solver. The complete loss can include the equation residual, light supervised
anchoring, a Hutchinson Jacobian penalty, latent consistency, and
encoder-decoder reconstruction.

```python
data = make_psi_poisson_grid(size=17)
model = SILVAPsiGNN(state_dim=32, processor=custom_processor)
result = model(
    data.initial_solution,
    data.forcing,
    data.coordinates,
    data.edge_index,
    data.node_types,
    boundary_values=data.boundary_values,
    normals=data.normals,
    return_result=True,
)
loss = model.loss(result, data.stiffness, data.rhs)
```

At inference, a trained model needs the mesh coordinates, directed graph,
forcing and boundary features, node types, and normals. The matrix $A$ is only
needed when measuring or training with the discretized physics residual.

<a id="ifno-material-operator"></a>

## IFNO for Heterogeneous Materials

IFNO begins with a pointwise lift $h_0=P[f]$, where $f$ may concatenate
coordinates, material descriptors, body forces, Dirichlet values, and traction.
It then reuses one Fourier residual increment
[[61]](../paper/references.md#ref-61){ .silva-cite }:

$$
h_{l+1}(x)=h_l(x)+\Delta t\,\sigma\left(
Wh_l(x)+\mathcal F^{-1}
\left(R_\theta\,\mathcal F[h_l]\right)(x)+c(x)
\right).
$$

The parameters are layer-independent. Dividing by $\Delta t$ shows that the
depth index discretizes a nonlocal evolution equation. At a converged deep
limit, the increment vanishes:

$$
\sigma\left(Wh^\star+
\mathcal F^{-1}(R_\theta\mathcal F[h^\star])+c\right)=0.
$$

SILVA supports both interpretations. `mode="unrolled"` applies a declared
number of shared residual steps, matching the finite-depth architecture.
`mode="equilibrium"` asks a root solver for the zero-increment state.

```python
model = SILVAIFNO(
    in_channels=8,
    state_channels=64,
    out_channels=2,
    depth=32,
    step_size=1 / 32,
    modes_height=16,
    modes_width=16,
    lift=material_lift,
    increment=shared_fourier_increment,
    boundary_projector=displacement_boundary,
    readout=displacement_and_damage_head,
)
```

For displacement and damage together, set `out_channels` to the total physical
channels and use a readout that applies suitable final activations to each
slice. For irregular domains, map the geometry to a structured computational
grid before the Fourier increment, or inject a nonuniform spectral operator.

<a id="snarf-style-forward-skinning"></a>

## SNARF Forward Skinning

Let the pose-independent canonical weight field satisfy

$$
w_b(x)\geq0,
\qquad
\sum_{b=1}^{B}w_b(x)=1.
$$

Given homogeneous bone transforms $B_b$, forward linear blend skinning is
[[62]](../paper/references.md#ref-62){ .silva-cite }

$$
d_w(x,B)=\sum_{b=1}^{B}w_b(x)B_b\bar x.
$$

A posed query $x'$ has canonical correspondences at every root

$$
d_w(x^\star,B)-x'=0.
$$

Because folds and contact can make this inverse one-to-many, SILVA initializes
one candidate with every inverse bone transform,

$$
x_b^0=B_b^{-1}\bar x',
$$

solves all candidates, rejects large residuals, queries the canonical occupancy
$f_\sigma(x^\star,p)$, and applies a differentiable soft union.

```python
model = SILVASNARF(
    coordinate_dim=3,
    bones=24,
    pose_dim=pose_dim,
    weight_field=canonical_weight_network,
    occupancy_field=pose_conditioned_occupancy,
    config=root_solver,
)
result = model(posed_queries, bone_transforms, pose=pose, return_result=True)
```

`sample_occupancy_grid` evaluates the posed field in chunks. A full mesh
pipeline can pass the returned scalar grid to a marching-cubes implementation;
keeping meshing outside the root module avoids forcing a geometry dependency on
users who only need occupancy or correspondences.

## Mesh Inference

For node $i$ and typed field $f$, let $a_i$ be a private anchor, $o_i$ an
admitted observation, $\lambda_i$ and $\tau_i$ their precisions, and $w_{ij}$ a
nonnegative receiver-controlled admission weight after source emission. The
linear-Gaussian equilibrium implemented in SILVA is
[[63]](../paper/references.md#ref-63){ .silva-cite }

$$
(\lambda_i+\tau_i+\sum_jw_{ij})z_i^\star
-\sum_jw_{ij}z_j^\star
=\lambda_i a_i+\tau_i o_i.
$$

Writing $Mz=b$, $M$ has positive diagonal and nonpositive off-diagonal entries.
It is weakly diagonally dominant, and anchoring makes the reachable component a
nonsingular M-matrix. Jacobi relaxation gives

$$
z_i^{k+1}=\frac{b_i+\sum_jw_{ij}z_j^k}
{\lambda_i+\tau_i+\sum_jw_{ij}}.
$$

The package reports the Z-matrix check, diagonal-dominance check, minimum real
eigenvalue, and Jacobi spectral radius, then compares the distributed result to
`torch.linalg.solve(M,b)`. These checks make a failed policy or disconnected
carrier visible instead of silently returning an answer.

```python
result = SILVAMeshInference()(
    anchors,
    anchor_precision,
    observations,
    observation_precision,
    admission,
    emission=emission,
    return_result=True,
)
assert result.certificate.jacobi_spectral_radius < 1
```

The implemented family covers the paper's linear-Gaussian mechanism. A
nonlinear associative-memory extension should retain typed observations,
receiver-controlled admission, lineage, source-novel forwarding, and an
explicit convergence diagnostic.

<a id="physics-guided-diffusion-pde-solver"></a>

## Physics-Guided Diffusion for PDEs

Let a standard field prior supply a reverse denoising proposal. Physics enters
only during inference through a residual energy
[[64]](../paper/references.md#ref-64){ .silva-cite }:

$$
E_{\mathrm{PDE}}(u)
=\frac12\lVert\mathcal L u+\mathcal N(u)-f\rVert_2^2.
$$

One SILVA reverse step is

$$
\widetilde u_t=\operatorname{Prior}(u_t,t),
$$

$$
\bar u_t=G_\sigma*\widetilde u_t,
$$

$$
u_{t-1}=\mathcal B\left(
\bar u_t-\eta_t\nabla E_{\mathrm{PDE}}(\bar u_t)
+\xi_t
\right).
$$

$G_\sigma$ suppresses unstable high-frequency residual gradients,
$\mathcal B$ enforces Dirichlet, Neumann, or periodic constraints after every
step, and $\xi_t$ is omitted for deterministic inference.

```python
sampler = SILVAPhysicsGuidedDiffusionPDE(
    energy=pde_energy,
    boundary_projector=hard_boundary_projection,
    noise_predictor=trained_field_prior,
    steps=1000,
    guidance_step=problem_step,
    prior_strength=prior_weight,
)
result = sampler(noise, condition=forcing, stochastic=True, return_result=True)
```

The prior, PDE residual, smoother, schedule, and boundary projector are
independent. A prior trained on normalized solution fields can therefore be
tested under new coefficients without changing its weights, provided the new
fields use the same representation and normalization.

## Thermodynamically Informed Material Equilibria

TherINO iterates directly on the physical strain rather than an unrelated
latent field [[73]](../paper/references.md#ref-73){ .silva-cite }. For local
stiffness $C(x)$ and strain $\varepsilon(x)$,

$$
\sigma(x)=C(x):\varepsilon(x),
\qquad
W(x)=\frac12\varepsilon(x):\sigma(x).
$$

The fixed encoder concatenates the current strain, constitutive stress, energy
density, and prescribed macroscopic strain:

$$
q(\varepsilon,C,\bar\varepsilon)
=\left[\varepsilon,\ C:\varepsilon,\
\frac12\varepsilon:(C:\varepsilon),\ \bar\varepsilon\right].
$$

The update is any differentiable shape-preserving operator $U_\theta$. The
SILVA transition solves

$$
\varepsilon^\star
=\Pi_{\bar\varepsilon}
\left(U_\theta(q(\varepsilon^\star,C,\bar\varepsilon))\right),
$$

where
$\Pi_{\bar\varepsilon}(v)=v-\langle v\rangle+\bar\varepsilon$ enforces the
macroscopic loading after every transition. The package default is a Fourier
point operator, but a convolutional hierarchy, U-Net, attention block, or
custom neural operator can be supplied through `update`.

```python
model = SILVATherINO(
    strain_components=6,
    encoder=constitutive_encoder,
    update=three_dimensional_operator,
    enforce_macro_strain=True,
    config=material_solver,
)
result = model(stiffness, macro_strain, return_result=True)
loss = model.loss(result, target_strain, stiffness)
```

The complete supervised objective separates strain, stress, and energy:

$$
\mathcal L
=\lambda_\varepsilon\|\varepsilon^\star-\varepsilon^{data}\|_2^2
+\lambda_\sigma\|C:\varepsilon^\star-\sigma^{data}\|_2^2
+\lambda_W\|W(\varepsilon^\star)-W^{data}\|_2^2.
$$

`make_therino_elastic_dataset` supplies an exact diagonal-elasticity cell. It
verifies constitutive contraction, mean loading, loss terms, and the root solve.
A source-scale result additionally requires the paper's periodic microstructure
generation, three-dimensional finite-element labels, Mandel representation,
contrast/loading split, normalization, Fourier configuration, optimizer, and
evaluation protocol.

## Fixed-Point Diffusion Denoisers

Fixed-Point Diffusion Models place an implicit block inside each denoising
network [[74]](../paper/references.md#ref-74){ .silva-cite }:

$$
h_t=f_{pre}(x_t),\qquad p_t=P(h_t),
$$

$$
z_t^\star=F_\theta(z_t^\star,p_t,e(t),c),
\qquad
\widehat\epsilon_t=f_{post}(z_t^\star).
$$

The reverse process is therefore a sequence of nearby fixed-point problems:

$$
x_{t_{k+1}}
=R(x_{t_k},\widehat\epsilon_{t_k},t_k,t_{k+1},\xi_k).
$$

`SILVAFixedPointDenoiser` exposes `preprocessor`, `projection`, `transition`,
`postprocessor`, and `config`. `SILVAFixedPointDiffusionModel` adds a descending
timestep sequence, one iteration budget per reverse step, a replaceable reverse
operator, and optional reuse of $z_{t_k}^\star$ as the next initialization.

```python
denoiser = SILVAFixedPointDenoiser(
    channels=latent_channels,
    preprocessor=pre_blocks,
    projection=input_injection,
    transition=timestep_conditioned_blocks,
    postprocessor=post_blocks,
    config=implicit_solver,
)
model = SILVAFixedPointDiffusionModel(
    denoiser,
    timesteps=reverse_timesteps,
    allocations=iterations_per_timestep,
    step_operator=reverse_process,
    reuse_equilibria=True,
)
```

The stochastic Jacobian-free route samples no-gradient and differentiable
transition counts:

$$
z_n=F_\theta^{\,n}(z_0,p,t),
\quad n\sim\mathcal U\{0,\ldots,N\},
$$

$$
\widetilde z
=F_\theta^{\,m}(\operatorname{stopgrad}(z_n),p,t),
\quad m\sim\mathcal U\{1,\ldots,M\}.
$$

This is distinct from `SILVADiffusionEquilibrium`, which represents all selected
reverse states as one triangular trajectory fixed point and provides
`step_operator` plus `data_consistency` for DeqIR-style joint restoration
[[49]](../paper/references.md#ref-49){ .silva-cite }. Both abstractions remain
available because they solve different implicit systems.

## Compact and Source-Scale Data

| Family | Compact builder | Source-scale route | Storage planning |
| --- | --- | --- | --- |
| C-DEQ | `make_consistency_teacher_dataset` | cache teacher trajectories for WikiText-103, ImageNet, or OGB | raw data plus `samples x stored_steps x state_elements x dtype_bytes` |
| Psi-GNN | `make_psi_poisson_grid` | regenerate the paper's 10,000 first-order meshes and mixed boundaries | sparse edges, node features, and optional FEM matrices dominate |
| IFNO | `make_ifno_material_dataset` | obtain or regenerate the selected material simulation/DIC fields | `samples x input/output channels x H x W x dtype_bytes` |
| SNARF | `make_snarf_stick_dataset` | use 2D Stick or obtain the licensed human-motion/mesh data | posed meshes and per-frame query samples dominate |
| Mesh | `make_mesh_gaussian_dataset` | generate carrier, forwarding, asymmetry, and noisy-estimation sweeps | normally below field-model datasets; policy traces dominate |
| PDE diffusion | `make_poisson_diffusion_dataset` | generate 4,000 normalized 64x64 solution fields per source setup | one scalar float32 snapshot set is about 66 MB before metadata/checkpoints |
| TherINO | `make_therino_elastic_dataset` | generate periodic microstructures and finite-element strain/stress localization labels | three-dimensional strain/stress fields, stiffness tensors, and solver labels dominate |
| fixed-point diffusion | `make_fixed_point_diffusion_dataset` | obtain one declared image task, encode latents, and reproduce the source schedule and checkpoints | raw images, latents, checkpoints, and 50,000 evaluation samples dominate |

The storage formula is more reliable than a single download number because
source repositories often provide several resolutions, variables, subjects,
or preprocessing variants. Record the exact archive checksum and split after
acquisition. Do not claim source-table reproduction from the compact builders.

## Source-Scale Run Plans

The reproduction registry now carries acquisition, access, storage, compact
data, and launch steps as executable metadata:

```python
from silva_networks import silva_reproduction_spec

plan = silva_reproduction_spec("silva_psi_gnn")
print(plan.data_sources)
print(plan.data_access)
print(plan.storage_plan)
print(plan.compact_data)
print(plan.source_scale_steps)
```

The same record is available from `silva-scale FAMILY --tier full --json`.
It does not download restricted assets or guess unpublished preprocessing.

### C-DEQ

The public route covers WikiText-103 [[65]](../paper/references.md#ref-65){
.silva-cite }, OGB node datasets [[66]](../paper/references.md#ref-66){
.silva-cite }, and registered ImageNet access [[67]](../paper/references.md#ref-67){
.silva-cite }, together with the reference implementation and checkpoints
[[59]](../paper/references.md#ref-59){ .silva-cite }. First reproduce the
teacher metric, freeze its initialization and solver, and cache the exact
states consumed by consistency training. A useful preflight uses 512-2,048
examples and two to eight stored states. This checks the complete trajectory,
loss, EMA, and few-step path before creating a large cache.

### Psi-GNN

The paper's benchmark is procedurally generated [[60]](../paper/references.md#ref-60){
.silva-cite }. Recreate first-order unstructured meshes with recorded Gmsh
[[72]](../paper/references.md#ref-72){ .silva-cite } parameters, preserve the
directed Dirichlet rows and mixed boundaries, and shard sparse graph and
finite-element objects. Begin with 32-128 meshes and verify the equation,
boundary, Jacobian, and root residuals. The complete route then uses the
paper's 6,000/2,000/2,000 split and variable-resolution evaluation.

### IFNO

The cited IFNO work uses generated constitutive simulations and experimental
DIC fields [[61]](../paper/references.md#ref-61){ .silva-cite }, but it does not
identify one public archive that can be redistributed by this package. The
reader can supply the source tensors, regenerate a declared material model, or
begin with the compact analytic heterogeneous field. A 64-256 sample,
32-by-32 preflight exercises loading, normalization, shared-depth training,
boundary projection, displacement/damage readout, and resolution transfer
before the reported grid, modes, and depth are restored.

### SNARF

The official repository provides preprocessing and checkpoint routes
[[62]](../paper/references.md#ref-62){ .silva-cite }. Human-model and motion
assets retain their own access terms: AMASS [[68]](../paper/references.md#ref-68){
.silva-cite }, D-FAUST [[69]](../paper/references.md#ref-69){ .silva-cite },
CAPE [[70]](../paper/references.md#ref-70){ .silva-cite }, and SMPL
[[71]](../paper/references.md#ref-71){ .silva-cite }. The package's articulated
stick is the unrestricted mechanism check. After acquisition, use one subject,
about ten frames, and a bounded query sample to verify preprocessing, root
success, occupancy, and mesh extraction before the complete subject protocol.

### Mesh Inference

The linear-Gaussian cases are generated from topology, typed observations,
precisions, policies, and seeds [[63]](../paper/references.md#ref-63){
.silva-cite }. No large external dataset is necessary. A compact sweep over
16-64 nodes and multiple seeds can already test centralized identity,
connectivity, asymmetry, anchor density, spectral radius, and message count.
Scaling increases cases and policy traces rather than changing the SILVA
operator.

### Physics-Guided Diffusion PDE

The source protocol generates normalized 64-by-64 Poisson, diffusion, and
Burgers fields [[64]](../paper/references.md#ref-64){ .silva-cite }. The package
includes an analytic Poisson subset with hard boundaries. Use 64-256 fields to
validate prior training, frozen-prior inference, Gaussian smoothing, energy
descent, and projection; then regenerate the source coefficient ranges and
4,000-field set. A scalar float32 set at that size is about 62.5 MiB before
conditioning, trajectories, optimizer state, and checkpoints.

### TherINO

The source task uses periodic heterogeneous elastic localization
[[73]](../paper/references.md#ref-73){ .silva-cite }. Begin with 16-64 small
two-dimensional generated cells and verify stiffness layout, mean loading,
stress, energy, and root residuals. The complete route restores the declared
three-dimensional microstructure generator, constituent tensors, finite-element
discretization, Mandel convention, loading directions, contrast split, Fourier
modes, solver, optimizer, and localization/homogenization metrics. Keep raw
geometry, constitutive tensors, numerical labels, normalization, and checkpoints
in separate versioned shards.

### Fixed-Point Diffusion

The source study evaluates ImageNet, FFHQ, CelebA-HQ, and LSUN Church with a
latent diffusion architecture [[74]](../paper/references.md#ref-74){
.silva-cite }. Dataset access terms differ. A preflight should encode 128-512
samples, use a short reverse schedule, and verify the internal block, timestep
conditioning, stochastic gradient route, allocation, and equilibrium reuse.
The complete experiment restores image preprocessing, latent encoder, diffusion
schedule, architecture, training budget, checkpoints, and FID-50K evaluation.
Report quality together with block evaluations, wall time, peak memory, and
per-timestep residuals.

## Full-Scale Checklist

1. Select the exact paper task and obtain its data under the source terms.
2. Record source revision, archive checksum, split, units, and normalization.
3. Instantiate the same SILVA class used in the compact notebook, replacing
   only widths, depths, task modules, solver budgets, and data loaders.
4. Save every configuration, seed, checkpoint, residual trace, and task metric.
5. Verify gradients and convergence on a small shard before distributed runs.
6. Compare against the source metric protocol and report deviations explicitly.

`silva_reproduction_spec(family)` returns these requirements programmatically.
`silva_family_guide(family)` returns the scale controls and extension points.

## Cross-Family Solver and Gradient Choices

The family determines what the equilibrium state means. The forward solver and
backward rule remain independent experimental variables when their assumptions
match that state.

| Family | Forward acceleration that can be studied | Backward choices | Required control experiment |
| --- | --- | --- | --- |
| C-DEQ | source consistency refiner, one/few-step policy, optional classical correction | exact implicit, JFB, or refiner differentiation | compare terminal error, residual, task metric, and latency against the same teacher path [[59]](../paper/references.md#ref-59){ .silva-cite } |
| IFNO and TherINO | classical Anderson/Broyden or a learned Anderson controller | exact implicit, JFB, SHINE | hold Fourier modes, grid, data, and transition weights fixed |
| fixed-point diffusion | warm starts, state reuse, learned allocation, or learned solver updates | exact implicit, JFB, SHINE, phantom | report quality and denoiser evaluations at every timestep |
| physics-guided diffusion PDE | schedule and guidance controls rather than one global root | differentiate through reverse steps or declared truncated path | preserve denoiser, PDE energy, smoothing, and boundary projection |
| SNARF | Broyden root and canonical-state warm starts | task-specific unrolling or implicit root gradients | compare correspondence success and mesh metrics under identical queries |
| Mesh Inference | local message schedule, damping, and stopping policy | explicit message differentiation or an implicit global formulation | compare centralized identity, message count, and topology conditions |

HyperDEQ [[87]](../paper/references.md#ref-87){ .silva-cite } learns the forward
solver; JFB [[88]](../paper/references.md#ref-88){ .silva-cite } and SHINE
[[89]](../paper/references.md#ref-89){ .silva-cite } change the backward path.
The [Equilibrium Expansion Atlas](equilibrium-expansion-atlas.md) derives these
axes and lists combinations that remain mathematically well defined.

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I inspect every constructor and result type? | [Emerging Equilibrium API](../api/emerging_equilibria.md) |
| Where are the compact known-solution builders? | [Emerging Equilibrium Data API](../api/emerging_data.md) |
| How do I reproduce a cited experiment responsibly? | [Reconstructing Paper Experiments](reconstructing-paper-experiments.md) |
| How do all families fit the SILVA grammar? | [Method Adaptation Atlas](method-adaptation-atlas.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->
