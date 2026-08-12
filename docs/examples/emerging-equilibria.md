# Emerging Equilibria Example

Run all eight compact mechanisms from public package imports:

```bash
python examples/emerging_equilibria.py
```

The command needs no external download. It evaluates a contractive teacher,
mixed-boundary Poisson grid, heterogeneous material field, articulated stick,
typed Gaussian mesh, and exact Poisson field. These are equation checks and
small simulations. The command also solves an exact thermodynamic elasticity
cell and a timestep-conditioned fixed-point diffusion sequence. These are not
substitutes for source-scale benchmark runs.

Every compact case instantiates a state-preserving SILVA transition,

$$
z^{k+1}=T_\theta(z^k;x,\mathcal C),
\qquad
r^k=T_\theta(z^k;x,\mathcal C)-z^k,
$$

or the corresponding finite tied-depth, root-search, or reverse-diffusion
specialization. The tensor contract is explicit in each result: the leading
batch or node dimensions are retained, the repeated state shape is preserved,
and the readout supplies the physical output shape.

## What Is Checked

| Family | Compact evidence | Full experiment handoff |
| --- | --- | --- |
| consistency DEQ | analytic teacher equilibrium and few-step state | cache source-model trajectories, train the refiner, compare task quality and latency |
| Psi-GNN | fixed Dirichlet values and finite-difference Poisson system | load unstructured meshes, preserve mixed boundary types, train with the source residual protocol |
| IFNO | tied increment and a known heterogeneous-bar response | load material simulation or DIC fields, match channels, splits, depth, modes, and metrics |
| SNARF | two-bone forward deformation and canonical root recovery | load articulated sequences, sample occupancy, retain multi-start roots, extract meshes |
| mesh inference | distributed and centralized Gaussian estimates agree | reproduce carrier policies, typed evidence, asynchronous sweeps, and communication metrics |
| physics-guided diffusion | PDE energy decreases and boundaries remain fixed | train or load the source field prior, then run the published reverse schedule and coefficient shifts |
| TherINO | exact strain, stress, energy, and prescribed mean strain agree | regenerate periodic microstructures and finite-element labels, then restore the source operator and contrast tests |
| fixed-point diffusion | per-timestep roots, allocation, reuse, and gradient route | restore the source latent encoder, image data, architecture, schedule, checkpoints, and FID-50K protocol |

Every row uses the same SILVA discipline: declare the state, condition,
repeated transition, numerical policy, and output diagnostic. The larger
protocol changes data volume and modules without changing that contract.

The compact mechanisms follow C-DEQ
[[59]](../paper/references.md#ref-59){ .silva-cite }, Psi-GNN
[[60]](../paper/references.md#ref-60){ .silva-cite }, IFNO
[[61]](../paper/references.md#ref-61){ .silva-cite }, SNARF
[[62]](../paper/references.md#ref-62){ .silva-cite }, Mesh Inference
[[63]](../paper/references.md#ref-63){ .silva-cite }, and physics-guided PDE
diffusion [[64]](../paper/references.md#ref-64){ .silva-cite }, TherINO
[[73]](../paper/references.md#ref-73){ .silva-cite }, and Fixed-Point Diffusion
Models [[74]](../paper/references.md#ref-74){ .silva-cite }. Each numbered
link opens the complete citation and primary external source.


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the latent vector or tensor z**, the condition is **the injected observation x**, and the
repeated map is **the tied map f_theta(z, x)**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **family-specific exact-solution, boundary, reconstruction, or trajectory checks**. The
invariants that must remain true are **state shape and a decreasing or bounded residual**.


### Run the Complete Example

```bash
python examples/emerging_equilibria.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
consistency: {'shape': (4, 3), 'teacher_error': 5.960464477539063e-08}
psi_gnn: {'shape': (25, 1), 'boundary_error': 0.0}
ifno: {'shape': (2, 1, 4, 8), 'final_increment': 3.803837776184082}
snarf: {'shape': (7, 1), 'root_residual': 8.068445911391109e-10}
mesh: {'shape': (5, 2), 'centralized_error': 1.8730469264482963e-07}
physics_diffusion: {'shape': (1, 1, 8, 8), 'final_energy': 40.260433197021484}
therino: {'shape': (2, 3, 6, 6), 'strain_error': 1.4901161193847656e-08}
fixed_point_diffusion: {'shape': (2, 1, 6, 6), 'reverse_steps': 3}
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: emerging-equilibria
  state: the latent vector or tensor z
  condition: the injected observation x
  repeated_transition: the tied map f_theta(z, x)
  invariant_checks: state shape and a decreasing or bounded residual
  compact_evidence: family-specific exact-solution, boundary, reconstruction, or trajectory checks
  scale_axes: latent width, solver tolerance, and iteration budget
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **the cited dataset and complete architecture for the selected family**. Increase only one of
**latent width, solver tolerance, and iteration budget** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all eight methods derived? | [Emerging Equilibrium Methods](../learn/emerging-equilibrium-methods.md) |
| Which classes and result records are public? | [Emerging Equilibria API](../api/emerging_equilibria.md) |
| Which exact compact datasets are available? | [Emerging Equilibrium Data API](../api/emerging_data.md) |
| Where are the executed simulations and plots? | [Notebook Library](../notebooks.md#advanced-equilibrium-and-physics-track) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
