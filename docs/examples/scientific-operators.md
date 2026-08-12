# Scientific Operators

This example executes six related but distinct scientific constructions through
the public package API:

1. a finite explicit ODE trajectory;
2. an implicit diffusion step;
3. reaction-diffusion with a projected Dirichlet boundary;
4. a viscous Burgers step;
5. a Fourier equilibrium operator on two grid resolutions;
6. graph diffusion on non-grid connectivity.

The learned spectral construction follows the Fourier Neural Operator lineage
[[31]](../paper/references.md#ref-31){ .silva-cite }, the function-to-function
view follows neural-operator theory
[[32]](../paper/references.md#ref-32){ .silva-cite }, and the structured
equilibrium composition follows SILVA
[[1]](../paper/references.md#ref-1){ .silva-cite }.

```bash
python examples/scientific_operators.py
```

## ODE Trajectory

For the relaxation law

$$
\frac{dh}{dt}=-\lambda(h-u),
$$

the exact solution is

$$
h(t)=u+(h_0-u)e^{-\lambda t}.
$$

`SILVAEulerFlowBlock` computes a finite explicit-Euler trajectory. The example
reports its terminal error against this analytic solution. This is a time
integration check, not an equilibrium solve.

## Implicit Diffusion

For

$$
\frac{\partial u}{\partial t}=D\Delta u,
$$

backward Euler gives

$$
u^{n+1}=u^n+\Delta t\,D\Delta_hu^{n+1}.
$$

`SILVAImplicitTimeStep` treats $u^n$ as the stimulus and the discrete Laplacian
as the recurrent local field. The solver residual measures how closely the
returned state satisfies this implicit equation.

## Reaction-Diffusion and Burgers

The two nonlinear right-hand sides are

$$
R_{\mathrm{RD}}(u)=D\Delta_hu+r(u)+s,
$$

and

$$
R_{\mathrm{B}}(u)=-uD_hu+\nu\Delta_hu+s.
$$

The reaction-diffusion example applies `SILVADirichletBoundary2D` after every
transition, so its outer nodes satisfy the prescribed value exactly. The
Burgers example uses periodic central differences on a one-dimensional field.
Both are deliberately one-step examples; a trajectory repeats the time-step
module and records the numerical state after each solve.

## Learned Fourier Operator

The operator model receives two channels, which can represent a coefficient
field $a(x)$ and source $q(x)$. It computes

$$
z^\star
=
\Psi\left[
R_\phi(a,q)
+B_{\mathrm{FNO},\theta}(z^\star)
+L_\theta(z^\star)
+G_\theta(z^\star)
\right],
\qquad
\widehat u=Q_\omega(z^\star).
$$

The same parameters run on two spatial resolutions. This verifies the tensor
and parameterization contract; learned resolution transfer must still be
evaluated on held-out data and with physical diagnostics.

## Irregular Graph PDE

On a graph, the local discrete Laplacian can be written

$$
(\Delta_Gz)_i
=
\sum_{j:(j,i)\in E}(z_j-z_i).
$$

The example supplies this field through `local_terms` of `SILVACortexLayer`.
Changing `edge_index` changes the sampled geometry without changing the solver
contract. Edge lengths, areas, conductivities, or learned messages can be added
as `edge_attr` in a problem-specific local module.

## What the Output Means

| Printed value | Interpretation |
| --- | --- |
| ODE Euler error | explicit trajectory error against the analytic terminal state |
| implicit-step residual | numerical self-consistency of the backward-Euler solve |
| boundary error | violation of the prescribed outer-node values |
| Fourier output shape | source-to-field tensor contract at one resolution |
| graph PDE shape | node-state contract on the selected connectivity |

These checks establish that each construction runs and differentiates. They are
not accuracy benchmarks. A scientific study should additionally report held-out
field error, PDE residual, boundary error, solver iterations, convergence rate,
runtime, and resolution or mesh transfer.

## Complete Source

```python
--8<-- "examples/scientific_operators.py"
```


<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the evolving or terminal physical state**, the condition is **time, initial condition, and external forcing**, and the
repeated map is **an explicit flow step or residual field T(z, x) - z**.

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
The reader-facing evidence for this route is **ODE error plus PDE, boundary, and equilibrium residuals**. The
invariants that must remain true are **time-step shape, initial condition, and integration consistency**.


### Run the Complete Example

```bash
python examples/scientific_operators.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
ODE Euler error 0.00819125771522522
implicit diffusion 40 1.2287812012345967e-07
reaction diffusion 4.5529768044616503e-07 boundary 0.0
Burgers 6 6.347658541017154e-07
Fourier operator (8, 8) (2, 1, 8, 8) 2.078301191329956
Fourier operator (12, 10) (2, 1, 12, 10) 2.870699644088745
graph PDE (8, 1) 9 4.807413347407419e-07
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
  example: scientific-operators
  state: the evolving or terminal physical state
  condition: time, initial condition, and external forcing
  repeated_transition: an explicit flow step or residual field T(z, x) - z
  invariant_checks: time-step shape, initial condition, and integration consistency
  compact_evidence: ODE error plus PDE, boundary, and equilibrium residuals
  scale_axes: time horizon, step count, state dimension, and stiffness
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

At full scale, move toward **the target mesh, time horizon, forcing distribution, and physical metric suite**. Increase only one of
**time horizon, step count, state dimension, and stiffness** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all equations and branch assignments derived? | [Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md) |
| Which numerical and model objects are public? | [Scientific Operators API](../api/scientific.md) |
| Where is the trained source-to-solution example? | [Neural Operators, ODEs, and PDEs Notebook](../package-notebooks/15_neural_operators_ode_pde.ipynb) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
