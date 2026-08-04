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

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all equations and branch assignments derived? | [Neural Operators, ODEs, PDEs, and SILVA](../learn/neural-operators-ode-pde.md) |
| Which numerical and model objects are public? | [Scientific Operators API](../api/scientific.md) |
| Where is the trained source-to-solution example? | [Neural Operators, ODEs, and PDEs Notebook](../package-notebooks/15_neural_operators_ode_pde.ipynb) |
