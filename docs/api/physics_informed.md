# Physics-Informed API

Poisson mirror equilibrium, physics-informed ODE equilibrium, implicit DAE
stage layer, and adversarial residual objective.

`SILVAPhysicsInformedEquilibrium` supports dense and matrix-free implicit time
derivatives. The matrix-free route solves

$$
(I-J_zf)\,\frac{dz}{dt}=J_tf
$$

from JVPs. `SILVAImplicitDAEStep` likewise offers dense Newton and
Newton-Krylov linear solves. The small-state `auto` defaults preserve the dense
educational path, while `build_scaled_silva` selects the matrix-free routes.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects physics-informed and algebraic equilibrium layers to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
\mathcal L=\lambda_{\mathrm{data}}\mathcal L_{\mathrm{data}}+\lambda_{\mathrm{phys}}\|\partial_t\hat y-\mathcal N(t,\hat y)\|_2^2+\lambda_J\mathcal R_J
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the implicit representation and the decoded physical state. |
| Condition | the physical residual callable is evaluated on the decoded state with declared boundary or algebraic constraints. |
| Diagnostic | physics residual, boundary residual, equilibrium residual, and adjoint residual. |
| Replacement point | the dynamics, residual, boundary projector, regularizer, stage equation, or discriminator. |
| Scale axes | collocation points, temporal horizon, stiffness, stage count, tolerance, and precision. |

The relevant method lineage is recorded in [[50]](../paper/references.md#ref-50) through [[52]](../paper/references.md#ref-52). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/advanced_equilibria.py"
```

```bash
python examples/advanced_equilibria.py
```

### Measured Compact Output

```text
monotone graph: (8, 1) 0.023554455488920212
equilibrium transformer: 0.18536624312400818
Poisson mirror: 0.005979819223284721
physics-informed loss: 0.8003759384155273
implicit DAE step: [0.4761904776096344] 1.862645149230957e-09
adversarial residual objective: 0.7888258695602417 1.3886094093322754
```

### Interpret the Output

The DAE row verifies its stage equation directly, while the physics-informed row is a weighted training objective. A full study must print the individual objective components rather than only their sum.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.physics_informed

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the ODE, DAE, and loss equations derived? | [Physics-Informed Equilibria](../learn/physics-informed-equilibria.md) |
| Where are the mechanisms executed together? | [Advanced Equilibria Example](../examples/advanced-equilibria.md) |
| Which analytic ODE and DAE batches are available? | [Advanced Equilibrium Datasets](../learn/advanced-equilibrium-datasets.md) |
| How are the matrix-free systems derived? | [Full-Scale SILVA](../learn/full-scale-silva.md#physics-informed-derivatives) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
