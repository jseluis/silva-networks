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
