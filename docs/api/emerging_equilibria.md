# Emerging Equilibrium Families

The public classes below implement eight additional equilibrium mechanisms with
replaceable transitions, physical operators, numerical methods, and readouts.
The compact defaults support inspection and tests; the same constructors accept
benchmark-scale modules and data.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects emerging equilibrium mechanisms to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
F_\theta(z;c)=T_\theta(z;c)-z=0
$$

| Part | What must remain inspectable |
| --- | --- |
| State | a family-specific implicit state with an explicit condition bundle. |
| Condition | calling the transition again at the returned state must reproduce that state within the solver tolerance. |
| Diagnostic | family invariant plus normalized fixed-point residual. |
| Replacement point | every default backbone, processor, increment, deformation, energy, constitutive map, or denoiser. |
| Scale axes | state dimension, discretization size, trajectory depth, solver policy, and checkpoint schedule. |

The relevant method lineage is recorded in [[59]](../paper/references.md#ref-59) through [[64]](../paper/references.md#ref-64), [[73]](../paper/references.md#ref-73), and [[74]](../paper/references.md#ref-74). Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/emerging_equilibria.py"
```

```bash
python examples/emerging_equilibria.py
```

### Measured Compact Output

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

The program validates all eight mechanisms independently. Each reported quantity has a family-specific meaning, so a scale study must retain both the shared fixed-point residual and the named physical or structural check.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.emerging_equilibria
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the equations derived and connected to SILVA? | [Emerging Equilibrium Methods](../learn/emerging-equilibrium-methods.md) |
| Which datasets and full-scale settings are required? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |
| Where are the executable compact studies? | [Notebook Overview](../notebooks.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
