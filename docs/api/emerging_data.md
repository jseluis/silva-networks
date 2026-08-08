# Emerging Equilibrium Data

These deterministic builders supply compact known-solution problems for
consistency trajectories, mixed-boundary Poisson graphs, heterogeneous material
operators, articulated forward skinning, typed mesh inference, and
physics-guided diffusion.
The same module also provides an exact periodic elasticity cell for the
thermodynamic operator family and seeded latent fields with an analytic
timestep-conditioned target for fixed-point diffusion. Source image datasets
retain their independent access terms.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects emerging-family data generators to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
r_{\mathrm{data}}=\|\mathcal A(x,y,c)\|,\qquad r_{\mathrm{data}}\rightarrow 0
$$

| Part | What must remain inspectable |
| --- | --- |
| State | boundary graphs, coefficient fields, canonical points, mesh observations, and diffusion trajectories. |
| Condition | each batch keeps the coordinates, masks, conditions, and targets required to recompute its governing residual. |
| Diagnostic | boundary error, constitutive error, root residual, or energy. |
| Replacement point | the compact generator with a source-dataset adapter that returns the same named fields. |
| Scale axes | mesh density, spatial resolution, number of poses, diffusion steps, and stored trajectory count. |

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

The exact checks expose different invariants: zero boundary error, a small deformation root residual, a small strain error, and the declared reverse-step count. The diffusion energy is an objective value and is not expected to be zero after this compact run.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.emerging_data
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Where to Go Next

| Question | Page |
| --- | --- |
| How is each compact problem derived? | [Emerging Equilibrium Methods](../learn/emerging-equilibrium-methods.md) |
| How do I replace compact data with a source benchmark? | [Reconstructing Paper Experiments](../learn/reconstructing-paper-experiments.md) |
| Which public models consume these batches? | [Emerging Equilibria API](emerging_equilibria.md) |
| Where are the retained simulations and plots? | [Notebook Library](../notebooks.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
