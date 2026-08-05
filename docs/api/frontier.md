# Recent Equilibrium API

`silva_networks.frontier` contains four research-derived mechanisms as SILVA
families. Each class keeps the source and state contracts explicit and returns
the diagnostic object appropriate to its solver.

## Family Map

| SILVA family | Transition or objective | Result object |
| --- | --- | --- |
| `SILVAFNODEQ` | input-injected Fourier block | `SILVAOperatorOutput` |
| `SILVAPhysicsGuidedGraphDEQ` | reaction, graph diffusion, directed transport | `SILVAPhysicsGraphOutput` |
| `SILVAHomotopyEquilibrium` | continuous residual flow $\dot z=T(z;x)-z$ | `SILVAHomotopyOutput` |
| `SILVADistributionalDEQ` | empirical-measure discrepancy descent | `SILVADistributionalResult` |

The mathematical derivations, citation mapping, extension boundaries, and
small reproductions are developed in
[Recent Equilibrium Families Inside SILVA](../learn/frontier-equilibrium-families.md).

## Constructor Selection

```python
from silva_networks import silva_equilibrium_model

model = silva_equilibrium_model(
    "silva_physics_graph_deq",
    in_dim=4,
    state_dim=16,
    out_dim=2,
)
```

Canonical keys are `silva_fno_deq`, `silva_physics_graph_deq`,
`silva_homotopy_equilibrium`, and `silva_distributional_deq`.

## Diagnostics

| Result | Main numerical fields |
| --- | --- |
| `SILVAOperatorOutput` | `state`, `solver_result.residuals`, `solver_result.converged` |
| `SILVAPhysicsGraphOutput` | `state`, `solver_result.residuals`, `solver_result.iterations` |
| `SILVAHomotopyOutput` | `state`, `terminal_residual`, `velocity_norms`, `steps`, `horizon` |
| `SILVADistributionalResult` | `state`, `transformed_state`, `discrepancies`, `converged` |

These quantities are numerical diagnostics. A task loss, PDE residual, physical
conservation error, or benchmark metric must be computed separately.
The [Full-Scale SILVA guide](../learn/full-scale-silva.md) carries these
families from compact checks to sharded and distributed dataset runs.

For large empirical measures, `distributional_discrepancy` and
`SILVADistributionalDEQ` accept `pairwise_chunk_size`. This retains the exact
energy-distance or Gaussian-MMD arithmetic while bounding the largest explicit
pair block. The arithmetic remains quadratic in particle count; chunking is a
memory control, not a complexity claim.

## API

::: silva_networks.frontier
    options:
      show_root_heading: true
      members_order: source
      show_signature_annotations: true

## Where to Go Next

| Question | Page |
| --- | --- |
| Where are all equations and branch mappings derived? | [Recent Equilibrium Families Inside SILVA](../learn/frontier-equilibrium-families.md) |
| Can I run all four cases together? | [Recent Equilibrium Examples](../examples/frontier-equilibria.md) |
| Can I execute each derivation cell by cell? | [Recent Equilibrium Families Notebook](../package-notebooks/16_frontier_equilibrium_families.ipynb) |
| Which family key should I select? | [Families API](families.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
