# Scaling

The scaling API connects every canonical family to its data contract,
literature, benchmark route, scale controls, extension points, numerical
defaults, and distributed runtime preparation. Task dimensions remain
explicit, and caller-provided constructor arguments always override the tier
defaults.

## Main Objects

| Object | Role |
| --- | --- |
| `SILVAFamilyGuide` | literature, benchmark, data, scaling, and extension contract |
| `full_scale_solver_config` | relative-residual forward and implicit-backward solver template |
| `build_scaled_silva` | canonical family factory with scale-sensitive numerical defaults |
| `SILVARuntimeConfig` | precision, batch, worker, checkpoint, distribution, and compilation choices |
| `prepare_silva_model` | device movement plus optional distributed and compiled wrapping |

The `smoke`, `workstation`, and `full` tiers alter numerical budgets and
runtime choices, not the SILVA state equation. Use the smoke tier to verify a
complete forward/loss/backward/checkpoint path before selecting a larger tier.

::: silva_networks.scaling

## Where to Go Next

| Question | Page |
| --- | --- |
| How are the scale equations and all 30 routes derived? | [Full-Scale SILVA](../learn/full-scale-silva.md) |
| How is sharded and distributed data loaded? | [Scaling Data API](scaling_data.md) |
| Can I execute the equivalence checks and training path? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |
| Which family key and constructor should I use? | [Families API](families.md) |
