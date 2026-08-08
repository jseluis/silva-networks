# Family Reproduction Dossiers

These 44 dossiers expose the complete path from a governing equation to source-scale
experimentation. Each page records what the implementation preserves, what SILVA makes
replaceable, which compact checks have run, and what remains before a publication-scale
reproduction can be claimed.

## Coverage

| Domain | Families |
| --- | ---: |
| SILVA composition | 6 |
| core equilibrium | 8 |
| geometry and distributions | 2 |
| graphs and distributed systems | 6 |
| optimization and certified equilibria | 7 |
| physics and differential systems | 4 |
| scientific operators | 6 |
| vision and generation | 5 |

## Evidence Vocabulary

| Status | Meaning |
| --- | --- |
| `contract-verified` | Equation, tensor shape, constructor, and required metadata are checked. |
| `compact-verified` | A deterministic mechanism run includes forward, diagnostics, and gradients. |
| `subset-verified` | The official data path and evaluation code have run on a recorded subset. |
| `source-scale-reproduced` | The complete cited protocol and metric have run with archived artifacts. |
| `planned` | The route is specified but has not been reported as completed. |

## All Families

| Family | Domain | Current package evidence | Source-scale stage |
| --- | --- | --- | --- |
| [`silva_layer`](silva_layer.md) | SILVA composition | `compact-verified` | `planned` |
| [`silva_graph`](silva_graph.md) | SILVA composition | `compact-verified` | `planned` |
| [`silva_graph_preset`](silva_graph_preset.md) | SILVA composition | `compact-verified` | `planned` |
| [`silva_cortex`](silva_cortex.md) | SILVA composition | `compact-verified` | `planned` |
| [`silva_cortex_network`](silva_cortex_network.md) | SILVA composition | `compact-verified` | `planned` |
| [`silva_image_cortex`](silva_image_cortex.md) | SILVA composition | `compact-verified` | `planned` |
| [`compact_deq`](compact_deq.md) | core equilibrium | `compact-verified` | `planned` |
| [`message_passing_deq`](message_passing_deq.md) | core equilibrium | `compact-verified` | `planned` |
| [`mdeq`](mdeq.md) | core equilibrium | `compact-verified` | `planned` |
| [`multiscale_vision_deq`](multiscale_vision_deq.md) | core equilibrium | `compact-verified` | `planned` |
| [`sequence_deq`](sequence_deq.md) | core equilibrium | `compact-verified` | `planned` |
| [`implicit_graph`](implicit_graph.md) | core equilibrium | `compact-verified` | `planned` |
| [`implicit_neural_representation`](implicit_neural_representation.md) | core equilibrium | `compact-verified` | `planned` |
| [`diffusion_equilibrium`](diffusion_equilibrium.md) | core equilibrium | `compact-verified` | `planned` |
| [`scientific_operator`](scientific_operator.md) | scientific operators | `compact-verified` | `planned` |
| [`fourier_operator_equilibrium`](fourier_operator_equilibrium.md) | scientific operators | `compact-verified` | `planned` |
| [`implicit_time_step`](implicit_time_step.md) | scientific operators | `compact-verified` | `planned` |
| [`silva_deq_flow`](silva_deq_flow.md) | vision and generation | `compact-verified` | `planned` |
| [`raft_deq_flow`](raft_deq_flow.md) | vision and generation | `compact-verified` | `planned` |
| [`quadratic_optimization`](quadratic_optimization.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_projected_qp`](silva_projected_qp.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_fno_deq`](silva_fno_deq.md) | scientific operators | `compact-verified` | `planned` |
| [`silva_physics_graph_deq`](silva_physics_graph_deq.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_homotopy_equilibrium`](silva_homotopy_equilibrium.md) | physics and differential systems | `compact-verified` | `planned` |
| [`silva_distributional_deq`](silva_distributional_deq.md) | geometry and distributions | `compact-verified` | `planned` |
| [`silva_monotone_graph_equilibrium`](silva_monotone_graph_equilibrium.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_generative_equilibrium_transformer`](silva_generative_equilibrium_transformer.md) | vision and generation | `compact-verified` | `planned` |
| [`silva_poisson_mirror_equilibrium`](silva_poisson_mirror_equilibrium.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_physics_informed_equilibrium`](silva_physics_informed_equilibrium.md) | physics and differential systems | `compact-verified` | `planned` |
| [`silva_implicit_dae_step`](silva_implicit_dae_step.md) | physics and differential systems | `compact-verified` | `planned` |
| [`silva_consistency_deq`](silva_consistency_deq.md) | vision and generation | `compact-verified` | `planned` |
| [`silva_psi_gnn`](silva_psi_gnn.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_ifno`](silva_ifno.md) | scientific operators | `compact-verified` | `planned` |
| [`silva_snarf`](silva_snarf.md) | geometry and distributions | `compact-verified` | `planned` |
| [`silva_mesh_inference`](silva_mesh_inference.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_physics_guided_diffusion_pde`](silva_physics_guided_diffusion_pde.md) | physics and differential systems | `compact-verified` | `planned` |
| [`silva_therino`](silva_therino.md) | scientific operators | `compact-verified` | `planned` |
| [`silva_fixed_point_diffusion`](silva_fixed_point_diffusion.md) | vision and generation | `compact-verified` | `planned` |
| [`silva_monotone_operator_equilibrium`](silva_monotone_operator_equilibrium.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_positive_concave_equilibrium`](silva_positive_concave_equilibrium.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_non_euclidean_equilibrium`](silva_non_euclidean_equilibrium.md) | optimization and certified equilibria | `compact-verified` | `planned` |
| [`silva_efficient_infinite_graph`](silva_efficient_infinite_graph.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_multiscale_graph_implicit`](silva_multiscale_graph_implicit.md) | graphs and distributed systems | `compact-verified` | `planned` |
| [`silva_delta_equilibrium`](silva_delta_equilibrium.md) | optimization and certified equilibria | `compact-verified` | `planned` |

## How to Use a Dossier

1. Run the compact stage and keep its result as a regression fixture.
2. Replace one configurable part at a time and record the deviation.
3. Validate the official data loader on a small subset before increasing scale.
4. Archive configuration, data receipt, checkpoint, metrics, runtime, and diagnostics.
5. Promote the evidence status only after every requirement for that level has run.

<!-- silva-extension-path:start -->
--8<-- "includes/extension/learn.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How do I build a new family? | [Advanced Extension Handbook](../learn/advanced-extension-handbook.md) |
| How do compatible families compare? | [Cross-Family Comparisons](../experiments/cross-family-comparisons.md) |
| Which objects expose these contracts? | [Research-Depth API](../api/research_depth.md) |
| Where is the executable dossier audit? | [Family Dossier Lab](../package-notebooks/42_family_reproduction_dossiers.ipynb) |
