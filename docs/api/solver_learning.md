# Learned Solver API

This module keeps the task transition, initializer, residual representation,
learned Anderson controller, readout, and distillation loss independently
replaceable. The mathematical derivation and source-scale protocol are in
[Learned Solvers and Backward Approximations](../learn/solver-learning-and-gradients.md).

::: silva_networks.solver_learning
    options:
      show_root_heading: true
      show_source: false
      members_order: source

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->

## Where to Go Next

| Question | Page |
| --- | --- |
| How are learned Anderson updates derived? | [Learned Solvers and Backward Approximations](../learn/solver-learning-and-gradients.md#hyperdeq-learn-the-forward-solver) |
| How is the family executed? | [Learned Solvers Lab](../package-notebooks/48_silva_learned_solvers.ipynb) |
| How do I replace the transition? | [Advanced Extension Handbook](../learn/advanced-extension-handbook.md) |
