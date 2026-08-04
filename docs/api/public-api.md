# Public API

This page is generated from the package root, `silva_networks`. It is the
canonical reference for names imported by:

```python
from silva_networks import *
```

Use the role-specific pages when you already know the area you need. Use this
page when you need to confirm whether a name is public, stable, and exported
from the top-level package.

## Import Contract

The public contract is the package `__all__` list. Every name documented here
is importable from `silva_networks`, and every role-specific API page documents
the source module behind that name.

## Recommended Import Style

Explicit imports make model definitions and experiment records easier to read:

```python
from silva_networks import SILVACortexLayer, SolverConfig, stability_report
```

The wildcard form above demonstrates export completeness; application code
should normally name the objects it uses.

## Route by Task

| Goal | Start with |
| --- | --- |
| solve and inspect a fixed point | [Solvers](solvers.md), [Diagnostics](diagnostics.md) |
| construct one structured equilibrium | [Layers](layers.md) |
| link several equilibrium points | [Architectures](architectures.md) |
| choose an internal point mapping | [Point Architectures](point_architectures.md) |
| adapt tables, images, or graphs | [Datasets](datasets.md) |
| train and evaluate | [Training](training.md) |
| reproduce a packaged run | [Public Experiments](public_experiments.md) |

Every equilibrium-facing route ultimately evaluates
\(z^\star=f_\theta(z^\star,x)\); the role-specific pages explain how the state,
transition, solver result, and diagnostics are represented for that task.

::: silva_networks
