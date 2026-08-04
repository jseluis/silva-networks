# Public Experiments

The public experiment runner executes the packaged validation configurations
used by the benchmark cards and release checks. Each JSON config selects a
dataset or synthetic problem, model family, solver settings, optimization
budget, and output path for reproducible local validation.

Use the console command when installed, or call the module directly:

```bash
silva-experiment --list-configs
silva-experiment --show-config solver_sweep
silva-experiment --config solver_sweep --output-dir outputs
python -m silva_networks.public_experiments --config graph_silva_smoke
```

Nested values can be changed without editing the packaged file:

```bash
silva-experiment \
  --config solver_sweep \
  --set scale=0.15 \
  --set solvers.0.max_iter=60 \
  --set solvers.0.tol=1e-7 \
  --output-dir outputs
```

Override values are parsed as JSON when possible, so numbers, booleans, arrays,
and objects retain their types.

## Configuration Contract

Every configuration contains `name` and `kind`. Model runs then add the
relevant state dimensions, branch choices, solver block, dataset fields, and
optimization budget. For example:

```json
{
  "name": "solver_sweep",
  "kind": "solver_sweep",
  "seed": 11,
  "dim": 6,
  "scale": 0.2,
  "solvers": [
    {"solver": "picard", "max_iter": 40, "tol": 1e-5, "alpha": 0.7}
  ]
}
```

For the solver family, the generated records contain `iterations`,
`converged`, `residual`, `spectral_radius`, and `jacobian_norm_estimate`. Model
families add loss, accuracy or regression metrics, tensor shapes, and the
configuration choices needed to interpret them.

## SILVA Interpretation

The runner does not define a second model API. It constructs the same public
SILVA layers and solves

$$
z^\star=f_\theta(z^\star,x)
$$

with a serialized `SolverConfig`. This keeps a JSON experiment tied directly
to the Python objects documented elsewhere in the API.

The runner writes JSON metrics and keeps benchmark claims separate from package
API behavior. For the curated results, see [Benchmark Cards](../experiments/benchmark-cards.md)
and [Results](../results.md).

::: silva_networks.public_experiments
