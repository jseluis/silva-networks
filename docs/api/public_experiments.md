# Public Experiments

The public experiment runner executes the packaged smoke configurations used by
the benchmark cards and release checks. It is intentionally small: each JSON
config selects a dataset or synthetic problem, model family, solver settings,
and output path for reproducible local validation.

Use the console command when installed, or call the module directly:

```bash
silva-experiment --list-configs
silva-experiment --show-config solver_sweep
silva-experiment --config solver_sweep --output-dir outputs
python -m silva_networks.public_experiments --config graph_silva_smoke
```

The runner writes JSON metrics and keeps benchmark claims separate from package
API behavior. For the curated results, see [Benchmark Cards](../experiments/benchmark-cards.md)
and [Results](../results.md).

::: silva_networks.public_experiments
