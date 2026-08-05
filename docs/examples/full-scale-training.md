# Full-Scale Training Example

This example begins with an equation-checked periodic PDE dataset, writes it as
lazy shards, trains an input-injected Fourier equilibrium through the common
SILVA training API, and records a resumable checkpoint. Replace the generated
fields with an official benchmark loader for a benchmark reproduction.

The target satisfies

$$
(-\Delta+m)u=f
$$

on a periodic grid. The model solves

$$
z^\star=F_\theta(z^\star;f),
\qquad \widehat u=R_\theta(z^\star).
$$

```python
from pathlib import Path

from silva_networks import (
    SILVAShardedTensorDataset,
    build_scaled_silva,
    fit_supervised,
    make_periodic_elliptic_dataset,
    make_silva_dataloader,
    runtime_for_tier,
    write_silva_tensor_shards,
)

root = Path("runs/periodic-fno-deq")
fields = make_periodic_elliptic_dataset(
    samples=4096,
    height=64,
    width=64,
    modes=12,
    seed=17,
)
manifest = write_silva_tensor_shards(
    {
        "x": fields.forcing[:, None],
        "y": fields.target[:, None],
    },
    root / "train-data",
    shard_size=256,
    overwrite=True,
)

runtime = runtime_for_tier(
    "workstation",
    mixed_precision="none",
    checkpoint_path=root / "checkpoint.pt",
)
dataset = SILVAShardedTensorDataset(manifest)
loader = make_silva_dataloader(
    dataset,
    runtime.data_config(shuffle=True),
)

model = build_scaled_silva(
    "silva_fno_deq",
    tier=runtime.tier,
    in_channels=1,
    state_channels=48,
    out_channels=1,
    modes_height=12,
    modes_width=12,
    block_depth=3,
)
result = fit_supervised(
    model,
    loader,
    config=runtime.train_config(
        task="regression",
        epochs=100,
        optimizer="adamw",
        lr=2e-4,
        weight_decay=1e-4,
        gradient_clipping=1.0,
    ),
)
print(result.history[-1])
```

## Resume

`runtime.train_config` sets `resume=True` when a checkpoint path is present.
Running the same program with a larger epoch count restores the optimizer,
scheduler, scaler, history, and random-number state before continuing.

## Move to a Published Benchmark

For Darcy flow or steady Navier-Stokes, preserve the official sample split and
normalization from the selected FNO/FNO-DEQ protocol
[[31]](../paper/references.md#ref-31){ .silva-cite }
[[43]](../paper/references.md#ref-43){ .silva-cite }. Replace only the dataset
construction above. Record the benchmark source, resolution, coefficients,
normalization, metric, and inverse transform used for reporting.

## Extend the SILVA Transition

`SILVAFNODEQ` is one family-specific constructor. For a new operator, use
`SILVAOperatorModel` or `SILVACortexLayer` and place the spectral block beside
boundary, local convolution, global context, or conservation-projection
branches. The solver and training code remain unchanged as long as the internal
mapping preserves the state shape.

## Where to Go Next

| Question | Page |
| --- | --- |
| What changes at full scale? | [Full-Scale SILVA](../learn/full-scale-silva.md) |
| How is FNO-DEQ derived? | [Recent Equilibrium Families](../learn/frontier-equilibrium-families.md#silva-fourier-equilibrium) |
| How do I build custom internal points? | [Point Architecture Catalog](point-architecture-catalog.md) |
| Which diagnostics should I retain? | [Interactive Diagnostics Lab](../learn/interactive-diagnostics-lab.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/examples.md"
<!-- silva-extension-path:end -->
