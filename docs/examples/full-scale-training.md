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

<!-- silva-worked-example:start -->
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**the tensor solved to equilibrium**, the condition is **the observed input or source tensor**, and the
repeated map is **the state-preserving transition evaluated by the root solver**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{\mathrm{task}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{\lVert T_\theta(z_K,x)-z_K\rVert_2}
{\lVert z_K\rVert_2+\varepsilon}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{\mathrm d z^\star}{\mathrm d\theta}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{-1}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **a measured training loss from the complete optimization path**. The
invariants that must remain true are **shape, device, dtype, finiteness, and differentiability**.


### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
--8<-- "examples/add_layers_on_top.py"
```

### Run the Complete Example

```bash
python examples/add_layers_on_top.py
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
final_loss 0.5183809995651245
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: full-scale-training
  state: the tensor solved to equilibrium
  condition: the observed input or source tensor
  repeated_transition: the state-preserving transition evaluated by the root solver
  invariant_checks: shape, device, dtype, finiteness, and differentiability
  compact_evidence: a measured training loss from the complete optimization path
  scale_axes: state width, batch size, and data volume
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **the configured sharded or distributed training run with resume checks**. Increase only one of
**state width, batch size, and data volume** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
<!-- silva-worked-example:end -->

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
