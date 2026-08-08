# Scaling Data

Lazy tensor shards and distributed loaders keep data execution independent from
the selected SILVA family. A JSON manifest records aligned tensor keys, sample
shapes, dtypes, order, and shard lengths. `SILVAShardedTensorDataset` then keeps
only one shard cached in each worker process.

## Data Flow

```python
from silva_networks import (
    SILVAShardedTensorDataset,
    make_silva_dataloader,
    runtime_for_tier,
    write_silva_tensor_shards,
)

manifest = write_silva_tensor_shards(
    {"x": inputs, "y": targets},
    "data/train",
    shard_size=512,
)
dataset = SILVAShardedTensorDataset(manifest)
runtime = runtime_for_tier("workstation")
loader = make_silva_dataloader(dataset, runtime.data_config())
```

Writing is atomic per shard and for the final manifest. Existing manifests or
shards are rejected unless `overwrite=True`. For distributed runs, the loader
uses one `DistributedSampler`; `fit_supervised` advances its epoch before each
training pass.

<!-- silva-api-study:start -->
## Operational Contract

This API surface connects coverage, reproduction, data, and scale configuration to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
F_\theta(z;x)=0,\qquad \widehat F_{\theta,s}(z;x)=0\ \text{uses the same mathematical contract at scale tier }s
$$

| Part | What must remain inspectable |
| --- | --- |
| State | the selected family, constructor contract, runtime tier, and data-loader configuration. |
| Condition | changing a runtime tier may change numerical budgets and resource use but must not silently change the family equation. |
| Diagnostic | coverage record, verification level, solver settings, effective batch size, and source-scale metrics. |
| Replacement point | compact defaults with family-specific modules, official data adapters, and an archived experiment configuration. |
| Scale axes | solver iterations, tolerance, model width, batch size, precision, workers, process count, and checkpoint interval. |

The relevant method lineage is recorded in the SILVA construction [[1]](../paper/references.md#ref-1) and the selected family's primary references. Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/api_scale_workflow.py"
```

```bash
python examples/api_scale_workflow.py
```

### Measured Compact Output

```text
family fno_deq
public objects 12
verification compact-verified
benchmark tasks 2
solver anderson
max iterations 12
runtime auto none
loader 4 0
```

### Interpret the Output

The family resolves through four independent registries: public coverage, source relation, scale guidance, and runtime/data configuration. The compact-verified label describes repository evidence; it does not convert the two listed benchmark tasks into claimed benchmark results.

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

<!-- silva-api-study:end -->

::: silva_networks.scaling_data

## Where to Go Next

| Question | Page |
| --- | --- |
| Where is a complete PDE sharding and training program? | [Full-Scale Training](../examples/full-scale-training.md) |
| Which runtime settings produce the loader configuration? | [Scaling API](scaling.md) |
| Can I execute a shard round trip and checkpoint resume? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |
| How are deterministic teaching datasets constructed? | [Dataset-Backed Equilibrium Labs](../learn/frontier-dataset-labs.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
