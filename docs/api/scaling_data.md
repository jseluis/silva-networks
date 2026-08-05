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
