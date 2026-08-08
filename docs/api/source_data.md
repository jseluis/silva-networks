# Source Data

The source-data API complements deterministic known-solution generators with
attributed real-data subsets, complete local loaders, and checksum-verified
snapshots. Compact subsets validate integration; complete benchmark claims
still require the official source protocol.

## Registry

```python
from silva_networks import available_source_datasets, source_dataset_info

for name in available_source_datasets():
    info = source_dataset_info(name)
    print(name, info.domain, info.expected_storage)
```

The registry covers CIFAR-10, MNIST, SVHN, Cora, CiteSeer, PubMed, MPI Sintel,
KITTI Flow, FlyingChairs, the small public motion source, and DarcyFlowSmall.
Dataset citations are numbered
[[81]](../paper/references.md#ref-81)-[[86]](../paper/references.md#ref-86).

## Snapshot Verification

```python
from silva_networks import load_source_snapshot

snapshot = load_source_snapshot(
    "docs/assets/source-data/cora-induced-96.pt",
    verify=True,
)
print(snapshot.receipt.content_sha256)
print(snapshot.tensors.keys())
```

The serializer accepts an ordered tensor mapping and rejects content that does
not match the receipt:

```python
from silva_networks import save_source_snapshot

save_source_snapshot(
    "compact.pt",
    tensors={"images": subset.images, "labels": subset.labels},
    receipt=subset.receipt,
)
```

## Shape Contracts

| Result object | Required tensors |
| --- | --- |
| `SILVAVisionSourceSubset` | `images: (B,C,H,W)`, `labels: (B,)` |
| `SILVAGraphSourceSubset` | graph features and edges, split masks, original node ids |
| `SILVAFlowSourceSubset` | `frame1`, `frame2`, optional `(B,2,H,W)` flow and valid mask |
| `SILVAOperatorSourceSubset` | input and target fields with the same sample count |
| `SILVASourceSnapshot` | named tensors plus a verified `SourceDataReceipt` |

## API

Installed packages retain the same three compact snapshots used by the
executed documentation. Load one without relying on a repository-relative
path:

```python
from silva_networks import load_bundled_source_snapshot

cora = load_bundled_source_snapshot("cora")
print(cora.receipt.content_sha256)
print(cora.tensors["x"].shape)
```

The names `cifar10`, `cora`, and `motion` are returned by
`available_bundled_source_snapshots()`. Complete experiments still use the
source-specific loaders and official local data.

::: silva_networks.source_data
    options:
      members_order: source
      show_root_heading: true
      show_source: true

## Where to Go Next

| Question | Page |
| --- | --- |
| How are compact and complete experiments separated? | [Real-Dataset Reproduction](../learn/real-dataset-reproduction.md) |
| How do the six families consume the snapshots? | [Source-Data Family Example](../examples/source-data.md) |
| Which family constructors use these tensors? | [Structured Equilibria](structured_equilibria.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/api.md"
<!-- silva-extension-path:end -->
