from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.source_data_families import run_source_subset_suite
from silva_networks import (
    SOURCE_DATASET_REGISTRY,
    available_bundled_source_snapshots,
    available_source_datasets,
    load_bundled_source_snapshot,
    load_darcy_source_subset,
    load_optical_flow_source_subset,
    load_planetoid_source_subset,
    load_source_snapshot,
    load_vision_source_subset,
    normalized_graph_operator,
    save_source_snapshot,
    source_dataset_info,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeVisionDataset:
    def __init__(self) -> None:
        self.targets = torch.arange(30) % 3
        self.images = torch.arange(30 * 3 * 8 * 8, dtype=torch.uint8).reshape(
            30, 3, 8, 8
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


class FakeGraphDataset:
    def __init__(self) -> None:
        nodes = 12
        forward = torch.stack(
            (torch.arange(nodes - 1), torch.arange(1, nodes))
        )
        reverse = forward.flip(0)
        self.data = SimpleNamespace(
            x=torch.arange(nodes * 4, dtype=torch.float32).reshape(nodes, 4),
            edge_index=torch.cat((forward, reverse), dim=1),
            y=torch.arange(nodes) % 3,
            train_mask=torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).bool(),
            val_mask=torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]).bool(),
            test_mask=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).bool(),
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.data


class FakeFlowDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        first = np.zeros((8, 10, 3), dtype=np.uint8)
        second = np.full((8, 10, 3), 127, dtype=np.uint8)
        flow = np.stack(
            (
                np.full((8, 10), 2.0, dtype=np.float32),
                np.full((8, 10), -4.0, dtype=np.float32),
            )
        )
        valid = np.ones((8, 10), dtype=bool)
        return first, second, flow, valid


def test_source_registry_exposes_metadata_and_domain_filter() -> None:
    assert len(SOURCE_DATASET_REGISTRY) >= 11
    assert "Cora" in available_source_datasets("graph")
    assert "Sintel" in available_source_datasets("optical flow")
    assert source_dataset_info("CIFAR10").official_protocol
    with pytest.raises(KeyError, match="Unknown source dataset"):
        source_dataset_info("missing")


def test_vision_subset_is_balanced_deterministic_and_receipted() -> None:
    dataset = FakeVisionDataset()
    first = load_vision_source_subset(
        "CIFAR10",
        dataset=dataset,
        samples_per_class=2,
        seed=19,
        image_size=(4, 4),
    )
    second = load_vision_source_subset(
        "CIFAR10",
        dataset=dataset,
        samples_per_class=2,
        seed=19,
        image_size=(4, 4),
    )

    assert first.images.shape == (6, 3, 4, 4)
    assert torch.bincount(first.labels).tolist() == [2, 2, 2]
    assert torch.equal(first.images, second.images)
    assert first.receipt == second.receipt
    assert len(first.receipt.content_sha256) == 64
    assert first.receipt.subset_size == 6
    assert first.receipt.as_dict()["selected_indices"] == list(
        first.receipt.selected_indices
    )


def test_vision_source_normalization_uses_registered_statistics() -> None:
    subset = load_vision_source_subset(
        "CIFAR10",
        dataset=FakeVisionDataset(),
        samples_per_class=1,
        normalization="source",
    )
    assert subset.images.shape == (3, 3, 8, 8)
    assert "CIFAR10" in subset.receipt.preprocessing[-1]
    assert torch.isfinite(subset.images).all()


def test_planetoid_full_graph_preserves_masks() -> None:
    subset = load_planetoid_source_subset(
        "Cora", dataset=FakeGraphDataset(), seed=4
    )
    assert subset.graph.x.shape == (12, 4)
    assert subset.train_mask.sum() == 3
    assert subset.validation_mask.sum() == 3
    assert subset.test_mask.sum() == 6
    assert torch.equal(subset.node_ids, torch.arange(12))
    assert subset.receipt.subset_size == 12
    assert subset.graph.metadata["subset_protocol"] == "full"


def test_planetoid_induced_subset_keeps_all_split_kinds_and_node_ids() -> None:
    first = load_planetoid_source_subset(
        "Cora", dataset=FakeGraphDataset(), subset_nodes=8, seed=7
    )
    second = load_planetoid_source_subset(
        "Cora", dataset=FakeGraphDataset(), subset_nodes=8, seed=7
    )
    assert first.graph.x.shape == (8, 4)
    assert first.graph.edge_index is not None
    assert int(first.graph.edge_index.max()) < 8
    assert first.train_mask.any()
    assert first.validation_mask.any()
    assert first.test_mask.any()
    assert torch.equal(first.node_ids, second.node_ids)
    assert first.receipt.content_sha256 == second.receipt.content_sha256
    assert "teaching protocol" in first.receipt.preprocessing[-1]


def test_normalized_graph_operator_supports_dense_and_sparse() -> None:
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    dense = normalized_graph_operator(edges, 3)
    sparse = normalized_graph_operator(edges, 3, dense=False)
    assert dense.shape == (3, 3)
    assert dense.isfinite().all()
    assert torch.allclose(dense, dense.T)
    assert sparse.layout == torch.sparse_coo
    assert torch.allclose(dense, sparse.to_dense())
    assert torch.linalg.eigvalsh(dense).abs().max() <= 1.0 + 1e-6


def test_optical_flow_subset_resizes_vectors_and_mask() -> None:
    subset = load_optical_flow_source_subset(
        "Sintel",
        root="unused",
        dataset=FakeFlowDataset(),
        image_size=(4, 5),
    )
    assert subset.frame1.shape == subset.frame2.shape == (1, 3, 4, 5)
    assert subset.flow is not None
    assert subset.flow.shape == (1, 2, 4, 5)
    assert torch.allclose(subset.flow[:, 0], torch.ones(1, 4, 5))
    assert torch.allclose(subset.flow[:, 1], -2.0 * torch.ones(1, 4, 5))
    assert subset.valid is not None
    assert subset.valid.shape == (1, 4, 5)
    assert subset.valid.all()


def test_darcy_npz_subset_is_deterministic(tmp_path) -> None:
    path = tmp_path / "darcy.npz"
    inputs = np.arange(10 * 2 * 4 * 4, dtype=np.float32).reshape(10, 2, 4, 4)
    targets = inputs[:, :1] * 0.25
    np.savez(path, coeff=inputs, solution=targets)

    first = load_darcy_source_subset(path, samples=4, seed=23)
    second = load_darcy_source_subset(path, samples=4, seed=23)
    assert first.inputs.shape == (4, 2, 4, 4)
    assert first.targets.shape == (4, 1, 4, 4)
    assert torch.equal(first.inputs, second.inputs)
    assert first.receipt == second.receipt


def test_source_loaders_validate_compact_selection_arguments(tmp_path) -> None:
    with pytest.raises(ValueError, match="samples_per_class"):
        load_vision_source_subset(
            "CIFAR10", dataset=FakeVisionDataset(), samples_per_class=0
        )
    with pytest.raises(ValueError, match="at least three"):
        load_planetoid_source_subset(
            "Cora", dataset=FakeGraphDataset(), subset_nodes=2
        )
    with pytest.raises(FileNotFoundError):
        load_darcy_source_subset(tmp_path / "missing.npz")


def test_source_snapshot_round_trip_verifies_tensor_content(tmp_path) -> None:
    subset = load_vision_source_subset(
        "CIFAR10",
        dataset=FakeVisionDataset(),
        samples_per_class=1,
        seed=3,
    )
    path = tmp_path / "source.pt"
    save_source_snapshot(
        path,
        tensors={"images": subset.images, "labels": subset.labels},
        receipt=subset.receipt,
    )
    restored = load_source_snapshot(path)
    assert restored.receipt == subset.receipt
    assert torch.equal(restored.tensors["images"], subset.images)
    assert torch.equal(restored.tensors["labels"], subset.labels)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["tensors"]["images"][0, 0, 0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        load_source_snapshot(path)


def test_source_snapshots_run_through_all_six_structured_families() -> None:
    metrics = run_source_subset_suite()
    assert set(metrics) == {
        "monotone_loss",
        "monotone_residual",
        "positive_loss",
        "positive_minimum_weight",
        "non_euclidean_logit_shift",
        "non_euclidean_measure",
        "eignn_loss",
        "eignn_residual",
        "mgnni_loss",
        "mgnni_mean_scale_entropy",
        "delta_cache_error",
        "delta_active_fraction",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["monotone_residual"] < 1e-5
    assert metrics["positive_minimum_weight"] >= 0.0
    assert metrics["non_euclidean_measure"] < 1.0
    assert metrics["eignn_residual"] < 1e-5
    assert 0.0 <= metrics["delta_active_fraction"] <= 1.0


def test_repository_source_snapshot_manifest_and_hashes() -> None:
    manifest_path = ROOT / "docs/assets/source-data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert set(manifest["snapshots"]) == {"cifar10", "cora", "motion"}
    for record in manifest["snapshots"].values():
        path = ROOT / record["path"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["file_sha256"]
        snapshot = load_source_snapshot(path)
        assert snapshot.receipt.as_dict() == record["receipt"]


def test_bundled_source_snapshots_match_documentation_copies() -> None:
    expected = {
        "cifar10": "cifar10-balanced-10.pt",
        "cora": "cora-induced-96.pt",
        "motion": "public-motion-frames-100-101.pt",
    }
    assert available_bundled_source_snapshots() == tuple(expected)
    for name, filename in expected.items():
        bundled = load_bundled_source_snapshot(name)
        documented = load_source_snapshot(ROOT / "docs/assets/source-data" / filename)
        assert bundled.receipt == documented.receipt
        assert bundled.tensors.keys() == documented.tensors.keys()
        for key in bundled.tensors:
            assert torch.equal(bundled.tensors[key], documented.tensors[key])
    with pytest.raises(KeyError, match="unknown bundled source snapshot"):
        load_bundled_source_snapshot("missing")


def test_structured_source_profiles_cover_all_six_families() -> None:
    config_root = ROOT / "src/silva_networks/configs"
    compact = json.loads(
        (config_root / "structured_real_subset_suite.json").read_text(
            encoding="utf-8"
        )
    )
    source_scale = json.loads(
        (config_root / "structured_source_scale_suite.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "silva_monotone_operator_equilibrium",
        "silva_positive_concave_equilibrium",
        "silva_non_euclidean_equilibrium",
        "silva_efficient_infinite_graph",
        "silva_multiscale_graph_implicit",
        "silva_delta_equilibrium",
    }
    assert set(compact["families"]) == expected
    assert set(source_scale["families"]) == expected
    assert compact["bundled_snapshots"] == {
        "cifar10": "cifar10",
        "cora": "cora",
        "motion": "motion",
    }
    assert compact["claims"]["benchmark_reproduction"] is False
    assert source_scale["claims"]["source_exact_defaults"] is False
