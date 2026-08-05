from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import TensorDataset

from silva_networks import (
    SILVADataLoaderConfig,
    SILVAShardedTensorDataset,
    make_silva_dataloader,
    write_silva_tensor_shards,
)


def test_sharded_tensor_dataset_round_trip_and_loader(tmp_path) -> None:
    x = torch.arange(33, dtype=torch.float32).reshape(11, 3)
    y = torch.arange(11)
    manifest = write_silva_tensor_shards(
        {"x": x, "y": y},
        tmp_path,
        shard_size=4,
    )
    dataset = SILVAShardedTensorDataset(manifest)
    loader = make_silva_dataloader(
        dataset,
        SILVADataLoaderConfig(batch_size=3, shuffle=False),
    )
    batches = list(loader)

    assert len(dataset) == 11
    assert torch.equal(dataset[0]["x"], x[0])
    assert torch.equal(dataset[-1]["y"], y[-1])
    assert torch.equal(torch.cat([batch["x"] for batch in batches]), x)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["length"] for entry in metadata["shards"]] == [4, 4, 3]


def test_sharded_data_validation_and_explicit_distributed_sampler(tmp_path) -> None:
    with pytest.raises(ValueError, match="first dimension"):
        write_silva_tensor_shards(
            {"x": torch.ones(3, 2), "y": torch.ones(4)},
            tmp_path,
            shard_size=2,
        )

    dataset = TensorDataset(torch.arange(12))
    loader = make_silva_dataloader(
        dataset,
        SILVADataLoaderConfig(
            batch_size=2,
            shuffle=False,
            distributed=True,
        ),
        rank=1,
        world_size=3,
    )

    assert len(loader.sampler) == 4


def test_existing_manifest_is_rejected_before_any_new_shard_is_written(tmp_path) -> None:
    manifest = tmp_path / "next-manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="manifest"):
        write_silva_tensor_shards(
            {"x": torch.ones(3, 2)},
            tmp_path,
            shard_size=2,
            prefix="next",
        )

    assert sorted(tmp_path.iterdir()) == [manifest]


def test_existing_later_shard_is_rejected_before_partial_write(tmp_path) -> None:
    existing = tmp_path / "next-00001.pt"
    existing.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="shard"):
        write_silva_tensor_shards(
            {"x": torch.ones(5, 2)},
            tmp_path,
            shard_size=2,
            prefix="next",
        )

    assert sorted(tmp_path.iterdir()) == [existing]
