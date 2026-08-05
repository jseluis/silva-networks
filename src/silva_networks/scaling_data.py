"""Sharded tensor datasets and distributed data loading for SILVA experiments."""

from __future__ import annotations

import bisect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

Tensor = torch.Tensor


@dataclass(frozen=True)
class SILVADataLoaderConfig:
    """Scale-aware options for a SILVA data loader."""

    batch_size: int
    workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = False
    distributed: bool | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        if self.persistent_workers and self.workers == 0:
            raise ValueError("persistent_workers requires workers > 0")


def write_silva_tensor_shards(
    tensors: Mapping[str, Tensor],
    directory: str | Path,
    *,
    shard_size: int,
    prefix: str = "shard",
    overwrite: bool = False,
) -> Path:
    """Write aligned tensors as independently loadable shards plus JSON metadata.

    Each tensor must have the same first dimension. The returned manifest can
    be opened by :class:`SILVAShardedTensorDataset` without loading every shard.
    """

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not tensors:
        raise ValueError("tensors cannot be empty")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a nonempty filename component")
    keys = tuple(tensors)
    if any(not key for key in keys):
        raise ValueError("tensor keys cannot be empty")
    if any(not torch.is_tensor(value) or value.dim() == 0 for value in tensors.values()):
        raise ValueError("every value must be a tensor with a sample dimension")
    lengths = {int(value.shape[0]) for value in tensors.values()}
    if len(lengths) != 1:
        raise ValueError("all tensors must share their first dimension")
    length = lengths.pop()
    if length < 1:
        raise ValueError("tensors must contain at least one sample")

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"{prefix}-manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    shard_plan = [
        (
            shard_index,
            start,
            min(start + shard_size, length),
            destination / f"{prefix}-{shard_index:05d}.pt",
        )
        for shard_index, start in enumerate(range(0, length, shard_size))
    ]
    if not overwrite:
        existing = next((path for _, _, _, path in shard_plan if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"refusing to overwrite existing shard: {existing}")
    shard_entries: list[dict[str, Any]] = []
    for _shard_index, start, stop, path in shard_plan:
        filename = path.name
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing shard: {path}")
        payload = {
            key: value[start:stop].detach().cpu().contiguous() for key, value in tensors.items()
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        shard_entries.append({"path": filename, "length": stop - start})

    metadata = {
        "format": "silva-tensor-shards",
        "version": 1,
        "length": length,
        "keys": list(keys),
        "tensors": {
            key: {
                "dtype": str(value.dtype).removeprefix("torch."),
                "sample_shape": list(value.shape[1:]),
            }
            for key, value in tensors.items()
        },
        "shards": shard_entries,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


class SILVAShardedTensorDataset(Dataset[dict[str, Tensor]]):
    """Lazy one-shard cache for aligned tensor datasets larger than memory."""

    def __init__(self, manifest: str | Path):
        self.manifest_path = Path(manifest)
        metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "silva-tensor-shards" or metadata.get("version") != 1:
            raise ValueError("unsupported SILVA tensor-shard manifest")
        self.keys = tuple(metadata.get("keys", ()))
        if not self.keys:
            raise ValueError("manifest must define tensor keys")
        self._shards = tuple(metadata.get("shards", ()))
        if not self._shards:
            raise ValueError("manifest must contain at least one shard")
        lengths = [int(entry["length"]) for entry in self._shards]
        if any(length < 1 for length in lengths):
            raise ValueError("every shard length must be positive")
        self._ends: list[int] = []
        running = 0
        for length in lengths:
            running += length
            self._ends.append(running)
        if running != int(metadata.get("length", -1)):
            raise ValueError("manifest length does not match shard lengths")
        self._length = running
        self._cached_index: int | None = None
        self._cached_payload: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        payload = self._load_shard(shard_index)
        local_index = index - start
        return {key: payload[key][local_index] for key in self.keys}

    def _load_shard(self, shard_index: int) -> dict[str, Tensor]:
        if self._cached_index == shard_index and self._cached_payload is not None:
            return self._cached_payload
        entry = self._shards[shard_index]
        path = self.manifest_path.parent / entry["path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != set(self.keys):
            raise ValueError(f"shard keys do not match manifest: {path}")
        expected = int(entry["length"])
        for key, value in payload.items():
            if not torch.is_tensor(value) or value.dim() == 0 or value.shape[0] != expected:
                raise ValueError(f"invalid tensor {key!r} in shard: {path}")
        self._cached_index = shard_index
        self._cached_payload = payload
        return payload


def make_silva_dataloader(
    dataset: Dataset[Any],
    config: SILVADataLoaderConfig,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader[Any]:
    """Build an ordinary or distributed data loader from one configuration."""

    distributed = (
        dist.is_available() and dist.is_initialized()
        if config.distributed is None
        else config.distributed
    )
    sampler: DistributedSampler[Any] | None = None
    if distributed:
        if rank is None or world_size is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "distributed loading requires an initialized process group or rank/world_size"
                )
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=config.shuffle,
            seed=config.seed,
            drop_last=config.drop_last,
        )
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": config.shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": config.workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
        "drop_last": config.drop_last,
    }
    if config.workers:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **kwargs)


__all__ = [
    "SILVADataLoaderConfig",
    "SILVAShardedTensorDataset",
    "make_silva_dataloader",
    "write_silva_tensor_shards",
]
