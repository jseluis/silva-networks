from __future__ import annotations

import csv
import math
import ssl
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import certifi
import numpy as np
import torch

TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a downloadable public dataset."""

    name: str
    domain: str
    task: TaskKind
    url: str
    file_name: str
    description: str
    source: str
    delimiter: str | None = ","
    has_header: bool = False
    target_column: int = -1
    feature_columns: tuple[int, ...] | None = None
    drop_columns: tuple[int, ...] = ()


@dataclass
class TabularDataset:
    """In-memory tabular dataset returned by ``load_tabular_dataset``."""

    name: str
    x: np.ndarray
    y: np.ndarray
    task: TaskKind
    feature_names: list[str]
    target_names: list[str]
    path: Path
    info: DatasetInfo

    def tensors(
        self, device: str | torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` as PyTorch tensors."""

        x = torch.as_tensor(self.x, dtype=torch.float32, device=device)
        if self.task == "classification":
            y = torch.as_tensor(self.y, dtype=torch.long, device=device)
        else:
            y = torch.as_tensor(self.y, dtype=torch.float32, device=device)
        return x, y


@dataclass(frozen=True)
class FeatureStandardization:
    """Training-split statistics for leakage-free NumPy preprocessing."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Impute and standardize features using these fitted statistics."""

        if x.ndim != 2 or x.shape[1] != self.mean.shape[1]:
            raise ValueError("x must be a 2D array with the fitted feature width")
        values = x.astype(np.float32, copy=True)
        values = np.where(np.isfinite(values), values, self.mean)
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class TensorStandardization:
    """Training-split statistics for leakage-free tensor preprocessing."""

    mean: torch.Tensor
    scale: torch.Tensor

    def to(self, device: str | torch.device) -> TensorStandardization:
        return TensorStandardization(self.mean.to(device), self.scale.to(device))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Impute and standardize features using these fitted statistics."""

        if x.dim() != 2 or x.shape[1] != self.mean.shape[1]:
            raise ValueError("x must be a 2D tensor with the fitted feature width")
        values = x.to(device=self.mean.device, dtype=self.mean.dtype)
        values = torch.where(torch.isfinite(values), values, self.mean)
        return (values - self.mean) / self.scale


@dataclass
class GraphTensorBatch:
    """SILVA-ready tensor container.

    The core SILVA graph and set APIs consume tensors with this structure. A
    dataset adapter may create these tensors from public datasets, private files,
    simulations, image grids, molecular tables, or user-defined objects.

    Attributes:
        x: Entity features with shape `(entities, features)`.
        edge_index: Optional source/destination edge tensor with shape
            `(2, edges)`.
        y: Optional labels or regression targets.
        batch: Optional graph/set id for each entity with shape `(entities,)`.
        edge_attr: Optional edge features with shape `(edges, edge_features)`.
        metadata: Optional descriptive fields such as feature names or source.
    """

    x: torch.Tensor
    edge_index: torch.Tensor | None = None
    y: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    @property
    def num_entities(self) -> int:
        """Number of entity rows in `x`."""

        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        """Number of edges in `edge_index`, or zero when no edges are present."""

        return 0 if self.edge_index is None else int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        """Number of graph/set ids represented by `batch`."""

        if self.batch is None or self.batch.numel() == 0:
            return 1
        return int(self.batch.max().item()) + 1

    def to(self, device: str | torch.device) -> GraphTensorBatch:
        """Move all tensor fields to a PyTorch device."""

        return GraphTensorBatch(
            x=self.x.to(device),
            edge_index=None if self.edge_index is None else self.edge_index.to(device),
            y=None if self.y is None else self.y.to(device),
            batch=None if self.batch is None else self.batch.to(device),
            edge_attr=None if self.edge_attr is None else self.edge_attr.to(device),
            metadata=self.metadata,
        )

    def model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return keyword tensors accepted by SILVA graph-style models."""

        kwargs: dict[str, torch.Tensor] = {"x": self.x}
        if self.edge_index is not None:
            kwargs["edge_index"] = self.edge_index
        if self.edge_attr is not None:
            kwargs["edge_attr"] = self.edge_attr
        if self.batch is not None:
            kwargs["batch"] = self.batch
        return kwargs

    def validate(self, raise_on_error: bool = True) -> bool:
        """Validate that the tensor fields satisfy the SILVA graph contract."""

        return validate_graph_tensor_batch(self, raise_on_error=raise_on_error)


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "iris": DatasetInfo(
        name="iris",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        file_name="iris.data",
        description="Fisher iris flower measurements with three species labels.",
        source="UCI Machine Learning Repository",
        target_column=4,
    ),
    "wine": DatasetInfo(
        name="wine",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        file_name="wine.data",
        description="Chemical measurements for wines from three cultivars.",
        source="UCI Machine Learning Repository",
        target_column=0,
        feature_columns=tuple(range(1, 14)),
    ),
    "wdbc": DatasetInfo(
        name="wdbc",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        file_name="wdbc.data",
        description="Wisconsin diagnostic breast cancer measurements.",
        source="UCI Machine Learning Repository",
        target_column=1,
        feature_columns=tuple(range(2, 32)),
        drop_columns=(0,),
    ),
    "seeds": DatasetInfo(
        name="seeds",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        file_name="seeds_dataset.txt",
        description="Wheat kernel geometric measurements.",
        source="UCI Machine Learning Repository",
        delimiter=None,
        target_column=7,
    ),
    "abalone": DatasetInfo(
        name="abalone",
        domain="tabular",
        task="regression",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        file_name="abalone.data",
        description="Abalone physical measurements with rings as age proxy.",
        source="UCI Machine Learning Repository",
        target_column=8,
    ),
    "yeast": DatasetInfo(
        name="yeast",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
        file_name="yeast.data",
        description="Protein localization measurements in yeast.",
        source="UCI Machine Learning Repository",
        delimiter=None,
        target_column=9,
        feature_columns=tuple(range(1, 9)),
        drop_columns=(0,),
    ),
    "airfoil_self_noise": DatasetInfo(
        name="airfoil_self_noise",
        domain="tabular",
        task="regression",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        file_name="airfoil_self_noise.dat",
        description="NASA airfoil self-noise measurements.",
        source="UCI Machine Learning Repository",
        delimiter=None,
        target_column=5,
    ),
    "wine_quality_red": DatasetInfo(
        name="wine_quality_red",
        domain="tabular",
        task="regression",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        file_name="winequality-red.csv",
        description="Red vinho verde physicochemical measurements and quality scores.",
        source="UCI Machine Learning Repository",
        delimiter=";",
        has_header=True,
        target_column=11,
    ),
    "wine_quality_white": DatasetInfo(
        name="wine_quality_white",
        domain="tabular",
        task="regression",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        file_name="winequality-white.csv",
        description="White vinho verde physicochemical measurements and quality scores.",
        source="UCI Machine Learning Repository",
        delimiter=";",
        has_header=True,
        target_column=11,
    ),
    "glass": DatasetInfo(
        name="glass",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        file_name="glass.data",
        description="Glass identification from oxide-content measurements.",
        source="UCI Machine Learning Repository",
        target_column=10,
        feature_columns=tuple(range(1, 10)),
        drop_columns=(0,),
    ),
    "banknote_authentication": DatasetInfo(
        name="banknote_authentication",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        file_name="data_banknote_authentication.txt",
        description="Wavelet features for banknote authentication.",
        source="UCI Machine Learning Repository",
        target_column=4,
    ),
    "forest_fires": DatasetInfo(
        name="forest_fires",
        domain="tabular",
        task="regression",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv",
        file_name="forestfires.csv",
        description="Meteorological forest-fire data with burned area target.",
        source="UCI Machine Learning Repository",
        has_header=True,
        target_column=12,
    ),
    "heart_cleveland": DatasetInfo(
        name="heart_cleveland",
        domain="tabular",
        task="classification",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        file_name="processed.cleveland.data",
        description="Cleveland heart-disease records with the processed 14-variable table.",
        source="UCI Machine Learning Repository",
        target_column=13,
    ),
}

TORCHVISION_DATASETS = (
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "EMNIST",
    "CIFAR10",
    "CIFAR100",
    "SVHN",
)


def available_datasets(domain: str | None = None) -> list[str]:
    """List available package-managed public datasets."""

    names = [
        name for name, info in DATASET_REGISTRY.items() if domain is None or info.domain == domain
    ]
    return sorted(names)


def available_torchvision_datasets() -> tuple[str, ...]:
    """Return torchvision dataset names accepted by `load_torchvision_dataset`.

    The vision datasets are loaded through the optional ``vision`` extra because
    they depend on TorchVision's dataset classes and download mirrors. The
    returned names include CIFAR10, CIFAR100, MNIST, FashionMNIST, KMNIST,
    EMNIST, and SVHN.
    """

    return TORCHVISION_DATASETS


def dataset_info(name: str) -> DatasetInfo:
    """Return metadata for one registered dataset."""

    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(available_datasets())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}") from exc


def dataset_path(name: str, root: str | Path = "data") -> Path:
    """Return the expected local path for a registered dataset file."""

    info = dataset_info(name)
    return Path(root) / info.name / info.file_name


def download_dataset(name: str, root: str | Path = "data", force: bool = False) -> Path:
    """Download one registered dataset into ``root/name/file``."""

    info = dataset_info(name)
    path = dataset_path(name, root)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_download_bytes(info.url))
    return path


def download_many(
    names: list[str] | tuple[str, ...] | None = None,
    root: str | Path = "data",
    force: bool = False,
) -> dict[str, Path]:
    """Download several datasets and return their local paths."""

    selected = list(names) if names is not None else available_datasets()
    return {name: download_dataset(name, root=root, force=force) for name in selected}


def load_tabular_dataset(
    name: str,
    root: str | Path = "data",
    download: bool = True,
    normalize: bool = False,
) -> TabularDataset:
    """Download, parse, and optionally standardize a tabular dataset.

    The default returns raw features so train/validation/test splits can be
    created before fitting `FeatureStandardization`. Set `normalize=True` only
    when fitting statistics on the complete table is intentional.
    """

    info = dataset_info(name)
    path = download_dataset(name, root=root) if download else dataset_path(name, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Use download=True or download_dataset().")
    rows, header = _read_rows(path, info)
    x, y, feature_names, target_names = _rows_to_arrays(rows, header, info)
    if normalize:
        x = standardize_features(x)
    return TabularDataset(
        name=info.name,
        x=x,
        y=y,
        task=info.task,
        feature_names=feature_names,
        target_names=target_names,
        path=path,
        info=info,
    )


def standardize_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-standardize a feature matrix after mean-imputing missing values."""

    return fit_feature_standardization(x, eps=eps).transform(x)


def fit_feature_standardization(
    x: np.ndarray,
    eps: float = 1e-8,
) -> FeatureStandardization:
    """Fit imputation and scaling statistics on a training feature matrix."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty 2D feature array")
    values = x.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0, keepdims=True)
    means = np.divide(
        np.where(finite, values, 0.0).sum(axis=0, keepdims=True),
        np.maximum(counts, 1),
    ).astype(np.float32)
    imputed = np.where(finite, values, means)
    scale = imputed.std(axis=0, keepdims=True)
    return FeatureStandardization(means, np.maximum(scale, eps).astype(np.float32))


def standardize_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Column-standardize a tensor after mean-imputing nonfinite entries.

    Args:
        x: Feature tensor with shape `(samples, features)`.
        eps: Minimum divisor used to avoid zero-variance division.

    Returns:
        Standardized floating tensor with the same shape as `x`.
    """

    return fit_tensor_standardization(x, eps=eps).transform(x)


def fit_tensor_standardization(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> TensorStandardization:
    """Fit imputation and scaling statistics on a training feature tensor."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if x.dim() != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty 2D feature tensor")
    values = x.float().clone()
    finite = torch.isfinite(values)
    safe_values = torch.where(finite, values, torch.zeros_like(values))
    counts = finite.sum(dim=0, keepdim=True).clamp_min(1)
    means = safe_values.sum(dim=0, keepdim=True) / counts
    values = torch.where(finite, values, means)
    centered = values - means
    std = centered.square().mean(dim=0, keepdim=True).sqrt()
    return TensorStandardization(means, std.clamp_min(eps))


def make_knn_edge_index(
    x: np.ndarray | torch.Tensor,
    k: int,
    *,
    batch: np.ndarray | torch.Tensor | None = None,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    include_self: bool = False,
    undirected: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    r"""Build a k-nearest-neighbor `edge_index` tensor.

    Edges are returned as `source -> destination`. For each destination entity
    \(i\), the selected sources are its nearest neighbors \(j\), so local message
    passing computes incoming neighbor information.

    Args:
        x: Entity features with shape `(entities, features)`.
        k: Number of neighbors per entity.
        batch: Optional graph id for each entity. Neighbor search is performed
            independently inside each graph id.
        metric: Distance geometry, either `euclidean` or `cosine`.
        include_self: If true, self-neighbors may be selected.
        undirected: If true, append the reverse of every edge and remove
            duplicate columns.
        device: Optional output device. Defaults to the feature tensor device.

    Returns:
        Long tensor with shape `(2, edges)`.
    """

    if k < 0:
        raise ValueError("k must be nonnegative")
    features = _as_float_tensor(x, device=device)
    if features.dim() != 2:
        raise ValueError("make_knn_edge_index expects x with shape (entities, features)")
    out_device = features.device
    if batch is None:
        batch_tensor = torch.zeros(features.shape[0], dtype=torch.long, device=out_device)
    else:
        batch_tensor = torch.as_tensor(batch, dtype=torch.long, device=out_device)
        if batch_tensor.shape != (features.shape[0],):
            raise ValueError("batch must have shape (entities,)")
    if k == 0 or features.shape[0] <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=out_device)

    sources: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    for graph_id in torch.unique(batch_tensor, sorted=True):
        members = torch.nonzero(batch_tensor == graph_id, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        local_x = features[members]
        if metric == "euclidean":
            distances = torch.cdist(local_x, local_x)
        elif metric == "cosine":
            normalized = torch.nn.functional.normalize(local_x, dim=-1)
            distances = 1.0 - normalized @ normalized.T
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        candidates = members.numel() if include_self else members.numel() - 1
        if candidates <= 0:
            continue
        k_eff = min(k, int(candidates))
        if not include_self:
            eye = torch.eye(members.numel(), dtype=torch.bool, device=out_device)
            distances = distances.masked_fill(eye, float("inf"))
        neighbor_idx = distances.topk(k_eff, largest=False).indices
        destination = members.repeat_interleave(k_eff)
        source = members[neighbor_idx.reshape(-1)]
        sources.append(source)
        destinations.append(destination)
    if not sources:
        return torch.empty(2, 0, dtype=torch.long, device=out_device)
    edge_index = torch.stack([torch.cat(sources), torch.cat(destinations)], dim=0)
    if undirected:
        reverse = edge_index.flip(0)
        edge_index = torch.unique(torch.cat([edge_index, reverse], dim=1), dim=1)
    return edge_index


def validate_graph_tensor_batch(data: GraphTensorBatch, raise_on_error: bool = True) -> bool:
    """Check the tensor contract expected by SILVA graph-style models.

    Args:
        data: Packed tensor batch.
        raise_on_error: If true, raise `ValueError` on the first invalid field.

    Returns:
        `True` when the batch is valid; `False` when invalid and
        `raise_on_error=False`.
    """

    if not isinstance(data.x, torch.Tensor):
        return _validation_error("x must be a torch.Tensor", raise_on_error)
    if data.x.dim() not in {1, 2}:
        return _validation_error(
            "x must have shape (entities,) for categorical ids or (entities, features)",
            raise_on_error,
        )
    if data.edge_index is not None:
        if not isinstance(data.edge_index, torch.Tensor):
            return _validation_error("edge_index must be a torch.Tensor", raise_on_error)
        if data.edge_index.dtype != torch.long:
            return _validation_error("edge_index must have dtype torch.long", raise_on_error)
        if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
            return _validation_error("edge_index must have shape (2, edges)", raise_on_error)
        if data.edge_index.device != data.x.device:
            return _validation_error("edge_index must be on the x device", raise_on_error)
        if data.edge_index.numel() > 0:
            if int(data.edge_index.min().item()) < 0:
                return _validation_error(
                    "edge_index contains a negative node index", raise_on_error
                )
            if int(data.edge_index.max().item()) >= data.x.shape[0]:
                return _validation_error("edge_index contains an index outside x", raise_on_error)
    if data.edge_attr is not None:
        if not isinstance(data.edge_attr, torch.Tensor) or data.edge_attr.dim() == 0:
            return _validation_error("edge_attr must be a non-scalar tensor", raise_on_error)
        if data.edge_index is None:
            return _validation_error("edge_attr requires edge_index", raise_on_error)
        if data.edge_attr.device != data.x.device:
            return _validation_error("edge_attr must be on the x device", raise_on_error)
        if data.edge_attr.shape[0] != data.edge_index.shape[1]:
            return _validation_error("edge_attr must have one row/value per edge", raise_on_error)
    if data.batch is not None:
        if not isinstance(data.batch, torch.Tensor):
            return _validation_error("batch must be a torch.Tensor", raise_on_error)
        if data.batch.dtype != torch.long:
            return _validation_error("batch must have dtype torch.long", raise_on_error)
        if data.batch.shape != (data.x.shape[0],):
            return _validation_error("batch must have shape (entities,)", raise_on_error)
        if data.batch.device != data.x.device:
            return _validation_error("batch must be on the x device", raise_on_error)
        if data.batch.numel() > 0:
            if int(data.batch.min().item()) < 0:
                return _validation_error("batch contains a negative graph id", raise_on_error)
            ids = torch.unique(data.batch, sorted=True)
            expected = torch.arange(ids.numel(), device=ids.device, dtype=ids.dtype)
            if not torch.equal(ids, expected):
                return _validation_error(
                    "batch graph ids must be contiguous and start at zero",
                    raise_on_error,
                )
    return True


def pyg_data_to_silva_graph(
    data: Any,
    *,
    device: str | torch.device | None = None,
) -> GraphTensorBatch:
    """Convert a PyG-like data object into `GraphTensorBatch`.

    The function does not require PyTorch Geometric as a runtime dependency. It
    reads the conventional attributes `x`, `edge_index`, `edge_attr`, `batch`,
    and `y` when they are present.
    """

    if not hasattr(data, "x"):
        raise ValueError("PyG-like data must have an x attribute")
    x = data.x
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    batch = getattr(data, "batch", None)
    y = getattr(data, "y", None)
    packed = GraphTensorBatch(
        x=torch.as_tensor(x, device=device),
        edge_index=None
        if edge_index is None
        else torch.as_tensor(edge_index, dtype=torch.long, device=device),
        edge_attr=None if edge_attr is None else torch.as_tensor(edge_attr, device=device),
        batch=None if batch is None else torch.as_tensor(batch, dtype=torch.long, device=device),
        y=None if y is None else torch.as_tensor(y, device=device),
        metadata={"adapter": "pyg_data_to_silva_graph"},
    )
    packed.validate()
    return packed


def tabular_to_silva_graph(
    data: TabularDataset | np.ndarray | torch.Tensor,
    *,
    y: np.ndarray | torch.Tensor | None = None,
    k: int = 8,
    batch: np.ndarray | torch.Tensor | None = None,
    normalize: bool = False,
    max_samples: int | None = None,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    undirected: bool = False,
    device: str | torch.device | None = None,
) -> GraphTensorBatch:
    """Convert tabular features into a SILVA-ready sample graph.

    Args:
        data: A `TabularDataset`, NumPy feature matrix, or tensor feature matrix.
        y: Optional target array when `data` is not a `TabularDataset`.
        k: Number of neighbors used for the sample interaction graph.
        batch: Optional graph id for each row; by default all rows form one graph.
        normalize: Whether to standardize features inside this adapter.
        max_samples: Optional prefix length for compact experiments.
        metric: Geometry used by kNN graph construction.
        undirected: Whether to include reverse edges.
        device: Optional output device.

    Returns:
        `GraphTensorBatch` with `x`, `edge_index`, optional `y`, and `batch`.
    """

    metadata: dict[str, Any] = {}
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    task: TaskKind | None = None
    if isinstance(data, TabularDataset):
        x_raw: np.ndarray | torch.Tensor = data.x
        y_raw: np.ndarray | torch.Tensor | None = data.y
        task = data.task
        metadata = {
            "name": data.name,
            "source": data.info.source,
            "feature_names": data.feature_names,
            "target_names": data.target_names,
            "task": data.task,
        }
    else:
        x_raw = data
        y_raw = y
    if max_samples is not None:
        x_raw = x_raw[:max_samples]
        if y_raw is not None:
            y_raw = y_raw[:max_samples]
        if batch is not None:
            batch = batch[:max_samples]
    if normalize:
        if isinstance(x_raw, torch.Tensor):
            x_tensor = standardize_tensor(x_raw).to(device=device)
        else:
            x_tensor = torch.as_tensor(
                standardize_features(np.asarray(x_raw)),
                dtype=torch.float32,
                device=device,
            )
    else:
        x_tensor = _as_float_tensor(x_raw, device=device)
    batch_tensor = (
        torch.zeros(x_tensor.shape[0], dtype=torch.long, device=x_tensor.device)
        if batch is None
        else torch.as_tensor(batch, dtype=torch.long, device=x_tensor.device)
    )
    edge_index = make_knn_edge_index(
        x_tensor,
        k,
        batch=batch_tensor,
        metric=metric,
        undirected=undirected,
    )
    y_tensor = _as_target_tensor(y_raw, task, device=x_tensor.device) if y_raw is not None else None
    packed = GraphTensorBatch(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        batch=batch_tensor,
        metadata=metadata,
    )
    packed.validate()
    return packed


def images_to_silva_vectors(
    images: np.ndarray | torch.Tensor,
    *,
    y: np.ndarray | torch.Tensor | None = None,
    scale_uint8: bool = True,
    channel_last: bool | None = None,
    device: str | torch.device | None = None,
) -> GraphTensorBatch:
    """Flatten image batches into vector features for vector SILVA models.

    Args:
        images: Image tensor/array with shape `(batch, channels, height, width)`
            or `(batch, height, width, channels)`.
        y: Optional labels.
        scale_uint8: Whether integer image arrays are divided by 255.
        channel_last: Explicitly interpret 4D input as NHWC (`True`) or NCHW
            (`False`). `None` uses a conservative shape heuristic.
        device: Optional output device.

    Returns:
        `GraphTensorBatch` whose `x` has shape `(batch, pixels_or_features)`.
    """

    tensor = torch.as_tensor(images, device=device)
    if tensor.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
        tensor = tensor.float()
        if scale_uint8:
            tensor = tensor / 255.0
    else:
        tensor = tensor.float()
    if tensor.dim() < 2:
        raise ValueError("images_to_silva_vectors expects a batch dimension")
    tensor = _channels_first_images(tensor, channel_last)
    x = tensor.flatten(1)
    y_tensor = _as_target_tensor(y, None, device=x.device) if y is not None else None
    return GraphTensorBatch(x=x, y=y_tensor, metadata={"adapter": "images_to_silva_vectors"})


def image_grid_edge_index(
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    include_diagonals: bool = False,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pixel-grid edges and a batch vector for image-as-graph models."""

    if height < 1 or width < 1 or batch_size < 1:
        raise ValueError("height, width, and batch_size must be positive")
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if include_diagonals:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    src: list[int] = []
    dst: list[int] = []
    nodes_per_image = height * width
    for batch_index in range(batch_size):
        base = batch_index * nodes_per_image
        for row in range(height):
            for col in range(width):
                receiver = base + row * width + col
                for drow, dcol in offsets:
                    nrow = row + drow
                    ncol = col + dcol
                    if 0 <= nrow < height and 0 <= ncol < width:
                        source = base + nrow * width + ncol
                        src.append(source)
                        dst.append(receiver)
    out_device = torch.device(device) if device is not None else None
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=out_device)
    batch = torch.arange(batch_size, dtype=torch.long, device=out_device).repeat_interleave(
        nodes_per_image
    )
    return edge_index, batch


def images_to_silva_pixel_graph(
    images: np.ndarray | torch.Tensor,
    *,
    y: np.ndarray | torch.Tensor | None = None,
    include_diagonals: bool = False,
    scale_uint8: bool = True,
    channel_last: bool | None = None,
    device: str | torch.device | None = None,
) -> GraphTensorBatch:
    """Convert images into pixel entities with grid-local edges.

    The resulting `x` has one row per pixel. For grayscale images the feature
    width is `1`; for color images it is the channel count.
    """

    tensor = torch.as_tensor(images, device=device)
    if tensor.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
        tensor = tensor.float()
        if scale_uint8:
            tensor = tensor / 255.0
    else:
        tensor = tensor.float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 4:
        raise ValueError("images_to_silva_pixel_graph expects 3D or 4D image input")
    tensor = _channels_first_images(tensor, channel_last)
    batch_size, channels, height, width = tensor.shape
    x = tensor.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
    edge_index, batch = image_grid_edge_index(
        height,
        width,
        batch_size=batch_size,
        include_diagonals=include_diagonals,
        device=x.device,
    )
    y_tensor = _as_target_tensor(y, None, device=x.device) if y is not None else None
    packed = GraphTensorBatch(
        x=x,
        edge_index=edge_index,
        y=y_tensor,
        batch=batch,
        metadata={"adapter": "images_to_silva_pixel_graph", "height": height, "width": width},
    )
    packed.validate()
    return packed


def molecular_to_silva_graph(
    *,
    x: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
    edge_attr: np.ndarray | torch.Tensor | None = None,
    batch: np.ndarray | torch.Tensor | None = None,
    y: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> GraphTensorBatch:
    """Pack atom, bond, and molecule-index tensors for molecular SILVA models."""

    x_tensor = torch.as_tensor(x, device=device)
    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index_tensor.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    edge_attr_tensor = None if edge_attr is None else torch.as_tensor(edge_attr, device=device)
    if edge_attr_tensor is not None and edge_attr_tensor.shape[0] != edge_index_tensor.shape[1]:
        raise ValueError("edge_attr must have one row/value per edge")
    batch_tensor = (
        torch.zeros(x_tensor.shape[0], dtype=torch.long, device=x_tensor.device)
        if batch is None
        else torch.as_tensor(batch, dtype=torch.long, device=x_tensor.device)
    )
    y_tensor = _as_target_tensor(y, None, device=x_tensor.device) if y is not None else None
    packed = GraphTensorBatch(
        x=x_tensor,
        edge_index=edge_index_tensor,
        y=y_tensor,
        batch=batch_tensor,
        edge_attr=edge_attr_tensor,
        metadata={"adapter": "molecular_to_silva_graph"},
    )
    packed.validate()
    return packed


def load_torchvision_dataset(
    name: str,
    root: str | Path = "data",
    train: bool = True,
    download: bool = True,
    transform: Any | None = None,
    **kwargs,
) -> Any:
    """Load a torchvision dataset by name when the optional vision extra is installed."""

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Install the vision extra before loading torchvision datasets: "
            'python -m pip install "silva-networks[vision]"'
        ) from exc

    if name not in TORCHVISION_DATASETS:
        available = ", ".join(TORCHVISION_DATASETS)
        raise KeyError(f"Unknown torchvision dataset {name!r}. Available datasets: {available}")
    dataset_cls = getattr(datasets, name)
    transform = transform or transforms.ToTensor()
    if name == "EMNIST":
        split = kwargs.pop("split", "balanced")
        return _instantiate_torchvision_dataset(
            dataset_cls,
            root=str(root),
            split=split,
            train=train,
            download=download,
            transform=transform,
            **kwargs,
        )
    if name == "SVHN":
        split = kwargs.pop("split", "train" if train else "test")
        return _instantiate_torchvision_dataset(
            dataset_cls,
            root=str(root),
            split=split,
            download=download,
            transform=transform,
            **kwargs,
        )
    return _instantiate_torchvision_dataset(
        dataset_cls,
        root=str(root),
        train=train,
        download=download,
        transform=transform,
        **kwargs,
    )


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "silva-networks/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return response.read()


def _instantiate_torchvision_dataset(dataset_cls, **kwargs):
    if kwargs.get("download", False):
        with _certifi_https_context():
            return dataset_cls(**kwargs)
    return dataset_cls(**kwargs)


@contextmanager
def _certifi_https_context():
    previous = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    try:
        yield
    finally:
        ssl._create_default_https_context = previous


def _read_rows(path: Path, info: DatasetInfo) -> tuple[list[list[str]], list[str]]:
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if info.delimiter is None:
        rows = [line.split() for line in lines]
    else:
        rows = list(csv.reader(lines, delimiter=info.delimiter))
    header: list[str] = []
    if info.has_header and rows:
        header = rows.pop(0)
    return rows, header


def _rows_to_arrays(
    rows: list[list[str]],
    header: list[str],
    info: DatasetInfo,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not rows:
        raise ValueError(f"Dataset {info.name!r} contains no rows")
    width = len(rows[0])
    target_column = _resolve_column(info.target_column, width)
    feature_columns = info.feature_columns or tuple(
        index
        for index in range(width)
        if index != target_column and index not in set(info.drop_columns)
    )
    feature_maps: dict[int, dict[str, float]] = {}
    target_map: dict[str, int] = {}
    x_values: list[list[float]] = []
    y_values: list[float | int] = []
    for row in rows:
        if len(row) != width:
            continue
        target_raw = row[target_column].strip()
        if _is_missing(target_raw):
            continue
        features = [_coerce_feature(row[index], index, feature_maps) for index in feature_columns]
        if info.task == "classification":
            y_values.append(_encode_label(target_raw, target_map))
        else:
            y_values.append(_coerce_float(target_raw))
        x_values.append(features)
    x = np.asarray(x_values, dtype=np.float32)
    if info.task == "classification":
        y = np.asarray(y_values, dtype=np.int64)
        target_names = [label for label, _ in sorted(target_map.items(), key=lambda item: item[1])]
    else:
        y = np.asarray(y_values, dtype=np.float32)
        target_names = [header[target_column] if header else "target"]
    feature_names = [
        header[index] if header else f"x{position}"
        for position, index in enumerate(feature_columns)
    ]
    return x, y, feature_names, target_names


def _resolve_column(index: int, width: int) -> int:
    return index if index >= 0 else width + index


def _coerce_feature(value: str, column: int, maps: dict[int, dict[str, float]]) -> float:
    value = value.strip()
    if _is_missing(value):
        return math.nan
    try:
        return float(value)
    except ValueError:
        mapping = maps.setdefault(column, {})
        if value not in mapping:
            mapping[value] = float(len(mapping))
        return mapping[value]


def _coerce_float(value: str) -> float:
    value = value.strip()
    if _is_missing(value):
        return math.nan
    return float(value)


def _encode_label(value: str, mapping: dict[str, int]) -> int:
    value = value.strip()
    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]


def _is_missing(value: str) -> bool:
    return value.strip() in {"", "?", "NA", "N/A", "nan", "NaN"}


def _channels_first_images(tensor: torch.Tensor, channel_last: bool | None) -> torch.Tensor:
    if channel_last is not None and tensor.dim() != 4:
        raise ValueError("channel_last can only be specified for 4D image batches")
    inferred_channel_last = (
        tensor.dim() == 4
        and tensor.shape[-1] in {1, 3, 4}
        and tensor.shape[1] not in {1, 3, 4}
    )
    if channel_last is True or (channel_last is None and inferred_channel_last):
        return tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def _as_float_tensor(
    data: np.ndarray | torch.Tensor, device: str | torch.device | None = None
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.float().to(device=device) if device is not None else data.float()
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def _as_target_tensor(
    y: np.ndarray | torch.Tensor,
    task: TaskKind | None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        tensor = y.to(device=device) if device is not None else y
    else:
        tensor = torch.as_tensor(y, device=device)
    if task == "classification":
        return tensor.long()
    if task == "regression":
        return tensor.float()
    return tensor


def _validation_error(message: str, raise_on_error: bool) -> bool:
    if raise_on_error:
        raise ValueError(message)
    return False
