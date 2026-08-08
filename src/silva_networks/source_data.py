"""Source-dataset adapters and reproducibility receipts for SILVA experiments.

The compact generators in :mod:`silva_networks.structured_data` provide known
solutions for mechanism tests. This module complements them with deterministic
subsets of public datasets used by source-scale experiments. The adapters never
claim that a subset reproduces a published benchmark; each result records the
exact data selection and preprocessing needed to distinguish a teaching run
from the complete protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .datasets import GraphTensorBatch, load_torchvision_dataset, pyg_data_to_silva_graph

VisionNormalization = Literal["unit", "source", "none"]
FlowDatasetName = Literal["Sintel", "KittiFlow", "FlyingChairs"]

_BUNDLED_SOURCE_SNAPSHOTS = {
    "cifar10": "cifar10-balanced-10.pt",
    "cora": "cora-induced-96.pt",
    "motion": "public-motion-frames-100-101.pt",
}


@dataclass(frozen=True)
class SourceDatasetInfo:
    """Stable metadata for a dataset used by a source-scale SILVA recipe."""

    name: str
    domain: str
    task: str
    source: str
    homepage: str
    citation_url: str
    access: str
    expected_storage: str
    official_protocol: str


@dataclass(frozen=True)
class SourceDataReceipt:
    """Machine-readable record of one deterministic source-data selection."""

    dataset: str
    split: str
    source: str
    homepage: str
    citation_url: str
    access: str
    version: str
    subset_size: int
    seed: int
    selected_indices: tuple[int, ...]
    content_sha256: str
    preprocessing: tuple[str, ...]
    adapter: str
    adapter_revision: str = "1"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable receipt."""

        return {
            "dataset": self.dataset,
            "split": self.split,
            "source": self.source,
            "homepage": self.homepage,
            "citation_url": self.citation_url,
            "access": self.access,
            "version": self.version,
            "subset_size": self.subset_size,
            "seed": self.seed,
            "selected_indices": list(self.selected_indices),
            "content_sha256": self.content_sha256,
            "preprocessing": list(self.preprocessing),
            "adapter": self.adapter,
            "adapter_revision": self.adapter_revision,
        }


@dataclass
class SILVAVisionSourceSubset:
    """Image tensor, labels, and source record for a deterministic subset."""

    images: Tensor
    labels: Tensor
    receipt: SourceDataReceipt

    def to(self, device: str | torch.device) -> SILVAVisionSourceSubset:
        return SILVAVisionSourceSubset(self.images.to(device), self.labels.to(device), self.receipt)


@dataclass
class SILVAGraphSourceSubset:
    """SILVA graph tensors, split masks, original node ids, and source record."""

    graph: GraphTensorBatch
    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor
    node_ids: Tensor
    receipt: SourceDataReceipt

    def to(self, device: str | torch.device) -> SILVAGraphSourceSubset:
        return SILVAGraphSourceSubset(
            graph=self.graph.to(device),
            train_mask=self.train_mask.to(device),
            validation_mask=self.validation_mask.to(device),
            test_mask=self.test_mask.to(device),
            node_ids=self.node_ids.to(device),
            receipt=self.receipt,
        )


@dataclass
class SILVAFlowSourceSubset:
    """One or more real image pairs with optional flow supervision."""

    frame1: Tensor
    frame2: Tensor
    flow: Tensor | None
    valid: Tensor | None
    receipt: SourceDataReceipt

    def to(self, device: str | torch.device) -> SILVAFlowSourceSubset:
        return SILVAFlowSourceSubset(
            frame1=self.frame1.to(device),
            frame2=self.frame2.to(device),
            flow=None if self.flow is None else self.flow.to(device),
            valid=None if self.valid is None else self.valid.to(device),
            receipt=self.receipt,
        )


@dataclass
class SILVAOperatorSourceSubset:
    """Input/output fields and source record for an operator-learning dataset."""

    inputs: Tensor
    targets: Tensor
    receipt: SourceDataReceipt

    def to(self, device: str | torch.device) -> SILVAOperatorSourceSubset:
        return SILVAOperatorSourceSubset(
            self.inputs.to(device), self.targets.to(device), self.receipt
        )


@dataclass
class SILVASourceSnapshot:
    """Named tensors and a verified source-data receipt stored with the repository."""

    tensors: dict[str, Tensor]
    receipt: SourceDataReceipt

    def to(self, device: str | torch.device) -> SILVASourceSnapshot:
        return SILVASourceSnapshot(
            {name: value.to(device) for name, value in self.tensors.items()},
            self.receipt,
        )


SOURCE_DATASET_REGISTRY: dict[str, SourceDatasetInfo] = {
    "CIFAR10": SourceDatasetInfo(
        name="CIFAR10",
        domain="vision",
        task="image classification",
        source="CIFAR-10",
        homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
        citation_url="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
        access="Public research dataset; consult the source page for current terms.",
        expected_storage="about 170 MB compressed",
        official_protocol="50,000 training and 10,000 test images with the source split.",
    ),
    "MNIST": SourceDatasetInfo(
        name="MNIST",
        domain="vision",
        task="image classification",
        source="MNIST",
        homepage="https://yann.lecun.com/exdb/mnist/",
        citation_url="https://doi.org/10.1109/5.726791",
        access="Public research dataset; consult the source page for current terms.",
        expected_storage="less than 100 MB",
        official_protocol="60,000 training and 10,000 test images with the source split.",
    ),
    "SVHN": SourceDatasetInfo(
        name="SVHN",
        domain="vision",
        task="image classification",
        source="Street View House Numbers",
        homepage="http://ufldl.stanford.edu/housenumbers/",
        citation_url="https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf",
        access="Public research dataset; consult the source page for current terms.",
        expected_storage="about 2.5 GB for train, test, and extra splits",
        official_protocol="Use the documented train, extra, and test split policy.",
    ),
    "Cora": SourceDatasetInfo(
        name="Cora",
        domain="graph",
        task="transductive node classification",
        source="Planetoid citation-network split",
        homepage="https://github.com/kimiyoung/planetoid",
        citation_url="https://proceedings.mlr.press/v48/yang16.html",
        access="Public research dataset distributed by the Planetoid repository.",
        expected_storage="less than 20 MB",
        official_protocol="Use the fixed Planetoid train, validation, and test masks.",
    ),
    "CiteSeer": SourceDatasetInfo(
        name="CiteSeer",
        domain="graph",
        task="transductive node classification",
        source="Planetoid citation-network split",
        homepage="https://github.com/kimiyoung/planetoid",
        citation_url="https://proceedings.mlr.press/v48/yang16.html",
        access="Public research dataset distributed by the Planetoid repository.",
        expected_storage="less than 20 MB",
        official_protocol="Use the fixed Planetoid train, validation, and test masks.",
    ),
    "PubMed": SourceDatasetInfo(
        name="PubMed",
        domain="graph",
        task="transductive node classification",
        source="Planetoid citation-network split",
        homepage="https://github.com/kimiyoung/planetoid",
        citation_url="https://proceedings.mlr.press/v48/yang16.html",
        access="Public research dataset distributed by the Planetoid repository.",
        expected_storage="less than 100 MB",
        official_protocol="Use the fixed Planetoid train, validation, and test masks.",
    ),
    "Sintel": SourceDatasetInfo(
        name="Sintel",
        domain="optical flow",
        task="dense motion estimation",
        source="MPI Sintel optical-flow benchmark",
        homepage="https://sintel.is.tue.mpg.de/",
        citation_url="https://doi.org/10.1007/978-3-642-33783-3_44",
        access="MPI Sintel terms apply; download and accept them at the source site.",
        expected_storage="about 5.3 GB for the complete benchmark download",
        official_protocol="Report clean/final validation and benchmark-test metrics separately.",
    ),
    "KittiFlow": SourceDatasetInfo(
        name="KittiFlow",
        domain="optical flow",
        task="dense motion estimation",
        source="KITTI 2015 optical-flow benchmark",
        homepage="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow",
        citation_url="https://doi.org/10.1109/CVPR.2012.6248074",
        access="KITTI terms apply; download the benchmark from the source site.",
        expected_storage="about 2 GB for the optical-flow image and label archives",
        official_protocol="Use training labels for validation and the test server for final results.",
    ),
    "FlyingChairs": SourceDatasetInfo(
        name="FlyingChairs",
        domain="optical flow",
        task="dense motion estimation",
        source="FlyingChairs optical-flow dataset",
        homepage="https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html",
        citation_url="https://doi.org/10.1109/ICCV.2015.316",
        access="Dataset-specific source terms apply.",
        expected_storage="about 22 GB",
        official_protocol="Use the published train/validation assignment file.",
    ),
    "PublicBasketballMotion": SourceDatasetInfo(
        name="PublicBasketballMotion",
        domain="optical flow",
        task="unsupervised real-image motion check",
        source="Public video used by the TorchVision optical-flow tutorial",
        homepage="https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html",
        citation_url="https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html",
        access="Freely usable Pexels video mirrored by the tutorial; not a benchmark dataset.",
        expected_storage="about 3.5 MB",
        official_protocol="Mechanism demonstration only; do not report it as a benchmark result.",
    ),
    "DarcyFlowSmall": SourceDatasetInfo(
        name="DarcyFlowSmall",
        domain="neural operator",
        task="Darcy-flow solution operator",
        source="NeuralOperator Darcy-flow small dataset",
        homepage="https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html",
        citation_url="https://doi.org/10.48550/arXiv.2010.08895",
        access="Use the dataset loader and terms documented by NeuralOperator.",
        expected_storage="loader-dependent; reserve at least 1 GB for cached resolutions",
        official_protocol="Preserve source train/test resolutions, normalization, and relative L2 metric.",
    ),
}


def available_source_datasets(domain: str | None = None) -> tuple[str, ...]:
    """Return registered source datasets, optionally filtered by domain."""

    return tuple(
        sorted(
            name
            for name, info in SOURCE_DATASET_REGISTRY.items()
            if domain is None or info.domain == domain
        )
    )


def source_dataset_info(name: str) -> SourceDatasetInfo:
    """Return stable source metadata for one dataset."""

    try:
        return SOURCE_DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(available_source_datasets())
        raise KeyError(f"Unknown source dataset {name!r}. Available datasets: {available}") from exc


def load_vision_source_subset(
    name: Literal["CIFAR10", "MNIST", "SVHN"],
    *,
    root: str | Path = "data",
    train: bool = True,
    samples_per_class: int = 4,
    seed: int = 0,
    image_size: tuple[int, int] | None = None,
    normalization: VisionNormalization = "unit",
    download: bool = False,
    dataset: Any | None = None,
) -> SILVAVisionSourceSubset:
    """Load a deterministic class-balanced vision subset.

    Passing ``dataset`` is useful for private mirrors and offline tests. When it
    is omitted, the corresponding TorchVision dataset is opened at ``root``.
    ``normalization='source'`` applies the conventional dataset statistics;
    ``'unit'`` keeps values in ``[0, 1]``.
    """

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if normalization not in {"unit", "source", "none"}:
        raise ValueError("normalization must be 'unit', 'source', or 'none'")
    info = source_dataset_info(name)
    loaded = (
        dataset
        if dataset is not None
        else load_torchvision_dataset(name, root=root, train=train, download=download)
    )
    labels = _dataset_labels(loaded)
    selected = _stratified_indices(labels, samples_per_class, seed)
    images: list[Tensor] = []
    selected_labels: list[int] = []
    for index in selected:
        image, label = loaded[index][:2]
        images.append(_image_tensor(image))
        selected_labels.append(int(torch.as_tensor(label).item()))
    batch = torch.stack(images)
    if image_size is not None and tuple(batch.shape[-2:]) != tuple(image_size):
        batch = F.interpolate(batch, size=image_size, mode="bilinear", align_corners=False)
    batch, normalization_step = _normalize_vision(batch, name, normalization)
    target = torch.tensor(selected_labels, dtype=torch.long)
    preprocessing = ["deterministic class-balanced selection", normalization_step]
    if image_size is not None:
        preprocessing.append(f"bilinear resize to {image_size[0]}x{image_size[1]}")
    receipt = _receipt(
        info,
        split="train" if train else "test",
        version=_package_version("torchvision"),
        seed=seed,
        indices=selected,
        tensors=(batch, target),
        preprocessing=preprocessing,
        adapter="load_vision_source_subset",
    )
    return SILVAVisionSourceSubset(batch, target, receipt)


def load_planetoid_source_subset(
    name: Literal["Cora", "CiteSeer", "PubMed"] = "Cora",
    *,
    root: str | Path = "data/planetoid",
    subset_nodes: int | None = None,
    seed: int = 0,
    download: bool = False,
    dataset: Any | None = None,
) -> SILVAGraphSourceSubset:
    """Load a Planetoid graph or a deterministic induced teaching subset.

    ``subset_nodes=None`` preserves the official full transductive graph and
    masks. A compact induced graph is intended for executable tutorials only;
    the receipt records its original node ids.
    """

    info = source_dataset_info(name)
    loaded = dataset
    if loaded is None:
        try:
            from torch_geometric.datasets import Planetoid
        except ImportError as exc:
            raise ImportError(
                "Install the benchmark extra before loading Planetoid data: "
                'python -m pip install "silva-networks[benchmarks]"'
            ) from exc
        dataset_root = Path(root) / name
        if not download and not _planetoid_present(dataset_root):
            raise FileNotFoundError(
                f"{name} was not found at {dataset_root}. Download it explicitly "
                "or call with download=True."
            )
        loaded = Planetoid(root=str(root), name=name)
    data = loaded if hasattr(loaded, "x") else loaded[0]
    graph = pyg_data_to_silva_graph(data)
    nodes = graph.num_entities
    train_mask = _mask_from_data(data, "train_mask", nodes)
    validation_mask = _mask_from_data(data, "val_mask", nodes)
    test_mask = _mask_from_data(data, "test_mask", nodes)
    node_ids = torch.arange(nodes, dtype=torch.long)
    preprocessing = ["source node features", "source edges", "source split masks"]
    if subset_nodes is not None:
        if subset_nodes < 3:
            raise ValueError("subset_nodes must be at least three")
        if subset_nodes > nodes:
            raise ValueError("subset_nodes cannot exceed the number of graph nodes")
        node_ids = _connected_subset_nodes(
            graph.edge_index,
            (train_mask, validation_mask, test_mask),
            subset_nodes,
            seed,
        )
        graph, train_mask, validation_mask, test_mask = _induced_graph(
            graph, node_ids, train_mask, validation_mask, test_mask
        )
        preprocessing.append(
            "deterministic connected induced subset; teaching protocol, not source benchmark"
        )
    selected = tuple(int(value) for value in node_ids.tolist())
    tensors = (
        graph.x,
        graph.edge_index if graph.edge_index is not None else torch.empty(2, 0),
        graph.y if graph.y is not None else torch.empty(0),
        train_mask,
        validation_mask,
        test_mask,
    )
    receipt = _receipt(
        info,
        split="public Planetoid masks",
        version=_package_version("torch-geometric"),
        seed=seed,
        indices=selected,
        tensors=tensors,
        preprocessing=preprocessing,
        adapter="load_planetoid_source_subset",
    )
    graph.metadata = {
        **(graph.metadata or {}),
        "dataset": name,
        "node_ids": selected,
        "subset_protocol": "full" if subset_nodes is None else "induced teaching subset",
    }
    return SILVAGraphSourceSubset(graph, train_mask, validation_mask, test_mask, node_ids, receipt)


def normalized_graph_operator(
    edge_index: Tensor,
    num_nodes: int,
    *,
    add_self_loops: bool = True,
    dense: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Build the symmetric normalized operator ``D^{-1/2} A D^{-1/2}``."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edges.dim() != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")
    if add_self_loops:
        diagonal = torch.arange(num_nodes, device=edges.device)
        edges = torch.cat((edges, torch.stack((diagonal, diagonal))), dim=1)
    values = torch.ones(edges.shape[1], dtype=dtype, device=edges.device)
    adjacency = torch.sparse_coo_tensor(
        edges, values, (num_nodes, num_nodes), device=edges.device
    ).coalesce()
    row, col = adjacency.indices()
    degree = torch.zeros(num_nodes, dtype=dtype, device=edges.device)
    degree.scatter_add_(0, row, adjacency.values())
    normalized_values = adjacency.values() * degree[row].clamp_min(1).rsqrt()
    normalized_values = normalized_values * degree[col].clamp_min(1).rsqrt()
    operator = torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape, device=edges.device
    ).coalesce()
    return operator.to_dense() if dense else operator


def load_optical_flow_source_subset(
    name: FlowDatasetName,
    *,
    root: str | Path,
    split: str = "train",
    index: int = 0,
    pass_name: str = "clean",
    image_size: tuple[int, int] | None = None,
    dataset: Any | None = None,
) -> SILVAFlowSourceSubset:
    """Load one local Sintel, KITTI, or FlyingChairs pair without downloading."""

    info = source_dataset_info(name)
    loaded = dataset
    if loaded is None:
        try:
            from torchvision import datasets
        except ImportError as exc:
            raise ImportError(
                "Install the vision extra before loading optical-flow data: "
                'python -m pip install "silva-networks[vision]"'
            ) from exc
        dataset_type = getattr(datasets, name)
        kwargs: dict[str, Any] = {"root": str(root), "split": split}
        if name == "Sintel":
            kwargs["pass_name"] = pass_name
        loaded = dataset_type(**kwargs)
    if not 0 <= index < len(loaded):
        raise IndexError(f"index {index} is outside a dataset of length {len(loaded)}")
    sample = loaded[index]
    if len(sample) not in {3, 4}:
        raise ValueError("optical-flow samples must contain images, flow, and optional mask")
    frame1, frame2, flow = sample[:3]
    valid = sample[3] if len(sample) == 4 else None
    first = _image_tensor(frame1).unsqueeze(0)
    second = _image_tensor(frame2).unsqueeze(0)
    flow_tensor = None if flow is None else _flow_tensor(flow).unsqueeze(0)
    valid_tensor = None if valid is None else torch.as_tensor(np.asarray(valid)).bool().unsqueeze(0)
    original_size = tuple(first.shape[-2:])
    if image_size is not None and original_size != tuple(image_size):
        first = F.interpolate(first, image_size, mode="bilinear", align_corners=False)
        second = F.interpolate(second, image_size, mode="bilinear", align_corners=False)
        if flow_tensor is not None:
            flow_tensor = F.interpolate(
                flow_tensor, image_size, mode="bilinear", align_corners=False
            )
            flow_tensor[:, 0] *= image_size[1] / original_size[1]
            flow_tensor[:, 1] *= image_size[0] / original_size[0]
        if valid_tensor is not None:
            valid_tensor = (
                F.interpolate(valid_tensor.float().unsqueeze(1), image_size, mode="nearest")
                .squeeze(1)
                .bool()
            )
    preprocessing = ["convert images to channel-first unit tensors"]
    if image_size is not None:
        preprocessing.append(
            f"resize images and flow to {image_size[0]}x{image_size[1]} with vector rescaling"
        )
    tensors = [first, second]
    if flow_tensor is not None:
        tensors.append(flow_tensor)
    if valid_tensor is not None:
        tensors.append(valid_tensor)
    receipt = _receipt(
        info,
        split=f"{split}:{pass_name}" if name == "Sintel" else split,
        version=_package_version("torchvision"),
        seed=0,
        indices=(index,),
        tensors=tensors,
        preprocessing=preprocessing,
        adapter="load_optical_flow_source_subset",
    )
    return SILVAFlowSourceSubset(first, second, flow_tensor, valid_tensor, receipt)


def load_public_motion_subset(
    video_path: str | Path,
    *,
    frame_indices: Sequence[int] = (100, 101),
    image_size: tuple[int, int] | None = (128, 192),
) -> SILVAFlowSourceSubset:
    """Load selected frames from the small real-motion tutorial video.

    The pair has no ground-truth flow and is therefore a qualitative mechanism
    check. It is deliberately distinguished from Sintel and KITTI evaluation.
    """

    if len(frame_indices) < 2:
        raise ValueError("frame_indices must contain at least two entries")
    indices = tuple(int(value) for value in frame_indices)
    if min(indices) < 0:
        raise ValueError("frame indices must be nonnegative")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install PyAV before reading a video source.") from exc
    requested = set(indices)
    decoded: dict[int, Tensor] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = str(stream.average_rate or "unknown")
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in requested:
                array = frame.to_ndarray(format="rgb24")
                decoded[frame_index] = torch.from_numpy(array).permute(2, 0, 1)
            if frame_index >= max(indices):
                break
    missing = sorted(requested.difference(decoded))
    if missing:
        raise IndexError(f"video ended before requested frames {missing}")
    selected = torch.stack([decoded[index] for index in indices]).float() / 255.0
    if image_size is not None and tuple(selected.shape[-2:]) != tuple(image_size):
        selected = F.interpolate(selected, image_size, mode="bilinear", align_corners=False)
    first = selected[:-1]
    second = selected[1:]
    info = source_dataset_info("PublicBasketballMotion")
    preprocessing = ["selected consecutive real-video frames", "scaled uint8 values to [0, 1]"]
    if image_size is not None:
        preprocessing.append(f"bilinear resize to {image_size[0]}x{image_size[1]}")
    receipt = _receipt(
        info,
        split=f"frames at {fps} fps",
        version=_package_version("av"),
        seed=0,
        indices=indices,
        tensors=(first, second),
        preprocessing=preprocessing,
        adapter="load_public_motion_subset",
    )
    return SILVAFlowSourceSubset(first, second, None, None, receipt)


def load_darcy_source_subset(
    path: str | Path,
    *,
    samples: int | None = None,
    seed: int = 0,
    input_key: str = "x",
    target_key: str = "y",
) -> SILVAOperatorSourceSubset:
    """Load a deterministic Darcy subset from a local ``.pt`` or ``.npz`` file.

    This format-neutral boundary lets a full experiment use the official
    NeuralOperator loader, a source archive, or a private mirror without
    changing the SILVA operator. Arrays may use ``x/y``, ``inputs/targets``,
    ``coeff/solution``, or caller-supplied keys.
    """

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix == ".pt":
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
    elif source_path.suffix == ".npz":
        with np.load(source_path) as archive:
            payload = {key: archive[key] for key in archive.files}
    else:
        raise ValueError("Darcy source files must use .pt or .npz")
    inputs, targets = _operator_arrays(payload, input_key, target_key)
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("Darcy inputs and targets must have the same sample count")
    count = inputs.shape[0] if samples is None else samples
    if count < 1 or count > inputs.shape[0]:
        raise ValueError("samples must be between one and the available sample count")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(inputs.shape[0], generator=generator)[:count]
    selected_inputs = inputs[indices].float()
    selected_targets = targets[indices].float()
    info = source_dataset_info("DarcyFlowSmall")
    receipt = _receipt(
        info,
        split="local source archive",
        version="source file",
        seed=seed,
        indices=tuple(int(value) for value in indices.tolist()),
        tensors=(selected_inputs, selected_targets),
        preprocessing=("preserve source field values", "deterministic sample selection"),
        adapter="load_darcy_source_subset",
    )
    return SILVAOperatorSourceSubset(selected_inputs, selected_targets, receipt)


def save_source_snapshot(
    path: str | Path,
    *,
    tensors: dict[str, Tensor],
    receipt: SourceDataReceipt,
) -> Path:
    """Write a compact source subset while preserving its tensor hash."""

    if not tensors:
        raise ValueError("a source snapshot must contain at least one tensor")
    values = {
        name: torch.as_tensor(value).detach().cpu() for name, value in tensors.items()
    }
    content_hash = _tensor_sha256(tuple(values.values()))
    if content_hash != receipt.content_sha256:
        raise ValueError(
            "snapshot tensors do not match the source receipt content checksum"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "tensor_keys": list(values),
            "tensors": values,
            "receipt": receipt.as_dict(),
        },
        target,
    )
    return target


def load_source_snapshot(
    path: str | Path, *, verify: bool = True
) -> SILVASourceSnapshot:
    """Load a compact source subset and verify its receipt checksum."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported source snapshot format")
    keys = payload.get("tensor_keys")
    tensors = payload.get("tensors")
    receipt_values = payload.get("receipt")
    if not isinstance(keys, list) or not isinstance(tensors, dict):
        raise TypeError("source snapshot tensor metadata is incomplete")
    if not isinstance(receipt_values, dict):
        raise TypeError("source snapshot receipt is missing")
    ordered = {str(key): torch.as_tensor(tensors[key]) for key in keys}
    normalized_receipt = dict(receipt_values)
    normalized_receipt["selected_indices"] = tuple(
        normalized_receipt["selected_indices"]
    )
    normalized_receipt["preprocessing"] = tuple(
        normalized_receipt["preprocessing"]
    )
    receipt = SourceDataReceipt(**normalized_receipt)
    if verify and _tensor_sha256(tuple(ordered.values())) != receipt.content_sha256:
        raise ValueError("source snapshot content checksum does not match its receipt")
    return SILVASourceSnapshot(ordered, receipt)


def available_bundled_source_snapshots() -> tuple[str, ...]:
    """Return compact attributed snapshots shipped with the package."""

    return tuple(_BUNDLED_SOURCE_SNAPSHOTS)


def load_bundled_source_snapshot(
    name: str, *, verify: bool = True
) -> SILVASourceSnapshot:
    """Load a packaged compact snapshot by registry name.

    The returned tensors use the same format and checksum verification as
    :func:`load_source_snapshot`. These compact records validate mechanisms;
    they are not replacements for official complete benchmark splits.
    """

    try:
        filename = _BUNDLED_SOURCE_SNAPSHOTS[name]
    except KeyError as exc:
        choices = ", ".join(_BUNDLED_SOURCE_SNAPSHOTS)
        raise KeyError(f"unknown bundled source snapshot {name!r}; choose from {choices}") from exc
    resource = importlib.resources.files("silva_networks").joinpath(
        "source_snapshots", filename
    )
    with importlib.resources.as_file(resource) as path:
        return load_source_snapshot(path, verify=verify)


def _dataset_labels(dataset: Any) -> Tensor:
    raw = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if raw is None:
        raw = [dataset[index][1] for index in range(len(dataset))]
    return torch.as_tensor(raw, dtype=torch.long).flatten()


def _stratified_indices(labels: Tensor, samples_per_class: int, seed: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for class_id in torch.unique(labels, sorted=True):
        candidates = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if candidates.numel() < samples_per_class:
            raise ValueError(
                f"class {int(class_id)} has {candidates.numel()} samples, fewer than "
                f"samples_per_class={samples_per_class}"
            )
        order = torch.randperm(candidates.numel(), generator=generator)[:samples_per_class]
        selected.extend(int(value) for value in candidates[order])
    return tuple(selected)


def _image_tensor(image: Any) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach().clone()
    else:
        array = np.asarray(image)
        if not array.flags.writeable:
            array = array.copy()
        tensor = torch.as_tensor(array)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] not in {1, 2, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    if tensor.dim() != 3:
        raise ValueError("images must have shape (H, W), (C, H, W), or (H, W, C)")
    integer = not tensor.is_floating_point()
    tensor = tensor.float()
    if integer or (tensor.numel() and float(tensor.max()) > 1.5):
        tensor = tensor / 255.0
    return tensor


def _flow_tensor(flow: Any) -> Tensor:
    tensor = torch.as_tensor(np.asarray(flow)).float()
    if tensor.dim() != 3:
        raise ValueError("flow must have shape (2, H, W) or (H, W, 2)")
    if tensor.shape[0] != 2 and tensor.shape[-1] == 2:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] != 2:
        raise ValueError("flow must have exactly two vector components")
    return tensor


def _normalize_vision(images: Tensor, name: str, mode: VisionNormalization) -> tuple[Tensor, str]:
    if mode == "none":
        return images, "preserve tensor scale"
    unit = images.clamp(0.0, 1.0)
    if mode == "unit":
        return unit, "scale image values to [0, 1]"
    statistics = {
        "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "MNIST": ((0.1307,), (0.3081,)),
        "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    }
    mean_values, std_values = statistics[name]
    mean = unit.new_tensor(mean_values).view(1, -1, 1, 1)
    std = unit.new_tensor(std_values).view(1, -1, 1, 1)
    return (unit - mean) / std, f"normalize with conventional {name} channel statistics"


def _mask_from_data(data: Any, name: str, nodes: int) -> Tensor:
    value = getattr(data, name, None)
    if value is None:
        raise ValueError(f"Planetoid-style data must provide {name}")
    mask = torch.as_tensor(value, dtype=torch.bool)
    if mask.dim() == 2:
        mask = mask[:, 0]
    if mask.shape != (nodes,):
        raise ValueError(f"{name} must have shape ({nodes},)")
    return mask


def _planetoid_present(path: Path) -> bool:
    raw = path / "raw"
    processed = path / "processed"
    return (raw.exists() and any(raw.iterdir())) or (
        processed.exists() and any(processed.iterdir())
    )


def _connected_subset_nodes(
    edge_index: Tensor | None,
    masks: tuple[Tensor, Tensor, Tensor],
    count: int,
    seed: int,
) -> Tensor:
    nodes = masks[0].numel()
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    selected_set: set[int] = set()
    for mask in masks:
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if candidates.numel():
            candidate = int(candidates[torch.randperm(candidates.numel(), generator=generator)[0]])
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
    if edge_index is not None and edge_index.numel():
        order = torch.randperm(edge_index.shape[1], generator=generator)
        progress = True
        while len(selected) < count and progress:
            progress = False
            for edge_id in order:
                source = int(edge_index[0, edge_id])
                target = int(edge_index[1, edge_id])
                if source in selected_set and target not in selected_set:
                    selected.append(target)
                    selected_set.add(target)
                    progress = True
                elif target in selected_set and source not in selected_set:
                    selected.append(source)
                    selected_set.add(source)
                    progress = True
                if len(selected) == count:
                    break
    if len(selected) < count:
        for value in torch.randperm(nodes, generator=generator):
            candidate = int(value)
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
            if len(selected) == count:
                break
    return torch.tensor(selected, dtype=torch.long)


def _induced_graph(
    graph: GraphTensorBatch,
    node_ids: Tensor,
    train_mask: Tensor,
    validation_mask: Tensor,
    test_mask: Tensor,
) -> tuple[GraphTensorBatch, Tensor, Tensor, Tensor]:
    inverse = torch.full((graph.num_entities,), -1, dtype=torch.long)
    inverse[node_ids] = torch.arange(node_ids.numel())
    edges = graph.edge_index
    if edges is None:
        induced_edges = None
    else:
        keep = (inverse[edges[0]] >= 0) & (inverse[edges[1]] >= 0)
        induced_edges = inverse[edges[:, keep]]
    induced = GraphTensorBatch(
        x=graph.x[node_ids],
        edge_index=induced_edges,
        y=None if graph.y is None else graph.y[node_ids],
        batch=torch.zeros(node_ids.numel(), dtype=torch.long),
        edge_attr=None,
        metadata=dict(graph.metadata or {}),
    )
    induced.validate()
    return (
        induced,
        train_mask[node_ids],
        validation_mask[node_ids],
        test_mask[node_ids],
    )


def _operator_arrays(payload: Any, input_key: str, target_key: str) -> tuple[Tensor, Tensor]:
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return torch.as_tensor(payload[0]), torch.as_tensor(payload[1])
    if not isinstance(payload, dict):
        raise TypeError("operator source file must contain a mapping or an (inputs, targets) pair")
    key_pairs = (
        (input_key, target_key),
        ("inputs", "targets"),
        ("coeff", "solution"),
        ("a", "u"),
    )
    for candidate_input, candidate_target in key_pairs:
        if candidate_input in payload and candidate_target in payload:
            return torch.as_tensor(payload[candidate_input]), torch.as_tensor(
                payload[candidate_target]
            )
    raise KeyError("could not find input/target field arrays in the source file")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _receipt(
    info: SourceDatasetInfo,
    *,
    split: str,
    version: str,
    seed: int,
    indices: Sequence[int],
    tensors: Sequence[Tensor],
    preprocessing: Sequence[str],
    adapter: str,
) -> SourceDataReceipt:
    return SourceDataReceipt(
        dataset=info.name,
        split=split,
        source=info.source,
        homepage=info.homepage,
        citation_url=info.citation_url,
        access=info.access,
        version=version,
        subset_size=len(indices),
        seed=seed,
        selected_indices=tuple(int(value) for value in indices),
        content_sha256=_tensor_sha256(tensors),
        preprocessing=tuple(preprocessing),
        adapter=adapter,
    )


def _tensor_sha256(tensors: Sequence[Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
