from __future__ import annotations

import numpy as np
import pytest
import torch

from silva_networks import (
    GraphTensorBatch,
    available_datasets,
    available_torchvision_datasets,
    dataset_info,
    image_grid_edge_index,
    images_to_silva_pixel_graph,
    images_to_silva_vectors,
    load_tabular_dataset,
    make_knn_edge_index,
    molecular_to_silva_graph,
    pyg_data_to_silva_graph,
    standardize_features,
    standardize_tensor,
    tabular_to_silva_graph,
    validate_graph_tensor_batch,
)


def test_dataset_registry_contains_real_cases() -> None:
    names = available_datasets()
    assert "iris" in names
    assert "wine" in names
    assert "wdbc" in names
    assert dataset_info("iris").task == "classification"


def test_torchvision_registry_contains_cifar_and_digit_cases() -> None:
    names = available_torchvision_datasets()
    assert "CIFAR10" in names
    assert "CIFAR100" in names
    assert "MNIST" in names
    assert "SVHN" in names


def test_load_tabular_dataset_from_existing_file(tmp_path) -> None:
    path = tmp_path / "iris" / "iris.data"
    path.parent.mkdir()
    path.write_text(
        "5.1,3.5,1.4,0.2,Iris-setosa\n"
        "7.0,3.2,4.7,1.4,Iris-versicolor\n"
        "6.3,3.3,6.0,2.5,Iris-virginica\n"
    )
    dataset = load_tabular_dataset("iris", root=tmp_path, download=False, normalize=True)
    assert dataset.x.shape == (3, 4)
    assert dataset.y.tolist() == [0, 1, 2]
    assert dataset.target_names == ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    raw = load_tabular_dataset("iris", root=tmp_path, download=False)
    assert np.allclose(raw.x[0], np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32))


def test_standardize_features_imputes_missing_values() -> None:
    x = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    z = standardize_features(x)
    assert not np.isnan(z).any()
    assert np.allclose(z.mean(axis=0), np.zeros(2), atol=1e-6)


def test_standardize_tensor_imputes_nonfinite_values() -> None:
    x = torch.tensor([[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0]])
    z = standardize_tensor(x)
    assert torch.isfinite(z).all()
    assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError, match="nonempty 2D"):
        standardize_tensor(torch.empty(0, 2))


def test_make_knn_edge_index_respects_batch_groups() -> None:
    x = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = make_knn_edge_index(x, k=1, batch=batch)
    assert edge_index.shape == (2, 4)
    for source, destination in edge_index.T.tolist():
        assert batch[source].item() == batch[destination].item()


def test_tabular_to_silva_graph_from_dataset(tmp_path) -> None:
    path = tmp_path / "iris" / "iris.data"
    path.parent.mkdir()
    path.write_text(
        "5.1,3.5,1.4,0.2,Iris-setosa\n"
        "5.2,3.4,1.5,0.2,Iris-setosa\n"
        "7.0,3.2,4.7,1.4,Iris-versicolor\n"
        "6.3,3.3,6.0,2.5,Iris-virginica\n"
    )
    dataset = load_tabular_dataset("iris", root=tmp_path, download=False, normalize=False)
    graph = tabular_to_silva_graph(dataset, k=2, normalize=True)
    assert graph.x.shape == (4, 4)
    assert graph.y is not None
    assert graph.edge_index is not None
    assert graph.batch is not None
    assert graph.edge_index.shape[0] == 2
    assert graph.metadata is not None
    assert graph.metadata["name"] == "iris"
    assert graph.validate()
    assert graph.num_entities == 4
    assert graph.num_edges == graph.edge_index.shape[1]


def test_image_adapters_create_vector_and_pixel_graphs() -> None:
    images = torch.arange(2 * 1 * 3 * 3, dtype=torch.float32).reshape(2, 1, 3, 3)
    vectors = images_to_silva_vectors(images, y=torch.tensor([0, 1]))
    assert vectors.x.shape == (2, 9)
    assert vectors.y is not None

    edge_index, batch = image_grid_edge_index(3, 3, batch_size=2)
    assert edge_index.shape[0] == 2
    assert batch.shape == (18,)

    pixels = images_to_silva_pixel_graph(images)
    assert pixels.x.shape == (18, 1)
    assert pixels.edge_index is not None
    assert pixels.batch is not None

    ambiguous_nhwc = torch.randn(2, 4, 4, 3)
    channel_last_pixels = images_to_silva_pixel_graph(
        ambiguous_nhwc,
        channel_last=True,
    )
    assert channel_last_pixels.x.shape == (32, 3)


def test_molecular_to_silva_graph_packs_optional_tensors() -> None:
    packed = molecular_to_silva_graph(
        x=np.array([0, 1, 2]),
        edge_index=np.array([[0, 1], [1, 2]]),
        edge_attr=np.array([0, 1]),
        y=np.array([0.5]),
    )
    assert packed.x.shape == (3,)
    assert packed.edge_index is not None
    assert packed.edge_attr is not None
    assert packed.batch is not None
    assert packed.y is not None
    assert packed.model_kwargs()["edge_index"].shape == (2, 2)
    assert validate_graph_tensor_batch(packed)


def test_graph_tensor_batch_validation_rejects_bad_edges() -> None:
    bad = GraphTensorBatch(
        x=torch.randn(2, 3),
        edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
    )
    assert not bad.validate(raise_on_error=False)
    bad_batch = GraphTensorBatch(
        x=torch.randn(2, 3),
        batch=torch.tensor([0, 2], dtype=torch.long),
    )
    assert not bad_batch.validate(raise_on_error=False)
    bad_attr = GraphTensorBatch(x=torch.randn(2, 3), edge_attr=torch.randn(1, 2))
    assert not bad_attr.validate(raise_on_error=False)


def test_pyg_like_data_adapter_reads_standard_fields() -> None:
    class Data:
        pass

    data = Data()
    data.x = torch.randn(3, 2)
    data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data.edge_attr = torch.randn(2, 1)
    data.y = torch.tensor([1])
    packed = pyg_data_to_silva_graph(data)
    assert packed.x.shape == (3, 2)
    assert packed.edge_attr is not None
    assert packed.metadata is not None
    assert packed.metadata["adapter"] == "pyg_data_to_silva_graph"
