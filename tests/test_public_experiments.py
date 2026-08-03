from __future__ import annotations

import importlib.util
import json
from importlib import metadata
from pathlib import Path

import torch

from silva_networks import public_experiments

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/public/run_experiment.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("public_experiment_runner", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_solver_sweep_config_runs() -> None:
    runner = load_runner()
    output = runner.run_config(
        {
            "name": "test_solver_sweep",
            "kind": "solver_sweep",
            "seed": 5,
            "device": "cpu",
            "dim": 4,
            "scale": 0.2,
            "jacobian_samples": 2,
            "power_iters": 4,
            "solvers": [{"solver": "picard", "max_iter": 5, "alpha": 0.5}],
        }
    )
    assert output["results"][0]["solver"] == "picard"
    assert output["results"][0]["residual"] >= 0


def test_cli_config_helpers_resolve_names_and_apply_overrides() -> None:
    runner = load_runner()
    rows = runner.list_configs()
    assert any(row["name"] == "solver_sweep" for row in rows)

    path = runner.resolve_config_path("solver_sweep")
    assert path.name == "solver_sweep.json"

    config = runner.load_config("solver_sweep")
    runner.apply_overrides(
        config,
        [
            "device=\"cpu\"",
            "solvers.0.max_iter=2",
            "solvers.0.alpha=0.25",
            "new_section.enabled=true",
        ],
    )
    assert config["device"] == "cpu"
    assert config["solvers"][0]["max_iter"] == 2
    assert config["solvers"][0]["alpha"] == 0.25
    assert config["new_section"]["enabled"] is True


def test_package_console_entry_points_and_configs_are_registered() -> None:
    scripts = {
        entry.name: entry.value
        for entry in metadata.entry_points(group="console_scripts")
        if entry.name.startswith("silva-")
    }
    assert scripts["silva-experiment"] == "silva_networks.public_experiments:main"
    assert scripts["silva-download-datasets"] == "silva_networks.dataset_cli:main"
    assert public_experiments.resolve_config_path("graph_silva_smoke").exists()


def test_graph_smoke_config_runs() -> None:
    runner = load_runner()
    output = runner.run_config(
        {
            "name": "test_graph",
            "kind": "graph_classification",
            "seed": 7,
            "device": "cpu",
            "graphs": 2,
            "nodes_per_graph": 4,
            "in_dim": 3,
            "hidden_dims": [5],
            "classes": 2,
            "solver": {"solver": "picard", "max_iter": 2, "alpha": 0.5},
            "steps": 2,
        }
    )
    assert output["output_shape"] == [2, 2]
    assert len(output["losses"]) == 2


def test_fully_configurable_graph_file_runs() -> None:
    runner = load_runner()
    path = ROOT / "experiments/public/configs/fully_configurable_graph.json"
    config = json.loads(path.read_text())
    config["device"] = "cpu"
    config["steps"] = 1
    config["solver"] = [
        {**item, "max_iter": min(int(item["max_iter"]), 2)}
        for item in config["solver"]
    ]
    output = runner.run_config(config)
    assert output["output_shape"] == [3, 2]
    assert output["state_shape"] == [18, 10]
    assert len(output["solver_residuals"]) == 3


def test_graph_operator_options_config_runs() -> None:
    runner = load_runner()
    output = runner.run_config(
        {
            "name": "test_graph_operator_options",
            "kind": "graph_operator_options",
            "seed": 8,
            "device": "cpu",
            "graphs": 2,
            "nodes_per_graph": 4,
            "in_dim": 4,
            "hidden_dim": 6,
            "classes": 2,
            "num_heads": 2,
            "k_neighbors": 3,
            "stack_alphas": [0.5, 0.2],
            "max_iter": 2,
            "steps": 1,
            "cases": [
                {"name": "full", "attention_mode": "simple", "graph_mode": "GAT"},
                {"name": "none", "attention_mode": "none", "graph_mode": "none"},
                {"name": "topk", "attention_mode": "topk", "graph_mode": "GAT"},
            ],
        }
    )
    assert [row["case"] for row in output["results"]] == ["full", "none", "topk"]
    assert all(row["output_shape"] == [8, 2] for row in output["results"])


def test_vision_vector_and_molecular_smokes_run() -> None:
    runner = load_runner()
    vision = runner.run_config(
        {
            "name": "test_vision_vector",
            "kind": "vision_vector_ablation",
            "seed": 9,
            "device": "cpu",
            "samples": 6,
            "in_dim": 16,
            "hidden_dim": 6,
            "num_heads": 2,
            "k_neighbors": 3,
            "max_iter": 2,
            "steps": 1,
            "cases": [
                {"name": "full", "attention_mode": "simple", "graph_mode": "GAT"},
                {"name": "none", "attention_mode": "none", "graph_mode": "none"},
            ],
        }
    )
    assert len(vision["results"]) == 2

    molecular = runner.run_config(
        {
            "name": "test_molecular",
            "kind": "molecular_smoke",
            "seed": 10,
            "device": "cpu",
            "graphs": 2,
            "nodes_per_graph": 4,
            "hidden_dim": 6,
            "num_heads": 2,
            "max_iter": 2,
            "steps": 1,
        }
    )
    assert molecular["output_shape"] == [2]
    assert molecular["state_shape"] == [8, 6]


class FakeTorchvisionDataset:
    def __init__(self, name: str):
        self.name = name
        self.channels = 1 if name in {"MNIST", "FashionMNIST", "KMNIST", "EMNIST"} else 3
        self.size = 8
        self.classes = [f"class_{index}" for index in range(3)]

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int):
        image = torch.full(
            (self.channels, self.size, self.size),
            float(index) / 12.0,
            dtype=torch.float32,
        )
        return image, index % len(self.classes)


def test_torchvision_image_classification_configs_run(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setattr(
        public_experiments,
        "load_torchvision_dataset",
        lambda name, **kwargs: FakeTorchvisionDataset(name),
    )

    cortex = runner.run_config(
        {
            "name": "test_cifar10_cortex",
            "kind": "torchvision_image_classification",
            "dataset": "CIFAR10",
            "download": False,
            "max_samples": 6,
            "preset": "cortex",
            "hidden_dim": [6, 4],
            "classes": 3,
            "attention_mode": "none",
            "graph_mode": "none",
            "num_heads": 1,
            "alphas": [0.5, 0.2],
            "max_iter": 2,
            "steps": 1,
            "device": "cpu",
        }
    )
    assert cortex["dataset"] == "CIFAR10"
    assert cortex["preset"] == "cortex"
    assert cortex["output_shape"] == [6, 3]

    suite = runner.run_config(
        {
            "name": "test_torchvision_suite",
            "kind": "torchvision_dataset_suite",
            "download": False,
            "max_samples": 4,
            "preset": "vector",
            "hidden_dim": 5,
            "classes": 3,
            "attention_mode": "none",
            "graph_mode": "none",
            "num_heads": 1,
            "alphas": [0.25],
            "max_iter": 2,
            "steps": 1,
            "device": "cpu",
            "datasets": [{"dataset": "MNIST"}, {"dataset": "CIFAR10"}],
        }
    )
    assert [row["dataset"] for row in suite["results"]] == ["MNIST", "CIFAR10"]
    assert all(row["output_shape"] == [4, 3] for row in suite["results"])
