from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from silva_networks import (
    SILVAConvVisionClassifier,
    SILVAGraphNetwork,
    SILVAGraphPresetNetwork,
    SILVAImageClassifier,
    SILVAImageCortexClassifier,
    SILVAMolecularRegressor,
    SILVAVisionVectorClassifier,
    SolverConfig,
    fixed_point,
    load_tabular_dataset,
    load_torchvision_dataset,
    make_knn_edge_index,
    move_to_device,
    resolve_device,
    stability_report,
    tabular_to_silva_graph,
)

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"


class SignedLocal(torch.nn.Module):
    """Small custom local branch used by the public custom-operator experiment."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        messages = torch.tanh(self.proj(z))
        if edge_index is None:
            return messages
        src, dst = edge_index
        out = torch.zeros_like(messages)
        out.index_add_(0, dst, messages[src])
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a public SILVA package experiment.")
    parser.add_argument(
        "--config",
        help="Path to a JSON config, or a built-in config name such as solver_sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List built-in public experiment configs and exit.",
    )
    parser.add_argument(
        "--show-config",
        help="Print a built-in or path-based JSON config and exit.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Override the config device field.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value. VALUE is parsed as JSON when possible; "
            "nested keys use dots, for example solver.max_iter=2."
        ),
    )
    args = parser.parse_args()

    if args.list_configs:
        print(json.dumps(list_configs(), indent=2))
        return
    if args.show_config:
        config = load_config(args.show_config)
        if args.device is not None:
            config["device"] = args.device
        apply_overrides(config, args.overrides)
        print(json.dumps(config, indent=2))
        return
    if args.config is None:
        parser.error("--config is required unless --list-configs or --show-config is used")

    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    apply_overrides(config, args.overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = run_config(config)
    out_path = args.output_dir / f"{config['name']}_metrics.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), **output}, indent=2))


def list_configs() -> list[dict[str, str]]:
    """Return built-in public config names, paths, and experiment kinds."""

    rows = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        config = json.loads(path.read_text())
        rows.append(
            {
                "name": config.get("name", path.stem),
                "kind": config.get("kind", "unknown"),
                "path": f"silva_networks/{path.relative_to(ROOT)}",
            }
        )
    return rows


def load_config(value: str | Path) -> dict[str, Any]:
    """Load a config by filesystem path or built-in config name."""

    path = resolve_config_path(value)
    return json.loads(path.read_text())


def resolve_config_path(value: str | Path) -> Path:
    """Resolve a path, built-in config stem, or built-in config filename."""

    raw = Path(value).expanduser()
    if raw.exists():
        return raw
    candidates = [CONFIG_DIR / raw.name]
    if raw.suffix != ".json":
        candidates.append(CONFIG_DIR / f"{raw.name}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = ", ".join(path.stem for path in sorted(CONFIG_DIR.glob("*.json")))
    raise FileNotFoundError(f"Could not find config {value!r}. Available configs: {available}")


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted CLI overrides to a JSON config in place."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form: {override!r}")
        key, raw_value = override.split("=", 1)
        set_config_value(config, key.split("."), parse_override_value(raw_value))


def parse_override_value(raw_value: str) -> Any:
    """Parse a CLI override value as JSON, falling back to a string."""

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def set_config_value(container: dict[str, Any] | list[Any], path: list[str], value: Any) -> None:
    """Set a nested config field. Numeric path parts address list indices."""

    if not path or any(part == "" for part in path):
        raise ValueError("Override keys must be non-empty")
    target: dict[str, Any] | list[Any] = container
    for part in path[:-1]:
        if isinstance(target, list):
            target = target[int(part)]
            continue
        if part not in target or target[part] is None:
            target[part] = {}
        target = target[part]
        if not isinstance(target, (dict, list)):
            raise TypeError(f"Cannot descend into non-container override path: {'.'.join(path)}")
    final = path[-1]
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config.get("seed", 0)))
    device = resolve_device(config.get("device", "auto"))
    kind = config["kind"]
    if kind == "solver_sweep":
        return run_solver_sweep(config, device)
    if kind == "graph_classification":
        return run_graph_classification(config, device)
    if kind == "graph_operator_options":
        return run_graph_operator_options(config, device)
    if kind == "vision_smoke":
        return run_vision_smoke(config, device)
    if kind == "vision_vector_ablation":
        return run_vision_vector_ablation(config, device)
    if kind == "torchvision_image_classification":
        return run_torchvision_image_classification(config, device)
    if kind == "torchvision_dataset_suite":
        return run_torchvision_dataset_suite(config, device)
    if kind == "molecular_smoke":
        return run_molecular_smoke(config, device)
    if kind == "tabular_node_classification":
        return run_tabular_node_classification(config, device)
    if kind == "tabular_dataset_suite":
        return run_tabular_dataset_suite(config, device)
    if kind == "custom_operator_experiment":
        return run_custom_operator_experiment(config, device)
    raise ValueError(f"Unknown experiment kind: {kind}")


def run_solver_sweep(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dim = int(config.get("dim", 6))
    scale = float(config.get("scale", 0.35))
    W = scale * torch.randn(dim, dim, device=device) / dim**0.5
    b = torch.linspace(-0.2, 0.2, dim, device=device)
    z0 = torch.zeros(dim, device=device)

    def transition(z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(W @ z + b)

    records = []
    for solver in config["solvers"]:
        solver_config = SolverConfig(**solver)
        result = fixed_point(transition, z0, solver_config)
        report = stability_report(
            transition,
            result.z,
            samples=int(config.get("jacobian_samples", 4)),
            iters=int(config.get("power_iters", 8)),
        )
        records.append(
            {
                "solver": solver_config.solver,
                "iterations": result.iterations,
                "converged": result.converged,
                "residual": result.residual,
                "stability_residual": report.residual,
                "spectral_radius": report.spectral_radius,
                "jacobian_norm_estimate": report.jacobian_norm_estimate,
            }
        )
    return {"name": config["name"], "kind": config["kind"], "device": device.type, "results": records}


def run_graph_classification(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = make_graph_batch(config, device)
    task = config.get("task", "graph")
    model = SILVAGraphNetwork(
        in_dim=int(config["in_dim"]),
        hidden_dims=config["hidden_dims"],
        out_dim=int(config.get("classes", 2)),
        task=task,
        pooling=config.get("pooling", "mean"),
        config=make_solver_configs(config),
        local=config.get("local", "graph"),
        global_term=config.get("global_term", "mean"),
        self_term=config.get("self_term"),
        head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
        dropout=float(config.get("dropout", 0.0)),
        normalize=bool(config.get("normalize_layers", True)),
        local_kwargs=make_local_kwargs(config, default_local="graph"),
        global_kwargs=make_global_kwargs(config, default_global="mean"),
        self_kwargs=config.get("self_kwargs"),
    ).to(device)
    return train_graph_model(config, model, data, device)


def run_graph_operator_options(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    records = []
    cases = config.get(
        "cases",
        [
            {"name": "full", "attention_mode": "simple", "graph_mode": "GAT"},
            {"name": "no_global", "attention_mode": "none", "graph_mode": "GAT"},
            {"name": "no_local", "attention_mode": "simple", "graph_mode": "none"},
            {"name": "none", "attention_mode": "none", "graph_mode": "none"},
            {"name": "static_global", "attention_mode": "static", "graph_mode": "GAT"},
            {"name": "topk_global", "attention_mode": "topk", "graph_mode": "GAT"},
        ],
    )
    for offset, case in enumerate(cases):
        torch.manual_seed(int(config.get("seed", 0)) + offset)
        data = make_graph_batch(config, device)
        task = config.get("task", "node")
        model = SILVAGraphPresetNetwork(
            in_dim=int(config["in_dim"]),
            hidden_dim=config.get("hidden_dim", 16),
            out_dim=int(config.get("classes", 2)),
            task=task,
            attention_mode=case.get("attention_mode", config.get("attention_mode", "simple")),
            graph_mode=case.get("graph_mode", config.get("graph_mode", "GAT")),
            num_heads=int(config.get("num_heads", 4)),
            k_neighbors=int(config.get("k_neighbors", 4)),
            local_depth=int(case.get("local_depth", config.get("local_depth", 1))),
            stack_alphas=config.get("stack_alphas", [0.5, 0.2]),
            max_iter=int(config.get("max_iter", 4)),
            solver=config.get("solver_name", "picard"),
            head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
        ).to(device)
        target = data["y_node"] if task == "node" else data["y"]
        result = train_graph_operator_case(config, model, data, target, device)
        result.update(
            {
                "case": case.get("name", f"case_{offset}"),
                "attention_mode": case.get("attention_mode", config.get("attention_mode", "simple")),
                "graph_mode": case.get("graph_mode", config.get("graph_mode", "GAT")),
                "local_depth": int(case.get("local_depth", config.get("local_depth", 1))),
            }
        )
        records.append(result)
    return {"name": config["name"], "kind": config["kind"], "device": device.type, "results": records}


def run_custom_operator_experiment(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = make_graph_batch(config, device)
    model = SILVAGraphNetwork(
        in_dim=int(config["in_dim"]),
        hidden_dims=config["hidden_dims"],
        out_dim=int(config.get("classes", 2)),
        task="graph",
        pooling=config.get("pooling", "mean"),
        config=make_solver_configs(config),
        local=lambda dim, index: SignedLocal(dim) if index % 2 == 1 else "topk",
        global_term=config.get("global_term", "mean"),
        head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
        local_kwargs={"k": int(config.get("k", 3))},
    ).to(device)
    result = train_graph_model(config, model, data, device)
    result["scope"] = "extensions"
    return result


def run_vision_smoke(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    samples = int(config.get("samples", 12))
    size = int(config.get("size", 8))
    x = torch.randn(samples, 1, size, size, device=device)
    left = x[:, :, :, : size // 2].mean(dim=(1, 2, 3))
    right = x[:, :, :, size // 2 :].mean(dim=(1, 2, 3))
    y = (right > left).long()
    model = SILVAImageClassifier(
        in_channels=1,
        hidden_channels=config["hidden_channels"],
        num_classes=2,
        config=make_solver_configs(config),
        head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    steps = int(config.get("steps", 5))
    for _ in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    accuracy = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
    return {
        "name": config["name"],
        "kind": config["kind"],
        "device": device.type,
        "losses": losses,
        "accuracy": accuracy,
        "output_shape": list(logits.shape),
    }


def run_vision_vector_ablation(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    records = []
    samples = int(config.get("samples", 16))
    in_dim = int(config.get("in_dim", 64))
    cases = config.get(
        "cases",
        [
            {"name": "full", "attention_mode": "simple", "graph_mode": "GAT"},
            {"name": "no_global", "attention_mode": "none", "graph_mode": "GAT"},
            {"name": "no_local", "attention_mode": "simple", "graph_mode": "none"},
            {"name": "none", "attention_mode": "none", "graph_mode": "none"},
            {"name": "static_global", "attention_mode": "static", "graph_mode": "GNN"},
        ],
    )
    x = torch.randn(samples, in_dim, device=device)
    y = (x[:, : in_dim // 2].mean(dim=1) > x[:, in_dim // 2 :].mean(dim=1)).long()
    for offset, case in enumerate(cases):
        torch.manual_seed(int(config.get("seed", 0)) + offset)
        model = SILVAVisionVectorClassifier(
            in_dim=in_dim,
            hidden_dim=config.get("hidden_dim", 16),
            num_classes=2,
            attention_mode=case.get("attention_mode", "simple"),
            graph_mode=case.get("graph_mode", "GAT"),
            k_neighbors=int(config.get("k_neighbors", 4)),
            num_heads=int(config.get("num_heads", 4)),
            alphas=tuple(config.get("alphas", [0.25])),
            max_iter=int(config.get("max_iter", 4)),
            solver=config.get("solver_name", "picard"),
            head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
        losses: list[float] = []
        for _ in range(int(config.get("steps", 3))):
            result = model(x, return_results=True)
            loss = F.cross_entropy(result.output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        records.append(
            {
                "case": case.get("name", f"case_{offset}"),
                "attention_mode": case.get("attention_mode", "simple"),
                "graph_mode": case.get("graph_mode", "GAT"),
                "losses": losses,
                "accuracy": float((result.output.argmax(dim=1) == y).float().mean().detach().cpu()),
                "solver_residuals": [item.residual for item in result.solver_results or []],
            }
        )
    return {"name": config["name"], "kind": config["kind"], "device": device.type, "results": records}


def run_torchvision_image_classification(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    x, y, metadata = load_torchvision_batch(config, device)
    preset = config.get("preset", "cortex")
    num_classes = int(config.get("classes", metadata["classes"]))
    hidden_dim = config.get("hidden_dim", [16, 12] if preset == "cortex" else 16)
    common = {
        "max_iter": int(config.get("max_iter", 3)),
        "solver": config.get("solver_name", "picard"),
        "attention_mode": config.get("attention_mode", "simple"),
        "graph_mode": config.get("graph_mode", "GAT"),
        "k_neighbors": int(config.get("k_neighbors", 4)),
        "num_heads": int(config.get("num_heads", 4)),
    }

    if preset == "cortex":
        model = SILVAImageCortexClassifier(
            in_channels=int(x.shape[1]),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            image_size=int(x.shape[-1]),
            alphas=tuple(config.get("alphas", [0.5, 0.2])),
            internal_depth=int(config.get("internal_depth", 1)),
            self_interaction=bool(config.get("self_interaction", True)),
            **common,
        ).to(device)
    elif preset == "conv":
        model = SILVAConvVisionClassifier(
            in_channels=int(x.shape[1]),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            image_size=int(x.shape[-1]),
            alphas=tuple(config.get("alphas", [0.5, 0.2])),
            **common,
        ).to(device)
    elif preset == "vector":
        model = SILVAVisionVectorClassifier(
            in_dim=int(x[0].numel()),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            alphas=tuple(config.get("alphas", [0.25])),
            **common,
        ).to(device)
    else:
        raise ValueError("preset must be one of: cortex, conv, vector")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    result = None
    steps = int(config.get("steps", 2))
    for _ in range(steps):
        result = model(x, return_results=True)
        loss = F.cross_entropy(result.output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert result is not None
    return {
        "name": config["name"],
        "kind": config["kind"],
        "dataset": metadata["dataset"],
        "source": metadata["source"],
        "preset": preset,
        "device": device.type,
        "samples": int(x.shape[0]),
        "image_shape": list(x.shape[1:]),
        "classes": num_classes,
        "losses": losses,
        "accuracy": float((result.output.argmax(dim=1) == y).float().mean().detach().cpu()),
        "output_shape": list(result.output.shape),
        "state_shape": list(result.state.shape),
        "solver_residuals": [item.residual for item in result.solver_results or []],
    }


def run_torchvision_dataset_suite(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    records = []
    defaults = {key: value for key, value in config.items() if key != "datasets"}
    for case in config["datasets"]:
        case_config = {**defaults, **case}
        case_config["kind"] = "torchvision_image_classification"
        case_config["name"] = f"{config['name']}_{case_config['dataset']}"
        records.append(run_torchvision_image_classification(case_config, device))
    return {"name": config["name"], "kind": config["kind"], "device": device.type, "results": records}


def run_molecular_smoke(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = make_molecular_batch(config, device)
    model = SILVAMolecularRegressor(
        hidden_dim=config.get("hidden_dim", 16),
        num_atom_types=int(config.get("num_atom_types", 21)),
        num_bond_types=int(config.get("num_bond_types", 4)),
        num_heads=int(config.get("num_heads", 4)),
        alphas=tuple(config.get("alphas", [0.5, 0.2])),
        max_iter=int(config.get("max_iter", 4)),
        solver=config.get("solver_name", "picard"),
        dropout=float(config.get("dropout", 0.0)),
        spectral_norm=bool(config.get("spectral_norm", False)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    result = None
    for _ in range(int(config.get("steps", 3))):
        result = model(
            x=data["x"],
            edge_index=data["edge_index"],
            edge_attr=data["edge_attr"],
            batch=data["batch"],
            return_results=True,
        )
        loss = F.l1_loss(result.output, data["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert result is not None
    return {
        "name": config["name"],
        "kind": config["kind"],
        "device": device.type,
        "losses": losses,
        "mae": losses[-1],
        "output_shape": list(result.output.shape),
        "state_shape": list(result.state.shape),
        "solver_residuals": [item.residual for item in result.solver_results or []],
    }


def run_tabular_node_classification(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dataset = load_tabular_dataset(
        config["dataset"],
        root=config.get("data_root", "data"),
        download=bool(config.get("download", True)),
        normalize=bool(config.get("normalize", True)),
    )
    if dataset.task != "classification":
        raise ValueError("tabular_node_classification requires a classification dataset")
    max_samples = int(config.get("max_samples", len(dataset.y)))
    if max_samples < len(dataset.y):
        indices = stratified_classification_indices(
            dataset.y,
            max_samples=max_samples,
            seed=int(config.get("seed", 0)),
        )
        graph_data = dataset.x[indices]
        graph_y = dataset.y[indices]
        graph = tabular_to_silva_graph(
            graph_data,
            y=graph_y,
            k=int(config.get("k", 6)),
            normalize=bool(config.get("renormalize_subset", True)),
            metric=config.get("metric", "euclidean"),
            undirected=bool(config.get("undirected", False)),
            device=device,
        )
        graph.metadata = {
            "name": dataset.name,
            "source": dataset.info.source,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
            "task": dataset.task,
            "sample_strategy": "stratified",
        }
    else:
        graph = tabular_to_silva_graph(
            dataset,
            k=int(config.get("k", 6)),
            normalize=bool(config.get("renormalize_subset", True)),
            max_samples=max_samples,
            metric=config.get("metric", "euclidean"),
            undirected=bool(config.get("undirected", False)),
            device=device,
        )
    x = graph.x
    edge_index = graph.edge_index
    y = graph.y
    if y is None:
        raise ValueError("tabular dataset adapter did not produce labels")

    model = SILVAGraphNetwork(
        in_dim=x.shape[1],
        hidden_dims=config["hidden_dims"],
        out_dim=int(y.max().item()) + 1,
        task="node",
        config=make_solver_configs(config),
        local=config.get("local", "topk"),
        global_term=config.get("global_term", "mean"),
        self_term=config.get("self_term"),
        head_hidden_dims=tuple(config.get("head_hidden_dims", [])),
        dropout=float(config.get("dropout", 0.0)),
        normalize=bool(config.get("normalize_layers", True)),
        local_kwargs=make_local_kwargs(config, default_local="topk"),
        global_kwargs=make_global_kwargs(config, default_global="mean"),
        self_kwargs=config.get("self_kwargs"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    steps = int(config.get("steps", 8))
    for _ in range(steps):
        result = model(x, edge_index=edge_index, return_results=True)
        loss = F.cross_entropy(result.output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    accuracy = float((result.output.argmax(dim=1) == y).float().mean().detach().cpu())
    return {
        "name": config["name"],
        "kind": config["kind"],
        "dataset": dataset.name,
        "source": dataset.info.source,
        "device": device.type,
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "classes": int(y.max().item()) + 1,
        "losses": losses,
        "accuracy": accuracy,
        "solver_residuals": [solver_result.residual for solver_result in result.solver_results or []],
    }


def stratified_classification_indices(
    y,
    *,
    max_samples: int,
    seed: int,
) -> list[int]:
    labels = torch.as_tensor(y, dtype=torch.long)
    classes = torch.unique(labels, sorted=True)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    per_class = max(1, max_samples // max(int(classes.numel()), 1))
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    used: set[int] = set()
    for label in classes.tolist():
        candidates = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        order = torch.randperm(candidates.numel(), generator=generator)
        take = min(per_class, candidates.numel(), max_samples - len(selected))
        chosen = candidates[order[:take]].tolist()
        selected.extend(int(index) for index in chosen)
        used.update(int(index) for index in chosen)
        if len(selected) >= max_samples:
            break
    if len(selected) < max_samples:
        remaining = [index for index in range(labels.numel()) if index not in used]
        if remaining:
            order = torch.randperm(len(remaining), generator=generator).tolist()
            selected.extend(remaining[index] for index in order[: max_samples - len(selected)])
    return sorted(selected[:max_samples])


def run_tabular_dataset_suite(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    records = []
    defaults = {key: value for key, value in config.items() if key != "datasets"}
    for case in config["datasets"]:
        case_config = {**defaults, **case}
        case_config["kind"] = "tabular_node_classification"
        case_config["name"] = f"{config['name']}_{case_config['dataset']}"
        records.append(run_tabular_node_classification(case_config, device))
    return {
        "name": config["name"],
        "kind": config["kind"],
        "device": device.type,
        "results": records,
    }


def load_torchvision_batch(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dataset_name = config["dataset"]
    train = bool(config.get("train", True))
    dataset_kwargs = {}
    if "split" in config:
        dataset_kwargs["split"] = config["split"]
    dataset = load_torchvision_dataset(
        dataset_name,
        root=config.get("data_root", "data"),
        train=train,
        download=bool(config.get("download", True)),
        **dataset_kwargs,
    )
    max_samples = min(int(config.get("max_samples", 32)), len(dataset))
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for index in range(max_samples):
        image, target = dataset[index]
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        image = image.float()
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() == 3 and image.shape[-1] in {1, 3} and image.shape[0] not in {1, 3}:
            image = image.permute(2, 0, 1)
        target_value = int(target.item()) if torch.is_tensor(target) else int(target)
        images.append(image)
        labels.append(target_value)

    x = torch.stack(images).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    image_size = config.get("image_size")
    if image_size is not None and int(image_size) != int(x.shape[-1]):
        x = F.interpolate(
            x,
            size=(int(image_size), int(image_size)),
            mode="bilinear",
            align_corners=False,
        )
    if bool(config.get("standardize", False)):
        flat = x.flatten(1)
        mean = flat.mean(dim=0, keepdim=True)
        scale = flat.std(dim=0, keepdim=True).clamp_min(1e-8)
        x = ((flat - mean) / scale).reshape_as(x)

    classes = config.get("classes")
    if classes is None:
        classes = len(getattr(dataset, "classes", [])) or int(y.max().item()) + 1
    return (
        x,
        y,
        {
            "dataset": dataset_name,
            "source": "TorchVision",
            "classes": int(classes),
            "train": train,
        },
    )


def train_graph_model(
    config: dict[str, Any],
    model: SILVAGraphNetwork,
    data: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    data = move_to_device(data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    steps = int(config.get("steps", 5))
    for _ in range(steps):
        result = model(
            data["x"],
            edge_index=data["edge_index"],
            batch=data["batch"],
            return_results=True,
        )
        target = data["y_node"] if getattr(model, "task", "graph") == "node" else data["y"]
        loss = F.cross_entropy(result.output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    accuracy = float((result.output.argmax(dim=1) == target).float().mean().detach().cpu())
    solver_residuals = [solver_result.residual for solver_result in result.solver_results or []]
    return {
        "name": config["name"],
        "kind": config["kind"],
        "device": device.type,
        "losses": losses,
        "accuracy": accuracy,
        "output_shape": list(result.output.shape),
        "state_shape": list(result.state.shape),
        "solver_residuals": solver_residuals,
    }


def train_graph_operator_case(
    config: dict[str, Any],
    model: SILVAGraphPresetNetwork,
    data: dict[str, torch.Tensor],
    target: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    data = move_to_device(data, device)
    target = target.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-2)))
    losses: list[float] = []
    result = None
    for _ in range(int(config.get("steps", 3))):
        result = model(
            data["x"],
            edge_index=data["edge_index"],
            batch=data["batch"],
            return_results=True,
        )
        loss = F.cross_entropy(result.output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert result is not None
    return {
        "losses": losses,
        "accuracy": float((result.output.argmax(dim=1) == target).float().mean().detach().cpu()),
        "output_shape": list(result.output.shape),
        "state_shape": list(result.state.shape),
        "solver_residuals": [solver_result.residual for solver_result in result.solver_results or []],
    }


def make_graph_batch(config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    graphs = int(config.get("graphs", 4))
    nodes_per_graph = int(config.get("nodes_per_graph", 6))
    in_dim = int(config["in_dim"])
    x = torch.randn(graphs * nodes_per_graph, in_dim, device=device)
    batch = torch.arange(graphs, device=device).repeat_interleave(nodes_per_graph)
    edges_src: list[int] = []
    edges_dst: list[int] = []
    for graph in range(graphs):
        offset = graph * nodes_per_graph
        for node in range(nodes_per_graph):
            edges_src.append(offset + node)
            edges_dst.append(offset + ((node + 1) % nodes_per_graph))
            edges_src.append(offset + ((node + 1) % nodes_per_graph))
            edges_dst.append(offset + node)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
    means = torch.stack([x[batch == graph, :2].mean() for graph in range(graphs)])
    y = (means > 0).long()
    node_signal = x[:, 0] + 0.25 * means[batch]
    y_node = (node_signal > 0).long()
    return {"x": x, "edge_index": edge_index, "batch": batch, "y": y, "y_node": y_node}


def make_molecular_batch(config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    graphs = int(config.get("graphs", 2))
    nodes_per_graph = int(config.get("nodes_per_graph", 5))
    atom_types = int(config.get("num_atom_types", 21))
    bond_types = int(config.get("num_bond_types", 4))
    total_nodes = graphs * nodes_per_graph
    x = torch.arange(total_nodes, device=device) % atom_types
    batch = torch.arange(graphs, device=device).repeat_interleave(nodes_per_graph)
    src: list[int] = []
    dst: list[int] = []
    edge_values: list[int] = []
    for graph in range(graphs):
        offset = graph * nodes_per_graph
        for node in range(nodes_per_graph):
            a = offset + node
            b = offset + ((node + 1) % nodes_per_graph)
            src.extend([a, b])
            dst.extend([b, a])
            edge_values.extend([(node + graph) % bond_types, (node + graph) % bond_types])
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_values, dtype=torch.long, device=device)
    target = torch.stack([x[batch == graph].float().mean() / atom_types for graph in range(graphs)])
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
        "target": target,
    }


def make_solver_configs(config: dict[str, Any]) -> SolverConfig | list[SolverConfig]:
    solver_config = config.get("solver", {"solver": "picard", "max_iter": 8, "alpha": 0.5})
    if isinstance(solver_config, list):
        return [SolverConfig(**item) for item in solver_config]
    return SolverConfig(**solver_config)


def make_local_kwargs(config: dict[str, Any], default_local: str) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    """Return local-operator kwargs from JSON, including top-k defaults."""

    if "local_kwargs" in config:
        return config["local_kwargs"]
    local = config.get("local", default_local)
    if "k" not in config:
        return None
    k = int(config["k"])
    if isinstance(local, str):
        return {"k": k} if local in {"topk", "channel_knn", "vision_knn"} else None
    if isinstance(local, list):
        return [
            {"k": k} if item in {"topk", "channel_knn", "vision_knn"} else None
            for item in local
        ]
    return None


def make_global_kwargs(
    config: dict[str, Any],
    default_global: str,
) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    """Return global-operator kwargs from JSON, including top-k defaults."""

    if "global_kwargs" in config:
        return config["global_kwargs"]
    global_term = config.get("global_term", default_global)
    if "global_k" in config:
        k = int(config["global_k"])
    elif "k_neighbors" in config:
        k = int(config["k_neighbors"])
    else:
        return None
    if isinstance(global_term, str):
        return {"k": k} if global_term in {"topk", "topk_attention"} else None
    if isinstance(global_term, list):
        return [
            {"k": k} if item in {"topk", "topk_attention"} else None
            for item in global_term
        ]
    return None


def make_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    return make_knn_edge_index(x, k)


if __name__ == "__main__":
    main()
