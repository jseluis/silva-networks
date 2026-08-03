"""Reference SILVA architectures.

This module provides clean PyTorch implementations of the SILVA model families
described in "SILVA Networks as Structured Implicit Layers and Vector
Attractors via Dynamic Interaction Fields" by Jose Luis Silva. The graph
branches use graph-attention ideas from Velickovic et al. (2018), the attention
branches use scaled dot-product attention from Vaswani et al. (2017), and the
fixed-point framing follows deep equilibrium models from Bai, Kolter, and
Koltun (2019).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .architectures import (
    SILVACortexLayer,
    SILVACortexNetwork,
    SILVANetworkOutput,
    build_mlp_head,
    pool_entities,
)
from .deq_engine import SILVAVariationalDropout
from .layers import (
    DynamicChannelLocal,
    GatedMeanFieldGlobal,
    GraphAttentionLocal,
    MeanFieldGlobal,
    MultiHeadChannelAttentionGlobal,
    StaticChannelGlobal,
    StaticMeanFieldGlobal,
    TopKGlobalAttention,
    ZeroTerm,
)
from .solvers import BackwardMode, BackwardSolverName, SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
AttentionMode = Literal["none", "static", "simple", "topk", "mean"]
GraphMode = Literal["none", "GAT", "gat", "graph_attention", "mean", "graph"]
VisionAttentionMode = Literal["none", "static", "simple", "multi-head"]
VisionGraphMode = Literal["none", "GAT", "gat", "knn", "GNN"]
PoolingMode = Literal["mean", "sum", "max"]
TaskMode = Literal["node", "graph"]
EquilibriumDropoutMode = Literal["independent", "variational", "disabled"]


@dataclass(frozen=True)
class SILVAGraphPresetConfig:
    """Reference defaults for a SILVA equilibrium layer.

    The defaults encode the common two-timescale SILVA settings used by the
    public examples: Picard iteration, a local graph branch, a
    global branch, and damping. Every field can be overridden at construction
    time or replaced with custom PyTorch modules in the lower-level API.
    """

    hidden_dim: int = 64
    attention_mode: AttentionMode = "simple"
    graph_mode: GraphMode = "GAT"
    num_heads: int = 4
    k_neighbors: int = 16
    local_depth: int = 1
    alpha: float = 0.5
    max_iter: int = 15
    solver: str = "picard"
    tol: float = 1e-6
    backward_mode: BackwardMode = "unrolled"
    backward_solver: BackwardSolverName = "gmres"
    backward_max_iter: int = 50
    backward_tol: float = 1e-6
    backward_stop_mode: Literal["absolute", "relative"] = "absolute"
    backward_relative_eps: float = 1e-8

    def solver_config(self) -> SolverConfig:
        return SolverConfig(
            solver=self.solver,  # type: ignore[arg-type]
            max_iter=self.max_iter,
            alpha=self.alpha,
            tol=self.tol,
            backward_mode=self.backward_mode,
            backward_solver=self.backward_solver,
            backward_max_iter=self.backward_max_iter,
            backward_tol=self.backward_tol,
            backward_stop_mode=self.backward_stop_mode,
            backward_relative_eps=self.backward_relative_eps,
        )


class SILVAGraphPresetLayer(nn.Module):
    """Reference graph/node SILVA equilibrium layer.

    The layer computes

    ``f(z, x) = LayerNorm(ReLU(W_stim x + L(tanh(z)) + G(tanh(z))))``

    and solves ``z = f(z, x)`` with the configured fixed-point solver. It is
    designed for citation graphs, CLUSTER/PATTERN-style node tasks, and any
    user dataset represented by feature matrix ``x`` and ``edge_index``.

    Args:
        in_dim: Number of input features per node/entity.
        hidden_dim: Recurrent state dimension.
        attention_mode: Global branch: `none`, `static`, `simple`, `topk`, or
            `mean`.
        graph_mode: Local branch: `GAT`, `graph_attention`, `mean`, `graph`, or
            `none`.
        num_heads: Number of graph-attention heads.
        k_neighbors: Top-k support size for bounded global attention.
        local_depth: Number of weight-tied local applications inside one
            solver step.
        config: Fixed-point solver configuration.
        normalize: If true, apply `LayerNorm` in the update block.

    Inputs:
        x: Tensor with shape `(nodes, in_dim)`.
        edge_index: Optional tensor with shape `(2, edges)`, source row first.
        batch: Optional graph id tensor with shape `(nodes,)`.
        z0: Optional initial state with shape `(nodes, hidden_dim)`.

    Output:
        Tensor: Tensor with shape `(nodes, hidden_dim)` or `SolverResult` when
        `return_result=True`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        *,
        attention_mode: AttentionMode = "simple",
        graph_mode: GraphMode = "GAT",
        num_heads: int = 4,
        k_neighbors: int = 16,
        local_depth: int = 1,
        config: SolverConfig | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        if local_depth < 1:
            raise ValueError("local_depth must be at least 1")
        self.input_injection = nn.Linear(in_dim, hidden_dim)
        self.local = _make_graph_local(graph_mode, hidden_dim, num_heads)
        self.global_term = _make_graph_global(attention_mode, hidden_dim, k_neighbors)
        self.norm = nn.LayerNorm(hidden_dim) if normalize else nn.Identity()
        self.config = config or SolverConfig(alpha=0.5, max_iter=15)
        self.attention_mode = attention_mode
        self.graph_mode = graph_mode
        self.k_neighbors = k_neighbors
        self.local_depth = local_depth
        self.hidden_dim = hidden_dim

    def f(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        stimulus = self.input_injection(x)
        signal = torch.tanh(z)
        local = self._local_term(signal, edge_index)
        global_update = self.global_term(signal, batch=batch)
        return self.norm(F.relu(stimulus + local + global_update))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        z_init = (
            torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)
            if z0 is None
            else z0
        )

        def transition(z: Tensor) -> Tensor:
            return self.f(z, x, edge_index=edge_index, batch=batch)

        result = solve_equilibrium(
            transition,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x),
        )
        return result if return_result else result.z

    def _local_term(self, signal: Tensor, edge_index: Tensor | None) -> Tensor:
        h = signal
        for hop in range(self.local_depth):
            h = self.local(h, edge_index=edge_index)
            if hop < self.local_depth - 1:
                h = torch.tanh(h)
        return h


class SILVAGraphPresetNetwork(nn.Module):
    """Stacked reference SILVA graph model.

    By default this is the fast/slow two-layer architecture used for the
    SILVA study's graph-node and CLUSTER/PATTERN-style cases:
    ``alpha_1 = 0.5`` and ``alpha_2 = 0.2``. Passing ``stack_alphas`` exposes
    the same hierarchy as an arbitrary-depth stack.

    Args:
        in_dim: Number of input features per node/entity.
        hidden_dim: Recurrent state dimension for every equilibrium layer, or
            one dimension per layer when `stack_alphas` is supplied.
        out_dim: Number of output classes or regression targets.
        task: `node` for per-node readout, `graph` for pooled graph readout.
        pooling: Graph pooling mode when `task="graph"`.
        attention_mode: Global branch mode.
        graph_mode: Local branch mode.
        num_heads: Number of graph-attention heads.
        k_neighbors: Top-k support size for bounded global attention.
        local_depth: Weight-tied local repetitions inside a solver step.
        layer1_alpha: Damping for the first layer when `stack_alphas` is absent.
        layer2_alpha: Damping for the second layer when `stack_alphas` is absent.
        stack_alphas: Optional damping values for an arbitrary-depth stack.
        max_iter: Solver iterations per equilibrium layer.
        solver: Fixed-point solver name.
        head_hidden_dims: Hidden dimensions for the readout head.
        dropout: Dropout probability inside the readout head.

    Inputs:
        x: Tensor with shape `(nodes, in_dim)`.
        edge_index: Optional tensor with shape `(2, edges)`.
        batch: Optional graph id tensor with shape `(nodes,)`.

    Output:
        Tensor | SILVANetworkOutput: Output tensor, or `SILVANetworkOutput` when state/results are requested.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | Sequence[int],
        out_dim: int,
        *,
        task: TaskMode = "node",
        pooling: PoolingMode = "mean",
        attention_mode: AttentionMode = "simple",
        graph_mode: GraphMode = "GAT",
        num_heads: int = 4,
        k_neighbors: int = 16,
        local_depth: int = 1,
        layer1_alpha: float = 0.5,
        layer2_alpha: float = 0.2,
        stack_alphas: Sequence[float] | None = None,
        max_iter: int = 15,
        solver: str = "picard",
        backward_mode: BackwardMode = "unrolled",
        backward_solver: BackwardSolverName = "gmres",
        backward_max_iter: int = 50,
        backward_tol: float = 1e-6,
        solver_configs: SolverConfig | Sequence[SolverConfig] | None = None,
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ):
        super().__init__()
        if task not in {"node", "graph"}:
            raise ValueError("task must be 'node' or 'graph'")
        alphas = list(stack_alphas) if stack_alphas is not None else [layer1_alpha, layer2_alpha]
        if len(alphas) < 1:
            raise ValueError("at least one equilibrium layer is required")
        hidden_dims = _normalize_hidden_dims(hidden_dim, len(alphas))
        configs = _preset_solver_configs(
            solver_configs,
            len(alphas),
            lambda index: SolverConfig(
                solver=solver,  # type: ignore[arg-type]
                max_iter=max_iter,
                alpha=alphas[index],
                backward_mode=backward_mode,
                backward_solver=backward_solver,
                backward_max_iter=backward_max_iter,
                backward_tol=backward_tol,
            ),
        )
        layers: list[SILVAGraphPresetLayer] = []
        previous = in_dim
        for index, layer_hidden_dim in enumerate(hidden_dims):
            layers.append(
                SILVAGraphPresetLayer(
                    previous,
                    layer_hidden_dim,
                    attention_mode=attention_mode,
                    graph_mode=graph_mode,
                    num_heads=num_heads,
                    k_neighbors=k_neighbors,
                    local_depth=local_depth,
                    config=configs[index],
                )
            )
            previous = layer_hidden_dim
        self.layers = nn.ModuleList(layers)
        self.head = build_mlp_head(hidden_dims[-1], out_dim, head_hidden_dims, dropout)
        self.task = task
        self.pooling = pooling
        self.hidden_dims = tuple(hidden_dims)
        self.hidden_dim = hidden_dims[-1]

    @property
    def layer1(self) -> SILVAGraphPresetLayer:
        return self.layers[0]

    @property
    def layer2(self) -> SILVAGraphPresetLayer | None:
        return self.layers[1] if len(self.layers) > 1 else None

    @property
    def extra_layers(self) -> list[SILVAGraphPresetLayer]:
        return list(self.layers[2:])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
        return_state: bool = False,
        return_results: bool = False,
    ):
        state = x
        results: list[SolverResult] = []
        for index, layer in enumerate(self.layers):
            layer_input = state if index == 0 else torch.tanh(state)
            result = layer(layer_input, edge_index=edge_index, batch=batch, return_result=True)
            state = result.z
            results.append(result)
        features = (
            state if self.task == "node" else pool_entities(state, batch=batch, mode=self.pooling)
        )
        output = self.head(torch.tanh(features))
        if return_state or return_results:
            return SILVANetworkOutput(
                output=output,
                state=state,
                solver_results=results if return_results else None,
            )
        return output


class SILVAVisionVectorLayer(nn.Module):
    """Hidden-channel SILVA equilibrium for flattened vision features.

    This layer mirrors the SILVA study's dynamic-channel vision family: local
    interaction is a state-dependent hidden-channel k-NN average, global
    interaction is per-sample channel attention, and the update map is a raw
    sum before the outer damped solver.

    Args:
        in_dim: Number of input features per sample.
        hidden_dim: Number of recurrent hidden channels.
        attention_mode: Global channel branch: `none`, `static`, `simple`, or
            `multi-head`.
        graph_mode: Local channel branch: `none`, `GAT`, `GNN`, or `knn`.
        k_neighbors: Number of hidden-channel neighbors.
        num_heads: Number of attention heads for `multi-head` mode.
        config: Fixed-point solver configuration.

    Inputs:
        x: Tensor with shape `(batch, in_dim)`.
        z0: Optional initial state with shape `(batch, hidden_dim)`.

    Output:
        Tensor: Tensor with shape `(batch, hidden_dim)` or `SolverResult` when
        `return_result=True`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        *,
        attention_mode: VisionAttentionMode = "simple",
        graph_mode: VisionGraphMode = "GAT",
        k_neighbors: int = 4,
        num_heads: int = 4,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        self.input_injection = nn.Linear(in_dim, hidden_dim)
        self.local = _make_vision_local(graph_mode, hidden_dim, k_neighbors)
        self.global_term = _make_vision_global(attention_mode, hidden_dim, num_heads)
        self.config = config or SolverConfig(alpha=0.25, max_iter=20)
        self.hidden_dim = hidden_dim
        self.attention_mode = attention_mode
        self.graph_mode = graph_mode

    def f(self, z: Tensor, x: Tensor, return_energy: bool = False):
        signal = torch.tanh(z)
        stimulus = self.input_injection(x)
        local = self.local(signal)
        global_update = self.global_term(signal)
        out = stimulus + local + global_update
        if return_energy:
            energy = quadratic_interaction_energy(signal, global_update)
            return out, energy
        return out

    def forward(
        self,
        x: Tensor,
        z0: Tensor | None = None,
        return_result: bool = False,
        return_energy_trace: bool = False,
    ):
        z_init = (
            torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)
            if z0 is None
            else z0
        )
        if not return_energy_trace:

            def transition(z: Tensor) -> Tensor:
                return self.f(z, x)

            result = solve_equilibrium(
                transition,
                z_init,
                self.config,
                params=tuple(self.parameters()),
                tensors=_differentiable_tensors(x),
            )
            return result if return_result else result.z
        energies: list[float] = []

        def step(z: Tensor) -> Tensor:
            out, energy = self.f(z, x, return_energy=True)
            energies.append(float(energy.detach().cpu()))
            return out

        result = solve_equilibrium(
            step,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x),
        )
        result.info["energy_trace"] = str(energies)
        return result if return_result else (result.z, energies)


class SILVAVisionVectorClassifier(nn.Module):
    """Vector-input vision classifier using one or more SILVA vector layers.

    Args:
        in_dim: Flattened input dimension.
        hidden_dim: Recurrent hidden-channel count, or one hidden-channel count
            per value in `alphas`.
        num_classes: Number of class logits.
        attention_mode: Global channel branch mode.
        graph_mode: Local channel branch mode.
        k_neighbors: Number of hidden-channel neighbors.
        num_heads: Number of channel-attention heads.
        alphas: Damping values, one per equilibrium layer.
        max_iter: Solver iterations per layer.
        solver: Fixed-point solver name.
        head_hidden_dims: Hidden dimensions for the readout head.
        dropout: Dropout probability inside the readout head.

    Inputs:
        x: Tensor with shape `(batch, in_dim)` or image-like tensor flattened
            internally.

    Output:
        Tensor | SILVANetworkOutput: Logit tensor, or `SILVANetworkOutput` when state/results are requested.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | Sequence[int],
        num_classes: int,
        *,
        attention_mode: VisionAttentionMode = "simple",
        graph_mode: VisionGraphMode = "GAT",
        k_neighbors: int = 4,
        num_heads: int = 4,
        alphas: Sequence[float] = (0.25,),
        max_iter: int = 20,
        solver: str = "picard",
        backward_mode: BackwardMode = "unrolled",
        backward_solver: BackwardSolverName = "gmres",
        backward_max_iter: int = 50,
        backward_tol: float = 1e-6,
        solver_configs: SolverConfig | Sequence[SolverConfig] | None = None,
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(alphas) < 1:
            raise ValueError("at least one equilibrium layer is required")
        layers: list[SILVAVisionVectorLayer] = []
        previous = in_dim
        hidden_dims = _normalize_hidden_dims(hidden_dim, len(alphas))
        configs = _preset_solver_configs(
            solver_configs,
            len(alphas),
            lambda index: SolverConfig(
                solver=solver,  # type: ignore[arg-type]
                max_iter=max_iter,
                alpha=alphas[index],
                backward_mode=backward_mode,
                backward_solver=backward_solver,
                backward_max_iter=backward_max_iter,
                backward_tol=backward_tol,
            ),
        )
        for index, layer_hidden_dim in enumerate(hidden_dims):
            layers.append(
                SILVAVisionVectorLayer(
                    previous,
                    layer_hidden_dim,
                    attention_mode=attention_mode,
                    graph_mode=graph_mode,
                    k_neighbors=k_neighbors,
                    num_heads=num_heads,
                    config=configs[index],
                )
            )
            previous = layer_hidden_dim
        self.layers = nn.ModuleList(layers)
        self.hidden_dims = tuple(hidden_dims)
        self.head = build_mlp_head(hidden_dims[-1], num_classes, head_hidden_dims, dropout)

    def forward(self, x: Tensor, return_state: bool = False, return_results: bool = False):
        state = x.flatten(1) if x.dim() > 2 else x
        results: list[SolverResult] = []
        for index, layer in enumerate(self.layers):
            layer_input = state if index == 0 else torch.tanh(state)
            result = layer(layer_input, return_result=True)
            state = result.z
            results.append(result)
        output = self.head(torch.tanh(state))
        if return_state or return_results:
            return SILVANetworkOutput(
                output=output,
                state=state,
                solver_results=results if return_results else None,
            )
        return output


class SILVAConvStem(nn.Module):
    """Two-block convolutional stem used before vector equilibria.

    Args:
        in_channels: Number of image input channels.
        hidden_dim: Output feature dimension.
        image_size: Square image size.
        dropout: Dropout probability after the convolution blocks.

    Inputs:
        x: Tensor with shape `(batch, in_channels, image_size, image_size)`.

    Output:
        Tensor: Tensor with shape `(batch, hidden_dim)`.
    """

    def __init__(
        self, in_channels: int, hidden_dim: int, image_size: int = 32, dropout: float = 0.3
    ):
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4 for two 2x2 pooling blocks")
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        reduced = image_size // 4
        self.projection = nn.Linear(64 * reduced * reduced, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.pool(F.relu(self.bn1(self.conv1(x))))
        h = self.pool(F.relu(self.bn2(self.conv2(h))))
        h = self.dropout(h)
        return self.projection(h.flatten(1))


class SILVAConvVisionClassifier(nn.Module):
    """CIFAR-style convolutional stem followed by vector SILVA layers.

    Args:
        in_channels: Number of image input channels.
        hidden_dim: Recurrent hidden-channel count, or one hidden-channel count
            per value in `alphas`.
        num_classes: Number of class logits.
        image_size: Square image size.
        attention_mode: Global channel branch mode.
        graph_mode: Local channel branch mode.
        k_neighbors: Number of hidden-channel neighbors.
        num_heads: Number of channel-attention heads.
        alphas: Damping values, one per equilibrium layer.
        max_iter: Solver iterations per layer.
        solver: Fixed-point solver name.
        dropout: Dropout probability in the convolutional stem.

    Inputs:
        x: Tensor with shape `(batch, in_channels, image_size, image_size)`.

    Output:
        Tensor | SILVANetworkOutput: Logit tensor, or `SILVANetworkOutput` when state/results are requested.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int | Sequence[int] = 64,
        num_classes: int = 10,
        *,
        image_size: int = 32,
        attention_mode: VisionAttentionMode = "simple",
        graph_mode: VisionGraphMode = "GAT",
        k_neighbors: int = 4,
        num_heads: int = 4,
        alphas: Sequence[float] = (0.5, 0.2),
        max_iter: int = 20,
        solver: str = "picard",
        backward_mode: BackwardMode = "unrolled",
        backward_solver: BackwardSolverName = "gmres",
        backward_max_iter: int = 50,
        backward_tol: float = 1e-6,
        solver_configs: SolverConfig | Sequence[SolverConfig] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(alphas) < 1:
            raise ValueError("at least one equilibrium layer is required")
        hidden_dims = _normalize_hidden_dims(hidden_dim, len(alphas))
        self.stem = SILVAConvStem(
            in_channels, hidden_dims[0], image_size=image_size, dropout=dropout
        )
        self.core = SILVAVisionVectorClassifier(
            hidden_dims[0],
            hidden_dims,
            num_classes,
            attention_mode=attention_mode,
            graph_mode=graph_mode,
            k_neighbors=k_neighbors,
            num_heads=num_heads,
            alphas=alphas,
            max_iter=max_iter,
            solver=solver,
            backward_mode=backward_mode,
            backward_solver=backward_solver,
            backward_max_iter=backward_max_iter,
            backward_tol=backward_tol,
            solver_configs=solver_configs,
        )

    def forward(self, x: Tensor, return_state: bool = False, return_results: bool = False):
        features = self.stem(x)
        return self.core(features, return_state=return_state, return_results=return_results)


class SILVAImageCortexClassifier(nn.Module):
    """Article-style retina plus linked SILVA cortex equilibrium points.

    This preset keeps the cortex hierarchy explicit: a convolutional front end
    maps an image to a vector stimulus, then each cortex point solves its own
    fixed point with its own damping value, solver, local branch, global branch,
    and optional internal transition network.

    The default two-point hierarchy uses alphas `(0.5, 0.2)`, matching the
    fast/slow structure used in the SILVA article code. Passing longer `alphas`
    and `hidden_dim` sequences creates deeper cortex hierarchies.

    Reference: Jose Luis Silva, "SILVA Networks as Structured Implicit Layers
    and Vector Attractors via Dynamic Interaction Fields", arXiv:2607.28989.

    Args:
        in_channels: Number of image input channels.
        hidden_dim: Recurrent state width, or one width per cortex point.
        num_classes: Number of class logits.
        image_size: Square image size.
        attention_mode: Global branch: `none`, `static`, `simple`, or
            `multi-head`.
        graph_mode: Local branch: `none`, `GAT`, `GNN`, or `knn`.
        k_neighbors: Hidden-channel neighbors for the dynamic local branch.
        num_heads: Attention heads for multi-head channel attention.
        alphas: One damping value per cortex point.
        max_iter: Solver iterations per cortex point.
        solver: Fixed-point solver name.
        backward_mode: `unrolled` or `implicit`.
        internal_depth: Number of linear/tanh blocks inside each cortex point.
        self_interaction: If true, add a learned self-interaction branch.
        dropout: Dropout probability in the convolutional stem.
        head_hidden_dims: Hidden widths for the classifier head.

    Inputs:
        x: Tensor with shape `(batch, in_channels, image_size, image_size)`.

    Output:
        Tensor | SILVACortexOutput: Logits, or structured cortex states and
        solver results when requested.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int | Sequence[int] = 64,
        num_classes: int = 10,
        *,
        image_size: int = 32,
        attention_mode: VisionAttentionMode = "simple",
        graph_mode: VisionGraphMode = "GAT",
        k_neighbors: int = 4,
        num_heads: int = 4,
        alphas: Sequence[float] = (0.5, 0.2),
        max_iter: int = 20,
        solver: str = "picard",
        backward_mode: BackwardMode = "unrolled",
        backward_solver: BackwardSolverName = "gmres",
        backward_max_iter: int = 50,
        backward_tol: float = 1e-6,
        solver_configs: SolverConfig | Sequence[SolverConfig] | None = None,
        internal_depth: int = 1,
        self_interaction: bool = False,
        dropout: float = 0.3,
        head_hidden_dims: Sequence[int] = (),
    ):
        super().__init__()
        if len(alphas) < 1:
            raise ValueError("at least one cortex point is required")
        hidden_dims = _normalize_hidden_dims(hidden_dim, len(alphas))
        configs = _preset_solver_configs(
            solver_configs,
            len(alphas),
            lambda index: SolverConfig(
                solver=solver,  # type: ignore[arg-type]
                max_iter=max_iter,
                alpha=alphas[index],
                backward_mode=backward_mode,
                backward_solver=backward_solver,
                backward_max_iter=backward_max_iter,
                backward_tol=backward_tol,
            ),
        )
        self.retina = SILVAConvStem(
            in_channels, hidden_dims[0], image_size=image_size, dropout=dropout
        )

        layers: list[SILVACortexLayer] = []
        previous = hidden_dims[0]
        for index, state_dim in enumerate(hidden_dims):
            encoder: nn.Module = (
                nn.Identity() if previous == state_dim else nn.Linear(previous, state_dim)
            )
            layers.append(
                SILVACortexLayer(
                    input_encoder=encoder,
                    state_dim=state_dim,
                    state_network=_make_cortex_state_network(state_dim, internal_depth),
                    self_terms=nn.Linear(state_dim, state_dim, bias=False)
                    if self_interaction
                    else None,
                    local_terms=_make_vision_local(graph_mode, state_dim, k_neighbors),
                    global_terms=_make_vision_global(attention_mode, state_dim, num_heads),
                    config=configs[index],
                    output_activation=nn.Identity(),
                    normalize=False,
                )
            )
            previous = state_dim
        self.cortex = SILVACortexNetwork(
            layers,
            links="tanh",
            head=build_mlp_head(hidden_dims[-1], num_classes, head_hidden_dims, dropout=0.0),
        )
        self.hidden_dims = tuple(hidden_dims)
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.attention_mode = attention_mode
        self.graph_mode = graph_mode

    def forward(self, x: Tensor, return_state: bool = False, return_results: bool = False):
        features = self.retina(x)
        return self.cortex(features, return_state=return_state, return_results=return_results)


class SILVAMolecularLayer(nn.Module):
    """Bond-aware graph SILVA layer for molecular node states.

    The local branch is edge-aware graph attention; the global branch is graph
    mean pooling followed by a learned broadcast, matching the ZINC-style
    molecular SILVA configuration.

    Args:
        hidden_dim: Atom-state dimension.
        num_heads: Number of local graph-attention heads.
        dropout: Dropout probability after the update block.
        spectral_norm: If true, apply spectral normalization to stimulus and
            global projections.
        config: Fixed-point solver configuration.

    Inputs:
        x_input: Tensor with shape `(atoms, hidden_dim)`.
        edge_index: Tensor with shape `(2, bonds)`.
        edge_attr: Tensor with shape `(bonds, hidden_dim)`.
        batch: Molecule id tensor with shape `(atoms,)`.

    Output:
        Tensor: Tensor with shape `(atoms, hidden_dim)` or `SolverResult` when
        `return_result=True`.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        dropout_mode: EquilibriumDropoutMode = "independent",
        spectral_norm: bool = True,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if dropout_mode not in {"independent", "variational", "disabled"}:
            raise ValueError(f"Unknown dropout_mode: {dropout_mode}")
        self.local = GraphAttentionLocal(
            hidden_dim,
            heads=num_heads,
            edge_dim=hidden_dim,
            concat=False,
            add_self_loops=False,
        )
        linear = nn.Linear(hidden_dim, hidden_dim)
        stim = nn.Linear(hidden_dim, hidden_dim)
        self.global_attr = nn.utils.spectral_norm(linear) if spectral_norm else linear
        self.W_stim = nn.utils.spectral_norm(stim) if spectral_norm else stim
        self.norm = nn.LayerNorm(hidden_dim)
        if dropout_mode == "variational":
            self.dropout = SILVAVariationalDropout(dropout)
        elif dropout_mode == "independent":
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.config = config or SolverConfig(alpha=0.5, max_iter=20)
        self.hidden_dim = hidden_dim
        self.dropout_mode = dropout_mode
        self.dropout_probability = float(dropout)
        self.spectral_norm = spectral_norm

    def f(
        self,
        z: Tensor,
        x_input: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        return_energy: bool = False,
    ):
        freeze_spectral_state = self.config.backward_mode != "unrolled"
        stimulus = _stable_module_call(self.W_stim, x_input, freeze_spectral_state)
        local = self.local(z, edge_index=edge_index, edge_attr=edge_attr)
        context = pool_entities(z, batch=batch, mode="mean")
        global_update = _stable_module_call(
            self.global_attr,
            context,
            freeze_spectral_state,
        )[batch]
        z_next = self.norm(F.relu(stimulus + local + global_update))
        z_next = self.dropout(z_next)
        if return_energy:
            energy = quadratic_interaction_energy(z, local + global_update)
            return z_next, energy
        return z_next

    def forward(
        self,
        x_input: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        if (
            self.training
            and self.dropout_probability > 0
            and self.dropout_mode == "independent"
            and self.config.backward_mode != "unrolled"
        ):
            raise RuntimeError(
                "independent equilibrium dropout is incompatible with implicit/phantom gradients; "
                "use dropout_mode='variational' or 'disabled'"
            )
        if isinstance(self.dropout, SILVAVariationalDropout):
            self.dropout.reset_mask()
        z_init = torch.zeros_like(x_input) if z0 is None else z0

        def transition(z: Tensor) -> Tensor:
            return self.f(z, x_input, edge_index, edge_attr, batch)

        result = solve_equilibrium(
            transition,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x_input, edge_attr),
        )
        return result if return_result else result.z


class SILVAMolecularRegressor(nn.Module):
    """Reference ZINC-style SILVA molecular regressor.

    Inputs can be passed either as a PyG-like object with ``x``, ``edge_index``,
    ``edge_attr``, and ``batch`` attributes, or directly as keyword tensors.

    Args:
        hidden_dim: Atom and bond embedding dimension, or one dimension per
            molecular equilibrium layer.
        num_atom_types: Number of categorical atom ids.
        num_bond_types: Number of categorical bond ids.
        atom_feature_dim: Input width for continuous atom features. When this is
            `None`, continuous atom features must already have width
            `hidden_dim[0]`.
        bond_feature_dim: Input width for continuous bond features. When this is
            `None`, continuous bond features must already have width
            `hidden_dim[0]`.
        num_heads: Number of local graph-attention heads.
        alphas: Damping values, one per molecular equilibrium layer.
        max_iter: Solver iterations per layer.
        solver: Fixed-point solver name.
        dropout: Dropout probability inside equilibrium layers.
        spectral_norm: If true, constrain stimulus/global projections with
            spectral normalization.
        out_dim: Number of graph-level regression targets.

    Inputs:
        x: Atom ids with shape `(atoms,)` or atom features `(atoms, hidden_dim)`.
        edge_index: Bond index tensor with shape `(2, bonds)`.
        edge_attr: Bond ids with shape `(bonds,)` or bond features
            `(bonds, hidden_dim)`.
        batch: Molecule id tensor with shape `(atoms,)`.

    Output:
        Tensor | SILVANetworkOutput: Graph-level prediction tensor, or `SILVANetworkOutput` when requested.
    """

    def __init__(
        self,
        hidden_dim: int | Sequence[int] = 128,
        *,
        num_atom_types: int = 21,
        num_bond_types: int = 4,
        atom_feature_dim: int | None = None,
        bond_feature_dim: int | None = None,
        num_heads: int = 4,
        alphas: Sequence[float] = (0.5, 0.2),
        max_iter: int = 20,
        solver: str = "picard",
        backward_mode: BackwardMode = "unrolled",
        backward_solver: BackwardSolverName = "gmres",
        backward_max_iter: int = 50,
        backward_tol: float = 1e-6,
        dropout: float = 0.1,
        dropout_mode: EquilibriumDropoutMode = "independent",
        spectral_norm: bool = True,
        solver_configs: SolverConfig | Sequence[SolverConfig] | None = None,
        out_dim: int = 1,
    ):
        super().__init__()
        if len(alphas) < 1:
            raise ValueError("at least one equilibrium layer is required")
        hidden_dims = _normalize_hidden_dims(hidden_dim, len(alphas))
        configs = _preset_solver_configs(
            solver_configs,
            len(alphas),
            lambda index: SolverConfig(
                solver=solver,  # type: ignore[arg-type]
                max_iter=max_iter,
                alpha=alphas[index],
                backward_mode=backward_mode,
                backward_solver=backward_solver,
                backward_max_iter=backward_max_iter,
                backward_tol=backward_tol,
            ),
        )
        self.atom_encoder = nn.Embedding(num_atom_types, hidden_dims[0])
        self.bond_encoder = nn.Embedding(num_bond_types, hidden_dims[0])
        self.atom_projector = (
            nn.Linear(atom_feature_dim, hidden_dims[0]) if atom_feature_dim is not None else None
        )
        self.bond_projector = (
            nn.Linear(bond_feature_dim, hidden_dims[0]) if bond_feature_dim is not None else None
        )
        layers: list[SILVAMolecularLayer] = []
        for index, alpha in enumerate(alphas):
            layer_hidden_dim = hidden_dims[index]
            layers.append(
                SILVAMolecularLayer(
                    layer_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    dropout_mode=dropout_mode,
                    spectral_norm=spectral_norm,
                    config=configs[index],
                )
            )
        self.layers = nn.ModuleList(layers)
        self.connectors = nn.ModuleList(
            [
                nn.Linear(hidden_dims[index], hidden_dims[index + 1])
                for index in range(len(hidden_dims) - 1)
            ]
        )
        self.edge_projectors = nn.ModuleList(
            [
                nn.Identity()
                if layer_hidden_dim == hidden_dims[0]
                else nn.Linear(hidden_dims[0], layer_hidden_dim)
                for layer_hidden_dim in hidden_dims
            ]
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], out_dim),
        )
        self.hidden_dims = tuple(hidden_dims)

    def forward(
        self,
        data=None,
        *,
        x: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        return_state: bool = False,
        return_results: bool = False,
    ):
        x, edge_index, edge_attr, batch = _unpack_graph_data(data, x, edge_index, edge_attr, batch)
        x_embed = self._encode_atoms(x)
        edge_embed = self._encode_bonds(edge_attr)
        state = x_embed
        results: list[SolverResult] = []
        for index, layer in enumerate(self.layers):
            layer_input = state if index == 0 else torch.tanh(state)
            if index > 0:
                layer_input = self.connectors[index - 1](layer_input)
            layer_edge_attr = self.edge_projectors[index](edge_embed)
            result = layer(
                layer_input,
                edge_index,
                layer_edge_attr,
                batch,
                return_result=True,
            )
            state = result.z
            results.append(result)
        graph_state = pool_entities(state, batch=batch, mode="mean")
        output = self.regressor(graph_state).squeeze(-1)
        if return_state or return_results:
            return SILVANetworkOutput(
                output=output,
                state=state,
                solver_results=results if return_results else None,
            )
        return output

    def _encode_atoms(self, x: Tensor) -> Tensor:
        if x.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
            encoded = self.atom_encoder(x.long())
            if encoded.dim() == 3:
                return encoded.sum(dim=1)
            return encoded
        features = x.float()
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        if self.atom_projector is not None:
            return self.atom_projector(features)
        if features.shape[-1] != self.hidden_dims[0]:
            raise ValueError(
                "continuous atom features must have width hidden_dim[0], "
                "or pass atom_feature_dim to project them"
            )
        return features

    def _encode_bonds(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}:
            encoded = self.bond_encoder(edge_attr.long())
            if encoded.dim() == 3:
                return encoded.sum(dim=1)
            return encoded
        features = edge_attr.float()
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        if self.bond_projector is not None:
            return self.bond_projector(features)
        if features.shape[-1] != self.hidden_dims[0]:
            raise ValueError(
                "continuous bond features must have width hidden_dim[0], "
                "or pass bond_feature_dim to project them"
            )
        return features


def silva_graph_preset(
    *,
    hidden_dim: int = 64,
    attention_mode: AttentionMode = "simple",
    graph_mode: GraphMode = "GAT",
    num_heads: int = 4,
    k_neighbors: int = 16,
    local_depth: int = 1,
    layer1_alpha: float = 0.5,
    layer2_alpha: float = 0.2,
    max_iter: int = 15,
    solver: str = "picard",
    backward_mode: BackwardMode = "unrolled",
    backward_solver: BackwardSolverName = "gmres",
    backward_max_iter: int = 50,
    backward_tol: float = 1e-6,
) -> dict[str, object]:
    """Return a serializable graph preset matching public defaults."""

    return {
        "hidden_dim": hidden_dim,
        "attention_mode": attention_mode,
        "graph_mode": graph_mode,
        "num_heads": num_heads,
        "k_neighbors": k_neighbors,
        "local_depth": local_depth,
        "layer1_alpha": layer1_alpha,
        "layer2_alpha": layer2_alpha,
        "max_iter": max_iter,
        "solver": solver,
        "backward_mode": backward_mode,
        "backward_solver": backward_solver,
        "backward_max_iter": backward_max_iter,
        "backward_tol": backward_tol,
    }


def quadratic_interaction_energy(z: Tensor, interaction: Tensor, reduction: str = "mean") -> Tensor:
    """Quadratic Lyapunov-style alignment energy.

    For each entity/sample row this computes

    ``E_i = ||z_i||^2 - <z_i, interaction_i>``.

    It is a diagnostic proxy, not a proof of global Lyapunov stability for an
    arbitrary nonlinear network.
    """

    values = (z * z).sum(dim=-1) - (z * interaction).sum(dim=-1)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def _differentiable_tensors(*values) -> tuple[Tensor, ...]:
    tensors: list[Tensor] = []

    def collect(value) -> None:
        if isinstance(value, torch.Tensor):
            if value.requires_grad and value.is_floating_point():
                tensors.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return tuple(tensors)


def _make_graph_local(graph_mode: GraphMode, hidden_dim: int, num_heads: int) -> nn.Module:
    if graph_mode in {"GAT", "gat", "graph_attention"}:
        return GraphAttentionLocal(hidden_dim, heads=num_heads, concat=True)
    if graph_mode in {"mean", "graph"}:
        from .layers import GraphLocal

        return GraphLocal(hidden_dim)
    if graph_mode == "none":
        return ZeroTerm()
    raise ValueError(f"Unknown graph_mode: {graph_mode}")


def _make_graph_global(
    attention_mode: AttentionMode,
    hidden_dim: int,
    k_neighbors: int,
) -> nn.Module:
    if attention_mode == "simple":
        return GatedMeanFieldGlobal(hidden_dim)
    if attention_mode == "static":
        return StaticMeanFieldGlobal(hidden_dim)
    if attention_mode == "topk":
        return TopKGlobalAttention(hidden_dim, k=k_neighbors)
    if attention_mode == "mean":
        return MeanFieldGlobal(hidden_dim)
    if attention_mode == "none":
        return ZeroTerm()
    raise ValueError(f"Unknown attention_mode: {attention_mode}")


def _make_vision_local(graph_mode: VisionGraphMode, hidden_dim: int, k_neighbors: int) -> nn.Module:
    if graph_mode in {"GAT", "gat", "knn", "GNN"}:
        return DynamicChannelLocal(hidden_dim, k=k_neighbors)
    if graph_mode == "none":
        return ZeroTerm()
    raise ValueError(f"Unknown vision graph_mode: {graph_mode}")


def _make_vision_global(
    attention_mode: VisionAttentionMode,
    hidden_dim: int,
    num_heads: int,
) -> nn.Module:
    if attention_mode == "simple":
        from .layers import ChannelSelfAttentionGlobal

        return ChannelSelfAttentionGlobal(hidden_dim)
    if attention_mode == "multi-head":
        return MultiHeadChannelAttentionGlobal(hidden_dim, heads=num_heads)
    if attention_mode == "static":
        return StaticChannelGlobal(hidden_dim)
    if attention_mode == "none":
        return ZeroTerm()
    raise ValueError(f"Unknown vision attention_mode: {attention_mode}")


def _make_cortex_state_network(dim: int, depth: int) -> nn.Module | None:
    if depth < 0:
        raise ValueError("internal_depth must be nonnegative")
    if depth == 0:
        return None
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)


def _stable_module_call(module: nn.Module, x: Tensor, freeze_updates: bool) -> Tensor:
    """Call spectral-normalized modules without mutating power-iteration state."""

    if not freeze_updates or not hasattr(module, "weight_orig"):
        return module(x)
    was_training = module.training
    module.training = False
    try:
        return module(x)
    finally:
        module.training = was_training


def _unpack_graph_data(
    data,
    x: Tensor | None,
    edge_index: Tensor | None,
    edge_attr: Tensor | None,
    batch: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if data is not None:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
    if x is None or edge_index is None or edge_attr is None:
        raise ValueError("x, edge_index, and edge_attr are required")
    if batch is None:
        batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
    return x, edge_index, edge_attr, batch


def _normalize_hidden_dims(hidden_dim: int | Sequence[int], num_layers: int) -> list[int]:
    if num_layers < 1:
        raise ValueError("at least one equilibrium layer is required")
    if isinstance(hidden_dim, int):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        return [hidden_dim for _ in range(num_layers)]
    hidden_dims = list(hidden_dim)
    if len(hidden_dims) != num_layers:
        raise ValueError("hidden_dim sequence length must match the number of alphas/layers")
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError("all hidden dimensions must be positive")
    return hidden_dims


def _preset_solver_configs(
    configs: SolverConfig | Sequence[SolverConfig] | None,
    count: int,
    default: Callable[[int], SolverConfig],
) -> list[SolverConfig]:
    if configs is None:
        return [default(index) for index in range(count)]
    if isinstance(configs, SolverConfig):
        return [configs for _ in range(count)]
    normalized = list(configs)
    if len(normalized) != count:
        raise ValueError("solver_configs must have one entry per equilibrium layer")
    if not all(isinstance(config, SolverConfig) for config in normalized):
        raise TypeError("solver_configs entries must be SolverConfig instances")
    return normalized
