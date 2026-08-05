from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .layers import (
    SILVAImageLayer,
    SILVALayer,
)
from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
Pooling = Literal["mean", "sum", "max"]
Task = Literal["node", "graph"]
CortexLink = Literal["identity", "tanh", "relu"]
CortexInitializer = Literal["zeros", "stimulus"]
TermFactory = Callable[[int], nn.Module] | Callable[[int, int], nn.Module]
SingleTermSpec = nn.Module | str | TermFactory
TermSpec = SingleTermSpec | Sequence[SingleTermSpec] | None
KwargsSpec = Mapping[str, object] | Sequence[Mapping[str, object] | None] | None
CortexModuleSpec = nn.Module | Sequence[nn.Module] | None


@dataclass
class SILVANetworkOutput:
    """Optional structured output for models that expose equilibrium states."""

    output: Tensor
    state: Tensor
    solver_results: list[SolverResult] | None = None


@dataclass
class SILVACortexOutput:
    """Structured output for linked cortex-style equilibrium points.

    Attributes:
        output: Final tensor after the optional readout head.
        state: Final equilibrium state.
        states: Equilibrium state produced by each cortex point.
        solver_results: Solver metadata for each cortex point.
    """

    output: Tensor
    state: Tensor
    states: list[Tensor]
    solver_results: list[SolverResult]


def build_mlp_head(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int] = (),
    dropout: float = 0.0,
    activation: Callable[[], nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a small readout head for node, graph, or image representations."""

    layers: list[nn.Module] = []
    previous = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(previous, hidden))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous = hidden
    layers.append(nn.Linear(previous, out_dim))
    return nn.Sequential(*layers)


def pool_entities(z: Tensor, batch: Tensor | None = None, mode: Pooling = "mean") -> Tensor:
    """Pool entity states into graph-level or set-level states."""

    if z.dim() != 2:
        raise ValueError("pool_entities expects z with shape (entities, dim)")
    if mode not in {"mean", "sum", "max"}:
        raise ValueError(f"Unknown pooling mode: {mode}")
    if batch is None:
        if mode == "mean":
            return z.mean(dim=0, keepdim=True)
        if mode == "sum":
            return z.sum(dim=0, keepdim=True)
        return z.max(dim=0, keepdim=True).values

    pooled: list[Tensor] = []
    for graph_id in torch.unique(batch, sorted=True):
        values = z[batch == graph_id]
        if mode == "mean":
            pooled.append(values.mean(dim=0))
        elif mode == "sum":
            pooled.append(values.sum(dim=0))
        else:
            pooled.append(values.max(dim=0).values)
    return torch.stack(pooled, dim=0)


class SILVAStack(nn.Module):
    """Stack multiple trainable SILVA equilibrium layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: int | Sequence[int],
        num_layers: int | None = None,
        config: SolverConfig | Sequence[SolverConfig] | None = None,
        local: TermSpec = "graph",
        global_term: TermSpec = "mean",
        self_term: TermSpec = None,
        normalize: bool = True,
        local_kwargs: KwargsSpec = None,
        global_kwargs: KwargsSpec = None,
        self_kwargs: KwargsSpec = None,
        layers: Sequence[nn.Module] | None = None,
    ):
        super().__init__()
        dims = _normalize_hidden_dims(hidden_dims, num_layers)
        configs = _normalize_configs(config, len(dims))
        local_kwargs_by_layer = _normalize_kwargs(local_kwargs, len(dims), "local_kwargs")
        global_kwargs_by_layer = _normalize_kwargs(global_kwargs, len(dims), "global_kwargs")
        self_kwargs_by_layer = _normalize_kwargs(self_kwargs, len(dims), "self_kwargs")

        if layers is not None and len(layers) != len(dims):
            raise ValueError("layers must match the requested number of hidden dimensions")
        built_layers: list[nn.Module] = []
        previous = in_dim
        for index, hidden in enumerate(dims):
            if layers is not None:
                built_layers.append(layers[index])
            else:
                local_spec = _resolve_term_spec(local, hidden, index)
                global_spec = _resolve_term_spec(global_term, hidden, index)
                self_spec = _resolve_term_spec(self_term, hidden, index)
                built_layers.append(
                    SILVALayer(
                        in_dim=previous,
                        hidden_dim=hidden,
                        local=local_spec,  # type: ignore[arg-type]
                        global_term=global_spec,  # type: ignore[arg-type]
                        self_term=self_spec,  # type: ignore[arg-type]
                        config=configs[index],
                        normalize=normalize,
                        local_kwargs=local_kwargs_by_layer[index]
                        if isinstance(local_spec, str)
                        else None,
                        global_kwargs=global_kwargs_by_layer[index]
                        if isinstance(global_spec, str)
                        else None,
                        self_kwargs=self_kwargs_by_layer[index]
                        if isinstance(self_spec, str)
                        else None,
                    )
                )
            previous = hidden
        self.layers = nn.ModuleList(built_layers)
        self.in_dim = in_dim
        self.hidden_dims = tuple(dims)
        self.out_dim = dims[-1]

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        return_results: bool = False,
    ):
        state = x
        results: list[SolverResult] = []
        for layer in self.layers:
            result = layer(
                state,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                return_result=True,
            )
            state = result.z
            results.append(result)
        return (state, results) if return_results else state


class SILVACortexLayer(nn.Module):
    r"""Flexible SILVA equilibrium point with arbitrary internal modules.

    A cortex layer first encodes the incoming object into a stimulus tensor,

    $$
    u = R_\phi(x),
    $$

    then solves one equilibrium point

    $$
    z^\star
    =
    \Psi\!\left[
      u
      + B_\theta(a(z^\star), u, x)
      + \sum_m I_{m,\theta}(a(z^\star), u, x, E, b)
    \right].
    $$

    The `state_network` term \(B_\theta\) may be a deep `nn.Sequential` or a
    list of modules. The interaction terms may be local, global, self, or any
    user-defined PyTorch modules. This covers the SILVA cortex hierarchy:
    a convolutional or linear front end, a fast first equilibrium point, a
    slower second equilibrium point, and different internal transition
    architectures at each point.

    Reference: Jose Luis Silva, "SILVA Networks as Structured Implicit Layers
    and Vector Attractors via Dynamic Interaction Fields", arXiv:2607.28989.

    Args:
        input_dim: Input width for the default linear encoder.
        state_dim: State width. Required when `input_encoder` is omitted or
            when `normalize=True`.
        input_encoder: Module mapping the incoming tensor to the recurrent
            state shape. If omitted, `nn.Linear(input_dim, state_dim)` is used.
        state_network: Module or sequence applied to the activated state inside
            each solver step.
        self_terms: Modules added as self-interaction branches.
        local_terms: Modules added as local interaction branches.
        global_terms: Modules added as global interaction branches.
        interaction_terms: Additional state-shaped interaction branches.
        output_network: Optional module applied after summing the stimulus and
            interactions and before the outer activation.
        normalizer: Optional normalization module. If omitted and
            `normalize=True`, `LayerNorm(state_dim)` is used.
        config: Fixed-point solver configuration.
        activation: State activation \(a\) applied before interactions.
        output_activation: Outer nonlinearity \(\Psi\).
        initializer: `zeros` starts from `zeros_like(u)`; `stimulus` starts
            from `u`.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        state_dim: int | None = None,
        *,
        input_encoder: nn.Module | None = None,
        state_network: CortexModuleSpec = None,
        self_terms: CortexModuleSpec = None,
        local_terms: CortexModuleSpec = None,
        global_terms: CortexModuleSpec = None,
        interaction_terms: CortexModuleSpec = None,
        output_network: nn.Module | None = None,
        normalizer: nn.Module | None = None,
        config: SolverConfig | None = None,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        output_activation: Callable[[Tensor], Tensor] = torch.tanh,
        normalize: bool = True,
        initializer: CortexInitializer = "zeros",
    ):
        super().__init__()
        if input_encoder is None:
            if input_dim is None or state_dim is None:
                raise ValueError(
                    "input_dim and state_dim are required when input_encoder is omitted"
                )
            input_encoder = nn.Linear(input_dim, state_dim)
        if normalize and normalizer is None and state_dim is None:
            raise ValueError(
                "state_dim is required when normalize=True and no normalizer is supplied"
            )
        if initializer not in {"zeros", "stimulus"}:
            raise ValueError("initializer must be 'zeros' or 'stimulus'")

        self.input_encoder = input_encoder
        self.state_network = _normalize_cortex_modules(state_network)
        self.self_terms = _normalize_cortex_modules(self_terms)
        self.local_terms = _normalize_cortex_modules(local_terms)
        self.global_terms = _normalize_cortex_modules(global_terms)
        self.interaction_terms = _normalize_cortex_modules(interaction_terms)
        self.output_network = output_network or nn.Identity()
        self.normalizer = (
            normalizer
            if normalizer is not None
            else (nn.LayerNorm(state_dim) if normalize else nn.Identity())
        )
        self.config = config or SolverConfig(alpha=0.5, max_iter=20)
        self.activation = activation
        self.output_activation = output_activation
        self.initializer = initializer
        self.input_dim = input_dim
        self.state_dim = state_dim

    def encode(self, x: Tensor) -> Tensor:
        """Encode the incoming tensor into the equilibrium stimulus shape."""

        return self.input_encoder(x)

    def initial_state(self, stimulus: Tensor, z0: Tensor | None = None) -> Tensor:
        """Return the initial solver state."""

        if z0 is not None:
            return z0
        if self.initializer == "stimulus":
            return stimulus
        return torch.zeros_like(stimulus)

    def f(
        self,
        z: Tensor,
        stimulus: Tensor,
        x: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the undamped cortex transition."""

        signal = self.activation(z)
        total = stimulus
        if len(self.state_network) > 0:
            state_field = _run_cortex_sequence(
                self.state_network,
                signal,
                stimulus=stimulus,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            total = _add_cortex_field(
                total,
                state_field,
                z,
                source="state_network",
            )
        for module in (
            *self.self_terms,
            *self.local_terms,
            *self.global_terms,
            *self.interaction_terms,
        ):
            field = _call_cortex_module(
                module,
                signal,
                stimulus=stimulus,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            total = _add_cortex_field(
                total,
                field,
                z,
                source=module.__class__.__name__,
            )
        total = _call_cortex_module(
            self.output_network,
            total,
            stimulus=stimulus,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        output = self.normalizer(self.output_activation(total))
        if output.shape != z.shape:
            raise ValueError(
                "cortex transition must preserve the equilibrium-state shape: "
                f"expected {tuple(z.shape)}, received {tuple(output.shape)}"
            )
        return output

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ):
        stimulus = self.encode(x)
        z_init = self.initial_state(stimulus, z0=z0)

        def transition(z: Tensor) -> Tensor:
            return self.f(
                z,
                stimulus,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )

        result = solve_equilibrium(
            transition,
            z_init,
            self.config,
            params=tuple(self.parameters()),
            tensors=_differentiable_tensors(x, edge_attr),
        )
        return result if return_result else result.z


class SILVACortexNetwork(nn.Module):
    """Link several `SILVACortexLayer` equilibrium points in one PyTorch model.

    Each layer may have its own encoder, internal transition network,
    interaction terms, and solver configuration. The link between equilibrium
    points is configurable; the SILVA fast/slow hierarchy uses
    `links="tanh"` with different `SolverConfig.alpha` values per layer.
    """

    def __init__(
        self,
        layers: Sequence[SILVACortexLayer],
        *,
        links: CortexLink | nn.Module | Sequence[CortexLink | nn.Module] = "tanh",
        head: nn.Module | None = None,
    ):
        super().__init__()
        if len(layers) < 1:
            raise ValueError("SILVACortexNetwork needs at least one cortex layer")
        self.layers = nn.ModuleList(layers)
        self.links = _normalize_cortex_links(links, len(layers) - 1)
        self.head = head or nn.Identity()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        return_state: bool = False,
        return_results: bool = False,
    ):
        state = x
        states: list[Tensor] = []
        results: list[SolverResult] = []
        for index, layer in enumerate(self.layers):
            result = layer(
                state,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                return_result=True,
            )
            state = result.z
            states.append(state)
            results.append(result)
            if index < len(self.layers) - 1:
                state = _apply_cortex_link(self.links[index], state)
        output = self.head(state)
        if return_state or return_results:
            return SILVACortexOutput(
                output=output,
                state=state,
                states=states,
                solver_results=results,
            )
        return output


def silva_cortex_layer(
    input_dim: int | None = None,
    state_dim: int | None = None,
    **kwargs,
) -> SILVACortexLayer:
    """Create a flexible cortex-style SILVA equilibrium point."""

    return SILVACortexLayer(input_dim=input_dim, state_dim=state_dim, **kwargs)


def silva_cortex_network(
    layers: Sequence[SILVACortexLayer],
    **kwargs,
) -> SILVACortexNetwork:
    """Create a linked hierarchy of cortex-style equilibrium points."""

    return SILVACortexNetwork(layers, **kwargs)


class SILVAGraphNetwork(nn.Module):
    """End-to-end graph or node model built from a SILVA stack and readout head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: int | Sequence[int],
        out_dim: int,
        num_layers: int | None = None,
        task: Task = "node",
        pooling: Pooling = "mean",
        config: SolverConfig | Sequence[SolverConfig] | None = None,
        local: TermSpec = "graph",
        global_term: TermSpec = "mean",
        self_term: TermSpec = None,
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        normalize: bool = True,
        local_kwargs: KwargsSpec = None,
        global_kwargs: KwargsSpec = None,
        self_kwargs: KwargsSpec = None,
        encoder: nn.Module | None = None,
        head: nn.Module | None = None,
    ):
        super().__init__()
        if task not in {"node", "graph"}:
            raise ValueError("task must be either 'node' or 'graph'")
        self.encoder = encoder or SILVAStack(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            config=config,
            local=local,
            global_term=global_term,
            self_term=self_term,
            normalize=normalize,
            local_kwargs=local_kwargs,
            global_kwargs=global_kwargs,
            self_kwargs=self_kwargs,
        )
        encoder_out_dim = getattr(self.encoder, "out_dim", None)
        if head is None and encoder_out_dim is None:
            raise ValueError("a custom encoder without out_dim requires a custom head")
        self.head = head or build_mlp_head(
            encoder_out_dim, out_dim, head_hidden_dims, dropout
        )
        self.task = task
        self.pooling = pooling

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        return_state: bool = False,
        return_results: bool = False,
    ):
        if return_results:
            state, results = self.encoder(
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                return_results=True,
            )
        else:
            state = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            results = None

        features = (
            state if self.task == "node" else pool_entities(state, batch=batch, mode=self.pooling)
        )
        output = self.head(features)
        if return_state or return_results:
            return SILVANetworkOutput(output=output, state=state, solver_results=results)
        return output


class SILVAImageClassifier(nn.Module):
    """Image classifier using one or more SILVA image equilibrium layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | Sequence[int],
        num_classes: int,
        num_layers: int | None = None,
        config: SolverConfig | Sequence[SolverConfig] | None = None,
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        layers: Sequence[nn.Module] | None = None,
        head: nn.Module | None = None,
    ):
        super().__init__()
        channels = _normalize_hidden_dims(hidden_channels, num_layers)
        configs = _normalize_configs(config, len(channels))
        if layers is not None and len(layers) != len(channels):
            raise ValueError("layers must match the requested number of hidden channels")
        built_layers: list[nn.Module] = []
        previous = in_channels
        for index, hidden in enumerate(channels):
            built_layers.append(
                layers[index]
                if layers is not None
                else SILVAImageLayer(previous, hidden, config=configs[index])
            )
            previous = hidden
        self.layers = nn.ModuleList(built_layers)
        self.head = head or build_mlp_head(
            channels[-1], num_classes, head_hidden_dims, dropout
        )
        self.out_channels = channels[-1]

    def forward(self, x: Tensor, return_state: bool = False, return_results: bool = False):
        state = x
        results: list[SolverResult] = []
        for layer in self.layers:
            result = layer(state, return_result=True)
            state = result.z
            results.append(result)
        features = F.adaptive_avg_pool2d(state, output_size=1).flatten(1)
        output = self.head(features)
        if return_state or return_results:
            return SILVANetworkOutput(
                output=output,
                state=state,
                solver_results=results if return_results else None,
            )
        return output


def _normalize_cortex_modules(modules: CortexModuleSpec) -> nn.ModuleList:
    if modules is None:
        return nn.ModuleList()
    if isinstance(modules, nn.Module):
        return nn.ModuleList([modules])
    return nn.ModuleList(list(modules))


def _run_cortex_sequence(
    modules: nn.ModuleList,
    z: Tensor,
    **kwargs,
) -> Tensor:
    state = z
    for module in modules:
        state = _call_cortex_module(module, state, **kwargs)
    return state


def _call_cortex_module(module: nn.Module, z: Tensor, **kwargs) -> Tensor:
    signature = inspect.signature(module.forward)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return module(z, **{key: value for key, value in kwargs.items() if value is not None})
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return module(z, **accepted)


def _add_cortex_field(
    total: Tensor,
    field: Tensor,
    state: Tensor,
    *,
    source: str,
) -> Tensor:
    if not isinstance(field, Tensor):
        raise TypeError(f"{source} must return a torch.Tensor, received {type(field).__name__}")
    try:
        combined = total + field
    except RuntimeError as exc:
        raise ValueError(
            f"{source} returned shape {tuple(field.shape)}, which cannot be added to "
            f"equilibrium state shape {tuple(state.shape)}"
        ) from exc
    if combined.shape != state.shape:
        raise ValueError(
            f"{source} changed the equilibrium-state shape from {tuple(state.shape)} "
            f"to {tuple(combined.shape)}"
        )
    return combined


def _normalize_cortex_links(
    links: CortexLink | nn.Module | Sequence[CortexLink | nn.Module],
    count: int,
) -> list[CortexLink | nn.Module]:
    if count == 0:
        return []
    if isinstance(links, (nn.Module, str)):
        return [links for _ in range(count)]
    normalized = list(links)
    if len(normalized) != count:
        raise ValueError("links must have one entry between each pair of cortex layers")
    return normalized


def _apply_cortex_link(link: CortexLink | nn.Module, state: Tensor) -> Tensor:
    if isinstance(link, nn.Module):
        return link(state)
    if link == "identity":
        return state
    if link == "tanh":
        return torch.tanh(state)
    if link == "relu":
        return F.relu(state)
    raise ValueError(f"Unknown cortex link: {link}")


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
        if isinstance(value, tuple | list):
            for item in value:
                collect(item)

    for value in values:
        collect(value)
    return tuple(tensors)


def _normalize_hidden_dims(hidden_dims: int | Sequence[int], num_layers: int | None) -> list[int]:
    if isinstance(hidden_dims, int):
        count = 1 if num_layers is None else num_layers
        if count < 1:
            raise ValueError("num_layers must be positive")
        return [hidden_dims for _ in range(count)]
    dims = list(hidden_dims)
    if not dims:
        raise ValueError("hidden_dims must contain at least one dimension")
    if num_layers is not None and num_layers != len(dims):
        raise ValueError("num_layers must match len(hidden_dims) when hidden_dims is a sequence")
    return dims


def _normalize_configs(
    config: SolverConfig | Sequence[SolverConfig] | None,
    num_layers: int,
) -> list[SolverConfig | None]:
    if config is None or isinstance(config, SolverConfig):
        return [config for _ in range(num_layers)]
    configs = list(config)
    if len(configs) != num_layers:
        raise ValueError("A config sequence must have one SolverConfig per layer")
    return configs


def _normalize_kwargs(
    kwargs: KwargsSpec,
    num_layers: int,
    name: str,
) -> list[dict | None]:
    if kwargs is None:
        return [None for _ in range(num_layers)]
    if isinstance(kwargs, Mapping):
        return [dict(kwargs) for _ in range(num_layers)]
    items = list(kwargs)
    if len(items) != num_layers:
        raise ValueError(f"{name} must have one entry per layer")
    normalized: list[dict | None] = []
    for item in items:
        if item is None:
            normalized.append(None)
        elif isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            raise TypeError(f"{name} entries must be mappings or None")
    return normalized


def _resolve_term_spec(spec: TermSpec, dim: int, index: int) -> nn.Module | str | None:
    if isinstance(spec, nn.Module) or spec is None or isinstance(spec, str):
        return spec
    if _is_term_sequence(spec):
        return _resolve_term_spec(spec[index], dim, index)
    if callable(spec):
        return _call_term_factory(spec, dim, index)
    raise TypeError(f"Unsupported SILVA term specification: {type(spec)!r}")


def _is_term_sequence(spec: TermSpec) -> bool:
    return isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray))


def _call_term_factory(factory: TermFactory, dim: int, index: int) -> nn.Module:
    signature = inspect.signature(factory)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return factory(dim, index)  # type: ignore[misc]
    if len(positional) >= 2:
        return factory(dim, index)  # type: ignore[misc]
    return factory(dim)  # type: ignore[misc]
