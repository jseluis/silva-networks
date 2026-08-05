"""Monotone graph and injected-transformer equilibria inside SILVA.

The modules in this file preserve the source/state/operator contracts of the
published mechanisms while exposing ordinary PyTorch modules and SILVA solver
diagnostics. Compact defaults are intended for teaching and small experiments;
benchmark-scale width, depth, data, and training protocols remain user choices.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor
AttentionMode = Literal["auto", "sdpa", "chunked", "manual"]


def _positive_integer(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _validate_edge_index(edge_index: Tensor, nodes: int, device: torch.device) -> None:
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edge_index.device != device:
        raise ValueError("edge_index and node state must be on the same device")
    if edge_index.numel() and (
        int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= nodes
    ):
        raise ValueError("edge_index contains a node outside the graph")


def normalized_laplacian_field(
    state: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> Tensor:
    r"""Apply one half of the symmetric normalized graph Laplacian.

    For an adjacency matrix ``A`` and degree matrix ``D``, the returned field is

    $$
    GZ=\frac12\left(I-D^{-1/2}AD^{-1/2}\right)Z.
    $$

    Bidirectional edges should be supplied when an undirected graph is wanted.
    Isolated nodes receive the identity contribution ``Z / 2``.
    """

    if state.dim() != 2 or not state.is_floating_point():
        raise ValueError("state must be a floating tensor with shape (nodes, channels)")
    nodes = state.shape[0]
    _validate_edge_index(edge_index, nodes, state.device)
    edges = edge_index.shape[1]
    weights = (
        torch.ones(edges, device=state.device, dtype=state.dtype)
        if edge_weight is None
        else edge_weight
    )
    if weights.shape not in {(edges,), (edges, 1)}:
        raise ValueError("edge_weight must have shape (edges,) or (edges, 1)")
    if weights.device != state.device or weights.dtype != state.dtype:
        raise ValueError("edge_weight must match the state device and dtype")
    if torch.any(weights < 0):
        raise ValueError("edge_weight must be nonnegative")

    source, destination = edge_index
    flat_weights = weights.reshape(edges)
    degree = torch.zeros(nodes, device=state.device, dtype=state.dtype)
    degree.index_add_(0, destination, flat_weights)
    inverse_sqrt = torch.where(
        degree > 0,
        degree.clamp_min(torch.finfo(state.dtype).tiny).rsqrt(),
        torch.zeros_like(degree),
    )
    normalized = flat_weights * inverse_sqrt[source] * inverse_sqrt[destination]
    neighbors = torch.zeros_like(state)
    neighbors.index_add_(0, destination, normalized[:, None] * state[source])
    return 0.5 * (state - neighbors)


class SILVAMonotoneGraphTransition(nn.Module):
    r"""Forward-backward monotone graph transition.

    The graph channel matrix is parameterized as

    $$
    W=(1-m)I-C C^T+F-F^T,\qquad m>0,
    $$

    and one forward-backward step is

    $$
    Z^+=\operatorname{prox}_{\alpha f}
    \left((1-\alpha)Z+\alpha(WGZ+B(X))\right).
    $$

    ``activation`` supplies the proximal map; ReLU is the default.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        margin: float = 0.1,
        step_size: float = 0.8,
        operator_rank: int | None = None,
        activation: Callable[[Tensor], Tensor] = F.relu,
    ):
        super().__init__()
        _positive_integer(in_dim, "in_dim")
        _positive_integer(state_dim, "state_dim")
        if not 0.0 < margin < 1.0:
            raise ValueError("margin must satisfy 0 < margin < 1")
        if not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must satisfy 0 < step_size <= 1")
        if operator_rank is not None:
            _positive_integer(operator_rank, "operator_rank")
            if operator_rank > state_dim:
                raise ValueError("operator_rank cannot exceed state_dim")
        rank = state_dim if operator_rank is None else operator_rank
        self.source = nn.Linear(in_dim, state_dim)
        self.c_factor = nn.Parameter(0.05 * torch.randn(state_dim, rank))
        if operator_rank is None:
            self.skew_factor = nn.Parameter(0.05 * torch.randn(state_dim, state_dim))
            self.register_parameter("skew_left", None)
            self.register_parameter("skew_right", None)
        else:
            self.register_parameter("skew_factor", None)
            self.skew_left = nn.Parameter(0.05 * torch.randn(state_dim, rank))
            self.skew_right = nn.Parameter(0.05 * torch.randn(state_dim, rank))
        self.margin = float(margin)
        self.step_size = float(step_size)
        self.operator_rank = operator_rank
        self.activation = activation
        self.in_dim = in_dim
        self.state_dim = state_dim

    def channel_weight(self) -> Tensor:
        """Return the constrained channel matrix ``W``."""

        identity = torch.eye(
            self.state_dim,
            device=self.c_factor.device,
            dtype=self.c_factor.dtype,
        )
        symmetric = self.c_factor @ self.c_factor.transpose(0, 1)
        if self.skew_factor is not None:
            skew = self.skew_factor - self.skew_factor.transpose(0, 1)
        else:
            assert self.skew_left is not None and self.skew_right is not None
            skew = self.skew_left @ self.skew_right.transpose(0, 1)
            skew = skew - self.skew_right @ self.skew_left.transpose(0, 1)
        return (1.0 - self.margin) * identity - symmetric + skew

    def apply_channel_weight(self, values: Tensor) -> Tensor:
        """Apply ``values @ W.T`` without materializing ``W`` when factorized."""

        if values.shape[-1] != self.state_dim:
            raise ValueError(f"values must have final dimension {self.state_dim}")
        symmetric = (values @ self.c_factor) @ self.c_factor.transpose(0, 1)
        if self.skew_factor is not None:
            skew = values @ self.skew_factor.transpose(0, 1)
            skew = skew - values @ self.skew_factor
        else:
            assert self.skew_left is not None and self.skew_right is not None
            skew = (values @ self.skew_right) @ self.skew_left.transpose(0, 1)
            skew = skew - (values @ self.skew_left) @ self.skew_right.transpose(0, 1)
        return (1.0 - self.margin) * values - symmetric + skew

    def monotonicity_lower_bound(self) -> Tensor:
        """Return the analytic lower bound supplied by the positive margin."""

        return self.c_factor.new_tensor(self.margin)

    def monotonicity_certificate(self) -> Tensor:
        r"""Return the smallest eigenvalue of ``I-(W+W^T)/2``."""

        weight = self.channel_weight()
        symmetric_operator = torch.eye(
            self.state_dim,
            device=weight.device,
            dtype=weight.dtype,
        ) - 0.5 * (weight + weight.transpose(0, 1))
        return torch.linalg.eigvalsh(symmetric_operator).min()

    def forward(
        self,
        state: Tensor,
        inputs: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if state.dim() != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape (nodes, {self.state_dim})")
        if inputs.dim() != 2 or inputs.shape != (state.shape[0], self.in_dim):
            raise ValueError(f"inputs must have shape (nodes, {self.in_dim})")
        if inputs.device != state.device or inputs.dtype != state.dtype:
            raise ValueError("inputs must match the state device and dtype")
        graph_state = normalized_laplacian_field(state, edge_index, edge_weight)
        field = self.apply_channel_weight(graph_state)
        proposal = (1.0 - self.step_size) * state
        proposal = proposal + self.step_size * (field + self.source(inputs))
        return self.activation(proposal)


@dataclass
class SILVAMonotoneGraphOutput:
    """Task output, equilibrium state, solver trace, and monotonicity certificate."""

    output: Tensor
    state: Tensor
    solver_result: SolverResult
    monotonicity_certificate: Tensor


class SILVAMonotoneGraphEquilibrium(nn.Module):
    """Monotone implicit graph network represented as a SILVA equilibrium."""

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        out_dim: int,
        *,
        margin: float = 0.1,
        step_size: float = 0.8,
        operator_rank: int | None = None,
        transition: SILVAMonotoneGraphTransition | None = None,
        readout: nn.Module | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        _positive_integer(out_dim, "out_dim")
        self.transition = transition or SILVAMonotoneGraphTransition(
            in_dim,
            state_dim,
            margin=margin,
            step_size=step_size,
            operator_rank=operator_rank,
        )
        if self.transition.in_dim != in_dim or self.transition.state_dim != state_dim:
            raise ValueError("transition dimensions must match in_dim and state_dim")
        self.readout = readout or nn.Linear(state_dim, out_dim)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-5,
            alpha=1.0,
            anderson_batch_dims=0,
        )
        self.in_dim = in_dim
        self.state_dim = state_dim
        self.out_dim = out_dim

    def forward(
        self,
        inputs: Tensor,
        edge_index: Tensor,
        *,
        edge_weight: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAMonotoneGraphOutput:
        if inputs.dim() != 2 or inputs.shape[-1] != self.in_dim:
            raise ValueError(f"inputs must have shape (nodes, {self.in_dim})")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        initial = inputs.new_zeros(inputs.shape[0], self.state_dim) if z0 is None else z0
        if initial.shape != (inputs.shape[0], self.state_dim):
            raise ValueError("z0 must have shape (nodes, state_dim)")
        if initial.device != inputs.device or initial.dtype != inputs.dtype:
            raise ValueError("z0 must match the input device and dtype")

        def fixed_map(state: Tensor) -> Tensor:
            return self.transition(state, inputs, edge_index, edge_weight)

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.transition.parameters()),
            tensors=(inputs,),
        )
        output = self.readout(result.z)
        expected = (inputs.shape[0], self.out_dim)
        if output.shape != expected:
            raise ValueError(f"readout must return shape {expected}")
        if return_result:
            return SILVAMonotoneGraphOutput(
                output,
                result.z,
                result,
                self.transition.monotonicity_certificate(),
            )
        return output


class SILVAInjectedSelfAttention(nn.Module):
    r"""Multi-head self-attention with a precomputed QKV source injection.

    For equilibrium state ``Z`` and one-time source injection ``U``, the three
    attention projections are

    $$
    (Q,K,V)=Z W_{qkv}+U.
    $$
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        *,
        attention_mode: AttentionMode = "auto",
        query_chunk_size: int | None = None,
    ):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads != 0:
            raise ValueError("dim must be positive and divisible by heads")
        if attention_mode not in {"auto", "sdpa", "chunked", "manual"}:
            raise ValueError("attention_mode must be auto, sdpa, chunked, or manual")
        if query_chunk_size is not None:
            _positive_integer(query_chunk_size, "query_chunk_size")
        if attention_mode == "chunked" and query_chunk_size is None:
            query_chunk_size = 256
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_mode = attention_mode
        self.query_chunk_size = query_chunk_size

    def forward(
        self,
        state: Tensor,
        qkv_injection: Tensor,
        class_injection: Tensor | None = None,
    ) -> Tensor:
        if state.dim() != 3 or state.shape[-1] != self.dim:
            raise ValueError(f"state must have shape (batch, tokens, {self.dim})")
        if qkv_injection.shape != (*state.shape[:-1], 3 * self.dim):
            raise ValueError("qkv_injection must have shape (batch, tokens, 3 * dim)")
        qkv = self.qkv(state) + qkv_injection
        if class_injection is not None:
            if class_injection.shape != (state.shape[0], 3 * self.dim):
                raise ValueError("class_injection must have shape (batch, 3 * dim)")
            qkv = qkv + class_injection[:, None, :]
        batch, tokens, _ = state.shape
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        mode = "sdpa" if self.attention_mode == "auto" else self.attention_mode
        if mode == "manual":
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        elif mode == "sdpa":
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            chunk_size = self.query_chunk_size or tokens
            attended = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[:, :, start : start + chunk_size],
                        k,
                        v,
                        dropout_p=0.0,
                    )
                    for start in range(0, tokens, chunk_size)
                ],
                dim=2,
            )
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.dim)
        return self.output(attended)


class SILVAEquilibriumTransformerBlock(nn.Module):
    """Injected attention and feed-forward branches reused by a fixed-point solve."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        expansion: int = 4,
        state_scale: float = 0.2,
        attention_mode: AttentionMode = "auto",
        query_chunk_size: int | None = None,
    ):
        super().__init__()
        _positive_integer(expansion, "expansion")
        if not 0.0 < state_scale <= 1.0:
            raise ValueError("state_scale must satisfy 0 < state_scale <= 1")
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = SILVAInjectedSelfAttention(
            dim,
            heads,
            attention_mode=attention_mode,
            query_chunk_size=query_chunk_size,
        )
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Linear(expansion * dim, dim),
        )
        self.state_scale = float(state_scale)
        self.dim = dim

    def forward(
        self,
        state: Tensor,
        qkv_injection: Tensor,
        class_injection: Tensor | None = None,
    ) -> Tensor:
        attended = state + self.attention(
            self.attention_norm(state),
            qkv_injection,
            class_injection,
        )
        proposal = attended + self.feed_forward(self.feed_forward_norm(attended))
        return torch.tanh(self.state_scale * proposal)


@dataclass
class SILVAGenerativeEquilibriumOutput:
    """Decoded image, token equilibrium, one-time injection, and solver trace."""

    output: Tensor
    state: Tensor
    injection: Tensor
    solver_result: SolverResult


def _sincos_2d_position(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for 2D sine/cosine positions")
    quarter = dim // 4
    frequency = torch.arange(quarter, device=device, dtype=dtype)
    frequency = torch.exp(-math.log(10000.0) * frequency / max(quarter - 1, 1))
    y = torch.arange(height, device=device, dtype=dtype)[:, None] * frequency[None]
    x = torch.arange(width, device=device, dtype=dtype)[:, None] * frequency[None]
    y_features = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
    x_features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    positions = torch.cat(
        [
            y_features[:, None, :].expand(height, width, -1),
            x_features[None, :, :].expand(height, width, -1),
        ],
        dim=-1,
    )
    return positions.reshape(1, height * width, dim)


class SILVAGenerativeEquilibriumTransformer(nn.Module):
    r"""One-time image injection followed by a weight-tied token equilibrium.

    Image patches are embedded and processed once by injection blocks. Their
    projections supply a distinct QKV injection to each block inside the tied
    equilibrium transition. The compact ``tanh`` stability envelope makes the
    teaching configuration numerically inspectable without changing the
    one-time-injection contract.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        *,
        patch_size: int = 2,
        hidden_dim: int = 32,
        heads: int = 4,
        injection_depth: int = 1,
        equilibrium_depth: int = 2,
        expansion: int = 4,
        state_scale: float = 0.2,
        classes: int | None = None,
        attention_mode: AttentionMode = "auto",
        query_chunk_size: int | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        for value, name in (
            (in_channels, "in_channels"),
            (out_channels, "out_channels"),
            (patch_size, "patch_size"),
            (hidden_dim, "hidden_dim"),
            (injection_depth, "injection_depth"),
            (equilibrium_depth, "equilibrium_depth"),
        ):
            _positive_integer(value, name)
        if classes is not None:
            _positive_integer(classes, "classes")
        if hidden_dim % 4 != 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by 4 and by heads")
        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.injection_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    heads,
                    dim_feedforward=expansion * hidden_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(injection_depth)
            ]
        )
        self.injection_projection = nn.Linear(
            hidden_dim,
            equilibrium_depth * 3 * hidden_dim,
        )
        self.equilibrium_blocks = nn.ModuleList(
            [
                SILVAEquilibriumTransformerBlock(
                    hidden_dim,
                    heads=heads,
                    expansion=expansion,
                    state_scale=state_scale,
                    attention_mode=attention_mode,
                    query_chunk_size=query_chunk_size,
                )
                for _ in range(equilibrium_depth)
            ]
        )
        self.class_embedding = (
            nn.Embedding(classes, equilibrium_depth * 3 * hidden_dim)
            if classes is not None
            else None
        )
        self.decoder = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=30,
            tol=1e-5,
            alpha=1.0,
            anderson_batch_dims=1,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.equilibrium_depth = equilibrium_depth

    def _tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        patches = self.patch_embed(images)
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        positions = _sincos_2d_position(
            height,
            width,
            self.hidden_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + positions, height, width

    def _decode(self, state: Tensor, height: int, width: int) -> Tensor:
        batch = state.shape[0]
        patch = self.patch_size
        values = self.decoder(state)
        values = values.view(batch, height, width, self.out_channels, patch, patch)
        return values.permute(0, 3, 1, 4, 2, 5).reshape(
            batch,
            self.out_channels,
            height * patch,
            width * patch,
        )

    def forward(
        self,
        images: Tensor,
        *,
        labels: Tensor | None = None,
        z0: Tensor | None = None,
        return_result: bool = False,
    ) -> Tensor | SILVAGenerativeEquilibriumOutput:
        if images.dim() != 4 or images.shape[1] != self.in_channels:
            raise ValueError(f"images must have shape (batch, {self.in_channels}, height, width)")
        if not images.is_floating_point():
            raise TypeError("images must have a floating-point dtype")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        tokens, height, width = self._tokens(images)
        for block in self.injection_blocks:
            tokens = block(tokens)
        injection = self.injection_projection(tokens)
        chunks = injection.chunk(self.equilibrium_depth, dim=-1)
        class_chunks: tuple[Tensor, ...] | tuple[None, ...]
        if self.class_embedding is None:
            if labels is not None:
                raise ValueError("labels require classes to be configured")
            class_chunks = (None,) * self.equilibrium_depth
        else:
            if labels is None or labels.shape != (images.shape[0],):
                raise ValueError("labels must have shape (batch,) for a class-conditioned model")
            class_chunks = self.class_embedding(labels).chunk(self.equilibrium_depth, dim=-1)

        initial = torch.zeros_like(tokens) if z0 is None else z0
        if initial.shape != tokens.shape:
            raise ValueError("z0 must match the patch-token shape")
        if initial.device != tokens.device or initial.dtype != tokens.dtype:
            raise ValueError("z0 must match the image device and dtype")

        def fixed_map(state: Tensor) -> Tensor:
            value = state
            for block, qkv_source, class_source in zip(
                self.equilibrium_blocks,
                chunks,
                class_chunks,
                strict=True,
            ):
                value = block(value, qkv_source, class_source)
            return value

        result = solve_equilibrium(
            fixed_map,
            initial,
            self.config,
            params=tuple(self.equilibrium_blocks.parameters()),
            tensors=(injection,),
        )
        output = self._decode(result.z, height, width)
        if return_result:
            return SILVAGenerativeEquilibriumOutput(output, result.z, injection, result)
        return output


def silva_distillation_loss(prediction: Tensor, teacher_target: Tensor) -> Tensor:
    """Return the mean-squared one-step teacher-matching objective."""

    if prediction.shape != teacher_target.shape:
        raise ValueError("prediction and teacher_target must have the same shape")
    return F.mse_loss(prediction, teacher_target)


def silva_monotone_graph_equilibrium(**kwargs) -> SILVAMonotoneGraphEquilibrium:
    """Create a monotone graph equilibrium inside SILVA."""

    return SILVAMonotoneGraphEquilibrium(**kwargs)


def silva_generative_equilibrium_transformer(
    **kwargs,
) -> SILVAGenerativeEquilibriumTransformer:
    """Create an injected equilibrium transformer inside SILVA."""

    return SILVAGenerativeEquilibriumTransformer(**kwargs)


__all__ = [
    "AttentionMode",
    "SILVAEquilibriumTransformerBlock",
    "SILVAGenerativeEquilibriumOutput",
    "SILVAGenerativeEquilibriumTransformer",
    "SILVAInjectedSelfAttention",
    "SILVAMonotoneGraphEquilibrium",
    "SILVAMonotoneGraphOutput",
    "SILVAMonotoneGraphTransition",
    "normalized_laplacian_field",
    "silva_distillation_loss",
    "silva_generative_equilibrium_transformer",
    "silva_monotone_graph_equilibrium",
]
