"""SILVA optical-flow modules based on equilibrium flow refinement.

This module combines all-pairs correlation, recurrent flow refinement, and
fixed-point solution of a flow field through the SILVA transition interface.

References:
    - Silva, "SILVA Networks as Structured Implicit Layers and Vector
      Attractors via Dynamic Interaction Fields", 2026.
    - Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical
      Flow", ECCV 2020.
    - Bai, Geng, Savani, and Kolter, "Deep Equilibrium Optical Flow
      Estimation", CVPR 2022.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .solvers import SolverConfig, SolverResult, solve_equilibrium

Tensor = torch.Tensor


@dataclass
class SILVAFlowBatch:
    """Synthetic or loaded optical-flow batch.

    Attributes:
        image1: First image tensor with shape `(batch, channels, height, width)`.
        image2: Second image tensor with the same shape.
        flow: Ground-truth forward flow with shape `(batch, 2, height, width)`.
        valid: Optional validity mask with shape `(batch, 1, height, width)`.
    """

    image1: Tensor
    image2: Tensor
    flow: Tensor
    valid: Tensor | None = None

    def to(self, device: str | torch.device) -> SILVAFlowBatch:
        """Move all tensors to a PyTorch device."""

        return SILVAFlowBatch(
            image1=self.image1.to(device),
            image2=self.image2.to(device),
            flow=self.flow.to(device),
            valid=None if self.valid is None else self.valid.to(device),
        )


@dataclass
class SILVAFlowResult:
    """Output from `SILVAOpticalFlowDEQ`.

    Attributes:
        flow: Estimated flow with shape `(batch, 2, height, width)`.
        solver_result: Fixed-point solver diagnostics.
        correlation: Optional all-pairs correlation volume.
    """

    flow: Tensor
    solver_result: SolverResult
    correlation: Tensor | None = None
    hidden: Tensor | None = None
    low_resolution_flow: Tensor | None = None
    flow_sequence: list[Tensor] | None = None
    cached_state: SILVARAFTState | None = None


@dataclass
class SILVARAFTState:
    """Reusable DEQ-Flow hidden state and low-resolution flow."""

    hidden: Tensor
    flow: Tensor


def silva_coords_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a pixel coordinate grid with channels `(x, y)`.

    Returns:
        Tensor with shape `(batch, 2, height, width)`.
    """

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=0).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1).contiguous()


def silva_flow_warp(
    tensor: Tensor,
    flow: Tensor,
    *,
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: bool = True,
) -> Tensor:
    r"""Warp `tensor` by a pixel-space flow field.

    The output at pixel \(p=(x,y)\) samples the input at \(p+u(p)\), where
    \(u=(u_x,u_y)\) is stored in `flow[:, 0]` and `flow[:, 1]`.

    Args:
        tensor: Image or feature tensor with shape `(batch, channels, height, width)`.
        flow: Flow tensor with shape `(batch, 2, height, width)`.
        padding_mode: Passed to `torch.nn.functional.grid_sample`.
        align_corners: Passed to `grid_sample`.

    Returns:
        Warped tensor with the same shape as `tensor`.
    """

    if tensor.dim() != 4:
        raise ValueError("tensor must have shape (batch, channels, height, width)")
    if flow.shape[:2] != (tensor.shape[0], 2) or flow.shape[-2:] != tensor.shape[-2:]:
        raise ValueError("flow must have shape (batch, 2, height, width)")
    batch_size, _, height, width = tensor.shape
    coords = silva_coords_grid(
        batch_size,
        height,
        width,
        device=tensor.device,
        dtype=tensor.dtype,
    ) + flow.to(dtype=tensor.dtype)
    if width > 1:
        x = 2.0 * coords[:, 0] / (width - 1) - 1.0
    else:
        x = torch.zeros_like(coords[:, 0])
    if height > 1:
        y = 2.0 * coords[:, 1] / (height - 1) - 1.0
    else:
        y = torch.zeros_like(coords[:, 1])
    grid = torch.stack([x, y], dim=-1)
    return F.grid_sample(
        tensor, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners
    )


def silva_all_pairs_correlation(
    fmap1: Tensor,
    fmap2: Tensor,
    *,
    normalize_features: bool = False,
) -> Tensor:
    r"""Compute a RAFT-style all-pairs correlation volume.

    For feature maps \(F_1,F_2\in\mathbb R^{B\times C\times H\times W}\), the
    correlation is

    $$
    C_{b,i,j,k,\ell}
    =
    \frac{\langle F_{1,b,:,i,j},F_{2,b,:,k,\ell}\rangle}{\sqrt C}.
    $$

    Args:
        fmap1: First feature map with shape `(batch, channels, height, width)`.
        fmap2: Second feature map with the same shape.
        normalize_features: If true, L2-normalize channel vectors before the
            dot product.

    Returns:
        Tensor with shape `(batch, height, width, height, width)`.
    """

    if fmap1.shape != fmap2.shape or fmap1.dim() != 4:
        raise ValueError("fmap1 and fmap2 must have the same 4D shape")
    if normalize_features:
        fmap1 = F.normalize(fmap1, dim=1)
        fmap2 = F.normalize(fmap2, dim=1)
    batch_size, channels, height, width = fmap1.shape
    flat1 = fmap1.flatten(2)
    flat2 = fmap2.flatten(2)
    corr = torch.einsum("bcn,bcm->bnm", flat1, flat2) / (channels**0.5)
    return corr.reshape(batch_size, height, width, height, width)


def silva_local_correlation_lookup(correlation: Tensor, flow: Tensor, radius: int = 2) -> Tensor:
    """Sample local correlation neighborhoods around current flow estimates.

    Args:
        correlation: All-pairs correlation with shape `(batch, height, width, height, width)`.
        flow: Flow tensor with shape `(batch, 2, height, width)`.
        radius: Lookup radius in pixels.

    Returns:
        Tensor with shape `(batch, (2*radius+1)^2, height, width)`.
    """

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if correlation.dim() != 5:
        raise ValueError("correlation must have shape (batch, height, width, height, width)")
    batch_size, height, width, height2, width2 = correlation.shape
    if (height, width) != (height2, width2):
        raise ValueError("only same-resolution correlation volumes are supported")
    if flow.shape != (batch_size, 2, height, width):
        raise ValueError("flow must have shape (batch, 2, height, width)")

    base = silva_coords_grid(batch_size, height, width, device=flow.device, dtype=flow.dtype)
    center = base + flow
    corr_maps = correlation.reshape(batch_size * height * width, 1, height, width)
    channels: list[Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            coords = center.clone()
            coords[:, 0] = coords[:, 0] + dx
            coords[:, 1] = coords[:, 1] + dy
            if width > 1:
                x = 2.0 * coords[:, 0] / (width - 1) - 1.0
            else:
                x = torch.zeros_like(coords[:, 0])
            if height > 1:
                y = 2.0 * coords[:, 1] / (height - 1) - 1.0
            else:
                y = torch.zeros_like(coords[:, 1])
            sample_grid = torch.stack([x, y], dim=-1).reshape(batch_size * height * width, 1, 1, 2)
            sampled = F.grid_sample(
                corr_maps,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            channels.append(sampled.reshape(batch_size, height, width))
    return torch.stack(channels, dim=1)


class SILVACorrelationPyramid:
    """RAFT all-pairs correlation pyramid with differentiable local lookup."""

    def __init__(self, fmap1: Tensor, fmap2: Tensor, *, levels: int = 4, radius: int = 4):
        if levels < 1:
            raise ValueError("levels must be positive")
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        correlation = silva_all_pairs_correlation(
            fmap1,
            fmap2,
            normalize_features=False,
        )
        batch, height, width, _, _ = correlation.shape
        current = correlation.reshape(batch * height * width, 1, height, width)
        self.maps = [current]
        for _ in range(1, levels):
            if min(current.shape[-2:]) <= 1:
                self.maps.append(current)
            else:
                current = F.avg_pool2d(current, 2, stride=2)
                self.maps.append(current)
        self.batch = batch
        self.height = height
        self.width = width
        self.radius = radius

    @property
    def channels(self) -> int:
        return len(self.maps) * (2 * self.radius + 1) ** 2

    def lookup(self, flow: Tensor) -> Tensor:
        if flow.shape != (self.batch, 2, self.height, self.width):
            raise ValueError("flow shape does not match the correlation source grid")
        base = silva_coords_grid(
            self.batch,
            self.height,
            self.width,
            device=flow.device,
            dtype=flow.dtype,
        )
        coordinates = (base + flow).permute(0, 2, 3, 1)
        offsets = torch.arange(
            -self.radius,
            self.radius + 1,
            device=flow.device,
            dtype=flow.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        delta = torch.stack([offset_x, offset_y], dim=-1)
        outputs: list[Tensor] = []
        for level, correlation in enumerate(self.maps):
            level_coordinates = coordinates.reshape(-1, 1, 1, 2) / (2**level)
            grid = level_coordinates + delta[None]
            grid = _pixel_to_normalized_grid(grid, correlation.shape[-2], correlation.shape[-1])
            sampled = F.grid_sample(
                correlation,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.view(
                self.batch,
                self.height,
                self.width,
                -1,
            )
            outputs.append(sampled)
        return torch.cat(outputs, dim=-1).permute(0, 3, 1, 2).contiguous()


class SILVARAFTEncoder(nn.Module):
    """Configurable convolutional encoder for RAFT/DEQ-Flow features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: Sequence[int] = (32, 64, 96),
        output_stride: Literal[1, 2, 4, 8] = 8,
        normalization: Literal["group", "instance", "batch", "none"] = "instance",
        dropout: float = 0.0,
        architecture: Literal["raft", "plain"] = "raft",
        residual_blocks: int | Sequence[int] = 2,
        stage_strides: Sequence[int] | None = None,
    ):
        super().__init__()
        if output_stride not in {1, 2, 4, 8}:
            raise ValueError("output_stride must be 1, 2, 4, or 8")
        if not hidden_channels:
            raise ValueError("hidden_channels must not be empty")
        if architecture not in {"raft", "plain"}:
            raise ValueError("architecture must be raft or plain")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if architecture == "raft":
            self.net = self._raft_encoder(
                in_channels,
                out_channels,
                hidden_channels,
                output_stride,
                normalization,
                dropout,
                residual_blocks,
                stage_strides,
            )
        else:
            self.net = self._plain_encoder(
                in_channels,
                out_channels,
                hidden_channels,
                output_stride,
                normalization,
                dropout,
            )
        self.output_stride = output_stride

    @staticmethod
    def _plain_encoder(
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        output_stride: int,
        normalization: Literal["group", "instance", "batch", "none"],
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = in_channels
        remaining_stride = output_stride
        for width in hidden_channels:
            stride = 2 if remaining_stride > 1 else 1
            remaining_stride //= stride
            layers.extend(
                [
                    nn.Conv2d(previous, width, 3, stride=stride, padding=1),
                    _flow_norm(normalization, width),
                    nn.ReLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    _flow_norm(normalization, width),
                    nn.ReLU(),
                ]
            )
            previous = width
        layers.append(nn.Conv2d(previous, out_channels, 1))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def _raft_encoder(
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        output_stride: int,
        normalization: Literal["group", "instance", "batch", "none"],
        dropout: float,
        residual_blocks: int | Sequence[int],
        stage_strides: Sequence[int] | None,
    ) -> nn.Sequential:
        block_counts = (
            [residual_blocks] * len(hidden_channels)
            if isinstance(residual_blocks, int)
            else list(residual_blocks)
        )
        if len(block_counts) != len(hidden_channels) or any(count < 1 for count in block_counts):
            raise ValueError("residual_blocks must provide one positive count per encoder stage")
        stem_stride = 2 if output_stride > 1 else 1
        if stage_strides is None:
            remaining_twos = int(math.log2(output_stride // stem_stride))
            if remaining_twos > len(hidden_channels) - 1:
                raise ValueError("hidden_channels has too few stages for output_stride")
            strides = [1] * len(hidden_channels)
            for index in range(len(hidden_channels) - remaining_twos, len(hidden_channels)):
                strides[index] = 2
        else:
            strides = list(stage_strides)
            if len(strides) != len(hidden_channels) or any(stride not in {1, 2} for stride in strides):
                raise ValueError("stage_strides must contain one 1 or 2 per encoder stage")
            if stem_stride * math.prod(strides) != output_stride:
                raise ValueError("stem and stage_strides must multiply to output_stride")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels[0], 7, stride=stem_stride, padding=3),
            _flow_norm(normalization, hidden_channels[0]),
            nn.ReLU(),
        ]
        previous = hidden_channels[0]
        for width, count, stride in zip(hidden_channels, block_counts, strides, strict=True):
            layers.append(
                SILVARAFTResidualBlock(
                    previous,
                    width,
                    stride=stride,
                    normalization=normalization,
                )
            )
            layers.extend(
                SILVARAFTResidualBlock(width, width, normalization=normalization)
                for _ in range(count - 1)
            )
            previous = width
        layers.append(nn.Conv2d(previous, out_channels, 1))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class SILVARAFTResidualBlock(nn.Module):
    """Two-convolution residual block used by RAFT feature encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        normalization: Literal["group", "instance", "batch", "none"] = "instance",
    ):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            _flow_norm(normalization, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _flow_norm(normalization, out_channels),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                _flow_norm(normalization, out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.main(x) + self.downsample(x))


class SILVASeparatedConvGRU(nn.Module):
    """Horizontal/vertical convolutional GRU used by RAFT update blocks."""

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        combined = hidden_dim + input_dim
        self.horizontal = nn.ModuleList(
            [
                nn.Conv2d(combined, hidden_dim, (1, kernel_size), padding=(0, padding))
                for _ in range(3)
            ]
        )
        self.vertical = nn.ModuleList(
            [
                nn.Conv2d(combined, hidden_dim, (kernel_size, 1), padding=(padding, 0))
                for _ in range(3)
            ]
        )

    def forward(self, hidden: Tensor, inputs: Tensor) -> Tensor:
        hidden = self._step(hidden, inputs, self.horizontal)
        return self._step(hidden, inputs, self.vertical)

    @staticmethod
    def _step(hidden: Tensor, inputs: Tensor, layers: nn.ModuleList) -> Tensor:
        combined = torch.cat([hidden, inputs], dim=1)
        update = torch.sigmoid(layers[0](combined))
        reset = torch.sigmoid(layers[1](combined))
        candidate = torch.tanh(layers[2](torch.cat([reset * hidden, inputs], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class SILVAGlobalMotionAggregator(nn.Module):
    """Optional GMA-style global attention over motion features."""

    def __init__(self, context_dim: int, motion_dim: int, heads: int = 1):
        super().__init__()
        if heads < 1 or motion_dim % heads != 0:
            raise ValueError("motion_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = motion_dim // heads
        self.query = nn.Conv2d(context_dim, motion_dim, 1)
        self.key = nn.Conv2d(context_dim, motion_dim, 1)
        self.value = nn.Conv2d(motion_dim, motion_dim, 1)
        self.output = nn.Conv2d(motion_dim, motion_dim, 1)

    def forward(self, context: Tensor, motion: Tensor) -> Tensor:
        batch, _, height, width = motion.shape
        positions = height * width
        query = self.query(context).view(batch, self.heads, self.head_dim, positions)
        key = self.key(context).view(batch, self.heads, self.head_dim, positions)
        value = self.value(motion).view(batch, self.heads, self.head_dim, positions)
        scores = torch.einsum("bhdi,bhdj->bhij", query, key) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        aggregate = torch.einsum("bhij,bhdj->bhdi", weights, value)
        return self.output(aggregate.reshape(batch, -1, height, width))


class SILVARAFTUpdateBlock(nn.Module):
    """RAFT motion encoder, recurrent hidden update, flow head, and upsampler mask."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        correlation_channels: int,
        *,
        motion_dim: int = 128,
        flow_head_dim: int = 256,
        gru_kernel_size: int = 5,
        global_motion: bool = False,
        global_motion_heads: int = 1,
        upsample_factor: int = 8,
        correlation_hidden_dims: Sequence[int] = (256, 192),
        flow_hidden_dims: Sequence[int] = (128, 64),
        upsampling_mask_scale: float = 0.25,
    ):
        super().__init__()
        if len(correlation_hidden_dims) != 2 or any(width < 1 for width in correlation_hidden_dims):
            raise ValueError("correlation_hidden_dims must contain two positive widths")
        if len(flow_hidden_dims) != 2 or any(width < 1 for width in flow_hidden_dims):
            raise ValueError("flow_hidden_dims must contain two positive widths")
        if motion_dim <= 2:
            raise ValueError("motion_dim must be greater than two")
        if upsampling_mask_scale <= 0:
            raise ValueError("upsampling_mask_scale must be positive")
        corr_first, corr_hidden = correlation_hidden_dims
        flow_first, flow_hidden = flow_hidden_dims
        self.correlation_encoder = nn.Sequential(
            nn.Conv2d(correlation_channels, corr_first, 1),
            nn.ReLU(),
            nn.Conv2d(corr_first, corr_hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, flow_first, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(flow_first, flow_hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.motion_projection = nn.Sequential(
            nn.Conv2d(corr_hidden + flow_hidden, motion_dim - 2, 3, padding=1),
            nn.ReLU(),
        )
        self.global_aggregator = (
            SILVAGlobalMotionAggregator(context_dim, motion_dim, global_motion_heads)
            if global_motion
            else None
        )
        recurrent_input = context_dim + motion_dim * (2 if global_motion else 1)
        self.gru = SILVASeparatedConvGRU(hidden_dim, recurrent_input, gru_kernel_size)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, flow_head_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(flow_head_dim, 2, 3, padding=1),
        )
        self.mask_head = (
            nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, upsample_factor * upsample_factor * 9, 1),
            )
            if upsample_factor > 1
            else None
        )
        self.upsampling_mask_scale = upsampling_mask_scale

    def forward(
        self,
        hidden: Tensor,
        context: Tensor,
        correlation: Tensor,
        flow: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        correlation_features = self.correlation_encoder(correlation)
        flow_features = self.flow_encoder(flow)
        motion = self.motion_projection(torch.cat([correlation_features, flow_features], dim=1))
        motion = torch.cat([motion, flow], dim=1)
        inputs = [context, motion]
        if self.global_aggregator is not None:
            inputs.append(self.global_aggregator(context, motion))
        hidden = self.gru(hidden, torch.cat(inputs, dim=1))
        return (
            hidden,
            self.flow_head(hidden),
            self.upsampling_mask(hidden),
        )

    def upsampling_mask(self, hidden: Tensor) -> Tensor | None:
        """Predict the convex upsampling mask from a recurrent state."""

        return (
            None
            if self.mask_head is None
            else self.upsampling_mask_scale * self.mask_head(hidden)
        )


class SILVARAFTDEQ(nn.Module):
    r"""Coupled hidden-state/flow equilibrium adapted from RAFT and DEQ-Flow.

    The solved SILVA state is ``(h, u)`` with transition

    ``h_next = ConvGRU(h, context, motion(u, Corr(u)))`` and
    ``u_next = u + delta_u(h_next)``.

    Feature/context encoder widths, output stride, correlation levels/radius,
    GRU, GMA-style global aggregation, initialization, solver, gradient mode,
    cached fixed-point reuse, sparse correction states, and upsampling are all
    public parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        context_dim: int = 128,
        encoder_channels: Sequence[int] = (64, 96, 128),
        output_stride: Literal[1, 2, 4, 8] = 8,
        corr_levels: int = 4,
        corr_radius: int = 4,
        motion_dim: int = 128,
        flow_head_dim: int = 256,
        gru_kernel_size: int = 5,
        global_motion: bool = False,
        global_motion_heads: int = 1,
        hidden_initialization: Literal["zeros", "context"] = "zeros",
        input_normalization: Literal["none", "zero_one", "minus_one_one"] = "minus_one_one",
        correction_steps: int = 1,
        correction_tau: float = 1.0,
        encoder_architecture: Literal["raft", "plain"] = "raft",
        encoder_residual_blocks: int | Sequence[int] = 2,
        encoder_stage_strides: Sequence[int] | None = None,
        encoder_dropout: float = 0.0,
        correlation_hidden_dims: Sequence[int] = (256, 192),
        flow_hidden_dims: Sequence[int] = (128, 64),
        upsampling_mask_scale: float = 0.25,
        feature_encoder_module: nn.Module | None = None,
        context_encoder_module: nn.Module | None = None,
        update_block: SILVARAFTUpdateBlock | None = None,
        config: SolverConfig | None = None,
    ):
        super().__init__()
        if correction_steps < 0:
            raise ValueError("correction_steps must be nonnegative")
        if correction_tau <= 0:
            raise ValueError("correction_tau must be positive")
        if hidden_initialization not in {"zeros", "context"}:
            raise ValueError("hidden_initialization must be zeros or context")
        if input_normalization not in {"none", "zero_one", "minus_one_one"}:
            raise ValueError("input_normalization must be none, zero_one, or minus_one_one")
        encoder_kwargs = {
            "hidden_channels": encoder_channels,
            "output_stride": output_stride,
            "architecture": encoder_architecture,
            "residual_blocks": encoder_residual_blocks,
            "stage_strides": encoder_stage_strides,
            "dropout": encoder_dropout,
        }
        self.feature_encoder = feature_encoder_module or SILVARAFTEncoder(
            in_channels,
            feature_dim,
            normalization="instance",
            **encoder_kwargs,
        )
        context_output = context_dim + (hidden_dim if hidden_initialization == "context" else 0)
        self.context_encoder = context_encoder_module or SILVARAFTEncoder(
            in_channels,
            context_output,
            normalization="batch",
            **encoder_kwargs,
        )
        correlation_channels = corr_levels * (2 * corr_radius + 1) ** 2
        self.update = update_block or SILVARAFTUpdateBlock(
            hidden_dim,
            context_dim,
            correlation_channels,
            motion_dim=motion_dim,
            flow_head_dim=flow_head_dim,
            gru_kernel_size=gru_kernel_size,
            global_motion=global_motion,
            global_motion_heads=global_motion_heads,
            upsample_factor=output_stride,
            correlation_hidden_dims=correlation_hidden_dims,
            flow_hidden_dims=flow_hidden_dims,
            upsampling_mask_scale=upsampling_mask_scale,
        )
        self.config = config or SolverConfig(
            solver="anderson",
            max_iter=24,
            tol=1e-4,
            history=5,
            alpha=1.0,
            anderson_batch_dims=0,
            return_best=True,
        )
        if self.config.anderson_batch_dims != 0:
            raise ValueError("coupled RAFT state requires anderson_batch_dims=0")
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.output_stride = output_stride
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_initialization = hidden_initialization
        self.input_normalization = input_normalization
        self.correction_steps = correction_steps
        self.correction_tau = correction_tau

    def forward(
        self,
        image1: Tensor,
        image2: Tensor,
        *,
        flow0: Tensor | None = None,
        cached_state: SILVARAFTState | None = None,
        return_result: bool = False,
        return_correlation: bool = False,
    ):
        if image1.shape != image2.shape or image1.dim() != 4:
            raise ValueError("image1 and image2 must have the same 4D shape")
        if image1.shape[-2] % self.output_stride or image1.shape[-1] % self.output_stride:
            raise ValueError("image height and width must be divisible by output_stride")
        normalized1 = self._normalize_image(image1)
        normalized2 = self._normalize_image(image2)
        fmap1 = self.feature_encoder(normalized1)
        fmap2 = self.feature_encoder(normalized2)
        expected_spatial = (
            image1.shape[-2] // self.output_stride,
            image1.shape[-1] // self.output_stride,
        )
        if fmap1.dim() != 4 or fmap1.shape[0] != image1.shape[0]:
            raise ValueError("feature encoder must return a 4D tensor with matching batch size")
        if fmap1.shape[-2:] != expected_spatial:
            raise ValueError("feature encoder spatial stride does not match output_stride")
        pyramid = SILVACorrelationPyramid(
            fmap1,
            fmap2,
            levels=self.corr_levels,
            radius=self.corr_radius,
        )
        context_encoded = self.context_encoder(normalized1)
        expected_context_channels = self.context_dim + (
            self.hidden_dim if self.hidden_initialization == "context" else 0
        )
        if context_encoded.shape != (
            image1.shape[0],
            expected_context_channels,
            *expected_spatial,
        ):
            raise ValueError(
                "context encoder output must match configured channels and output_stride"
            )
        if self.hidden_initialization == "context":
            hidden_seed, context = torch.split(
                context_encoded,
                [self.hidden_dim, self.context_dim],
                dim=1,
            )
            hidden_seed = torch.tanh(hidden_seed)
        else:
            context = context_encoded
            hidden_seed = torch.zeros(
                image1.shape[0],
                self.hidden_dim,
                *fmap1.shape[-2:],
                device=image1.device,
                dtype=fmap1.dtype,
            )
        context = F.relu(context)
        flow_seed = self._initial_flow(image1, fmap1, flow0)
        if cached_state is not None:
            if (
                cached_state.hidden.shape != hidden_seed.shape
                or cached_state.flow.shape != flow_seed.shape
            ):
                raise ValueError("cached_state shapes do not match the current image pair")
            hidden_seed = cached_state.hidden.to(device=fmap1.device, dtype=fmap1.dtype)
            flow_seed = cached_state.flow.to(device=fmap1.device, dtype=fmap1.dtype)

        hidden_numel = hidden_seed.numel()
        initial = torch.cat([hidden_seed.reshape(-1), flow_seed.reshape(-1)])

        def unpack(state: Tensor) -> tuple[Tensor, Tensor]:
            hidden = state[:hidden_numel].reshape_as(hidden_seed)
            flow = state[hidden_numel:].reshape_as(flow_seed)
            return hidden, flow

        def transition(state: Tensor) -> Tensor:
            hidden, flow = unpack(state)
            local_correlation = pyramid.lookup(flow)
            next_hidden, delta_flow, _ = self.update(hidden, context, local_correlation, flow)
            next_flow = flow + delta_flow
            return torch.cat([next_hidden.reshape(-1), next_flow.reshape(-1)])

        result = solve_equilibrium(
            transition,
            initial,
            self.config,
            params=tuple(self.parameters()),
            tensors=tuple(
                tensor
                for tensor in (image1, image2)
                if tensor.requires_grad and tensor.is_floating_point()
            ),
        )
        hidden, low_flow = unpack(result.z)
        mask = self.update.upsampling_mask(hidden)
        final_flow = self._upsample_flow(low_flow, mask, image1.shape[-2:])
        correction_transition = transition
        if result.states and self.correction_steps > 0:
            correction_fmap1 = self.feature_encoder(normalized1)
            correction_fmap2 = self.feature_encoder(normalized2)
            correction_pyramid = SILVACorrelationPyramid(
                correction_fmap1,
                correction_fmap2,
                levels=self.corr_levels,
                radius=self.corr_radius,
            )
            correction_context = self.context_encoder(normalized1)
            if self.hidden_initialization == "context":
                correction_context = correction_context[:, self.hidden_dim :]
            correction_context = F.relu(correction_context)

            def correction_transition(state: Tensor) -> Tensor:
                correction_hidden, correction_flow = unpack(state)
                correction_corr = correction_pyramid.lookup(correction_flow)
                next_hidden, delta_flow, _ = self.update(
                    correction_hidden,
                    correction_context,
                    correction_corr,
                    correction_flow,
                )
                next_flow = correction_flow + delta_flow
                return torch.cat([next_hidden.reshape(-1), next_flow.reshape(-1)])

        corrections = self._correction_predictions(
            result.states,
            correction_transition,
            unpack,
            image1.shape[-2:],
        )
        corrections.append(final_flow)
        cached = SILVARAFTState(hidden.detach(), low_flow.detach())
        if return_result:
            correlation = (
                silva_all_pairs_correlation(fmap1, fmap2, normalize_features=False)
                if return_correlation
                else None
            )
            return SILVAFlowResult(
                flow=final_flow,
                solver_result=result,
                correlation=correlation,
                hidden=hidden,
                low_resolution_flow=low_flow,
                flow_sequence=corrections,
                cached_state=cached,
            )
        return final_flow

    def _correction_predictions(
        self,
        states: list[Tensor],
        transition: Callable[[Tensor], Tensor],
        unpack: Callable[[Tensor], tuple[Tensor, Tensor]],
        output_size: tuple[int, int],
    ) -> list[Tensor]:
        predictions: list[Tensor] = []
        for stored in states:
            state = stored.detach()
            for _ in range(self.correction_steps):
                updated = transition(state)
                state = (1.0 - self.correction_tau) * state + self.correction_tau * updated
            hidden, flow = unpack(state)
            mask = self.update.upsampling_mask(hidden)
            predictions.append(self._upsample_flow(flow, mask, output_size))
        return predictions

    def _initial_flow(self, image: Tensor, fmap: Tensor, flow0: Tensor | None) -> Tensor:
        if flow0 is None:
            return torch.zeros(
                image.shape[0],
                2,
                *fmap.shape[-2:],
                device=image.device,
                dtype=fmap.dtype,
            )
        if flow0.shape == (image.shape[0], 2, *fmap.shape[-2:]):
            return flow0.to(device=fmap.device, dtype=fmap.dtype)
        if flow0.shape != (image.shape[0], 2, *image.shape[-2:]):
            raise ValueError("flow0 must have full-image or encoder resolution")
        return (
            F.interpolate(
                flow0.to(device=fmap.device, dtype=fmap.dtype),
                size=fmap.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            / self.output_stride
        )

    def _upsample_flow(
        self,
        flow: Tensor,
        mask: Tensor | None,
        output_size: tuple[int, int],
    ) -> Tensor:
        if self.output_stride == 1:
            return flow
        if mask is None:
            return self.output_stride * F.interpolate(
                flow,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )
        batch, _, height, width = flow.shape
        factor = self.output_stride
        weights = mask.view(batch, 1, 9, factor, factor, height, width)
        weights = F.softmax(weights, dim=2)
        neighborhoods = F.unfold(factor * flow, (3, 3), padding=1)
        neighborhoods = neighborhoods.view(batch, 2, 9, 1, 1, height, width)
        upsampled = (weights * neighborhoods).sum(dim=2)
        upsampled = upsampled.permute(0, 1, 4, 2, 5, 3)
        return upsampled.reshape(batch, 2, factor * height, factor * width)

    def _normalize_image(self, image: Tensor) -> Tensor:
        values = image.float()
        if self.input_normalization == "none":
            return values
        if self.input_normalization == "zero_one":
            return values / 255.0 if values.detach().amax() > 1.5 else values
        values = values / 255.0 if values.detach().amax() > 1.5 else values
        return 2.0 * values - 1.0


def silva_flow_fixed_point_correction_loss(
    predictions: Sequence[Tensor],
    target: Tensor,
    *,
    valid: Tensor | None = None,
    gamma: float = 0.8,
) -> Tensor:
    """Exponentially weighted sparse fixed-point correction loss."""

    if not predictions:
        raise ValueError("predictions must not be empty")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must satisfy 0 < gamma <= 1")
    if target.dim() != 4 or target.shape[1] != 2:
        raise ValueError("target must have shape (batch, 2, height, width)")
    loss = target.new_zeros(())
    for index, prediction in enumerate(predictions):
        if prediction.shape != target.shape:
            raise ValueError("every prediction must have the same shape as target")
        weight = gamma ** (len(predictions) - index - 1)
        error = (prediction - target).abs()
        if valid is not None:
            one_channel_shape = (target.shape[0], 1, *target.shape[-2:])
            if valid.shape not in {target.shape, one_channel_shape}:
                raise ValueError("valid must have one or two channels at the target resolution")
            mask = valid.to(device=error.device, dtype=error.dtype)
            if mask.shape[1] == 2:
                mask = mask.amin(dim=1, keepdim=True)
            error = error * mask
            denominator = (mask.sum() * error.shape[1]).clamp_min(1.0)
            term = error.sum() / denominator
        else:
            term = error.mean()
        loss = loss + weight * term
    return loss


def _pixel_to_normalized_grid(grid: Tensor, height: int, width: int) -> Tensor:
    x = torch.zeros_like(grid[..., 0]) if width <= 1 else 2 * grid[..., 0] / (width - 1) - 1
    y = torch.zeros_like(grid[..., 1]) if height <= 1 else 2 * grid[..., 1] / (height - 1) - 1
    return torch.stack([x, y], dim=-1)


def _flow_norm(
    kind: Literal["group", "instance", "batch", "none"],
    channels: int,
) -> nn.Module:
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, channels), channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown flow normalization: {kind}")


class SILVAFlowFeatureEncoder(nn.Module):
    """Small feature encoder for SILVA optical-flow validation experiments."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, feature_dim),
            nn.GELU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)


class SILVAFlowUpdateBlock(nn.Module):
    """RAFT-style recurrent update block for a SILVA flow field.

    The block consumes the current flow, encoded image features, warped second
    features, residual features, and local all-pairs correlation lookups. It
    predicts a bounded flow increment.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, correlation_channels: int = 9):
        super().__init__()
        in_channels = 2 + 3 * feature_dim + correlation_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )

    def forward(
        self,
        flow: Tensor,
        fmap1: Tensor,
        warped_fmap2: Tensor,
        residual: Tensor,
        local_correlation: Tensor,
    ) -> Tensor:
        features = torch.cat([flow, fmap1, warped_fmap2, residual, local_correlation], dim=1)
        return self.net(features)


class SILVAOpticalFlowDEQ(nn.Module):
    r"""DEQ-Flow/RAFT-style optical-flow estimator ported to SILVA.

    The model estimates a flow field \(u^\star\) by solving

    $$
    u^\star
    =
    u^\star
    +
    \gamma\,\tanh\Delta_\theta(u^\star, F_1, F_2, C),
    $$

    where \(F_1,F_2\) are image features, \(C\) is an all-pairs correlation
    volume, and \(\Delta_\theta\) is a convolutional update block. In practice,
    damping from `SolverConfig` and the bounded `tanh` increment control the
    finite solve.

    Args:
        in_channels: Number of image channels.
        feature_dim: Feature encoder width.
        hidden_dim: Update block width.
        corr_radius: Local correlation lookup radius.
        update_scale: Scale applied to bounded flow increments.
        config: Fixed-point solver configuration.
        reengage: Whether to apply one differentiable transition after the
            numerical solve.

    Inputs:
        image1: Tensor with shape `(batch, channels, height, width)`.
        image2: Tensor with the same shape.
        flow0: Optional initial flow with shape `(batch, 2, height, width)`.

    Output:
        Flow tensor, or `SILVAFlowResult` when `return_result=True`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 8,
        hidden_dim: int = 32,
        corr_radius: int = 1,
        update_scale: float = 0.25,
        config: SolverConfig | None = None,
        reengage: bool = True,
    ):
        super().__init__()
        if corr_radius < 0:
            raise ValueError("corr_radius must be nonnegative")
        if update_scale <= 0:
            raise ValueError("update_scale must be positive")
        self.encoder = SILVAFlowFeatureEncoder(in_channels, feature_dim)
        corr_channels = (2 * corr_radius + 1) ** 2
        self.update = SILVAFlowUpdateBlock(feature_dim, hidden_dim, corr_channels)
        self.corr_radius = corr_radius
        self.update_scale = update_scale
        self.config = config or SolverConfig(solver="anderson", max_iter=12, alpha=0.6, history=4)
        self.reengage = reengage

    def transition(self, flow: Tensor, fmap1: Tensor, fmap2: Tensor, correlation: Tensor) -> Tensor:
        """Return one RAFT-style SILVA flow-refinement step."""

        warped = silva_flow_warp(fmap2, flow)
        residual = fmap1 - warped
        local_corr = silva_local_correlation_lookup(correlation, flow, radius=self.corr_radius)
        update = self.update(flow, fmap1, warped, residual, local_corr)
        return flow + self.update_scale * torch.tanh(update)

    def forward(
        self,
        image1: Tensor,
        image2: Tensor,
        *,
        flow0: Tensor | None = None,
        return_result: bool = False,
        return_correlation: bool = False,
    ):
        if image1.shape != image2.shape or image1.dim() != 4:
            raise ValueError("image1 and image2 must have the same 4D shape")
        fmap1 = self.encoder(image1)
        fmap2 = self.encoder(image2)
        correlation = silva_all_pairs_correlation(fmap1, fmap2)
        if flow0 is None:
            flow_init = torch.zeros(
                image1.shape[0],
                2,
                image1.shape[2],
                image1.shape[3],
                device=image1.device,
                dtype=fmap1.dtype,
            )
        else:
            expected = (image1.shape[0], 2, image1.shape[2], image1.shape[3])
            if flow0.shape != expected:
                raise ValueError("flow0 must have shape (batch, 2, height, width)")
            flow_init = flow0.to(device=image1.device, dtype=fmap1.dtype)
        solver_result = solve_equilibrium(
            lambda flow: self.transition(flow, fmap1, fmap2, correlation),
            flow_init,
            replace(self.config, reengage=self.reengage),
            params=tuple(self.parameters()),
            tensors=(image1, image2),
        )
        if return_result:
            return SILVAFlowResult(
                flow=solver_result.z,
                solver_result=solver_result,
                correlation=correlation if return_correlation else None,
            )
        return solver_result.z


class SILVADEQFlow(SILVAOpticalFlowDEQ):
    r"""SILVA-style public name for the optical-flow equilibrium layer.

    This class is equivalent to `SILVAOpticalFlowDEQ`. The name emphasizes the
    package convention: the model is a SILVA fixed-point flow estimator whose
    lineage includes RAFT all-pairs correlation and DEQ-Flow equilibrium
    solving.
    """


def silva_optical_flow_deq(
    *,
    in_channels: int = 1,
    feature_dim: int = 8,
    hidden_dim: int = 32,
    corr_radius: int = 1,
    update_scale: float = 0.25,
    config: SolverConfig | None = None,
    reengage: bool = True,
) -> SILVAOpticalFlowDEQ:
    """Create a SILVA optical-flow DEQ model."""

    return SILVAOpticalFlowDEQ(
        in_channels=in_channels,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        corr_radius=corr_radius,
        update_scale=update_scale,
        config=config,
        reengage=reengage,
    )


def silva_deq_flow(
    *,
    in_channels: int = 1,
    feature_dim: int = 8,
    hidden_dim: int = 32,
    corr_radius: int = 1,
    update_scale: float = 0.25,
    config: SolverConfig | None = None,
    reengage: bool = True,
) -> SILVADEQFlow:
    """Create a SILVA DEQ-flow model.

    This is the preferred SILVA-style factory. The older
    `silva_optical_flow_deq` name remains available for compatibility.
    """

    return SILVADEQFlow(
        in_channels=in_channels,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        corr_radius=corr_radius,
        update_scale=update_scale,
        config=config,
        reengage=reengage,
    )


def silva_raft_deq(**kwargs) -> SILVARAFTDEQ:
    """Create the coupled RAFT/DEQ-Flow SILVA architecture."""

    return SILVARAFTDEQ(**kwargs)


def make_silva_translation_flow_batch(
    *,
    batch_size: int = 2,
    channels: int = 1,
    height: int = 16,
    width: int = 16,
    shift: tuple[float, float] = (1.0, 0.0),
    noise: float = 0.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> SILVAFlowBatch:
    """Create a small synthetic optical-flow batch.

    The second image is generated by translating the first image. The stored
    flow is the forward displacement from `image1` to `image2`, using `(dx, dy)`
    channel order.
    """

    if height < 2 or width < 2:
        raise ValueError("height and width must be at least 2")
    if batch_size < 1 or channels < 1:
        raise ValueError("batch_size and channels must be positive")
    if noise < 0:
        raise ValueError("noise must be nonnegative")
    image1 = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
    image1 = F.avg_pool2d(image1, kernel_size=3, stride=1, padding=1)
    flow = torch.zeros(batch_size, 2, height, width, device=device, dtype=dtype)
    flow[:, 0].fill_(shift[0])
    flow[:, 1].fill_(shift[1])
    image2 = silva_flow_warp(image1, -flow, padding_mode="zeros")
    if noise > 0:
        image2 = image2 + noise * torch.randn_like(image2)
    valid = _translation_valid_mask(batch_size, height, width, shift, device=device, dtype=dtype)
    return SILVAFlowBatch(image1=image1, image2=image2, flow=flow, valid=valid)


def silva_endpoint_error(
    pred_flow: Tensor,
    target_flow: Tensor,
    valid: Tensor | None = None,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Compute optical-flow endpoint error."""

    if pred_flow.shape != target_flow.shape or pred_flow.dim() != 4 or pred_flow.shape[1] != 2:
        raise ValueError("flow tensors must both have shape (batch, 2, height, width)")
    epe = torch.linalg.norm(pred_flow - target_flow, dim=1, keepdim=True)
    if valid is not None:
        if valid.shape not in {epe.shape, pred_flow.shape}:
            raise ValueError("valid must have one or two channels at the flow resolution")
        mask = valid.to(device=epe.device, dtype=epe.dtype)
        if mask.shape[1] == 2:
            mask = mask.amin(dim=1, keepdim=True)
        epe = epe * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(epe.numel(), device=epe.device, dtype=epe.dtype)
    if reduction == "none":
        return epe
    if reduction == "sum":
        return epe.sum()
    if reduction == "mean":
        return epe.sum() / denom
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def silva_flow_smoothness_loss(
    flow: Tensor, *, reduction: Literal["mean", "sum"] = "mean"
) -> Tensor:
    """First-order smoothness penalty for flow fields."""

    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape (batch, 2, height, width)")
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    value = dx.abs().sum() + dy.abs().sum()
    if reduction == "sum":
        return value
    if reduction == "mean":
        denom = max(1, dx.numel() + dy.numel())
        return value / denom
    raise ValueError("reduction must be 'mean' or 'sum'")


def _translation_valid_mask(
    batch_size: int,
    height: int,
    width: int,
    shift: tuple[float, float],
    *,
    device: str | torch.device | None,
    dtype: torch.dtype,
) -> Tensor:
    coords = silva_coords_grid(batch_size, height, width, device=device, dtype=dtype)
    x = coords[:, 0] + shift[0]
    y = coords[:, 1] + shift[1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    return valid.unsqueeze(1).to(dtype=dtype)


__all__ = [
    "SILVARAFTDEQ",
    "SILVACorrelationPyramid",
    "SILVADEQFlow",
    "SILVAFlowBatch",
    "SILVAFlowFeatureEncoder",
    "SILVAFlowResult",
    "SILVAFlowUpdateBlock",
    "SILVAGlobalMotionAggregator",
    "SILVAOpticalFlowDEQ",
    "SILVARAFTEncoder",
    "SILVARAFTResidualBlock",
    "SILVARAFTState",
    "SILVARAFTUpdateBlock",
    "SILVASeparatedConvGRU",
    "make_silva_translation_flow_batch",
    "silva_all_pairs_correlation",
    "silva_coords_grid",
    "silva_deq_flow",
    "silva_endpoint_error",
    "silva_flow_fixed_point_correction_loss",
    "silva_flow_smoothness_loss",
    "silva_flow_warp",
    "silva_local_correlation_lookup",
    "silva_optical_flow_deq",
    "silva_raft_deq",
]
