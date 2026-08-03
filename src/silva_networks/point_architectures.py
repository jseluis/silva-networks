"""Shape-preserving internal architectures for SILVA equilibrium points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

Tensor = torch.Tensor
SILVAPointArchitectureName = Literal[
    "mlp",
    "residual_mlp",
    "residual_cnn",
    "unet",
    "dense_cnn",
    "transformer",
    "inverted_residual",
    "fourier_operator",
    "mlp_mixer",
    "convnext_v2",
]


@dataclass(frozen=True)
class SILVAPointArchitectureInfo:
    """Description of one built-in SILVA point architecture.

    Attributes:
        name: Stable name accepted by :func:`silva_point_architecture`.
        state_layout: Tensor layout expected by the module.
        introduced: Publication year of the source architecture, when applicable.
        reference_url: Primary source for the architecture, when applicable.
        summary: Short description of the internal computation.
    """

    name: SILVAPointArchitectureName
    state_layout: str
    introduced: int | None
    reference_url: str | None
    summary: str


def _check_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _check_rank(z: Tensor, rank: int, name: str, layout: str) -> None:
    if z.dim() != rank:
        raise ValueError(f"{name} expects {layout}; received shape {tuple(z.shape)}")


class SILVAMLPPointArchitecture(nn.Module):
    """Feed-forward field for vector or token SILVA states."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(dim, "dim")
        _check_positive(depth, "depth")
        hidden_dim = hidden_dim or 2 * dim
        _check_positive(hidden_dim, "hidden_dim")

        layers: list[nn.Module] = [nn.Linear(dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, dim))
        self.network = nn.Sequential(*layers)
        self.dim = dim
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"SILVAMLPPointArchitecture expects last dimension {self.dim}; "
                f"received shape {tuple(z.shape)}"
            )
        return self.scale * self.network(z)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return z + self.network(self.norm(z))


class SILVAResidualMLPPointArchitecture(nn.Module):
    """Residual multilayer field for vector or token SILVA states."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(dim, "dim")
        _check_positive(depth, "depth")
        hidden_dim = hidden_dim or 2 * dim
        _check_positive(hidden_dim, "hidden_dim")
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(dim, hidden_dim) for _ in range(depth)]
        )
        self.dim = dim
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"SILVAResidualMLPPointArchitecture expects last dimension {self.dim}; "
                f"received shape {tuple(z.shape)}"
            )
        state = z
        for block in self.blocks:
            state = block(state)
        return self.scale * state


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, z: Tensor) -> Tensor:
        update = self.conv1(F.gelu(self.norm1(z)))
        update = self.conv2(F.gelu(self.norm2(update)))
        return z + update


class SILVAResidualConvPointArchitecture(nn.Module):
    """Residual convolutional field for spatial SILVA states in NCHW layout."""

    def __init__(
        self,
        channels: int,
        depth: int = 2,
        kernel_size: int = 3,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(channels, "channels")
        _check_positive(depth, "depth")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve spatial shape")
        self.blocks = nn.ModuleList(
            [_ResidualConvBlock(channels, kernel_size) for _ in range(depth)]
        )
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        state = z
        for block in self.blocks:
            state = block(state)
        return self.scale * state


class SILVAUNetPointArchitecture(nn.Module):
    """Compact U-Net-shaped field that restores the spatial SILVA state shape."""

    def __init__(
        self,
        channels: int,
        base_channels: int | None = None,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(channels, "channels")
        base_channels = base_channels or 2 * channels
        _check_positive(base_channels, "base_channels")
        self.encoder = _ResidualConvBlock(channels)
        self.down = nn.Conv2d(channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.bottleneck = _ResidualConvBlock(base_channels)
        self.up = nn.ConvTranspose2d(base_channels, channels, kernel_size=2, stride=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        skip = self.encoder(z)
        low = self.bottleneck(F.gelu(self.down(skip)))
        up = self.up(low)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.scale * self.decoder(torch.cat([skip, up], dim=1))


class SILVADenseConvPointArchitecture(nn.Module):
    """DenseNet-style concatenated convolutional field for spatial SILVA states."""

    def __init__(
        self,
        channels: int,
        growth_rate: int | None = None,
        depth: int = 3,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(channels, "channels")
        _check_positive(depth, "depth")
        growth_rate = growth_rate or channels
        _check_positive(growth_rate, "growth_rate")
        self.layers = nn.ModuleList()
        width = channels
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                    nn.Conv2d(width, growth_rate, kernel_size=3, padding=1),
                )
            )
            width += growth_rate
        self.project = nn.Conv2d(width, channels, kernel_size=1)
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        features = [z]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.scale * self.project(torch.cat(features, dim=1))


class SILVATransformerPointArchitecture(nn.Module):
    """Transformer encoder field for token SILVA states in BND layout."""

    def __init__(
        self,
        dim: int,
        heads: int = 2,
        hidden_dim: int | None = None,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(dim, "dim")
        _check_positive(heads, "heads")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        hidden_dim = hidden_dim or 4 * dim
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dim = dim
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 3, self.__class__.__name__, "(batch, tokens, channels)")
        if z.shape[-1] != self.dim:
            raise ValueError(f"expected channel width {self.dim}; received {z.shape[-1]}")
        return self.scale * self.layer(z)


class SILVAInvertedResidualPointArchitecture(nn.Module):
    """MobileNetV2-style inverted residual field for spatial SILVA states."""

    def __init__(self, channels: int, expansion: int = 4, scale: float = 0.1):
        super().__init__()
        _check_positive(channels, "channels")
        _check_positive(expansion, "expansion")
        expanded = expansion * channels
        self.expand = nn.Conv2d(channels, expanded, kernel_size=1)
        self.depthwise = nn.Conv2d(
            expanded,
            expanded,
            kernel_size=3,
            padding=1,
            groups=expanded,
        )
        self.norm = nn.GroupNorm(1, expanded)
        self.project = nn.Conv2d(expanded, channels, kernel_size=1)
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        update = F.gelu(self.expand(z))
        update = F.gelu(self.norm(self.depthwise(update)))
        return self.scale * (z + self.project(update))


class _SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes_height: int, modes_width: int):
        super().__init__()
        _check_positive(modes_height, "modes_height")
        _check_positive(modes_width, "modes_width")
        weight_shape = (channels, channels, modes_height, modes_width, 2)
        weight_scale = 1.0 / max(1, channels * channels)
        self.weight_top = nn.Parameter(weight_scale * torch.randn(*weight_shape))
        self.weight_bottom = nn.Parameter(weight_scale * torch.randn(*weight_shape))
        self.modes_height = modes_height
        self.modes_width = modes_width

    @staticmethod
    def _multiply(values: Tensor, weights: Tensor) -> Tensor:
        complex_weights = torch.view_as_complex(weights.contiguous())
        return torch.einsum("bixy,ioxy->boxy", values, complex_weights)

    def forward(self, z: Tensor) -> Tensor:
        height, width = z.shape[-2:]
        spectrum = torch.fft.rfft2(z, norm="ortho")
        output = torch.zeros(
            z.shape[0],
            z.shape[1],
            height,
            width // 2 + 1,
            dtype=spectrum.dtype,
            device=z.device,
        )
        modes_height = min(self.modes_height, height // 2)
        modes_width = min(self.modes_width, width // 2 + 1)
        if modes_height > 0 and modes_width > 0:
            output[:, :, :modes_height, :modes_width] = self._multiply(
                spectrum[:, :, :modes_height, :modes_width],
                self.weight_top[:, :, :modes_height, :modes_width],
            )
            output[:, :, -modes_height:, :modes_width] = self._multiply(
                spectrum[:, :, -modes_height:, :modes_width],
                self.weight_bottom[:, :, :modes_height, :modes_width],
            )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class SILVAFourierOperatorPointArchitecture(nn.Module):
    """Fourier-operator field with spectral and local spatial branches."""

    def __init__(
        self,
        channels: int,
        modes_height: int = 4,
        modes_width: int = 4,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(channels, "channels")
        self.spectral = _SpectralConv2d(channels, modes_height, modes_width)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        return self.scale * (self.spectral(z) + self.local(z))


class _MLPMixerBlock(nn.Module):
    def __init__(
        self,
        tokens: int,
        dim: int,
        token_hidden_dim: int,
        channel_hidden_dim: int,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, token_hidden_dim),
            nn.GELU(),
            nn.Linear(token_hidden_dim, tokens),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden_dim),
            nn.GELU(),
            nn.Linear(channel_hidden_dim, dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        mixed_tokens = self.token_mlp(self.token_norm(z).transpose(1, 2)).transpose(1, 2)
        z = z + mixed_tokens
        return z + self.channel_mlp(self.channel_norm(z))


class SILVAMLPMixerPointArchitecture(nn.Module):
    """MLP-Mixer field for fixed-length token SILVA states in BND layout."""

    def __init__(
        self,
        tokens: int,
        dim: int,
        token_hidden_dim: int | None = None,
        channel_hidden_dim: int | None = None,
        depth: int = 1,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(tokens, "tokens")
        _check_positive(dim, "dim")
        _check_positive(depth, "depth")
        token_hidden_dim = token_hidden_dim or 2 * tokens
        channel_hidden_dim = channel_hidden_dim or 2 * dim
        self.blocks = nn.ModuleList(
            [
                _MLPMixerBlock(tokens, dim, token_hidden_dim, channel_hidden_dim)
                for _ in range(depth)
            ]
        )
        self.tokens = tokens
        self.dim = dim
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 3, self.__class__.__name__, "(batch, tokens, channels)")
        if z.shape[1:] != (self.tokens, self.dim):
            raise ValueError(
                f"expected token shape ({self.tokens}, {self.dim}); "
                f"received {tuple(z.shape[1:])}"
            )
        state = z
        for block in self.blocks:
            state = block(state)
        return self.scale * state


class _GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, z: Tensor) -> Tensor:
        response = torch.linalg.vector_norm(z, ord=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return z + self.gamma * (z * normalized) + self.beta


class _ConvNeXtV2Block(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        expanded = expansion * channels
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, expanded)
        self.grn = _GlobalResponseNorm(expanded)
        self.project = nn.Linear(expanded, channels)

    def forward(self, z: Tensor) -> Tensor:
        update = self.depthwise(z).permute(0, 2, 3, 1)
        update = self.expand(self.norm(update))
        update = self.grn(F.gelu(update))
        update = self.project(update).permute(0, 3, 1, 2)
        return z + update


class SILVAConvNeXtV2PointArchitecture(nn.Module):
    """ConvNeXt V2-style depthwise and response-normalized spatial field."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        depth: int = 1,
        scale: float = 0.1,
    ):
        super().__init__()
        _check_positive(channels, "channels")
        _check_positive(expansion, "expansion")
        _check_positive(depth, "depth")
        self.blocks = nn.ModuleList(
            [_ConvNeXtV2Block(channels, expansion) for _ in range(depth)]
        )
        self.channels = channels
        self.scale = float(scale)

    def forward(self, z: Tensor) -> Tensor:
        _check_rank(z, 4, self.__class__.__name__, "(batch, channels, height, width)")
        if z.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels; received {z.shape[1]}")
        state = z
        for block in self.blocks:
            state = block(state)
        return self.scale * state


_ARCHITECTURE_CLASSES: dict[SILVAPointArchitectureName, type[nn.Module]] = {
    "mlp": SILVAMLPPointArchitecture,
    "residual_mlp": SILVAResidualMLPPointArchitecture,
    "residual_cnn": SILVAResidualConvPointArchitecture,
    "unet": SILVAUNetPointArchitecture,
    "dense_cnn": SILVADenseConvPointArchitecture,
    "transformer": SILVATransformerPointArchitecture,
    "inverted_residual": SILVAInvertedResidualPointArchitecture,
    "fourier_operator": SILVAFourierOperatorPointArchitecture,
    "mlp_mixer": SILVAMLPMixerPointArchitecture,
    "convnext_v2": SILVAConvNeXtV2PointArchitecture,
}

_ARCHITECTURE_INFO: dict[SILVAPointArchitectureName, SILVAPointArchitectureInfo] = {
    "mlp": SILVAPointArchitectureInfo(
        "mlp",
        "(..., channels)",
        1986,
        "https://doi.org/10.1038/323533a0",
        "Classic feed-forward channel mixing.",
    ),
    "residual_mlp": SILVAPointArchitectureInfo(
        "residual_mlp",
        "(..., channels)",
        2015,
        "https://arxiv.org/abs/1512.03385",
        "Residual channel-mixing blocks.",
    ),
    "residual_cnn": SILVAPointArchitectureInfo(
        "residual_cnn",
        "(batch, channels, height, width)",
        2015,
        "https://arxiv.org/abs/1512.03385",
        "Residual local convolutional blocks.",
    ),
    "unet": SILVAPointArchitectureInfo(
        "unet",
        "(batch, channels, height, width)",
        2015,
        "https://arxiv.org/abs/1505.04597",
        "Contracting and expanding paths with a skip connection.",
    ),
    "dense_cnn": SILVAPointArchitectureInfo(
        "dense_cnn",
        "(batch, channels, height, width)",
        2016,
        "https://arxiv.org/abs/1608.06993",
        "Densely concatenated local convolutional features.",
    ),
    "transformer": SILVAPointArchitectureInfo(
        "transformer",
        "(batch, tokens, channels)",
        2017,
        "https://arxiv.org/abs/1706.03762",
        "Multi-head token attention and feed-forward mixing.",
    ),
    "inverted_residual": SILVAPointArchitectureInfo(
        "inverted_residual",
        "(batch, channels, height, width)",
        2018,
        "https://arxiv.org/abs/1801.04381",
        "Expanded depthwise convolution with a narrow residual state.",
    ),
    "fourier_operator": SILVAPointArchitectureInfo(
        "fourier_operator",
        "(batch, channels, height, width)",
        2020,
        "https://arxiv.org/abs/2010.08895",
        "Low-frequency spectral mixing plus a local projection.",
    ),
    "mlp_mixer": SILVAPointArchitectureInfo(
        "mlp_mixer",
        "(batch, tokens, channels)",
        2021,
        "https://arxiv.org/abs/2105.01601",
        "Alternating token and channel MLPs.",
    ),
    "convnext_v2": SILVAPointArchitectureInfo(
        "convnext_v2",
        "(batch, channels, height, width)",
        2023,
        "https://arxiv.org/abs/2301.00808",
        "Depthwise convolution, channel expansion, and global response normalization.",
    ),
}


def available_silva_point_architectures() -> tuple[SILVAPointArchitectureName, ...]:
    """Return the stable names of the ten built-in point architectures."""

    return tuple(_ARCHITECTURE_CLASSES)


def silva_point_architecture_info(
    name: SILVAPointArchitectureName | str,
) -> SILVAPointArchitectureInfo:
    """Return tensor-layout and source metadata for one point architecture."""

    try:
        return _ARCHITECTURE_INFO[name]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(available_silva_point_architectures())
        raise ValueError(f"Unknown SILVA point architecture '{name}'. Choose from: {choices}") from exc


def silva_point_architecture(
    name: SILVAPointArchitectureName | str,
    **kwargs,
) -> nn.Module:
    """Build a shape-preserving internal architecture for a SILVA point.

    Constructor arguments are forwarded to the selected architecture class.
    Use :func:`silva_point_architecture_info` to inspect the expected state
    layout before construction.
    """

    try:
        architecture = _ARCHITECTURE_CLASSES[name]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(available_silva_point_architectures())
        raise ValueError(f"Unknown SILVA point architecture '{name}'. Choose from: {choices}") from exc
    return architecture(**kwargs)


__all__ = [
    "SILVAConvNeXtV2PointArchitecture",
    "SILVADenseConvPointArchitecture",
    "SILVAFourierOperatorPointArchitecture",
    "SILVAInvertedResidualPointArchitecture",
    "SILVAMLPMixerPointArchitecture",
    "SILVAMLPPointArchitecture",
    "SILVAPointArchitectureInfo",
    "SILVAPointArchitectureName",
    "SILVAResidualConvPointArchitecture",
    "SILVAResidualMLPPointArchitecture",
    "SILVATransformerPointArchitecture",
    "SILVAUNetPointArchitecture",
    "available_silva_point_architectures",
    "silva_point_architecture",
    "silva_point_architecture_info",
]
