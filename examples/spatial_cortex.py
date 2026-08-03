from __future__ import annotations

import torch
import torch.nn.functional as F

from silva_networks import SILVACortexLayer, SolverConfig, resolve_device, silva_equilibrium_model


class SILVAResidualConvTransition(torch.nn.Module):
    """Deterministic residual block that preserves a spatial equilibrium state."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(1, channels)
        self.norm2 = torch.nn.GroupNorm(1, channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        update = F.gelu(self.norm1(self.conv1(z)))
        update = self.norm2(self.conv2(update))
        return z + 0.25 * update


class SILVATinyUNetTransition(torch.nn.Module):
    """Small U-Net-shaped transition whose output matches its input shape."""

    def __init__(self, channels: int):
        super().__init__()
        expanded = 2 * channels
        self.encoder = SILVAResidualConvTransition(channels)
        self.down = torch.nn.Conv2d(channels, expanded, kernel_size=3, stride=2, padding=1)
        self.bottleneck = SILVAResidualConvTransition(expanded)
        self.up = torch.nn.ConvTranspose2d(expanded, channels, kernel_size=2, stride=2)
        self.decoder = torch.nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(z)
        low = self.bottleneck(F.gelu(self.down(skip)))
        up = self.up(low)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return 0.25 * torch.tanh(self.decoder(torch.cat([skip, up], dim=1)))


class SILVASpatialToVectorLink(torch.nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.flatten(start_dim=1)


def make_tiny_pattern_dataset(
    samples: int = 24,
    image_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create horizontal/vertical bar images without downloads."""

    generator = torch.Generator().manual_seed(57)
    images = 0.04 * torch.randn(samples, 1, image_size, image_size, generator=generator)
    labels = torch.arange(samples) % 2
    center = image_size // 2
    for index, label in enumerate(labels):
        if int(label) == 0:
            images[index, 0, :, center - 1 : center + 1] += 1.0
        else:
            images[index, 0, center - 1 : center + 1, :] += 1.0
    return images, labels


def build_model(image_size: int = 8):
    channels = 4
    spatial_point = SILVACortexLayer(
        input_encoder=torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
        state_network=torch.nn.Sequential(
            SILVAResidualConvTransition(channels),
            SILVATinyUNetTransition(channels),
        ),
        normalizer=torch.nn.GroupNorm(1, channels),
        config=SolverConfig(solver="picard", max_iter=3, alpha=0.35),
    )
    vector_point = SILVACortexLayer(
        input_dim=channels * image_size * image_size,
        state_dim=12,
        state_network=torch.nn.Sequential(
            torch.nn.Linear(12, 24),
            torch.nn.GELU(),
            torch.nn.Linear(24, 12),
        ),
        config=SolverConfig(solver="anderson", max_iter=3, alpha=0.2, history=2),
    )
    return silva_equilibrium_model(
        "silva_cortex_network",
        layers=[spatial_point, vector_point],
        links=[SILVASpatialToVectorLink()],
        head=torch.nn.Linear(12, 2),
    )


def main() -> None:
    torch.manual_seed(58)
    device = resolve_device("auto")
    images, labels = make_tiny_pattern_dataset()
    images = images.to(device)
    labels = labels.to(device)
    model = build_model(image_size=images.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    for _ in range(4):
        result = model(images, return_results=True)
        loss = F.cross_entropy(result.output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (result.output.argmax(dim=1) == labels).float().mean()
    spatial_gradient = model.layers[0].input_encoder.weight.grad is not None
    vector_gradient = model.layers[1].input_encoder.weight.grad is not None
    if not spatial_gradient or not vector_gradient:
        raise RuntimeError("gradients did not reach both SILVA equilibrium points")
    print("device", device.type)
    print("state_shapes", [tuple(state.shape) for state in result.states])
    print("solvers", [solver.solver for solver in result.solver_results])
    print("loss", float(loss.detach().cpu()))
    print("accuracy", float(accuracy.detach().cpu()))
    print("point_gradients", [spatial_gradient, vector_gradient])


if __name__ == "__main__":
    main()
