from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
]
NAME = "14_point_architecture_catalog.ipynb"
_CELL_COUNTER = 0


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"point-architecture-{_CELL_COUNTER:04d}"


def md(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _next_cell_id(),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = """
from pathlib import Path
import importlib.util
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jseluis/silva-networks.git"

def find_local_silva_root():
    candidates = [Path.cwd(), Path("/content/silva-networks")]
    root = Path.cwd()
    while root != root.parent:
        candidates.append(root)
        root = root.parent
    for candidate in candidates:
        if (candidate / "src" / "silva_networks").exists():
            return candidate
    return None

root = find_local_silva_root()
if root is not None:
    sys.path.insert(0, str(root / "src"))
elif IN_COLAB and importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


NB = notebook(
    [
        md(
            r"""
# Ten Internal Architectures for SILVA Points

A SILVA equilibrium point separates the solver from the architecture evaluated
inside each transition:

$$
z^\star
=
\Psi\!\left[
u + B_\theta(a(z^\star))
+ \sum_m I_{m,\theta}(a(z^\star),u,x,E,b)
\right].
$$

The internal architecture must return a field with the same shape as the
equilibrium state:

$$
B_\theta:\mathbb R^{\mathcal S}\rightarrow\mathbb R^{\mathcal S}.
$$

This notebook exercises ten compact reference architectures on vector, token,
and spatial states. It checks shape preservation, fixed-point execution,
residuals, and gradients. The checks are compatibility tests, not comparative
accuracy benchmarks.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from silva_networks import (
    SILVACortexLayer,
    SILVACortexNetwork,
    SolverConfig,
    available_silva_point_architectures,
    silva_point_architecture,
    silva_point_architecture_info,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(140)
"""
        ),
        md(
            """
## Catalog and Tensor Contracts

The catalog is intentionally representative rather than ranked. Each entry
contributes a distinct computation pattern while preserving the state layout
needed by an equilibrium point.
"""
        ),
        code(
            """
for name in available_silva_point_architectures():
    info = silva_point_architecture_info(name)
    year = info.introduced if info.introduced is not None else "classic"
    print(f"{name:18s} | {info.state_layout:36s} | {year} | {info.summary}")
"""
        ),
        code(
            """
SPECS = {
    "mlp": ((4, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_mlp": ((4, 8), {"dim": 8, "hidden_dim": 12}),
    "residual_cnn": ((3, 4, 8, 8), {"channels": 4, "depth": 1}),
    "unet": ((3, 4, 8, 8), {"channels": 4, "base_channels": 6}),
    "dense_cnn": ((3, 4, 8, 8), {"channels": 4, "growth_rate": 3, "depth": 2}),
    "transformer": ((3, 6, 8), {"dim": 8, "heads": 2, "hidden_dim": 12}),
    "inverted_residual": ((3, 4, 8, 8), {"channels": 4, "expansion": 2}),
    "fourier_operator": (
        (3, 4, 8, 8),
        {"channels": 4, "modes_height": 3, "modes_width": 3},
    ),
    "mlp_mixer": ((3, 6, 8), {"tokens": 6, "dim": 8}),
    "convnext_v2": ((3, 4, 8, 8), {"channels": 4, "expansion": 2}),
}


def check_architecture(name):
    shape, kwargs = SPECS[name]
    architecture = silva_point_architecture(name, **kwargs)
    stimulus = torch.randn(*shape, requires_grad=True)
    point = SILVACortexLayer(
        input_encoder=torch.nn.Identity(),
        state_network=architecture,
        normalize=False,
        config=SolverConfig(solver="picard", max_iter=2, alpha=0.25),
    )
    result = point(stimulus, return_result=True)
    loss = result.z.square().mean()
    loss.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.square().sum())
            for parameter in architecture.parameters()
            if parameter.grad is not None
        )
    )
    assert result.z.shape == stimulus.shape
    assert torch.isfinite(result.z).all()
    assert gradient_norm > 0.0
    return {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in architecture.parameters()),
        "start": float(result.residuals[0]),
        "end": float(result.residuals[-1]),
        "gradient_norm": gradient_norm,
    }
"""
        ),
        md(
            """
## Vector-State Architectures

MLP and residual-MLP fields act on the final channel dimension. They support a
state shaped `(batch, channels)` and can also be applied independently at every
token when the leading dimensions are retained.
"""
        ),
        code(
            """
vector_results = [check_architecture(name) for name in ("mlp", "residual_mlp")]
vector_results
"""
        ),
        md(
            r"""
## Token-State Architectures

The Transformer and MLP-Mixer entries use `(batch, tokens, channels)` states.
Attention learns content-dependent token interactions, while MLP-Mixer uses a
fixed token count and alternates token and channel mixing.
"""
        ),
        code(
            """
token_results = [check_architecture(name) for name in ("transformer", "mlp_mixer")]
token_results
"""
        ),
        md(
            r"""
## Spatial-State Architectures

The six spatial entries use `(batch, channels, height, width)` states. U-Net may
downsample internally, and the Fourier operator moves through frequency space,
but both restore the original state shape before returning to the solver.
"""
        ),
        code(
            """
spatial_names = (
    "residual_cnn",
    "unet",
    "dense_cnn",
    "inverted_residual",
    "fourier_operator",
    "convnext_v2",
)
spatial_results = [check_architecture(name) for name in spatial_names]
spatial_results
"""
        ),
        md(
            """
## Compact Validation Summary

For a two-step damped solve, the final residual should be finite and lower than
the first-step residual. A nonzero gradient confirms that the selected module
participates in the differentiable transition.
"""
        ),
        code(
            """
results = vector_results + token_results + spatial_results
for row in results:
    ratio = row["end"] / row["start"]
    assert math.isfinite(ratio) and ratio < 1.0
    print(
        f"{row['name']:18s} parameters={row['parameters']:5d} "
        f"residual ratio={ratio:.3f} gradient={row['gradient_norm']:.3e}"
    )
"""
        ),
        code(
            """
names = [row["name"] for row in results]
ratios = [row["end"] / row["start"] for row in results]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(names, ratios, color=["#2474b5", "#239b56", "#d97706", "#8e44ad", "#c0392b"] * 2)
ax.axhline(1.0, color="#222222", linewidth=1)
ax.set_ylabel("final / initial residual")
ax.set_title("Two-step SILVA point compatibility check")
ax.tick_params(axis="x", rotation=55)
fig.tight_layout()
"""
        ),
        md(
            r"""
## Several Architectures Inside One Point

`state_network` may be a sequence. The modules are evaluated in order during
every solver step, so each module must preserve the shared state shape:

$$
B_\theta
=
B_{\theta,3}\circ B_{\theta,2}\circ B_{\theta,1}.
$$
"""
        ),
        code(
            """
composed_point = SILVACortexLayer(
    input_encoder=torch.nn.Identity(),
    state_network=[
        silva_point_architecture("residual_cnn", channels=4, depth=1),
        silva_point_architecture("convnext_v2", channels=4, expansion=2),
        silva_point_architecture("unet", channels=4, base_channels=6),
    ],
    normalizer=torch.nn.GroupNorm(1, 4),
    config=SolverConfig(solver="picard", max_iter=3, alpha=0.25),
)
composed_result = composed_point(torch.randn(3, 4, 8, 8), return_result=True)
print("composed state:", tuple(composed_result.z.shape))
print("residuals:", [f"{value:.3e}" for value in composed_result.residuals])
"""
        ),
        md(
            r"""
## Different Architectures Across Linked Points

The next model links two equilibrium points. The first uses a residual CNN with
Picard iteration; the second uses U-Net with Anderson acceleration. A tiny bar
dataset checks the complete forward and backward path.
"""
        ),
        code(
            """
def make_bar_data(samples=12, size=8):
    generator = torch.Generator().manual_seed(141)
    images = 0.04 * torch.randn(samples, 4, size, size, generator=generator)
    labels = torch.arange(samples) % 2
    for index, label in enumerate(labels):
        if int(label) == 0:
            images[index, :, :, 3:5] += 1.0
        else:
            images[index, :, 3:5, :] += 1.0
    return images, labels


network = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture("residual_cnn", channels=4, depth=2),
            normalizer=torch.nn.GroupNorm(1, 4),
            config=SolverConfig(solver="picard", max_iter=3, alpha=0.35),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Identity(),
            state_network=silva_point_architecture("unet", channels=4, base_channels=8),
            normalizer=torch.nn.GroupNorm(1, 4),
            config=SolverConfig(solver="anderson", max_iter=3, alpha=0.2, history=2),
        ),
    ],
    links="tanh",
    head=torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    ),
)

images, labels = make_bar_data()
network_result = network(images, return_results=True)
loss = F.cross_entropy(network_result.output, labels)
loss.backward()

print("logits:", tuple(network_result.output.shape))
print("states:", [tuple(state.shape) for state in network_result.states])
print("solvers:", [result.solver for result in network_result.solver_results])
print("loss:", f"{float(loss):.4f}")
print(
    "first point gradient:",
    network.layers[0].state_network[0].blocks[0].conv1.weight.grad is not None,
)
"""
        ),
        md(
            """
## Selection Rules

- Use vector MLPs for tabular or pooled feature states.
- Use Transformer when token interactions should depend on content.
- Use MLP-Mixer when token count is fixed and attention is unnecessary.
- Use residual, dense, inverted-residual, or ConvNeXt V2 blocks for local image structure.
- Use U-Net when one equilibrium evaluation needs multiple spatial scales.
- Use the Fourier operator when global spectral modes are part of the modeling assumption.
- Keep every module shape-preserving at the point boundary and tune solver damping after changing the internal architecture.

Primary sources: [ResNet](https://arxiv.org/abs/1512.03385),
[U-Net](https://arxiv.org/abs/1505.04597),
[DenseNet](https://arxiv.org/abs/1608.06993),
[Transformer](https://arxiv.org/abs/1706.03762),
[MobileNetV2](https://arxiv.org/abs/1801.04381),
[Fourier Neural Operator](https://arxiv.org/abs/2010.08895),
[MLP-Mixer](https://arxiv.org/abs/2105.01601), and
[ConvNeXt V2](https://arxiv.org/abs/2301.00808).
"""
        ),
    ]
)


def main() -> None:
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / NAME
        path.write_text(json.dumps(NB, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
