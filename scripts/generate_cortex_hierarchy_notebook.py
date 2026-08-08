from __future__ import annotations

import re
import textwrap
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
]
NAME = "11_cortex_hierarchy.ipynb"
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE
)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"cortex-{_CELL_COUNTER:04d}"


def _matplotlib_import_with_publication_dpi(match: re.Match[str]) -> str:
    indent = match.group("indent")
    return (
        f"{indent}import matplotlib.pyplot as plt\n\n"
        f'{indent}plt.rcParams.update({{"figure.dpi": 300, "savefig.dpi": 300}})'
    )


def md(source: str) -> dict:
    source = _INLINE_MATH_RE.sub(r"$\1$", textwrap.dedent(source).strip())
    return {
        "cell_type": "markdown",
        "id": _next_cell_id(),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str) -> dict:
    source = textwrap.dedent(source).strip()
    if "import matplotlib.pyplot as plt" in source and "figure.dpi" not in source:
        source = _MATPLOTLIB_IMPORT_RE.sub(
            _matplotlib_import_with_publication_dpi,
            source,
            count=1,
        )
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
    candidates = [
        Path.cwd(),
        Path("/content/silva-networks"),
        Path("/content/drive/MyDrive/silva-networks"),
    ]
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
# Cortex Hierarchy with the SILVA Package

This notebook checks the cortex-style architecture path:

$$
x
\xrightarrow{R_\phi}
u_0
\xrightarrow{\alpha_1}
z_1^\star
\xrightarrow{\tanh}
u_1
\xrightarrow{\alpha_2}
z_2^\star
\xrightarrow{R_\psi}
\hat y.
$$

Each cortex point solves

$$
z_\ell^\star
=
F_{\theta_\ell}(z_\ell^\star,u_{\ell-1}),
$$

with damped solver steps

$$
z_{\ell,k+1}
=
(1-\alpha_\ell)z_{\ell,k}
+\alpha_\ell F_{\theta_\ell}(z_{\ell,k},u_{\ell-1}).
$$

The goal is to verify package behavior: flexible internal modules, different
architectures at different points, different solvers, different alphas,
gradients, and an image-cortex preset.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from silva_networks import (
    SILVACortexLayer,
    SILVACortexNetwork,
    SILVAImageCortexClassifier,
    SolverConfig,
    resolve_device,
    silva_equilibrium_model,
)

torch.manual_seed(11)
device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
device
"""
        ),
        md(
            r"""
## A Deep Internal Network Inside One Equilibrium Point

The internal transition network \(B_\theta\) can contain many trainable layers:

$$
B_\theta(a)
=
B_{\theta,10}\circ\cdots\circ B_{\theta,1}(a).
$$

The cortex point then adds this field to the stimulus and the optional
interaction branches before the solver damping is applied.
"""
        ),
        code(
            """
def deep_state_network(dim, depth):
    modules = []
    for _ in range(depth):
        modules += [torch.nn.Linear(dim, dim), torch.nn.Tanh()]
    modules.append(torch.nn.Linear(dim, dim))
    return torch.nn.Sequential(*modules)


class StimulusGate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, z, stimulus):
        return torch.sigmoid(self.gate(stimulus)) * z
"""
        ),
        code(
            """
model = SILVACortexNetwork(
    [
        SILVACortexLayer(
            input_dim=5,
            state_dim=14,
            state_network=deep_state_network(14, depth=10),
            self_terms=torch.nn.Linear(14, 14, bias=False),
            interaction_terms=[StimulusGate(14)],
            config=SolverConfig(solver="picard", max_iter=6, alpha=0.5),
        ),
        SILVACortexLayer(
            input_encoder=torch.nn.Linear(14, 10),
            state_dim=10,
            state_network=torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.GELU(),
                torch.nn.Linear(20, 10),
            ),
            config=SolverConfig(solver="anderson", max_iter=6, alpha=0.2, history=3),
            normalize=False,
        ),
    ],
    links="tanh",
    head=torch.nn.Linear(10, 2),
).to(device)

x = torch.randn(8, 5, device=device)
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)

result = model(x, return_results=True)
loss = torch.nn.functional.cross_entropy(result.output, y)
loss.backward()

print("logits shape:", tuple(result.output.shape))
print("state shapes:", [tuple(state.shape) for state in result.states])
print("solvers:", [r.solver for r in result.solver_results])
print("alphas:", [layer.config.alpha for layer in model.layers])
print("gradient reaches layer 1 encoder:", model.layers[0].input_encoder.weight.grad is not None)
"""
        ),
        md(
            r"""
## Residual Curves

The two equilibrium points solve different equations. Their residual curves are
not expected to be identical because their transition maps, solvers, and alphas
are different.
"""
        ),
        code(
            """
plt.figure(figsize=(6, 3))
for i, solver_result in enumerate(result.solver_results, start=1):
    plt.plot(solver_result.residuals, marker="o", label=f"point {i}: {solver_result.solver}")
plt.yscale("log")
plt.xlabel("solver step")
plt.ylabel("residual")
plt.title("Cortex hierarchy residuals")
plt.legend()
plt.tight_layout()
"""
        ),
        md(
            r"""
## Spatial Architecture Inside a SILVA Point

The equilibrium state may be an image tensor rather than a feature vector. In
this example, the first point has state shape `(batch, 4, 8, 8)` and evaluates
a residual convolutional block plus a U-Net-shaped transition during every
solver iteration. The U-Net may downsample internally, but it restores the
equilibrium-state shape before returning:

$$
F_{\theta_1}:\mathbb R^{4\times8\times8}
\rightarrow\mathbb R^{4\times8\times8}.
$$

The solved spatial state is flattened and passed to a second SILVA point with
a different vector architecture and solver.
"""
        ),
        code(
            """
class SILVAResidualConvTransition(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = torch.nn.GroupNorm(1, channels)
        self.norm2 = torch.nn.GroupNorm(1, channels)

    def forward(self, z):
        update = F.gelu(self.norm1(self.conv1(z)))
        return z + 0.25 * self.norm2(self.conv2(update))


class SILVATinyUNetTransition(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded = 2 * channels
        self.encoder = SILVAResidualConvTransition(channels)
        self.down = torch.nn.Conv2d(channels, expanded, 3, stride=2, padding=1)
        self.bottleneck = SILVAResidualConvTransition(expanded)
        self.up = torch.nn.ConvTranspose2d(expanded, channels, 2, stride=2)
        self.decoder = torch.nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, z):
        skip = self.encoder(z)
        low = self.bottleneck(F.gelu(self.down(skip)))
        up = self.up(low)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return 0.25 * torch.tanh(self.decoder(torch.cat([skip, up], dim=1)))


class SILVASpatialToVectorLink(torch.nn.Module):
    def forward(self, z):
        return z.flatten(start_dim=1)


def make_tiny_pattern_dataset(samples=24, image_size=8):
    generator = torch.Generator().manual_seed(57)
    images = 0.04 * torch.randn(samples, 1, image_size, image_size, generator=generator)
    labels = torch.arange(samples) % 2
    center = image_size // 2
    for index, label in enumerate(labels):
        if int(label) == 0:
            images[index, 0, :, center - 1:center + 1] += 1.0
        else:
            images[index, 0, center - 1:center + 1, :] += 1.0
    return images, labels
"""
        ),
        code(
            """
channels = 4
spatial_point = SILVACortexLayer(
    input_encoder=torch.nn.Conv2d(1, channels, 3, padding=1),
    state_network=torch.nn.Sequential(
        SILVAResidualConvTransition(channels),
        SILVATinyUNetTransition(channels),
    ),
    normalizer=torch.nn.GroupNorm(1, channels),
    config=SolverConfig(solver="picard", max_iter=3, alpha=0.35),
)
vector_point = SILVACortexLayer(
    input_dim=channels * 8 * 8,
    state_dim=12,
    state_network=torch.nn.Sequential(
        torch.nn.Linear(12, 24),
        torch.nn.GELU(),
        torch.nn.Linear(24, 12),
    ),
    config=SolverConfig(solver="anderson", max_iter=3, alpha=0.2, history=2),
)
spatial_model = silva_equilibrium_model(
    "silva_cortex_network",
    layers=[spatial_point, vector_point],
    links=[SILVASpatialToVectorLink()],
    head=torch.nn.Linear(12, 2),
).to(device)

pattern_images, pattern_labels = make_tiny_pattern_dataset()
pattern_images = pattern_images.to(device)
pattern_labels = pattern_labels.to(device)
optimizer = torch.optim.Adam(spatial_model.parameters(), lr=2e-2)

for _ in range(4):
    spatial_result = spatial_model(pattern_images, return_results=True)
    spatial_loss = F.cross_entropy(spatial_result.output, pattern_labels)
    optimizer.zero_grad()
    spatial_loss.backward()
    optimizer.step()

spatial_accuracy = (spatial_result.output.argmax(dim=1) == pattern_labels).float().mean()
spatial_gradients = [
    spatial_model.layers[0].input_encoder.weight.grad is not None,
    spatial_model.layers[1].input_encoder.weight.grad is not None,
]
if not all(spatial_gradients):
    raise RuntimeError("gradients did not reach both SILVA equilibrium points")
print("state shapes:", [tuple(state.shape) for state in spatial_result.states])
print("solvers:", [item.solver for item in spatial_result.solver_results])
print("loss:", float(spatial_loss.detach().cpu()))
print("accuracy:", float(spatial_accuracy.detach().cpu()))
print("point gradients:", spatial_gradients)
"""
        ),
        md(
            r"""
The module contract for one SILVA point is:

1. the completed transition returns the equilibrium-state shape;
2. the transition is deterministic during one solve;
3. tensors remain on the same device and dtype;
4. every operation supports the selected backward mode.

For spatial states, `GroupNorm` avoids mutable batch statistics. Random masks
must remain consistent during repeated transition evaluations. Larger internal
architectures may require stronger damping, residual scaling, or spectral
normalization; residual curves remain the direct convergence check.
"""
        ),
        md(
            r"""
## Alpha Sweep

For a fixed transition, smaller \(\alpha\) mixes less of the new field value at
each iteration:

$$
z_{k+1}-z_k
=
\alpha(F_\theta(z_k,x)-z_k).
$$

The step size is proportional to \(\alpha\). This is why a fast first point and
a slower second point can be encoded directly by separate `SolverConfig`
objects.
"""
        ),
        code(
            """
base_x = torch.randn(4, 3, device=device)
alpha_summaries = []
for alpha in [0.2, 0.5, 0.8]:
    layer = SILVACortexLayer(
        input_dim=3,
        state_dim=6,
        state_network=torch.nn.Linear(6, 6),
        config=SolverConfig(solver="picard", max_iter=5, alpha=alpha),
        normalize=False,
    ).to(device)
    out = layer(base_x, return_result=True)
    alpha_summaries.append((alpha, out.residuals))

for alpha, residuals in alpha_summaries:
    print(alpha, [round(v, 6) for v in residuals])
"""
        ),
        code(
            """
plt.figure(figsize=(6, 3))
for alpha, residuals in alpha_summaries:
    plt.plot(residuals, marker="o", label=f"alpha={alpha}")
plt.yscale("log")
plt.xlabel("solver step")
plt.ylabel("residual")
plt.title("Effect of solver damping")
plt.legend()
plt.tight_layout()
"""
        ),
        md(
            r"""
## Image Cortex Preset

`SILVAImageCortexClassifier` packages the convolutional-retina path:

$$
u_0=C_\psi(x),
\qquad
z_1^\star
\xrightarrow{\tanh}
z_2^\star.
$$

The local branch can be the dynamic hidden-channel kNN term, and the global
branch can be per-sample channel attention. Both remain configurable.
"""
        ),
        code(
            """
image_model = SILVAImageCortexClassifier(
    in_channels=3,
    hidden_dim=[8, 6],
    num_classes=2,
    image_size=8,
    attention_mode="simple",
    graph_mode="GAT",
    k_neighbors=2,
    alphas=(0.5, 0.2),
    max_iter=3,
    internal_depth=2,
    self_interaction=True,
    dropout=0.0,
).to(device)

images = torch.randn(4, 3, 8, 8, device=device)
labels = torch.tensor([0, 1, 0, 1], device=device)
image_result = image_model(images, return_results=True)
image_loss = torch.nn.functional.cross_entropy(image_result.output, labels)
image_loss.backward()

print("image logits:", tuple(image_result.output.shape))
print("image states:", [tuple(state.shape) for state in image_result.states])
print("image solvers:", [r.solver for r in image_result.solver_results])
print("retina gradient:", image_model.retina.conv1.weight.grad is not None)
"""
        ),
        md(
            r"""
## What This Enables

The same grammar covers the article configurations and extensions:

| Need | Package control |
| --- | --- |
| convolutional front end | `SILVAImageCortexClassifier.retina` or custom `input_encoder` |
| many layers inside one point | `state_network=nn.Sequential(...)` or a module list |
| different point architectures | one `SILVACortexLayer` per point |
| different alphas | one `SolverConfig(alpha=...)` per point |
| different solvers | Picard, Anderson, or Broyden per point |
| local/global/self ablations | pass modules, omit modules, or use zero modules |
| user datasets | adapt to tensors, then call the same PyTorch modules |

## Citation

Dr. Jose Luis Silva. SILVA Networks. Version 1.2.1. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770098
"""
        ),
    ]
)


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_notebook(out_dir / NAME, NB, indent=1)
        print(out_dir / NAME)


if __name__ == "__main__":
    main()
