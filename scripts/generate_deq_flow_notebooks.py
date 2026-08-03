from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/implicit_bridge",
    ROOT / "docs/implicit-bridge-notebooks",
    ROOT / "colab/implicit_bridge",
]
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"flow-{_CELL_COUNTER:04d}"


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
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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


COMMON_IMPORTS = """
import numpy as np
import torch
import matplotlib.pyplot as plt

from silva_networks import resolve_device, SolverConfig

torch.manual_seed(11)
np.random.seed(11)
device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
device
"""


CITATION = r"""
## Citation and Sources

If this package or notebook is used, cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
```

When the work uses SILVA methodology, also cite:

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

Additional sources for this notebook:

- TorchDEQ: https://github.com/locuslab/torchdeq
- Deep Equilibrium Optical Flow Estimation: https://arxiv.org/abs/2204.08442
- DEQ-Flow: https://github.com/locuslab/deq-flow
- RAFT: https://arxiv.org/abs/2003.12039
- RAFT repository: https://github.com/princeton-vl/RAFT

The implementation used here is package-native and follows the package solver API.
"""


NOTEBOOKS = {
    "07_silva_deq_engine_torchdeq_bridge.ipynb": notebook(
        [
            md(
                r"""
# SILVA DEQ Engine: TorchDEQ-Style Interface

A general DEQ engine separates the transition from the solver:

$$
z^\star=f_\theta(z^\star,x).
$$

The transition can be any PyTorch callable whose output has the same structure
as its input state. The solver is controlled by `SILVADEQConfig`.
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Single-State System

Use an affine-tanh transition:

$$
f_\theta(z,x)=\tanh(W_xx+W_zz).
$$
"""
            ),
            code(
                """
from silva_networks import SILVADEQConfig, silva_deq, silva_residual_ratio

x = torch.randn(10, 3, device=device)
z0 = torch.zeros(10, 8, device=device)
input_proj = torch.nn.Linear(3, 8).to(device)
state_proj = torch.nn.Linear(8, 8, bias=False).to(device)
with torch.no_grad():
    state_proj.weight.mul_(0.2)

def transition(z):
    return torch.tanh(input_proj(x) + state_proj(z))

single = silva_deq(
    transition,
    z0,
    config=SILVADEQConfig(forward_solver="anderson", forward_max_iter=10, alpha=0.7, history=4),
    return_result=True,
)
single.state.shape, single.solver_result.iterations, single.solver_result.residual
"""
            ),
            code(
                """
plt.figure(figsize=(5.0, 3.0))
plt.plot(single.solver_result.residuals, marker="o")
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.title(f"ratio={silva_residual_ratio(single.solver_result.residuals):.2e}")
plt.tight_layout()
"""
            ),
            md(
                r"""
## Multi-State System

For a two-state equilibrium,

$$
a^\star=\tanh(W_xx+W_aa^\star+W_{ba}b^\star),
$$

$$
b^\star=\tanh(Pa^\star).
$$

The engine packs \((a,b)\), solves one vector fixed point, and unpacks the
result.
"""
            ),
            code(
                """
a0 = torch.zeros(10, 6, device=device)
b0 = torch.zeros(10, 4, device=device)
Wa = torch.nn.Linear(6, 6, bias=False).to(device)
Wb = torch.nn.Linear(4, 6, bias=False).to(device)
Px = torch.nn.Linear(3, 6).to(device)
P = torch.nn.Linear(6, 4).to(device)
with torch.no_grad():
    Wa.weight.mul_(0.15)
    Wb.weight.mul_(0.15)
    P.weight.mul_(0.15)

def multi_transition(state):
    a, b = state
    next_a = torch.tanh(Px(x) + Wa(a) + Wb(b))
    next_b = torch.tanh(P(a))
    return next_a, next_b

multi = silva_deq(
    multi_transition,
    (a0, b0),
    config=SILVADEQConfig(forward_solver="picard", forward_max_iter=12, alpha=0.6),
    return_result=True,
)
a_star, b_star = multi.state
a_star.shape, b_star.shape, multi.info
"""
            ),
            md(
                r"""
## Variational Dropout

Inside a fixed-point solve, a fixed dropout mask keeps the transition map
consistent:

$$
z_{k+1}=f_{\theta,\omega}(z_k,x).
$$
"""
            ),
            code(
                """
from silva_networks import SILVAVariationalDropout, reset_silva_deq

dropout = SILVAVariationalDropout(0.4).to(device)
dropout.train()
ones = torch.ones(4, 5, device=device)
first = dropout(ones)
second = dropout(ones)
reset_silva_deq(dropout)
third = dropout(ones)
torch.allclose(first, second), torch.allclose(first, third)
"""
            ),
            md(CITATION),
        ]
    ),
    "08_silva_optical_flow_deq_raft_bridge.ipynb": notebook(
        [
            md(
                r"""
# SILVA Optical Flow: RAFT and DEQ-Flow Bridge

RAFT introduced all-pairs correlation and recurrent flow refinement. DEQ-Flow
casts optical flow as a fixed-point solve. The SILVA package version is compact
and designed for tutorials and smoke tests:

$$
u^\star=T_\theta(u^\star,I_1,I_2).
$$
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Synthetic Translation Pair

The package generator creates two images and a known forward flow
\(u=(u_x,u_y)\). This keeps the notebook small enough for CPU and Colab.
"""
            ),
            code(
                """
from silva_networks import make_silva_translation_flow_batch

batch = make_silva_translation_flow_batch(
    batch_size=1,
    channels=1,
    height=12,
    width=12,
    shift=(0.75, 0.25),
    device=device,
)
batch.image1.shape, batch.image2.shape, batch.flow[:, :, 0, 0]
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(batch.image1[0, 0].detach().cpu(), cmap="gray")
axes[0].set_title("image1")
axes[1].imshow(batch.image2[0, 0].detach().cpu(), cmap="gray")
axes[1].set_title("image2")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
"""
            ),
            md(
                r"""
## Correlation and Local Lookup

Given feature maps \(F_1,F_2\),

$$
C_{i,j,k,\ell}
=
\frac{\langle F_{1,:,i,j},F_{2,:,k,\ell}\rangle}{\sqrt C}.
$$
"""
            ),
            code(
                """
from silva_networks import (
    SILVAFlowFeatureEncoder,
    silva_all_pairs_correlation,
    silva_local_correlation_lookup,
)

encoder = SILVAFlowFeatureEncoder(in_channels=1, feature_dim=4).to(device)
f1 = encoder(batch.image1)
f2 = encoder(batch.image2)
corr = silva_all_pairs_correlation(f1, f2)
local = silva_local_correlation_lookup(corr, torch.zeros_like(batch.flow), radius=1)
corr.shape, local.shape
"""
            ),
            md(
                r"""
## Flow Fixed Point

The transition has the form

$$
T_\theta(u)
=
u+\gamma\tanh\Delta_\theta(u,F_1,\tilde F_2,F_1-\tilde F_2,C[u]).
$$
"""
            ),
            code(
                """
from silva_networks import (
    silva_deq_flow,
    silva_endpoint_error,
    silva_flow_smoothness_loss,
)

model = silva_deq_flow(
    feature_dim=4,
    hidden_dim=10,
    corr_radius=1,
    update_scale=0.2,
    config=SolverConfig(solver="picard", max_iter=5, alpha=0.4),
).to(device)

result = model(batch.image1, batch.image2, return_result=True, return_correlation=True)
epe = silva_endpoint_error(result.flow, batch.flow, batch.valid)
smooth = silva_flow_smoothness_loss(result.flow)
float(epe.detach().cpu()), float(smooth.detach().cpu()), result.solver_result.residuals
"""
            ),
            code(
                """
plt.figure(figsize=(5.0, 3.0))
plt.plot(result.solver_result.residuals, marker="o")
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("flow fixed-point residual")
plt.tight_layout()
"""
            ),
            code(
                """
flow = result.flow[0].detach().cpu()
step = 3
y, x_grid = np.mgrid[0:flow.shape[1]:step, 0:flow.shape[2]:step]
u = flow[0, ::step, ::step].numpy()
v = flow[1, ::step, ::step].numpy()
plt.figure(figsize=(4.2, 4.2))
plt.imshow(batch.image1[0, 0].detach().cpu(), cmap="gray")
plt.quiver(x_grid, y, u, v, color="tab:cyan")
plt.axis("off")
plt.tight_layout()
"""
            ),
            md(
                r"""
## One Training Step

The model is an ordinary PyTorch module. A supervised smoke loss can combine
endpoint error and smoothness.
"""
            ),
            code(
                """
optim = torch.optim.Adam(model.parameters(), lr=1e-2)
optim.zero_grad()
train_result = model(batch.image1, batch.image2, return_result=True)
loss = silva_endpoint_error(train_result.flow, batch.flow, batch.valid)
loss = loss + 0.01 * silva_flow_smoothness_loss(train_result.flow)
loss.backward()
optim.step()
float(loss.detach().cpu()), model.update.net[-1].weight.grad is not None
"""
            ),
            md(CITATION),
        ]
    ),
}


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        for out_dir in OUT_DIRS:
            (out_dir / name).write_text(json.dumps(nb, indent=2) + "\n")
    print(f"Wrote {len(NOTEBOOKS)} notebooks into {len(OUT_DIRS)} locations.")


if __name__ == "__main__":
    main()
