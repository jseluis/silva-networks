from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
]
NAME = "09_family_selector_and_projected_qp.ipynb"
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"family-{_CELL_COUNTER:04d}"


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
# Family Selector and Projected QP Smoke Tutorial

This notebook checks two package design promises:

1. SILVA implementations can be selected by family name.
2. The projected quadratic-program layer solves real constrained fixed-point
   problems with gradients.

The shared equilibrium contract is

$$
z^\star=f_\theta(z^\star,x),
\qquad
r(z^\star,x)=f_\theta(z^\star,x)-z^\star.
$$

For the projected QP case, the state is an optimizer variable:

$$
z_i^\star
=
\arg\min_{z\in C}
\frac12 z^\top A z-b_i^\top z,
\qquad
A=L L^\top+\lambda I.
$$

The fixed-point map is

$$
T(z)=\Pi_C[z-\eta(Az-b_i)].
$$
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import torch
import matplotlib.pyplot as plt

from silva_networks import (
    SILVADEQFlow,
    SILVAProjectedQPLayer,
    SolverConfig,
    available_silva_families,
    make_silva_translation_flow_batch,
    resolve_device,
    silva_deq_flow,
    silva_endpoint_error,
    silva_equilibrium_model,
    silva_family_description,
    silva_projected_qp_layer,
)

torch.manual_seed(9)
device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
device
"""
        ),
        md(
            r"""
## Selectable Families

The canonical public names favor SILVA-style entry points. Compatibility aliases
such as `"optical_flow_deq"` and `"constrained_quadratic_optimization"` still
resolve to the same package-native implementations.
"""
        ),
        code(
            """
for name in available_silva_families():
    print(f"{name:28s} -> {silva_family_description(name)}")
"""
        ),
        code(
            """
flow_from_alias = silva_equilibrium_model(
    "optical_flow_deq",
    feature_dim=2,
    hidden_dim=4,
    config=SolverConfig(solver="picard", max_iter=1, alpha=0.4),
)
qp_from_alias = silva_equilibrium_model(
    "constrained_quadratic_optimization",
    in_dim=3,
    state_dim=4,
    constraint="simplex",
    config=SolverConfig(solver="picard", max_iter=3, alpha=1.0),
)

isinstance(flow_from_alias, SILVADEQFlow), isinstance(qp_from_alias, SILVAProjectedQPLayer)
"""
        ),
        md(
            r"""
## Simplex Projection

The simplex constraint is

$$
\Delta_m=\{z\in\mathbb R^d:z_j\ge 0,\ \sum_jz_j=m\}.
$$

After the solve, every row should be nonnegative and should sum to `m`.
"""
        ),
        code(
            """
x = torch.randn(8, 3, device=device)
simplex_layer = silva_projected_qp_layer(
    in_dim=3,
    state_dim=4,
    constraint="simplex",
    simplex_mass=1.0,
    step_size=0.08,
    config=SolverConfig(solver="picard", max_iter=30, alpha=1.0, tol=1e-7),
).to(device)

simplex_result = simplex_layer(x, return_result=True)
simplex_energy = simplex_layer.energy(simplex_result.z, x).mean()
simplex_energy.backward()

print("shape:", tuple(simplex_result.z.shape))
print("row sums:", simplex_result.z.sum(dim=-1).detach().cpu().round(decimals=6).tolist())
print("minimum entry:", float(simplex_result.z.min().detach().cpu()))
print("projected residual:", simplex_result.residual)
print("gradient reached B:", simplex_layer.b_proj.weight.grad is not None)
"""
        ),
        code(
            """
plt.figure(figsize=(5, 3))
plt.plot(simplex_result.residuals, marker="o")
plt.yscale("log")
plt.xlabel("solver step")
plt.ylabel("projected residual")
plt.title("SILVA projected QP residual")
plt.tight_layout()
"""
        ),
        md(
            r"""
## Box and Affine Constraints

For a box,

$$
C=[\ell,u]^d.
$$

For an affine equality,

$$
C=\{z:A_{\rm eq}z=b_{\rm eq}\}.
$$

The smoke checks below verify the constraints directly from the returned
states.
"""
        ),
        code(
            """
box_layer = silva_projected_qp_layer(
    in_dim=3,
    state_dim=5,
    constraint="box",
    lower_bound=-0.25,
    upper_bound=0.25,
    step_size=0.08,
    config=SolverConfig(solver="picard", max_iter=20, alpha=1.0),
).to(device)
z_box = box_layer(x)
print(float(z_box.min().detach().cpu()), float(z_box.max().detach().cpu()))

Aeq = torch.tensor([[1.0, 1.0, 0.0]], device=device)
beq = torch.tensor([1.0], device=device)
affine_layer = silva_projected_qp_layer(
    in_dim=3,
    state_dim=3,
    constraint="affine",
    equality_matrix=Aeq,
    equality_rhs=beq,
    step_size=0.08,
    config=SolverConfig(solver="picard", max_iter=20, alpha=1.0),
).to(device)
z_affine = affine_layer(x)
print((z_affine @ Aeq.T).detach().cpu().round(decimals=6).tolist())
"""
        ),
        md(
            r"""
## Flow Family Smoke Check

The flow model has a different state shape, but the same solver pattern:

$$
u^\star=T_\theta(u^\star,I_1,I_2).
$$

The package-native `silva_deq_flow` name keeps the SILVA convention, while the
documentation cites RAFT for all-pairs correlation and DEQ-Flow for the
equilibrium optical-flow framing.
"""
        ),
        code(
            """
batch = make_silva_translation_flow_batch(batch_size=1, height=6, width=6, device=device)
flow_model = silva_deq_flow(
    feature_dim=2,
    hidden_dim=4,
    config=SolverConfig(solver="picard", max_iter=2, alpha=0.4),
).to(device)
flow_result = flow_model(batch.image1, batch.image2, return_result=True)
epe = silva_endpoint_error(flow_result.flow, batch.flow, batch.valid)
print(tuple(flow_result.flow.shape), float(epe.detach().cpu()), flow_result.solver_result.residual)
"""
        ),
        md(
            r"""
## Citation

If this notebook or package is used, cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
```

When the work is connected to the SILVA Networks paper, cite the paper as well.

Additional method citations:

- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- TorchDEQ: https://github.com/locuslab/torchdeq
- RAFT: https://arxiv.org/abs/2003.12039
- Deep Equilibrium Optical Flow Estimation: https://arxiv.org/abs/2204.08442
- OptNet: https://arxiv.org/abs/1703.00443
- Differentiable Convex Optimization Layers: https://arxiv.org/abs/1910.12430
- CVXPYlayers: https://github.com/cvxpy/cvxpylayers
"""
        ),
    ]
)


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / NAME).write_text(json.dumps(NB, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NAME} to {len(OUT_DIRS)} locations")


if __name__ == "__main__":
    main()
