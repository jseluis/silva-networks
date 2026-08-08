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
NAME = "08_equation_to_code_walkthrough.ipynb"
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE
)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"walk-{_CELL_COUNTER:04d}"


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


NB = notebook(
    [
        md(
            r"""
# Equation-to-Code Walkthrough

This notebook follows one thread from the full SILVA equation to executable
package objects:

$$
z^\star
=
\Psi_\theta\!\left(
S_\theta(x)
+H_\theta(a(z^\star))
+L_\theta(a(z^\star),E)
+G_\theta(a(z^\star),b)
\right).
$$

By setting terms to zero or choosing different operators, the same package API
recovers compact DEQ layers, message-passing DEQs, and full SILVA graph layers.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
import torch
import matplotlib.pyplot as plt

from silva_networks import (
    SILVAGraphNetwork,
    SolverConfig,
    full_jacobian,
    make_knn_edge_index,
    resolve_device,
    silva_deq_reduction_layer,
    silva_message_passing_reduction_layer,
    stability_report,
)

torch.manual_seed(8)
device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
device
"""
        ),
        md(
            r"""
## Step 1: Compact DEQ as a SILVA Reduction

Start from the full equation and set

$$
L_\theta=0,\qquad G_\theta=0,\qquad a(z)=z,\qquad
H_\theta(z)=W_z z.
$$

With \(S_\theta(x)=W_xx+b\) and \(\Psi=\tanh\), substitution gives

$$
z^\star
=
\tanh(W_xx+b+W_z z^\star).
$$
"""
        ),
        code(
            """
x = torch.randn(12, 4, device=device)
deq_layer = silva_deq_reduction_layer(
    in_dim=4,
    hidden_dim=6,
    config=SolverConfig(solver="anderson", max_iter=12, alpha=0.6, history=4),
).to(device)

z = torch.randn(12, 6, device=device)
manual = torch.tanh(deq_layer.stimulus(x) + deq_layer.self_term(z))
from_layer = deq_layer.f(z, x)
float((manual - from_layer).abs().max().detach().cpu())
"""
        ),
        code(
            """
deq_result = deq_layer(x, return_result=True)
deq_result.z.shape, deq_result.iterations, deq_result.residual
"""
        ),
        md(
            r"""
## Step 2: Message Passing as a Local SILVA Field

Keep a local operator and remove the learned self and global branches:

$$
z^\star=\Psi_\theta(S_\theta(x)+L_\theta(a(z^\star),E)).
$$

For a graph, \(E\) is encoded by `edge_index`. The first row stores sources and
the second row stores destinations.
"""
        ),
        code(
            """
edge_index = make_knn_edge_index(x, k=3, undirected=True, device=device)
mp_layer = silva_message_passing_reduction_layer(
    in_dim=4,
    hidden_dim=6,
    local="gat",
    local_kwargs={"heads": 2},
    config=SolverConfig(solver="picard", max_iter=8, alpha=0.5),
).to(device)

mp_result = mp_layer(x, edge_index=edge_index, return_result=True)
mp_result.z.shape, edge_index.shape, mp_result.residual
"""
        ),
        md(
            r"""
## Step 3: Turn the Full SILVA Field Back On

A stack can use different branch choices and solver settings per layer:

$$
z_{\ell}^{\star}
=
\Psi_{\theta_\ell}
\left(
S_{\theta_\ell}(x_\ell)
+H_{\theta_\ell}
+L_{\theta_\ell}
+G_{\theta_\ell}
\right).
$$
"""
        ),
        code(
            """
model = SILVAGraphNetwork(
    in_dim=4,
    hidden_dims=[10, 8],
    out_dim=3,
    task="node",
    local=["graph", "gat"],
    global_term=["mean", "topk"],
    self_term=["none", "linear"],
    local_kwargs=[None, {"heads": 2}],
    global_kwargs=[None, {"k": 4}],
    config=[
        SolverConfig(solver="picard", max_iter=6, alpha=0.5),
        SolverConfig(solver="anderson", max_iter=8, alpha=0.35, history=4),
    ],
).to(device)

out = model(x, edge_index=edge_index, return_results=True)
out.output.shape, out.state.shape, [r.solver for r in out.solver_results]
"""
        ),
        md(
            r"""
## Step 4: Diagnose the Equilibrium

The residual is

$$
\|f_\theta(z^\star,x)-z^\star\|_2.
$$

For small states, the full Jacobian can be materialized. For larger states,
matrix-free `vjp` and `jvp` calls are preferred.
"""
        ),
        code(
            """
small_x = x[:2]
small_layer = silva_deq_reduction_layer(
    in_dim=4,
    hidden_dim=3,
    config=SolverConfig(solver="picard", max_iter=10, alpha=0.5),
).to(device)
small_result = small_layer(small_x, return_result=True)

J = full_jacobian(lambda state: small_layer.f(state, small_x), small_result.z)
report = stability_report(
    lambda state: small_layer.f(state, small_x),
    small_result.z,
    samples=2,
    iters=4,
)
J.shape, report
"""
        ),
        code(
            """
plt.figure(figsize=(5, 3))
plt.plot(deq_result.residuals, marker="o", label="DEQ reduction")
plt.plot(mp_result.residuals, marker="o", label="message-passing reduction")
for index, solver_result in enumerate(out.solver_results):
    plt.plot(solver_result.residuals, marker="o", label=f"full SILVA layer {index}")
plt.yscale("log")
plt.xlabel("solver step")
plt.ylabel("residual")
plt.legend()
plt.tight_layout()
"""
        ),
        md(
            r"""
## Citation

If this notebook or package is used, cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.2.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770098
```

When the work is connected to the SILVA Networks methodology, cite the SILVA
Networks paper as well:

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

Related sources:

- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- TorchDEQ: https://github.com/locuslab/torchdeq
"""
        ),
    ]
)


def main() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_notebook(out_dir / NAME, NB)
    print(f"Wrote {NAME} into {len(OUT_DIRS)} locations.")


if __name__ == "__main__":
    main()
