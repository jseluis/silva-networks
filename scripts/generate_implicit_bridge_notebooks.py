from __future__ import annotations

import re
import textwrap
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "notebooks/implicit_bridge",
    ROOT / "docs/implicit-bridge-notebooks",
    ROOT / "colab/implicit_bridge",
]
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE
)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"implicit-{_CELL_COUNTER:04d}"


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
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
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
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from silva_networks import resolve_device, SolverConfig

torch.manual_seed(7)
np.random.seed(7)
device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
device
"""


CITATION = r"""
## Citation and Sources

If this package or notebook is used, cite the software repository:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.2.2. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770098
```

When the work is connected to the SILVA methodology, cite the SILVA Networks
paper as well:

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

Background sources:

- Deep Implicit Layers tutorial: https://implicit-layers-tutorial.org/
- LocusLab DEQ repository: https://github.com/locuslab/deq
- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- Multiscale Deep Equilibrium Models: https://arxiv.org/abs/2006.08656
- Stabilizing Equilibrium Models by Jacobian Regularization: https://arxiv.org/abs/2106.14342

The notebook is adapted to the `silva_networks` public API. It links to the
sources above and keep the examples package-native.
"""


NOTEBOOKS = {
    "01_introduction_fixed_points.ipynb": notebook(
        [
            md(
                r"""
# Introduction: Fixed Points as Layers

The entry point is the equation

$$
z^\star=f_\theta(z^\star,x).
$$

The symbol \(x\) is the external stimulus and \(z^\star\) is the hidden state
that is consistent with the transition \(f_\theta\). A finite neural network
layer applies a map once. A deep equilibrium layer asks for the state that would
remain unchanged if the map were applied again.

This notebook follows the introduction theme from the Deep Implicit Layers
tutorial and runs it through the `silva_networks` solver API.
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Scalar Warm-Up

Start with a scalar contraction:

$$
f(z)=\tanh(az+b),\qquad |a|<1.
$$

The Picard step with damping is

$$
z_{k+1}=(1-\alpha)z_k+\alpha f(z_k).
$$

The residual at step \(k\) is

$$
r_k=|f(z_k)-z_k|.
$$
"""
            ),
            code(
                """
from silva_networks import fixed_point, silva_residual_ratio

a = torch.tensor(0.55, device=device)
b = torch.tensor(0.25, device=device)
z0 = torch.zeros((), device=device)

def scalar_f(z):
    return torch.tanh(a * z + b)

configs = [
    SolverConfig(solver="picard", max_iter=25, alpha=0.8),
    SolverConfig(solver="anderson", max_iter=12, alpha=0.8, history=4),
    SolverConfig(solver="broyden", max_iter=12, alpha=0.8),
]

solves = {cfg.solver: fixed_point(scalar_f, z0, cfg) for cfg in configs}
[(name, float(result.z.detach().cpu()), result.iterations, result.residual) for name, result in solves.items()]
"""
            ),
            code(
                """
plt.figure(figsize=(5.5, 3.2))
for name, result in solves.items():
    plt.plot(result.residuals, marker="o", label=f"{name}, ratio={silva_residual_ratio(result.residuals):.2e}")
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.legend()
plt.tight_layout()
"""
            ),
            md(
                r"""
## Package Fixed-Point Block

The scalar map becomes a vector map by replacing \(a\) with \(W_z\), \(b\) with
an input-dependent term, and solving

$$
z^\star=\tanh(W_z z^\star + W_x x + b).
$$

`silva_fixed_point_block` exposes this equation directly.
"""
            ),
            code(
                """
from silva_networks import silva_fixed_point_block

x = torch.randn(8, 3, device=device)
block = silva_fixed_point_block(
    in_dim=3,
    state_dim=6,
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.7, history=3),
).to(device)

result = block(x, return_result=True)
result.z.shape, result.iterations, result.residuals[:3], result.residual
"""
            ),
            md(
                r"""
## A Tiny Classifier

The equilibrium state can feed an ordinary PyTorch head:

$$
\hat y = W_o z^\star + c.
$$

The code below uses a synthetic two-class task so the notebook is quick on CPU
and also runs unchanged on a Colab GPU runtime.
"""
            ),
            code(
                """
from silva_networks import silva_fixed_point_classifier

x_train = torch.randn(32, 4, device=device)
y_train = (x_train[:, 0] + 0.5 * x_train[:, 1] > 0).long()
model = silva_fixed_point_classifier(
    in_features=4,
    state_dim=12,
    num_classes=2,
    config=SolverConfig(solver="picard", max_iter=8, alpha=0.6),
).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.03)

losses = []
for _ in range(8):
    optim.zero_grad()
    logits = model(x_train)
    loss = torch.nn.functional.cross_entropy(logits, y_train)
    loss.backward()
    optim.step()
    losses.append(float(loss.detach().cpu()))

losses[:3], losses[-1]
"""
            ),
            code(
                """
plt.figure(figsize=(4.8, 3.0))
plt.plot(losses, marker="o")
plt.xlabel("training step")
plt.ylabel("cross entropy")
plt.tight_layout()
"""
            ),
            md(CITATION),
        ]
    ),
    "02_implicit_autodiff.ipynb": notebook(
        [
            md(
                r"""
# Implicit Functions and Automatic Differentiation

Let \(z^\star\) solve

$$
F(z^\star,\theta)=f_\theta(z^\star,x)-z^\star=0.
$$

Differentiate \(F=0\):

$$
\frac{\partial f_\theta}{\partial z}\,dz
+\frac{\partial f_\theta}{\partial \theta}\,d\theta
-dz=0.
$$

Collect the \(dz\) terms:

$$
(I-J_f)\,dz=\frac{\partial f_\theta}{\partial \theta}\,d\theta.
$$

Reverse-mode differentiation solves the transposed linear system

$$
(I-J_f^\top)u=g,
$$

where \(g=\partial \ell/\partial z^\star\).
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Materialized Jacobian on a Small State

For a tiny state it is useful to materialize \(J_f\). This makes the
matrix-free `vjp` and `jvp` calls easy to verify.
"""
            ),
            code(
                """
from silva_networks import (
    SILVAImplicitTransition,
    fixed_point,
    full_jacobian,
    vjp,
    jvp,
    implicit_adjoint_solve,
)

transition = SILVAImplicitTransition(2, 2, spectral_scale=0.35).to(device)
x = torch.tensor([[0.3, -0.5]], device=device)
z0 = torch.zeros(1, 2, device=device)

def f(z):
    return transition(z, x)

solve = fixed_point(f, z0, SolverConfig(max_iter=20, alpha=0.7))
J = full_jacobian(f, solve.z)
J
"""
            ),
            code(
                """
probe = torch.randn_like(solve.z)
_, jvp_value = jvp(f, solve.z, probe)
vjp_value = vjp(f, solve.z, probe)

checks = {
    "Jv_close": torch.allclose(J @ probe.reshape(-1), jvp_value.reshape(-1), atol=1e-5),
    "JT_v_close": torch.allclose(J.T @ probe.reshape(-1), vjp_value.reshape(-1), atol=1e-5),
}
checks
"""
            ),
            md(
                r"""
## Adjoint Solve

The explicit matrix solution is

$$
u=(I-J_f^\top)^{-1}g.
$$

The package helper computes the same object with VJP-backed GMRES.
"""
            ),
            code(
                """
g = torch.tensor([[1.0, -0.25]], device=device)
explicit_u = torch.linalg.solve(torch.eye(2, device=device) - J.T, g.reshape(-1)).reshape_as(g)
gmres_u = implicit_adjoint_solve(f, solve.z, g, max_iter=8, tol=1e-7)

explicit_u.detach().cpu().numpy(), gmres_u.x.detach().cpu().numpy(), gmres_u.residuals
"""
            ),
            md(
                r"""
## Jacobian Spectrum

The spectral radius \(\rho(J_f)\) is a local fixed-point diagnostic. A value
below one is compatible with local contraction. A value near or above one means
the solver may need damping, a different solver, or regularization.
"""
            ),
            code(
                """
from silva_networks import spectral_radius, hutchinson_jacobian_norm

rho = spectral_radius(f, solve.z, iters=12)
fro = hutchinson_jacobian_norm(f, solve.z, samples=2, squared=False)
float(rho), float(fro.detach().cpu())
"""
            ),
            md(CITATION),
        ]
    ),
    "03_neural_odes_as_implicit_layers.ipynb": notebook(
        [
            md(
                r"""
# Neural ODEs as a Bridge to Implicit Layers

A neural ODE represents hidden evolution by

$$
\frac{dh(t)}{dt}=v_\theta(h(t),t).
$$

With explicit Euler and a time-independent vector field,

$$
h_{k+1}=h_k+\Delta t\,v_\theta(h_k).
$$

This is not a DEQ solve by itself, but it gives a useful bridge: both ODEs and
DEQs describe computation through an operator applied repeatedly. ODEs track a
finite-time path; DEQs solve for a time-independent equilibrium.
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## A Rotating-Damped Vector Field

For a hand-checkable example, let

$$
v(h)=Ah,
\qquad
A=\begin{bmatrix}-0.15 & -1\\ 1 & -0.15\end{bmatrix}.
$$

The first Euler update is

$$
h_1
=h_0+\Delta t Ah_0.
$$

With \(h_0=(1,0)\) and \(\Delta t=0.1\),

$$
Ah_0=(-0.15,1),
\qquad
h_1=(0.985,0.1).
$$
"""
            ),
            code(
                """
from silva_networks import SILVAEulerFlowBlock

class LinearVectorField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor([[-0.15, -1.0], [1.0, -0.15]]), requires_grad=False)

    def forward(self, h):
        return h @ self.A.T

ode = SILVAEulerFlowBlock(2, steps=50, step_size=0.1, vector_field=LinearVectorField()).to(device)
h0 = torch.tensor([[1.0, 0.0]], device=device)
terminal, trajectory = ode(h0, return_trajectory=True)
trajectory_xy = trajectory[:, 0].detach().cpu().numpy()
trajectory_xy[:3], terminal.detach().cpu().numpy()
"""
            ),
            code(
                """
plt.figure(figsize=(4.2, 4.0))
plt.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], marker="o", markersize=2)
plt.scatter([trajectory_xy[0, 0]], [trajectory_xy[0, 1]], label="start")
plt.scatter([trajectory_xy[-1, 0]], [trajectory_xy[-1, 1]], label="end")
plt.axis("equal")
plt.xlabel("h1")
plt.ylabel("h2")
plt.legend()
plt.tight_layout()
"""
            ),
            md(
                r"""
## Trainable ODE Block

`SILVAEulerFlowBlock` accepts a trainable vector field. It is an ordinary
PyTorch module, so gradients flow through every Euler step.
"""
            ),
            code(
                """
x = torch.randn(16, 3, device=device)
y = (x[:, 0] > 0).long()
feature = torch.nn.Linear(3, 8).to(device)
flow = SILVAEulerFlowBlock(8, hidden_dim=16, steps=5, step_size=0.15).to(device)
head = torch.nn.Linear(8, 2).to(device)
optim = torch.optim.Adam(list(feature.parameters()) + list(flow.parameters()) + list(head.parameters()), lr=0.02)

losses = []
for _ in range(6):
    optim.zero_grad()
    h = torch.tanh(feature(x))
    logits = head(flow(h))
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optim.step()
    losses.append(float(loss.detach().cpu()))

losses
"""
            ),
            code(
                """
plt.figure(figsize=(4.8, 3.0))
plt.plot(losses, marker="o")
plt.xlabel("training step")
plt.ylabel("loss")
plt.tight_layout()
"""
            ),
            md(CITATION),
        ]
    ),
    "04_deq_and_silva.ipynb": notebook(
        [
            md(
                r"""
# Deep Equilibrium Models and SILVA

The DEQ baseline in this notebook solves

$$
z^\star=\tanh(W_z z^\star+W_x x+b).
$$

SILVA keeps the same fixed-point idea but splits the transition into named
branches:

$$
z^\star
=\sigma\{S_\theta(x)+H_\theta(z^\star)+L_\theta(z^\star)+G_\theta(z^\star)\}.
$$

The names are operational:

- \(S_\theta\): stimulus from data.
- \(H_\theta\): optional learned self-interaction inside the transition.
- \(L_\theta\): local interaction, for example graph messages or dynamic kNN.
- \(G_\theta\): global context, for example mean field or bounded attention.
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## DEQ MLP Baseline

This mirrors the compact DEQ experiment: solve one equilibrium state and train a
linear readout on top.
"""
            ),
            code(
                """
from silva_networks import silva_fixed_point_classifier

x = torch.randn(40, 6, device=device)
y = (x[:, :3].sum(dim=1) > x[:, 3:].sum(dim=1)).long()
deq_model = silva_fixed_point_classifier(
    in_features=6,
    state_dim=16,
    num_classes=2,
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.6, history=4),
).to(device)

out = deq_model(x, return_result=True)
loss = torch.nn.functional.cross_entropy(out.output, y)
loss.backward()
out.output.shape, out.solver_result.solver, out.solver_result.residual
"""
            ),
            md(
                r"""
## SILVA Graph Variant

Now convert the same rows into graph entities. A k-nearest-neighbor graph
defines local messages, while a mean-field or attention term supplies global
context.
"""
            ),
            code(
                """
from silva_networks import SILVAGraphNetwork, make_knn_edge_index

edge_index = make_knn_edge_index(x.detach(), k=4, metric="cosine", undirected=True, device=device)
silva = SILVAGraphNetwork(
    in_dim=6,
    hidden_dims=[16, 16],
    out_dim=2,
    task="node",
    local=["graph", "gat"],
    global_term=["mean", "topk"],
    self_term=["none", "linear"],
    local_kwargs=[{}, {"heads": 4, "add_self_loops": True}],
    global_kwargs=[{}, {"k": 6}],
    config=[
        SolverConfig(solver="picard", max_iter=6, alpha=0.55),
        SolverConfig(solver="anderson", max_iter=6, alpha=0.45, history=3),
    ],
).to(device)

result = silva(x, edge_index=edge_index, return_results=True)
silva_loss = torch.nn.functional.cross_entropy(result.output, y)
silva_loss.backward()
result.output.shape, result.state.shape, [r.solver for r in result.solver_results]
"""
            ),
            md(
                r"""
## Residual Diagnostics

Both models are controlled by `SolverConfig`. The solver choice, damping,
history, tolerance, and iteration budget can be changed per layer.
"""
            ),
            code(
                """
plt.figure(figsize=(5.5, 3.2))
plt.plot(out.solver_result.residuals, marker="o", label="DEQ MLP")
for index, solver_result in enumerate(result.solver_results):
    plt.plot(solver_result.residuals, marker="o", label=f"SILVA layer {index}")
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("residual")
plt.legend()
plt.tight_layout()
"""
            ),
            md(CITATION),
        ]
    ),
    "05_differentiable_optimization.ipynb": notebook(
        [
            md(
                r"""
# Differentiable Optimization

An optimization problem can be a layer. For a quadratic objective

$$
\phi(z,x)=\frac12 z^\top A z-b_\theta(x)^\top z,
\qquad A=L L^\top+\lambda I,
$$

the stationarity condition is

$$
\nabla_z\phi(z,x)=Az-b_\theta(x)=0.
$$

The solution can be obtained directly by \(Az=b_\theta(x)\), or by applying the
fixed-point map

$$
T(z)=z-\eta(Az-b_\theta(x)).
$$
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Exact Solve and Fixed-Point Solve

The package layer exposes both the direct linear solve and the iterative
fixed-point version. This makes it easy to debug the optimization layer before
using it inside a larger model.
"""
            ),
            code(
                """
from silva_networks import silva_quadratic_optimization_layer

layer = silva_quadratic_optimization_layer(
    in_dim=3,
    state_dim=2,
    ridge=1.5,
    step_size=0.2,
    config=SolverConfig(max_iter=60, alpha=1.0, tol=1e-7),
    reengage=False,
).to(device)
x = torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.4, 0.7]], device=device)
iterative = layer(x)
exact = layer.exact_solution(x)
torch.max(torch.abs(iterative - exact)).detach().cpu()
"""
            ),
            md(
                r"""
## Energy Decrease

The fixed-point map is gradient descent on \(\phi\). The sequence below records
the objective value by hand, one update at a time.
"""
            ),
            code(
                """
z = torch.zeros_like(exact)
energies = []
for _ in range(25):
    energies.append(float(layer.energy(z, x).mean().detach().cpu()))
    z = layer.transition(z, x)
energies.append(float(layer.energy(z, x).mean().detach().cpu()))
energies[:3], energies[-1]
"""
            ),
            code(
                """
plt.figure(figsize=(5.0, 3.0))
plt.plot(energies, marker="o")
plt.xlabel("gradient step")
plt.ylabel("mean quadratic energy")
plt.tight_layout()
"""
            ),
            md(
                r"""
## Gradients Through the Layer

The objective parameters are PyTorch parameters. A downstream loss can update
the input-to-\(b\) map and the positive-definite matrix factor.
"""
            ),
            code(
                """
target = torch.zeros_like(iterative)
optim = torch.optim.Adam(layer.parameters(), lr=0.01)
optim.zero_grad()
solution = layer(x)
loss = (solution - target).square().mean()
loss.backward()
optim.step()

float(loss.detach().cpu()), layer.b_proj.weight.grad is not None, layer.factor.grad is not None
"""
            ),
            md(CITATION),
        ]
    ),
    "06_mdeq_jacobian_regularization.ipynb": notebook(
        [
            md(
                r"""
# Multiscale Equilibria and Jacobian Regularization

A multiscale equilibrium solves several coupled states at once. In a compact
two-scale notation,

$$
z^\star=(z_\ell^\star,z_h^\star),
\qquad
z^\star=f_\theta(z^\star,x).
$$

The tutorial block uses

$$
z_\ell^+=\tanh(S_\ell(x)+A_{\ell\ell}z_\ell+A_{h\ell}z_h),
$$

$$
z_h^+=\tanh(S_h(x)+A_{hh}z_h+A_{\ell h}z_\ell).
$$

Jacobian regularization adds a penalty such as

$$
\lambda_J\|J_f(z^\star)\|_F^2
$$

to encourage stable local dynamics.
"""
            ),
            code(BOOTSTRAP),
            code(COMMON_IMPORTS),
            md(
                r"""
## Two Coupled Scales

The state is stored as one tensor for solver compatibility, then split into
low and high scales when the transition is evaluated.
"""
            ),
            code(
                """
from silva_networks import (
    silva_jacobian_regularization_loss,
    silva_multiscale_deq_block,
    stability_report,
)

x = torch.randn(6, 4, device=device)
block = silva_multiscale_deq_block(
    in_dim=4,
    low_dim=3,
    high_dim=5,
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.6, history=3),
).to(device)

result = block(x, return_result=True)
z_low, z_high = block.split_state(result.z)
result.z.shape, z_low.shape, z_high.shape, result.residual
"""
            ),
            md(
                r"""
## Hutchinson Estimate

For a random Rademacher vector \(v\),

$$
\mathbb E\|J_f^\top v\|_2^2=\|J_f\|_F^2.
$$

The estimator uses VJP calls and does not materialize the full Jacobian.
"""
            ),
            code(
                """
penalty = silva_jacobian_regularization_loss(
    lambda z: block.transition(z, x),
    result.z,
    samples=2,
    squared=True,
    weight=0.01,
)
penalty.detach().cpu()
"""
            ),
            md(
                r"""
## Stability Report

The residual and spectral radius summarize the fixed point locally:

$$
\rho(J_f(z^\star))<1
$$

is the familiar contraction-style target for stable Picard dynamics.
"""
            ),
            code(
                """
report = stability_report(lambda z: block.transition(z, x), result.z, samples=1, iters=6)
report
"""
            ),
            md(
                r"""
## Training Loss with a Jacobian Penalty

This cell separates the task loss from the regularizer so the effect of the
penalty is visible in the output.
"""
            ),
            code(
                """
head = torch.nn.Linear(block.state_dim, 2).to(device)
y = torch.randint(0, 2, (x.shape[0],), device=device)
optim = torch.optim.Adam(list(block.parameters()) + list(head.parameters()), lr=0.01)

losses = []
penalties = []
for _ in range(4):
    optim.zero_grad()
    solve = block(x, return_result=True)
    logits = head(solve.z)
    task_loss = torch.nn.functional.cross_entropy(logits, y)
    jac_loss = silva_jacobian_regularization_loss(lambda z: block.transition(z, x), solve.z, samples=1, weight=0.01)
    loss = task_loss + jac_loss
    loss.backward()
    optim.step()
    losses.append(float(task_loss.detach().cpu()))
    penalties.append(float(jac_loss.detach().cpu()))

losses, penalties
"""
            ),
            code(
                """
plt.figure(figsize=(5.2, 3.0))
plt.plot(losses, marker="o", label="task loss")
plt.plot(penalties, marker="s", label="Jacobian penalty")
plt.xlabel("training step")
plt.legend()
plt.tight_layout()
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
            write_notebook(out_dir / name, nb)
    print(f"Wrote {len(NOTEBOOKS)} notebooks into {len(OUT_DIRS)} locations.")


if __name__ == "__main__":
    main()
