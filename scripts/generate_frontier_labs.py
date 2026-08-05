"""Generate focused dataset-backed notebooks for recent SILVA families."""

from __future__ import annotations

import textwrap
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
)
_CELL_COUNTER = 0


def _cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:04d}"


def md(prefix: str, source: str) -> dict[str, object]:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _cell_id(prefix),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(prefix: str, source: str) -> dict[str, object]:
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _cell_id(prefix),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
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


BOOTSTRAP = r"""
from pathlib import Path
import importlib.util
import subprocess
import sys

IN_HOSTED_RUNTIME = "google.colab" in sys.modules
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
elif IN_HOSTED_RUNTIME and importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


def fno_lab() -> dict[str, object]:
    p = "silva-fno-lab"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Fourier Equilibrium: PDE Dataset Lab

This lab develops an input-injected Fourier equilibrium from a periodic
elliptic equation, verifies the generated data against the equation, trains a
small SILVA model, and separates prediction error from equilibrium and
physical residuals.

The state is a field $v\in\mathbb R^{B\times C\times H\times W}$ and the
canonical SILVA family is `silva_fno_deq` [43].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                """
import torch
import matplotlib.pyplot as plt

from silva_networks import (
    SILVAFNODEQ,
    SolverConfig,
    make_periodic_elliptic_dataset,
    silva_equilibrium_model,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(170)
""",
            ),
            md(
                p,
                r"""
## 1. Exact Periodic Elliptic Data

On the unit torus, consider

$$
(-\Delta+m)u=f,
\qquad m>0.
$$

For Fourier wave vector $k$, differentiation gives

$$
\widehat{-\Delta u}(k)=|k|^2\widehat u(k).
$$

Therefore every Fourier coefficient has the exact solution

$$
\widehat u(k)
=\frac{\widehat f(k)}{|k|^2+m}.
$$

`make_periodic_elliptic_dataset` samples a low-mode forcing and evaluates this
formula. It is a deterministic equation dataset rather than a collection of
unverified random input/target pairs.
""",
            ),
            code(
                p,
                """
data = make_periodic_elliptic_dataset(
    samples=8,
    height=8,
    width=8,
    modes=2,
    mass=1.0,
    seed=17,
)
equation_error = data.equation_residual().abs().max()

assert data.forcing.shape == data.target.shape == (8, 1, 8, 8)
assert equation_error < 2e-5
print("forcing:", tuple(data.forcing.shape))
print("maximum dataset equation residual:", float(equation_error))
""",
            ),
            md(
                p,
                r"""
## 2. From a Fourier Layer to a SILVA Equilibrium

A retained-mode convolution has the form

$$
\mathcal K_\theta v
=\mathcal F^{-1}
\left(R_\theta(k)\mathcal F(v)(k)\right).
$$

One input-injected internal layer is

$$
v_{j+1}
=g+\sigma\left(W_jv_j+\mathcal K_jv_j+b_j\right),
\qquad g=P_\phi(f).
$$

After composing the internal layers into $B_\theta$, SILVA solves

$$
v^\star=B_\theta(v^\star,P_\phi(f)),
\qquad \widehat u=Q_\psi(v^\star).
$$

The forcing is the source branch, the pointwise channel map is the self branch,
and the spectral convolution is a global field interaction [31, 32, 43].
""",
            ),
            code(
                p,
                """
config = SolverConfig(
    solver="picard",
    max_iter=6,
    tol=1e-5,
    alpha=0.7,
    backward_mode="unrolled",
)
model = SILVAFNODEQ(
    in_channels=1,
    state_channels=4,
    out_channels=1,
    modes_height=2,
    modes_width=2,
    block_depth=1,
    state_scale=0.04,
    config=config,
)
initial = model(data.forcing[:2], return_result=True)

assert initial.output.shape == data.target[:2].shape
print("state:", tuple(initial.state.shape))
print("solver iterations:", initial.solver_result.iterations)
print("equilibrium residual:", initial.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 3. Three Errors, Three Questions

For prediction $\widehat u$, report at least

$$
\mathcal L_{\mathrm{data}}
=\frac1{BHW}\|\widehat u-u\|_2^2,
$$

$$
\epsilon_{\mathrm{eq}}
=\|B_\theta(v^\star,g)-v^\star\|,
$$

and the physical residual

$$
\epsilon_{\mathrm{PDE}}
=\|(-\Delta+m)\widehat u-f\|.
$$

The first measures target fit, the second measures numerical equilibrium, and
the third checks the governing equation. None can substitute for the others.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for epoch in range(5):
    optimizer.zero_grad()
    prediction = model(data.forcing[:6])
    loss = torch.nn.functional.mse_loss(prediction, data.target[:6])
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

evaluation = model(data.forcing[6:], return_result=True)
test_mse = torch.nn.functional.mse_loss(evaluation.output, data.target[6:])
test_pde = data.equation_residual(
    torch.cat([data.target[:6], evaluation.output], dim=0)
)[6:].square().mean().sqrt()

assert all(torch.isfinite(torch.tensor(losses)))
assert torch.isfinite(test_mse) and torch.isfinite(test_pde)
print("training losses:", losses)
print("held-out MSE:", float(test_mse))
print("held-out PDE residual:", float(test_pde))
print("held-out equilibrium residual:", evaluation.solver_result.residual)
""",
            ),
            code(
                p,
                """
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.3))
axes[0].imshow(data.forcing[6, 0], cmap="viridis")
axes[0].set_title("forcing")
axes[1].imshow(data.target[6, 0], cmap="coolwarm")
axes[1].set_title("exact field")
axes[2].imshow(evaluation.output[0, 0].detach(), cmap="coolwarm")
axes[2].set_title("SILVA field")
axes[3].plot(range(1, len(losses) + 1), losses, marker="o")
axes[3].set_yscale("log")
axes[3].set_xlabel("epoch")
axes[3].set_title("training loss")
for axis in axes[:3]:
    axis.set_xticks([])
    axis.set_yticks([])
fig.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Resolution Transfer Is a Testable Contract

The spectral weights are indexed by retained modes, not by the full grid size.
The same model can therefore accept another resolution. This checks shape
compatibility; discretization-invariant accuracy still requires training and
evaluation across resolutions.
""",
            ),
            code(
                p,
                """
fine = make_periodic_elliptic_dataset(
    samples=2,
    height=10,
    width=12,
    modes=2,
    mass=1.0,
    seed=18,
)
fine_result = model(fine.forcing, return_result=True)
assert fine_result.output.shape == (2, 1, 10, 12)
print("new resolution:", tuple(fine_result.output.shape[-2:]))
print("new-resolution residual:", fine_result.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 5. Factory and Extension Points

The canonical factory key keeps the model under SILVA. A custom readout may
enforce boundary values or decode several physical variables. A custom block
must preserve field shape and continue to inject the forcing during every
tied application.
""",
            ),
            code(
                p,
                """
factory_model = silva_equilibrium_model(
    "silva_fno_deq",
    in_channels=1,
    state_channels=3,
    out_channels=1,
    modes_height=2,
    modes_width=2,
    config=SolverConfig(max_iter=4, alpha=0.7),
)
factory_output = factory_model(data.forcing[:1])
assert factory_output.shape == data.target[:1].shape
print(type(factory_model).__name__)
""",
            ),
            md(
                p,
                r"""
## 6. Practical Guidance

| Symptom | Check first | Typical response |
|---|---|---|
| equilibrium residual stalls | state scale and solver damping | reduce recurrent scale or damping step |
| PDE residual is high but MSE is low | physical discretization and loss | add equation-aware validation or training |
| fine-grid error grows | train/test resolution distribution | train across grids and report each resolution |
| boundary artifacts appear | boundary representation | use padding, coordinate channels, or an explicit boundary map |

The compact dataset validates implementation and teaching claims. Large Darcy
or steady Navier-Stokes comparisons require the exact benchmark split,
normalization, architecture scale, and reporting protocol used by the study
[43].
""",
            ),
        ]
    )


def graph_lab() -> dict[str, object]:
    p = "silva-graph-transport-lab"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Physics Graph Equilibrium: Transport Dataset Lab

This lab derives graph diffusion and directed transport, verifies a generated
steady-state dataset, trains a node-level SILVA equilibrium, and tests node
relabeling. The canonical family is `silva_physics_graph_deq` [44].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                """
import torch
import matplotlib.pyplot as plt

from silva_networks import (
    SILVAPhysicsGuidedGraphDEQ,
    SolverConfig,
    graph_convection_diffusion,
    make_graph_transport_dataset,
    silva_equilibrium_model,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(180)
""",
            ),
            md(
                p,
                r"""
## 1. Continuous and Discrete Transport

A steady convection-diffusion-reaction equation can be written

$$
0=s+\gamma_r u+\gamma_d\Delta u
-\gamma_a\mathbf v\cdot\nabla u-u.
$$

For incoming edge $i\rightarrow j$, SILVA uses

$$
(\mathcal L_GZ)_j
=\frac1{d_j}\sum_{i\rightarrow j}
w_{ij}(Z_i-Z_j),
$$

$$
(\nabla_VZ)_j
=\frac1{d_j}\sum_{i\rightarrow j}
v_{ij}(Z_j-Z_i).
$$

The first is symmetric when conductances and reverse edges agree. The second
retains orientation through the signed edge velocity [44].
""",
            ),
            code(
                p,
                """
data = make_graph_transport_dataset(
    samples=4,
    nodes=10,
    reaction_scale=0.05,
    diffusion_scale=0.2,
    advection_scale=0.05,
    seed=18,
)
physical_error = data.equation_residual().abs().max()

assert data.x.shape == (40, 3)
assert data.edge_index.shape == (2, 80)
assert physical_error < 2e-6
print("nodes:", data.x.shape[0])
print("graphs:", int(data.batch.max()) + 1)
print("maximum dataset equation residual:", float(physical_error))
""",
            ),
            md(
                p,
                r"""
## 2. SILVA Branch Equation

Node observations $X$ enter the source branch. Reaction, diffusion, and
advection receive distinct learned channel maps:

$$
\begin{aligned}
T(Z;X)
&=\phi\left[S(X)+\gamma_rR(Z)\right.\\
&\qquad+\gamma_dD(\mathcal L_GZ)\\
&\qquad\left.-\gamma_aA(\nabla_VZ)\right].
\end{aligned}
$$

The node equilibrium is $Z^\star=T(Z^\star;X)$. A node readout predicts one
value per node; graph pooling is applied only when the task has one target per
graph.
""",
            ),
            code(
                p,
                """
probe = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
one_graph_edges = data.edge_index[:, :20]
diffusion, gradient = graph_convection_diffusion(
    probe,
    one_graph_edges,
    edge_weight=data.edge_weight[:20],
    edge_velocity=data.edge_velocity[:20],
)
print("diffusion shape:", tuple(diffusion.shape))
print("directed-gradient shape:", tuple(gradient.shape))
print("constant-field diffusion check:", float(
    graph_convection_diffusion(torch.ones_like(probe), one_graph_edges)[0].abs().max()
))
""",
            ),
            md(
                p,
                r"""
## 3. Train a Small Node Field

The dataset target solves the discrete linear equation exactly. The learned
transition is nonlinear, so training asks the SILVA equilibrium and readout to
approximate that solution operator over four source fields.
""",
            ),
            code(
                p,
                """
model = SILVAPhysicsGuidedGraphDEQ(
    in_dim=3,
    state_dim=6,
    out_dim=1,
    config=SolverConfig(
        solver="picard",
        max_iter=7,
        tol=1e-5,
        alpha=0.7,
        backward_mode="unrolled",
    ),
)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
losses = []
for epoch in range(6):
    optimizer.zero_grad()
    prediction = model(
        data.x,
        data.edge_index,
        edge_weight=data.edge_weight,
        edge_velocity=data.edge_velocity,
    )
    loss = torch.nn.functional.mse_loss(prediction, data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

result = model(
    data.x,
    data.edge_index,
    edge_weight=data.edge_weight,
    edge_velocity=data.edge_velocity,
    return_result=True,
)
assert result.output.shape == data.target.shape
assert all(torch.isfinite(torch.tensor(losses)))
print("training losses:", losses)
print("fixed-point residual:", result.solver_result.residual)
print("prediction physical residual:", float(
    data.equation_residual(result.output).square().mean().sqrt()
))
""",
            ),
            code(
                p,
                """
nodes = 10
fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.4))
axes[0].plot(data.coordinates[:nodes, 0], data.x[:nodes, 0], marker="o")
axes[0].set_title("source")
axes[1].plot(data.coordinates[:nodes, 0], data.target[:nodes, 0], label="exact")
axes[1].plot(
    data.coordinates[:nodes, 0],
    result.output[:nodes, 0].detach(),
    "--",
    label="SILVA",
)
axes[1].legend()
axes[1].set_title("steady field")
axes[2].plot(range(1, len(losses) + 1), losses, marker="o")
axes[2].set_yscale("log")
axes[2].set_xlabel("epoch")
axes[2].set_title("training loss")
fig.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Node Relabeling

Let $P$ be a node permutation. A graph transition should satisfy

$$
T(PZ;PX,PE)=P\,T(Z;X,E),
$$

where $PE$ means that both rows of `edge_index` are relabeled consistently.
This is a structural test, not a statistical expectation.
""",
            ),
            code(
                p,
                """
single_x = data.x[:10]
single_edges = data.edge_index[:, :20]
single_weight = data.edge_weight[:20]
single_velocity = data.edge_velocity[:20]
baseline = model(
    single_x,
    single_edges,
    edge_weight=single_weight,
    edge_velocity=single_velocity,
)
permutation = torch.tensor([3, 0, 8, 1, 6, 2, 9, 5, 7, 4])
inverse = torch.empty_like(permutation)
inverse[permutation] = torch.arange(permutation.numel())
permuted_edges = inverse[single_edges]
permuted = model(
    single_x[permutation],
    permuted_edges,
    edge_weight=single_weight,
    edge_velocity=single_velocity,
)
relabel_error = (permuted - baseline[permutation]).abs().max()
assert relabel_error < 2e-5
print("node relabeling error:", float(relabel_error))
""",
            ),
            md(
                p,
                r"""
## 5. Node Tasks and Graph Tasks

For node prediction, the readout receives every equilibrium node. For graph
prediction, SILVA first computes a mask-free `mean`, `sum`, or `max` pooling
inside each graph id. Choose pooling from the units of the target: means are
intensive, sums are extensive, and maxima represent extremes.
""",
            ),
            code(
                p,
                """
graph_model = silva_equilibrium_model(
    "silva_physics_graph_deq",
    in_dim=3,
    state_dim=4,
    out_dim=1,
    task="graph",
    pooling="mean",
    config=SolverConfig(max_iter=4, alpha=0.7),
)
graph_values = graph_model(
    data.x,
    data.edge_index,
    edge_weight=data.edge_weight,
    edge_velocity=data.edge_velocity,
    batch=data.batch,
)
assert graph_values.shape == (4, 1)
print("graph-level output:", tuple(graph_values.shape))
""",
            ),
            md(
                p,
                r"""
## 6. Practical Guidance

| Problem | Diagnostic | Response |
|---|---|---|
| transport direction is reversed | inspect one directed edge by hand | document `source -> destination` and velocity sign |
| node values oversmooth | compare reaction, diffusion, and transport ablations | reduce diffusion or retain source injection |
| graph residual is low but physics residual is high | evaluate the discrete equation separately | constrain or supervise the physical branches |
| batched graphs interact accidentally | compare graph ids at both edge endpoints | offset edges and validate every batch |

The small ring dataset tests the implementation and physical bookkeeping.
Environmental benchmark claims require the measurement network, missing-data
rules, temporal split, and preprocessing used by the cited study [44].
""",
            ),
        ]
    )


def homotopy_lab() -> dict[str, object]:
    p = "silva-homotopy-lab"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Homotopy Equilibrium: Continuation Lab

This lab connects a SILVA fixed point to a continuous residual path, compares
Euler and Runge-Kutta integration with an analytic solution, and trains a small
conditioned transition. The canonical family is
`silva_homotopy_equilibrium` [46].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                """
import torch
import matplotlib.pyplot as plt
from torch import nn

from silva_networks import (
    SILVAHomotopyEquilibrium,
    make_affine_homotopy_dataset,
    silva_equilibrium_model,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(190)
""",
            ),
            md(
                p,
                r"""
## 1. Fixed-Point Residual and Classical Homotopy

For transition $T(z;x)$, define

$$
r(z;x)=z-T(z;x).
$$

The equilibrium satisfies $r(z^\star;x)=0$. A continuation equation between
an easy root $z_0$ and the target residual is

$$
H(z,\lambda;x)
=(1-\lambda)(z-z_0)+\lambda r(z;x)=0.
$$

Differentiating along a zero path gives

$$
\frac{\partial H}{\partial z}\frac{dz}{ds}
+\frac{\partial H}{\partial\lambda}
\frac{d\lambda}{ds}=0.
$$

This expresses root finding as path following [46].
""",
            ),
            md(
                p,
                r"""
## 2. The SILVA Residual Flow

The package exposes the direct continuous path

$$
\frac{dz}{dt}=T(z;x)-z=-r(z;x).
$$

Every stationary state of this flow is a SILVA fixed point. With step size
$h=T_f/K$, Euler uses

$$
z_{k+1}=z_k+h\,[T(z_k;x)-z_k],
$$

while fourth-order Runge-Kutta evaluates the vector field four times per step.
The continuous path and its terminal residual remain observable.
""",
            ),
            code(
                p,
                """
data = make_affine_homotopy_dataset(
    samples=24,
    dimension=2,
    contraction=0.5,
    seed=19,
)
assert data.fixed_point_residual().abs().max() < 1e-6
print("conditions:", tuple(data.condition.shape))
print("maximum exact fixed-point residual:", float(
    data.fixed_point_residual().abs().max()
))
""",
            ),
            md(
                p,
                r"""
## 3. Analytic Affine Path

For $T(z;x)=az+x$ with $|a|<1$,

$$
z^\star=\frac{x}{1-a}.
$$

The residual flow is $\dot z=x-(1-a)z$, with solution

$$
z(t)=z^\star+(z_0-z^\star)e^{-(1-a)t}.
$$

This provides both an endpoint and complete-trajectory reference.
""",
            ),
            code(
                p,
                """
class AffineTransition(nn.Module):
    def __init__(self, contraction):
        super().__init__()
        self.contraction = contraction

    def forward(self, state, condition):
        return self.contraction * state + condition

models = {
    integrator: SILVAHomotopyEquilibrium(
        in_dim=2,
        state_dim=2,
        out_dim=2,
        transition=AffineTransition(data.contraction),
        readout=nn.Identity(),
        steps=32,
        horizon=10.0,
        integrator=integrator,
        learnable_initial=False,
    )
    for integrator in ("euler", "rk4")
}
results = {
    name: model(data.condition[:4], return_result=True)
    for name, model in models.items()
}
for name, result in results.items():
    error = (result.output - data.target[:4]).abs().max()
    print(name, "endpoint error:", float(error))
    print(name, "terminal residual:", result.terminal_residual)
""",
            ),
            code(
                p,
                """
fig, axis = plt.subplots(figsize=(4.8, 2.8))
for name, result in results.items():
    axis.semilogy(
        range(len(result.velocity_norms)),
        result.velocity_norms,
        marker="o",
        markersize=2,
        label=name,
    )
axis.set_xlabel("integration step")
axis.set_ylabel("maximum velocity norm")
axis.legend()
fig.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Train a Conditioned SILVA Transition

The transition may be learned while the integration rule remains explicit.
For parameters $\theta$, differentiation follows every numerical step:

$$
\frac{d\mathcal L}{d\theta}
=\frac{\partial\mathcal L}{\partial z_K}
\frac{\partial z_K}{\partial\theta}.
$$

Memory therefore grows with the number of retained integration steps. This is
different from an implicit adjoint at a converged algebraic fixed point.
""",
            ),
            code(
                p,
                """
learned = SILVAHomotopyEquilibrium(
    in_dim=2,
    state_dim=6,
    out_dim=2,
    steps=10,
    horizon=4.0,
    integrator="rk4",
)
optimizer = torch.optim.Adam(learned.parameters(), lr=4e-3)
losses = []
for epoch in range(8):
    optimizer.zero_grad()
    prediction = learned(data.condition[:20])
    loss = torch.nn.functional.mse_loss(prediction, data.target[:20])
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

held_out = learned(data.condition[20:], return_result=True)
held_out_loss = torch.nn.functional.mse_loss(held_out.output, data.target[20:])
assert all(torch.isfinite(torch.tensor(losses)))
assert torch.isfinite(held_out_loss)
print("training losses:", losses)
print("held-out loss:", float(held_out_loss))
print("held-out terminal residual:", held_out.terminal_residual)
""",
            ),
            md(
                p,
                r"""
## 5. Choosing the Horizon and Integrator

For the affine example, endpoint error contains the finite-horizon factor
$e^{-(1-a)T_f}$. Increasing the number of steps reduces discretization error,
whereas increasing the horizon reduces truncation error. These controls solve
different problems.

| Control | Primary effect | Cost |
|---|---|---|
| larger horizon | gets closer to a stationary state | may require more steps for accuracy |
| more steps | reduces integration error | more transition evaluations and memory |
| Euler | transparent first-order path | smaller stable step sizes |
| RK4 | higher accuracy per step | four transition evaluations per step |
""",
            ),
            code(
                p,
                """
factory_model = silva_equilibrium_model(
    "silva_homotopy_equilibrium",
    in_dim=2,
    state_dim=4,
    out_dim=2,
    steps=4,
    horizon=2.0,
)
factory_output = factory_model(data.condition[:2])
assert factory_output.shape == (2, 2)
print(type(factory_model).__name__)
""",
            ),
            md(
                p,
                r"""
## 6. Practical Guidance

Report the task loss, terminal fixed-point residual, horizon, number of steps,
integrator, and transition evaluations. A low supervised loss does not imply
that the terminal state is near equilibrium. A low terminal residual does not
show that the continuous path matches measured dynamics.

The affine dataset verifies the path equations exactly. Reproducing the cited
vision experiments requires their architecture, data augmentation, schedules,
and evaluation protocol [46].
""",
            ),
        ]
    )


def distribution_lab() -> dict[str, object]:
    p = "silva-distribution-lab"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Distributional Equilibrium: Empirical-Measure Lab

This lab derives measure discrepancies, verifies permutation behavior and
masks, runs particle descent, and trains a small task readout from a
distributional SILVA state. The canonical family is
`silva_distributional_deq` [45].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                """
import torch
import matplotlib.pyplot as plt
from torch import nn

from silva_networks import (
    SILVADistributionalDEQ,
    distributional_discrepancy,
    make_variable_measure_dataset,
    silva_equilibrium_model,
)

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(200)
""",
            ),
            md(
                p,
                r"""
## 1. Matrices Represent Empirical Measures

Input rows $X=(x_1,\ldots,x_M)$ represent

$$
\rho_X=\frac1M\sum_{i=1}^{M}\delta_{x_i}.
$$

Latent rows $Z=(z_1,\ldots,z_N)$ represent

$$
\mu_Z=\frac1N\sum_{j=1}^{N}\delta_{z_j}.
$$

Permuting rows does not change either measure. Padded storage therefore needs
a boolean mask so padding contributes neither to attention nor to empirical
expectations.
""",
            ),
            code(
                p,
                """
data = make_variable_measure_dataset(
    samples=6,
    min_particles=5,
    max_particles=9,
    dimension=2,
    components=2,
    seed=20,
)
assert torch.equal(data.context_mask.sum(dim=1), data.counts)
assert torch.allclose(data.empirical_mean(), data.target_mean)
print("context:", tuple(data.context.shape))
print("particle counts:", data.counts.tolist())
print("target means:", data.target_mean)
""",
            ),
            md(
                p,
                r"""
## 2. Gaussian MMD and Energy Distance

For kernel $k$, the biased squared maximum mean discrepancy is

$$
\begin{aligned}
\operatorname{MMD}^2(\mu,\nu)
&=\mathbb E_{x,x'\sim\mu}k(x,x')\\
&\quad+\mathbb E_{y,y'\sim\nu}k(y,y')\\
&\quad-2\mathbb E_{x\sim\mu,y\sim\nu}k(x,y).
\end{aligned}
$$

SILVA provides the Gaussian kernel

$$
k_\ell(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\ell^2}\right)
$$

and the energy distance

$$
D_E^2
=2\mathbb E\|x-y\|
-\mathbb E\|x-x'\|
-\mathbb E\|y-y'\|.
$$
""",
            ),
            code(
                p,
                """
permutation = torch.tensor([4, 0, 3, 1, 2, 5, 6, 7, 8])
original = distributional_discrepancy(
    data.context,
    data.context,
    kernel="gaussian",
    left_mask=data.context_mask,
    right_mask=data.context_mask,
)
permuted = distributional_discrepancy(
    data.context,
    data.context[:, permutation],
    kernel="gaussian",
    left_mask=data.context_mask,
    right_mask=data.context_mask[:, permutation],
)
assert torch.allclose(original, permuted, atol=1e-6)
print("self discrepancy:", float(original))
print("permuted discrepancy:", float(permuted))
""",
            ),
            md(
                p,
                r"""
## 3. Distributional SILVA Objective

The transition maps a latent measure and input measure to a transformed latent
measure. The equilibrium objective is

$$
G_{\theta,X}(Z)
=\frac12D^2\left(\mu_Z,\mu_{F_\theta(Z,X)}\right).
$$

Particle descent applies

$$
z_j^{k+1}
=z_j^k-\eta\nabla_{z_j}G_{\theta,X}(Z^k).
$$

For latent permutation $P$ and context permutation $Q$, the built-in
transition satisfies

$$
F_\theta(PZ,QX)=P F_\theta(Z,X).
$$

This is equivariance in latent order and invariance in context order [45].
""",
            ),
            code(
                p,
                """
model = SILVADistributionalDEQ(
    input_dim=2,
    latent_dim=4,
    particles=5,
    heads=2,
    kernel="gaussian",
    bandwidth=1.0,
    step_size=0.12,
    max_iter=3,
)
result = model(
    data.context[:2],
    context_mask=data.context_mask[:2],
    return_result=True,
)
assert result.state.shape == (2, 5, 4)
assert result.discrepancies[-1] <= result.discrepancies[0] + 1e-6
print("discrepancy path:", result.discrepancies)
print("converged to configured tolerance:", result.converged)
""",
            ),
            md(
                p,
                r"""
## 4. Train a Readout from the Equilibrium Measure

The equilibrium state remains a set. A task head must pool in a way consistent
with the target. Here a pointwise decoder maps latent particles back to two
dimensions, and their mean predicts the empirical input mean:

$$
\widehat m
=\frac1N\sum_{j=1}^{N}Q_\psi(z_j^\star).
$$
""",
            ),
            code(
                p,
                """
decoder = nn.Linear(4, 2)
optimizer = torch.optim.Adam(
    [*model.parameters(), *decoder.parameters()],
    lr=3e-3,
)
losses = []
for epoch in range(4):
    optimizer.zero_grad()
    state = model(data.context, context_mask=data.context_mask)
    decoded_particles = decoder(state)
    prediction = decoded_particles.mean(dim=1)
    task_loss = torch.nn.functional.mse_loss(prediction, data.target_mean)
    transformed = model.transition(
        state,
        data.context,
        context_mask=data.context_mask,
    )
    equilibrium_loss = distributional_discrepancy(
        state,
        transformed,
        kernel="gaussian",
    )
    loss = task_loss + 0.05 * equilibrium_loss
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

assert all(torch.isfinite(torch.tensor(losses)))
print("training losses:", losses)
print("task loss:", float(task_loss.detach()))
print("distributional discrepancy:", float(equilibrium_loss.detach()))
""",
            ),
            code(
                p,
                """
with torch.no_grad():
    final_state = model(data.context[:1], context_mask=data.context_mask[:1])
    decoded = decoder(final_state)[0]
valid_context = data.context[0, data.context_mask[0]]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
axes[0].scatter(valid_context[:, 0], valid_context[:, 1], label="context")
axes[0].scatter(decoded[:, 0], decoded[:, 1], marker="x", label="latent readout")
axes[0].legend()
axes[0].set_title("empirical measures")
axes[1].plot(range(1, len(losses) + 1), losses, marker="o")
axes[1].set_yscale("log")
axes[1].set_xlabel("epoch")
axes[1].set_title("training objective")
fig.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 5. Fixed and Invalid Particles

`latent_mask` identifies valid latent rows. `fixed_mask` is a subset that
remains equal to its initial value during particle descent. This supports
observed anchors or boundary particles without allowing padding rows to enter
the measure.
""",
            ),
            code(
                p,
                """
z0 = torch.randn(1, 5, 4)
latent_mask = torch.tensor([[True, True, True, True, False]])
fixed_mask = torch.tensor([[True, False, False, False, False]])
anchored = model(
    data.context[:1],
    z0=z0,
    context_mask=data.context_mask[:1],
    latent_mask=latent_mask,
    fixed_mask=fixed_mask,
    return_result=True,
)
assert torch.equal(anchored.state[:, 0], z0[:, 0])
assert torch.equal(anchored.state[:, 4], torch.zeros_like(anchored.state[:, 4]))
print("fixed particle preserved and padding excluded")
""",
            ),
            code(
                p,
                """
factory_model = silva_equilibrium_model(
    "silva_distributional_deq",
    input_dim=2,
    latent_dim=4,
    particles=4,
    heads=2,
    max_iter=1,
)
factory_state = factory_model(
    data.context[:1],
    context_mask=data.context_mask[:1],
)
assert factory_state.shape == (1, 4, 4)
print(type(factory_model).__name__)
""",
            ),
            md(
                p,
                r"""
## 6. Practical Guidance

| Problem | Diagnostic | Response |
|---|---|---|
| result changes after row permutation | run the EI check | remove positional row encodings and order-dependent pooling |
| padding changes the result | compare variable-length and padded forms | pass masks through attention and discrepancy terms |
| discrepancy oscillates | inspect every descent step | reduce particle step size or change bandwidth |
| task loss is low but measures disagree | report task and measure losses separately | retain an explicit equilibrium diagnostic |

The generated mixtures validate variable-size storage, masks, invariance, and
outer gradients. Reproducing point-cloud benchmarks requires their official
splits, augmentations, sample counts, and task metrics [45].
""",
            ),
        ]
    )


LABS = {
    "17_silva_fno_equilibrium_lab.ipynb": fno_lab,
    "18_silva_graph_transport_lab.ipynb": graph_lab,
    "19_silva_homotopy_equilibrium_lab.ipynb": homotopy_lab,
    "20_silva_distributional_equilibrium_lab.ipynb": distribution_lab,
}


def main() -> None:
    for name, build in LABS.items():
        notebook_payload = build()
        for directory in OUT_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            write_notebook(directory / name, notebook_payload)
        print(name)


if __name__ == "__main__":
    main()
