"""Generate executable labs for eight emerging SILVA equilibrium families."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (
    ROOT / "notebooks/package_api",
    ROOT / "docs/package-notebooks",
    ROOT / "colab",
)
_COUNTERS: defaultdict[str, int] = defaultdict(int)


def _id(prefix: str) -> str:
    _COUNTERS[prefix] += 1
    return f"{prefix}-{_COUNTERS[prefix]:04d}"


def md(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _id(prefix),
        "metadata": {},
        "source": value.splitlines(True),
    }


def code(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _id(prefix),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": value.splitlines(True),
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
elif importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


PLOT_SETUP = """
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(33)
"""


def consistency_lab() -> dict[str, object]:
    p = "silva-cdeq"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Consistency DEQ

This lab derives solver-time consistency distillation, constructs an exact
contractive teacher, trains the refiner, and measures the one/few-step tradeoff.
The mechanism follows Consistency Deep Equilibrium Models [59]; the teacher,
student, solver, time map, and readout remain replaceable SILVA components.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from torch import nn
from silva_networks import (
    SILVAConsistencyDEQ,
    SolverConfig,
    make_consistency_teacher_dataset,
    silva_consistency_loss,
)
""",
            ),
            md(
                p,
                r"""
## 1. Teacher Equilibrium and Solver-Time Path

For a SILVA transition $f_\theta$,

$$z^\star=f_\theta(z^\star,x),\qquad F_\theta(z;x)=f_\theta(z,x)-z=0.$$

Fixing $z_0$ and the solver selects one trajectory $\{z_k\}_{k=0}^{K}$.
The compact teacher is affine, so its exact equilibrium is

$$z^\star=(I-A)^{-1}(Bx+b).$$
""",
            ),
            code(
                p,
                """
data = make_consistency_teacher_dataset(
    samples=48, state_dim=6, condition_dim=4, seed=33
)

class AffineTeacher(nn.Module):
    def __init__(self, matrix, source, bias):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("source", source)
        self.register_buffer("bias", bias)

    def forward(self, state, condition):
        return state @ self.matrix.T + condition @ self.source.T + self.bias

teacher = AffineTeacher(data.matrix, data.source_matrix, data.bias)
model = SILVAConsistencyDEQ(
    6,
    4,
    teacher_transition=teacher,
    teacher_config=SolverConfig(
        solver="anderson", max_iter=14, tol=1e-7, anderson_batch_dims=1
    ),
)
trajectory = model.teacher_trajectory(data.condition)
teacher_error = torch.linalg.vector_norm(
    trajectory.equilibrium - data.equilibrium, dim=-1
).max()
assert teacher_error < 3e-5
print("teacher states:", len(trajectory.states))
print("maximum exact-equilibrium error:", float(teacher_error))
print("terminal solver residual:", trajectory.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 2. Terminally Anchored Consistency Map

$$
g_\phi(z_t,t,x)=c_{\rm skip}(t)z_t+c_{\rm out}(t)P_\phi(z_{\leq t},t,x),
$$

$$
c_{\rm skip}(t)=\left(\frac{t-\epsilon}{T-\epsilon}\right)^\gamma,
\quad c_{\rm out}(t)=1-c_{\rm skip}(t),
\quad t_k=\epsilon+(1-e^{-\rho k})(T-\epsilon).
$$

At $T$, the map is the identity on the teacher endpoint. Earlier times permit
large learned corrections. A two-state history activates the
Anderson-structured proposal.
""",
            ),
            code(
                p,
                """
times = torch.linspace(model.epsilon, model.terminal_time, 100)
skip, out = model.boundary_coefficients(times)
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
axes[0].plot(times, skip, label="skip")
axes[0].plot(times, out, label="refiner")
axes[0].set(xlabel="virtual time", ylabel="coefficient")
axes[0].legend()

solver_errors = [
    float(torch.linalg.vector_norm(state - data.equilibrium, dim=-1).mean())
    for state in trajectory.states
]
axes[1].semilogy(solver_errors, marker="o", markersize=2)
axes[1].set(xlabel="teacher sample", ylabel="mean equilibrium error")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. Global and Local Distillation

Global consistency maps every sampled state to $z_K$. Local consistency makes
adjacent states agree under the current and exponential-moving-average models:

$$
\mathcal L=\lambda d(g_\phi(z_k),z_K)
+(1-\lambda)d(g_\phi(z_k),g_{\phi^-}(z_{k-1})).
$$

The compact optimization below uses the global term to isolate terminal
matching. The API also accepts the adjacent prediction and task loss.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.refiner.parameters(), lr=2e-2)
losses = []
for _ in range(80):
    optimizer.zero_grad()
    prediction = model(data.condition, steps=1, return_result=True)
    objective = silva_consistency_loss(
        prediction.state, trajectory.equilibrium, global_weight=1.0
    )
    objective.total.backward()
    optimizer.step()
    losses.append(float(objective.total.detach()))

errors = {}
for steps in (1, 2, 4):
    value = model(data.condition, steps=steps)
    errors[steps] = float(
        torch.linalg.vector_norm(value - trajectory.equilibrium, dim=-1).mean()
    )
print("initial/final distillation loss:", losses[0], losses[-1])
print("mean equilibrium error by evaluations:", errors)
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
axes[0].semilogy(losses)
axes[0].set(xlabel="optimization step", ylabel="global consistency loss")
axes[1].bar([str(key) for key in errors], list(errors.values()))
axes[1].set(xlabel="student evaluations", ylabel="mean equilibrium error")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Scaling and Reproduction

For a source task, first train or obtain the teacher, then cache solver states
with the exact initial state and solver. Cache size is

$$N_{samples}\,N_{stored}\,N_{state}\,N_{bytes}.$$

Use the source tokenization or image/graph preprocessing, train local and global
terms with an EMA target, and report task quality against network evaluations,
latency, and teacher-equilibrium error. WikiText-103, ImageNet, ogbn-arxiv, and
ogbn-products require their own task heads and data loaders; the consistency
module itself does not change.
""",
            ),
        ]
    )


def psi_lab() -> dict[str, object]:
    p = "silva-psi"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Psi-GNN for Mixed-Boundary Poisson Problems

This lab derives the boundary-aware graph, typed message maps, fixed-point
processor, residual objective, and source-scale route of Psi-GNN [60]. It uses
a known mixed-boundary field, then trains a compact SILVA model while retaining
the equation and boundary diagnostics.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from silva_networks import (
    SILVAPsiGNN,
    SILVAPsiGNNProcessor,
    SolverConfig,
    make_psi_poisson_grid,
)
""",
            ),
            md(
                p,
                r"""
## 1. PDE, Discretization, and Directed Boundary Graph

$$
-\Delta u=f\ \text{in }\Omega,
\quad u=g\ \text{on }\partial\Omega_D,
\quad \partial_nu=0\ \text{on }\partial\Omega_N.
$$

First-order discretization gives $AU=B$ and
$\mathcal L_{res}=N^{-1}\lVert AU-B\rVert_2^2$. Replacing Dirichlet rows by
identity rows makes Dirichlet nodes send known values without receiving graph
updates. Interior and Neumann stencils remain bidirectional.
""",
            ),
            code(
                p,
                """
data = make_psi_poisson_grid(size=11)
equation_error = (data.stiffness @ data.target - data.rhs).abs().max()
assert equation_error < 1e-6
print("nodes/edges:", data.coordinates.shape[0], data.edge_index.shape[1])
print("interior/Dirichlet/Neumann:", torch.bincount(data.node_types).tolist())
print("exact discrete residual:", float(equation_error))
""",
            ),
            code(
                p,
                """
size = int(data.coordinates.shape[0] ** 0.5)
figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
for axis, values, title in zip(
    axes,
    [data.target, data.forcing, data.node_types[:, None].float()],
    ["exact solution", "forcing", "node type"],
):
    image = axis.imshow(values.reshape(size, size), origin="lower", cmap="viridis")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 2. Typed Processor

Interior nodes receive separate incoming and outgoing messages,

$$\phi^I_{\leftarrow,i}=\sum_j\Phi^I_{\leftarrow}(H_i,H_j,d_{ji},\|d_{ji}\|),$$

$$\phi^I_{\rightarrow,i}=\sum_j\Phi^I_{\rightarrow}(H_i,H_j,d_{ij},\|d_{ij}\|),$$

and a residual update $z_i^I=H_i+\Lambda^I(H_i,b_i,\phi^I_\leftarrow,
\phi^I_\rightarrow)$. Neumann updates use their own message map and outward
normal. Dirichlet latent values stay equal to the encoded initial condition.
""",
            ),
            code(
                p,
                """
processor = SILVAPsiGNNProcessor(12, update_scale=0.12, normalize=False)
model = SILVAPsiGNN(
    12,
    processor=processor,
    config=SolverConfig(
        solver="picard", max_iter=12, tol=1e-5, backward_mode="unrolled"
    ),
)

def solve():
    return model(
        data.initial_solution,
        data.forcing,
        data.coordinates,
        data.edge_index,
        data.node_types,
        boundary_values=data.boundary_values,
        normals=data.normals,
        return_result=True,
    )

initial = solve()
assert initial.boundary_error == 0
print("output/state shapes:", initial.output.shape, initial.state.shape)
print("boundary error:", float(initial.boundary_error))
print("root residual:", initial.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 3. Complete Training Objective

The full construction combines equation residual, optional light supervision,
Jacobian stabilization, latent consistency, and encoder-decoder reconstruction.
The compact run emphasizes the known solution so progress is visible quickly;
the source experiment emphasizes the finite-element residual.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for _ in range(35):
    optimizer.zero_grad()
    result = solve()
    terms = model.loss(
        result,
        data.stiffness,
        data.rhs,
        exact=data.target,
        supervised_weight=1.0,
    )
    terms.total.backward()
    optimizer.step()
    losses.append(float(terms.total.detach()))

trained = solve()
field_error = torch.mean((trained.output - data.target).square()).sqrt()
print("initial/final objective:", losses[0], losses[-1])
print("field RMSE:", float(field_error))
print("boundary error:", float(trained.boundary_error))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))
axes[0].semilogy(losses)
axes[0].set(xlabel="optimization step", ylabel="objective")
predicted = trained.output.detach().reshape(size, size)
error = (predicted - data.target.reshape(size, size)).abs()
images = [predicted, error]
for axis, values, title in zip(axes[1:], images, ["prediction", "absolute error"]):
    image = axis.imshow(values, origin="lower", cmap="viridis")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. From This Grid to the Source Experiment

Regenerate first-order unstructured meshes, mark Dirichlet/interior/Neumann
nodes, derive directed edges from the boundary-modified stencil, and retain
coordinates, distances, normals, forcing, and boundary features. Use the FEM
matrices only for $AU-B$ during training and evaluation. Match the source
6000/2000/2000 split, approximately 500-node training meshes, optimizer groups,
Jacobian penalty, variable-resolution tests, residual, and MSE against the LU
solution. The same processor accepts each graph size without padding.
""",
            ),
        ]
    )


def ifno_lab() -> dict[str, object]:
    p = "silva-ifno"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA IFNO for Heterogeneous Materials

This lab derives the layer-independent Fourier residual increment, trains a
compact coefficient-to-displacement operator, compares shared depths, and maps
the implementation to displacement/damage studies. The mechanism follows IFNO
[61] and remains a configurable SILVA field point.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from silva_networks import SILVAIFNO, make_ifno_material_dataset
""",
            ),
            md(
                p,
                r"""
## 1. Material Operator Contract

The input field may concatenate coordinates $x$, material descriptor $b(x)$,
body force, padded Dirichlet values, and padded traction. IFNO lifts it to
$h_0=P[f]$ and reuses

$$
h_{l+1}=h_l+\Delta t\,\sigma\left(
Wh_l+\mathcal F^{-1}(R_\theta\mathcal F[h_l])+c\right).
$$

For the compact heterogeneous bar, equilibrium gives
$du/dx=T/E(x)$ and $u(0)=0$, so cumulative quadrature supplies an exact target.
""",
            ),
            code(
                p,
                """
data = make_ifno_material_dataset(samples=16, height=8, width=24, seed=30)
assert data.inputs.shape == (16, 4, 8, 24)
print("input/target:", data.inputs.shape, data.target.shape)
print("modulus range:", float(data.modulus.min()), float(data.modulus.max()))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
for axis, values, title in zip(
    axes,
    [data.modulus[0, 0], data.target[0, 0], data.inputs[0, 3]],
    ["heterogeneous modulus", "exact displacement", "traction field"],
):
    image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 2. Tied Depth and Deep Limit

The same increment module is called at every depth. Dividing by $\Delta t$
shows a nonlocal evolution equation. The `unrolled` mode matches finite shared
depth. The `equilibrium` mode solves for zero increment and is intended for
deep-limit studies where that root is well posed.
""",
            ),
            code(
                p,
                """
model = SILVAIFNO(
    in_channels=4,
    state_channels=12,
    out_channels=1,
    depth=6,
    step_size=0.08,
    modes_height=3,
    modes_width=6,
)
result = model(data.inputs[:2], return_result=True)
assert len(result.increment_norms) == 6
print("shared increment object:", id(model.increment))
print("increment norms:", [round(value, 3) for value in result.increment_norms])
""",
            ),
            md(
                p,
                r"""
## 3. Compact Coefficient-to-Displacement Training

This task validates the data route, spectral/local gradient path, shared-depth
reuse, and physical readout. It is not a heterogeneous-material benchmark.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
losses = []
for _ in range(45):
    optimizer.zero_grad()
    prediction = model(data.inputs)
    loss = torch.nn.functional.mse_loss(prediction, data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

with torch.no_grad():
    prediction = model(data.inputs[:1])
relative = torch.linalg.vector_norm(prediction - data.target[:1])
relative = relative / torch.linalg.vector_norm(data.target[:1])
print("initial/final loss:", losses[0], losses[-1])
print("sample relative L2:", float(relative))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))
axes[0].semilogy(losses)
axes[0].set(xlabel="optimization step", ylabel="MSE")
for axis, values, title in zip(
    axes[1:],
    [prediction[0, 0], (prediction[0, 0] - data.target[0, 0]).abs()],
    ["predicted displacement", "absolute error"],
):
    image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Displacement, Damage, and Source-Scale Runs

Use `out_channels=2` for scalar displacement plus damage, or the physical
spatial dimension plus damage for vector displacement. A custom readout can
leave displacement unbounded and apply a sigmoid only to damage. Reproduce a
source task by matching its simulated/experimental fields, boundary padding,
normalization, Fourier modes, shared depth, shallow-to-deep initialization,
optimizer, split, and relative field metrics. Storage follows
`samples x channels x H x W x dtype_bytes`; record each source archive and
preprocessing revision because material datasets vary by task.
""",
            ),
        ]
    )


def snarf_lab() -> dict[str, object]:
    p = "silva-snarf"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA SNARF Forward Skinning

This lab derives canonical blend weights, forward deformation, multi-start
implicit correspondences, occupancy composition, and the route to mesh-scale
experiments. The mechanism follows SNARF [62] and uses SILVA's configurable
root solver without hiding the geometric fields.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from torch import nn
from silva_networks import (
    SILVASNARF,
    SolverConfig,
    make_snarf_stick_dataset,
    silva_forward_skinning,
)
""",
            ),
            md(
                p,
                r"""
## 1. Forward Skinning Is the Model

Canonical weights obey $w_b(x)\geq0$ and $\sum_bw_b(x)=1$. Bone transforms
produce

$$d_w(x,B)=\sum_bw_b(x)B_b\bar x.$$

Unlike a pose-dependent backward field, $w(x)$ lives in canonical space. The
compact two-bone stick supplies known weights and deformed points.
""",
            ),
            code(
                p,
                """
data = make_snarf_stick_dataset(points=61)
reconstructed = silva_forward_skinning(
    data.canonical_points, data.transforms, data.blend_weights
)
assert torch.allclose(reconstructed, data.deformed_points)
print("points/bones:", data.canonical_points.shape[0], data.transforms.shape[0])
print("forward error:", float((reconstructed - data.deformed_points).abs().max()))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].plot(data.canonical_points[:, 0], data.canonical_points[:, 1], "o-", ms=2)
axes[0].set(title="canonical stick", aspect="equal")
axes[1].plot(data.deformed_points[:, 0], data.deformed_points[:, 1], "o-", ms=2)
axes[1].set(title="posed stick", aspect="equal")
for axis in axes:
    axis.grid(alpha=0.2)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 2. Canonical Correspondences as Roots

For posed query $x'$, each canonical correspondence satisfies

$$d_w(x^\star,B)-x'=0.$$

Every inverse bone transform supplies a starting point
$x_b^0=B_b^{-1}\bar x'$. Multiple valid roots are retained because folds and
self-contact can make the inverse one-to-many.
""",
            ),
            code(
                p,
                """
class ExactStickWeights(nn.Module):
    def forward(self, points):
        left = torch.sigmoid(-8.0 * points[..., 0])
        return torch.stack([left, 1.0 - left], dim=-1)

class StickOccupancy(nn.Module):
    def forward(self, points, pose=None):
        return torch.sigmoid(20.0 * (0.12 - points[..., 1].abs())).unsqueeze(-1)

model = SILVASNARF(
    coordinate_dim=2,
    bones=2,
    weight_field=ExactStickWeights(),
    occupancy_field=StickOccupancy(),
    correspondence_tol=2e-3,
    config=SolverConfig(
        solver="broyden",
        max_iter=35,
        tol=1e-7,
        history=8,
        backward_mode="unrolled",
        return_best=True,
    ),
)
result = model(data.deformed_points, data.transforms, return_result=True)
best_residual = result.residuals.min(dim=1).values
assert best_residual.max() < 2e-3
print("valid candidates:", int(result.valid.sum()), "/", result.valid.numel())
print("maximum best-root residual:", float(best_residual.max()))
print("occupancy range:", float(result.occupancy.min()), float(result.occupancy.max()))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
axes[0].semilogy(best_residual.detach(), marker="o", ms=2, lw=0.8)
axes[0].set(xlabel="posed query", ylabel="best root residual")
axes[1].plot(data.deformed_points[:, 0], result.occupancy.detach(), lw=1)
axes[1].set(xlabel="posed x coordinate", ylabel="soft occupancy")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. What the Advanced User Replaces

`weight_field` maps arbitrary leading point dimensions to simplex-valued bone
weights. `occupancy_field` maps canonical points and optional pose to occupancy.
The transform count, root solver, residual threshold, and soft-union temperature
are independent. `sample_occupancy_grid` evaluates the posed field in chunks;
its output can be passed to marching cubes when a mesh is required.
""",
            ),
            md(
                p,
                r"""
## 4. Source-Scale Route

For 2D Stick, match the source canonical geometry, topology-change object,
poses, query distribution, and occupancy labels. For human experiments, obtain
the required DFaust/AMASS or CAPE data under their terms, reproduce subject and
sequence splits, bone transforms, canonical pose, 20K-point frame sampling,
near-surface noise, bootstrap losses, and unseen-pose metrics. Store source
archive checksums and mesh preprocessing. Large human-motion collections and
derived meshes may require substantial local storage; inspect the selected
dataset release before acquisition instead of assuming one fixed size.
""",
            ),
        ]
    )


def mesh_lab() -> dict[str, object]:
    p = "silva-mesh"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Mesh Inference

This lab derives typed center-free relaxation in the linear-Gaussian regime,
verifies the M-matrix certificate, compares against the centralized optimum,
and studies directed admission. The mechanism follows Mesh Inference [63].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from silva_networks import (
    SILVAMeshInference,
    SolverConfig,
    make_mesh_gaussian_dataset,
)
""",
            ),
            md(
                p,
                r"""
## 1. Typed Anchors, Evidence, and Policy

For receiver $i$ and field $f$,

$$
(\lambda_i+\tau_i+\sum_jw_{ij})z_i^\star-\sum_jw_{ij}z_j^\star
=\lambda_i a_i+\tau_i o_i.
$$

$w_{ij}\geq0$ combines receiver admission and source emission. Fields are
independent typed coordinates; no model parameter or gradient is part of this
wire-level state.
""",
            ),
            code(
                p,
                """
data = make_mesh_gaussian_dataset(nodes=7, fields=3, asymmetric=True)
model = SILVAMeshInference(
    SolverConfig(solver="picard", max_iter=600, tol=1e-8, return_best=True)
)
result = model(
    data.anchors,
    data.anchor_precision,
    data.observations,
    data.observation_precision,
    data.admission,
    emission=data.emission,
    return_result=True,
)
print("distributed/centralized agreement:", float(result.agreement_error))
print("certificate:", result.certificate)
assert result.agreement_error < 5e-5
assert result.certificate.is_z_matrix
assert result.certificate.jacobi_spectral_radius < 1
""",
            ),
            md(
                p,
                r"""
## 2. Why the Iteration Converges

The system matrix has positive diagonal and nonpositive off-diagonal entries.
Anchoring makes the reachable component nonsingular. Jacobi relaxation is

$$
z_i^{k+1}=\frac{b_i+\sum_jw_{ij}z_j^k}
{\lambda_i+\tau_i+\sum_jw_{ij}}.
$$

The reported certificate checks the Z-matrix property, weak diagonal
dominance, minimum real eigenvalue, and Jacobi spectral radius.
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
nodes = torch.arange(data.anchors.shape[0])
for field in range(data.anchors.shape[1]):
    axes[0].plot(nodes, result.output[:, field], "o-", label=f"field {field}")
    axes[0].plot(nodes, result.centralized[:, field], "k.", ms=3)
axes[0].set(xlabel="node", ylabel="equilibrium estimate")
axes[0].legend(fontsize=7)
axes[1].semilogy(result.solver_result.residuals)
axes[1].set(xlabel="relaxation step", ylabel="fixed-point residual")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. Directed Admission Sweep

Asymmetry can remove a common quadratic potential without removing the
M-matrix convergence structure. The sweep below varies one direction while
checking distributed/centralized agreement and the Jacobi rate.
""",
            ),
            code(
                p,
                """
gains = torch.linspace(0.1, 1.5, 12)
agreement, radii = [], []
for gain in gains:
    admission = data.admission.clone()
    admission[1, 0, :] = gain
    value = model(
        data.anchors,
        data.anchor_precision,
        data.observations,
        data.observation_precision,
        admission,
        emission=data.emission,
        return_result=True,
    )
    agreement.append(float(value.agreement_error))
    radii.append(value.certificate.jacobi_spectral_radius)

figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
axes[0].semilogy(gains, agreement, marker="o", ms=3)
axes[0].set(xlabel="directed admission gain", ylabel="centralized agreement")
axes[1].plot(gains, radii, marker="o", ms=3)
axes[1].axhline(1.0, color="black", lw=0.7)
axes[1].set(xlabel="directed admission gain", ylabel="Jacobi spectral radius")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Source-Scale Verification

Reproduce the paper's carrier chain, forwarding, asymmetry, anchor-density,
latency, noisy estimation, and confidentiality-probe sweeps. Preserve typed
lineage and source-novel forwarding in the policy layer. For every run, save the
admission/emission support, centralized optimum, M-matrix certificate, recovery
error, spectral gap, message count, and random seed. The compact builder is
synthetic by design, so large source-scale runs require compute rather than a
large external dataset.
""",
            ),
        ]
    )


def diffusion_lab() -> dict[str, object]:
    p = "silva-pde-diffusion"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Physics-Guided Diffusion for PDEs

This lab derives inference-time residual guidance, Gaussian smoothing, hard
boundary projection, and deterministic/stochastic reverse paths. It solves a
compact Poisson field from random initialization and maps the same interface to
the source Poisson, diffusion, and Burgers studies [64].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from silva_networks import (
    SILVAPhysicsGuidedDiffusionPDE,
    finite_difference_poisson_energy,
    make_poisson_diffusion_dataset,
    project_homogeneous_dirichlet,
)
""",
            ),
            md(
                p,
                r"""
## 1. Residual Energy

For $-\Delta u=f$ with hard boundary conditions,

$$E(u)=\frac12\lVert-\Delta_hu-f\rVert_2^2.$$

The general implementation accepts any differentiable
$E_{PDE}(u)=\frac12\lVert\mathcal Lu+\mathcal N(u)-f\rVert_2^2$ and an
independent boundary projector.
""",
            ),
            code(
                p,
                """
data = make_poisson_diffusion_dataset(size=12, seed=33)

def energy(field, forcing):
    return finite_difference_poisson_energy(field, forcing, data.spacing)

initial = project_homogeneous_dirichlet(data.initial)
print("initial residual energy:", float(energy(initial, data.forcing)))
print("target residual energy:", float(energy(data.target, data.forcing)))
""",
            ),
            md(
                p,
                r"""
## 2. Reverse Step

$$\widetilde u_t=\operatorname{Prior}(u_t,t),$$

$$\bar u_t=G_\sigma*\widetilde u_t,$$

$$u_{t-1}=\mathcal B(\bar u_t-\eta\nabla E(\bar u_t)+\xi_t).$$

The neutral prior below isolates the physical correction. Replacing it with a
trained noise predictor changes no energy or boundary code.
""",
            ),
            code(
                p,
                """
sampler = SILVAPhysicsGuidedDiffusionPDE(
    energy,
    project_homogeneous_dirichlet,
    steps=45,
    guidance_step=7e-6,
    prior_strength=0.0,
    smoothing_sigma=0.55,
)
result = sampler(initial, condition=data.forcing, return_result=True)
print("initial/final energy:", float(energy(initial, data.forcing)), result.energies[-1])
print("reverse states:", len(result.states))
assert result.energies[-1] < float(energy(initial, data.forcing))
assert torch.count_nonzero(result.output[..., 0, :]) == 0
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 4, figsize=(10.2, 2.5))
fields = [initial[0, 0], result.output.detach()[0, 0], data.target[0, 0]]
titles = ["random start", "guided field", "exact field"]
for axis, values, title in zip(axes[:3], fields, titles):
    image = axis.imshow(values, origin="lower", cmap="viridis")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
axes[3].semilogy(result.energies)
axes[3].set(xlabel="reverse step", ylabel="PDE energy")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. Prior Independence and Stochasticity

`noise_predictor(state, time, condition)` may be a U-Net, neural operator, or
other field prior trained without the target equation. `prior_mode="noise"`
uses a DDPM-style clean estimate; `prior_mode="clean"` accepts a direct field
estimate. Set `stochastic=True` to add scheduled perturbations. Always report
both residual-energy and boundary traces: a visually plausible field is not a
PDE solution unless those diagnostics agree.
""",
            ),
            md(
                p,
                r"""
## 4. Source-Scale Route

Generate or obtain the source 64x64 fields and preserve the reported 4,000
snapshot split, global max-absolute normalization, three-level 64-channel
prior, linear variance schedule, and equation-specific guidance steps. A scalar
float32 array of 4,000 x 64 x 64 values is about 66 MB before conditions,
time fields, metadata, and checkpoints. For transient diffusion and Burgers,
treat physical time as an additional grid axis, hard-project the initial and
spatial boundaries after every reverse step, and report relative field error,
PDE residual, boundary error, convergence, and coefficient-shift tests.
""",
            ),
        ]
    )


def therino_lab() -> dict[str, object]:
    p = "silva-therino"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Thermodynamically Informed Neural Operator

This lab derives a physical-strain equilibrium, verifies it on an exact
periodic elastic cell, exposes every replaceable component, and maps the compact
case to the source-scale mechanics protocol. The thermodynamic construction
follows TherINO [73]; SILVA provides the common solver, transition, diagnostics,
scaling, and reproduction contracts.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from torch import nn
from silva_networks import (
    SILVATherINO,
    SILVAThermodynamicEncoder,
    SILVAThermodynamicUpdate,
    SolverConfig,
    make_therino_elastic_dataset,
)
""",
            ),
            md(
                p,
                r"""
## 1. Physical State, Constitutive Map, and Equilibrium

Let $arepsilon(x)\in\mathbb R^{d_s}$ be the strain components and
$C(x)\in\mathbb R^{d_s\times d_s}$ the local stiffness. Linear elasticity
gives

$$
\sigma(x)=C(x):\varepsilon(x),\qquad
W(x)=\frac12\varepsilon(x):\sigma(x).
$$

The prescribed loading is a volume average,

$$
\langle\varepsilon\rangle
=\frac{1}{|\Omega|}\int_\Omega\varepsilon(x)\,dx
=\bar\varepsilon.
$$

Instead of solving an unrelated latent state, the transition acts on the
physical strain itself. A fixed thermodynamic encoder forms

$$
q(\varepsilon,C,\bar\varepsilon)
=\left[\varepsilon,\ C:\varepsilon,\
\frac12\varepsilon:(C:\varepsilon),\ \bar\varepsilon\right],
$$

and the shared operator solves

$$
\varepsilon^\star
=\Pi_{\bar\varepsilon}
\left(U_\theta(q(\varepsilon^\star,C,\bar\varepsilon))\right).
$$

The projection
$\Pi_{\bar\varepsilon}(v)=v-\langle v\rangle+\bar\varepsilon$
enforces the macroscopic strain after every transition.
""",
            ),
            md(
                p,
                r"""
## 2. Exact Periodic Verification Cell

The compact dataset uses uncoupled diagonal elasticity. Mechanical equilibrium
requires constant stress in each component. With compliance
$S_i(x)=C_i(x)^{-1}$,

$$
\sigma_i=\frac{\bar\varepsilon_i}{\langle S_i\rangle},
\qquad
\varepsilon_i(x)=S_i(x)\sigma_i.
$$

This is a known solution, not a benchmark claim. It checks tensor layout,
constitutive contraction, bulk loading, the root solve, and all three losses
before expensive finite-element data are introduced.
""",
            ),
            code(
                p,
                """
data = make_therino_elastic_dataset(
    samples=4,
    size=24,
    strain_components=3,
    contrast=8.0,
    seed=73,
)
computed_stress = torch.einsum(
    "bijxy,bjxy->bixy", data.stiffness, data.target_strain
)
assert torch.allclose(computed_stress, data.target_stress, atol=1e-6)
assert torch.allclose(
    data.target_strain.mean(dim=(-2, -1)), data.macro_strain, atol=1e-7
)
print("stiffness:", tuple(data.stiffness.shape))
print("target strain:", tuple(data.target_strain.shape))
print("macroscopic strain:", data.macro_strain[0])
""",
            ),
            code(
                p,
                """
diagonal = torch.diagonal(
    data.stiffness[0], dim1=0, dim2=1
).permute(2, 0, 1)
figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
images = (
    axes[0].imshow(diagonal[0], cmap="viridis"),
    axes[1].imshow(data.target_strain[0, 0], cmap="coolwarm"),
    axes[2].imshow(data.target_stress[0, 0], cmap="magma"),
)
titles = (r"stiffness $C_{11}$", r"strain $\\varepsilon_1$", r"stress $\\sigma_1$")
for axis, image, title in zip(axes, images, titles):
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. Define What Lives Inside the SILVA Point

`SILVATherINO` separates five decisions:

| Role | Default | Replaceable contract |
|---|---|---|
| constitutive encoder | `SILVAThermodynamicEncoder` | `(strain, stiffness, macro) -> encoded field` |
| spatial update | Fourier point operator | `encoded field -> candidate strain` |
| loading constraint | mean-strain projection | candidate -> admissible candidate |
| equilibrium solver | Anderson | transition plus initial strain -> root |
| supervision | strain, stress, energy | result plus target -> scalar terms |

The exact relaxation below is intentionally transparent. It is a verification
operator whose fixed point is known. Replacing it with
`SILVAThermodynamicUpdate`, a U-Net, a convolutional hierarchy, or a custom
neural operator changes the internal architecture without changing the outer
SILVA contract.
""",
            ),
            code(
                p,
                """
class ExactMaterialRelaxation(nn.Module):
    def __init__(self, target, gain=0.35):
        super().__init__()
        self.register_buffer("target", target)
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, encoded):
        current_strain = encoded[:, : self.target.shape[1]]
        return self.gain * current_strain + (1.0 - self.gain) * self.target

update = ExactMaterialRelaxation(data.target_strain)
model = SILVATherINO(
    strain_components=3,
    encoder=SILVAThermodynamicEncoder(3),
    update=update,
    enforce_macro_strain=True,
    config=SolverConfig(
        solver="picard",
        max_iter=14,
        tol=1e-8,
        backward_mode="unrolled",
        anderson_batch_dims=1,
        return_best=False,
    ),
)
result = model(data.stiffness, data.macro_strain, return_result=True)
objective = model.loss(result, data.target_strain, data.stiffness)
assert torch.allclose(result.strain, data.target_strain, atol=2e-6)
print("iterations:", result.solver_result.iterations)
print("terminal residual:", result.solver_result.residual)
print("strain/stress/energy losses:",
      float(objective.strain), float(objective.stress), float(objective.energy))
""",
            ),
            code(
                p,
                """
figure, axis = plt.subplots(figsize=(4.6, 2.8))
axis.semilogy(result.solver_result.residuals, marker="o", markersize=2.5)
axis.set(
    xlabel="fixed-point iteration",
    ylabel=r"$\\|T(\\varepsilon)-\\varepsilon\\|_2$",
)
axis.grid(alpha=0.25)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Trainable Finite-Iteration Objective

For targets $(\varepsilon^{data},\sigma^{data},W^{data})$, the complete compact
objective is

$$
\mathcal L
=\lambda_\varepsilon\|\varepsilon^\star-\varepsilon^{data}\|_2^2
+\lambda_\sigma\|C:\varepsilon^\star-\sigma^{data}\|_2^2
+\lambda_W\|W(\varepsilon^\star)-W^{data}\|_2^2.
$$

The next cell deliberately uses only three unrolled iterations. The remaining
finite-iteration error gives a nonzero signal for the relaxation parameter and
demonstrates the complete gradient path.
""",
            ),
            code(
                p,
                """
train_update = ExactMaterialRelaxation(data.target_strain, gain=0.85)
train_model = SILVATherINO(
    update=train_update,
    config=SolverConfig(
        solver="picard", max_iter=3, tol=1e-12,
        backward_mode="unrolled", return_best=False
    ),
)
optimizer = torch.optim.Adam(train_model.parameters(), lr=4e-2)
losses = []
for _ in range(35):
    optimizer.zero_grad()
    prediction = train_model(data.stiffness, data.macro_strain, return_result=True)
    loss = train_model.loss(prediction, data.target_strain, data.stiffness)
    loss.total.backward()
    optimizer.step()
    losses.append(float(loss.total.detach()))

assert losses[-1] < losses[0]
figure, axis = plt.subplots(figsize=(4.6, 2.8))
axis.semilogy(losses)
axis.set(xlabel="optimizer step", ylabel="strain + stress + energy loss")
axis.grid(alpha=0.25)
figure.tight_layout()
plt.show()
print("learned relaxation gain:", float(train_update.gain.detach()))
""",
            ),
            md(
                p,
                r"""
## 5. Move from the Exact Cell to Full Mechanics

For a source-scale study [73], replace only the components demanded by the
physics:

1. Generate periodic two- or multi-phase microstructures and retain constituent
   stiffness tensors, geometry parameters, load cases, and seeds.
2. Produce strain and stress labels with a declared finite-element solver,
   discretization, tolerance, and periodic boundary construction.
3. Use the Fourier update below or provide another differentiable operator with
   the same shape contract.
4. Train strain, stress, and energy terms; report localization and homogenized
   response on held-out geometries, loadings, resolutions, and stiffness
   contrasts.
5. Record root residuals, iterations, wall time, memory, normalization, model
   width/modes, and every checkpoint revision.

The default operator is directly inspectable and can be replaced at
construction time:
""",
            ),
            code(
                p,
                """
full_update = SILVAThermodynamicUpdate(
    strain_components=3,
    hidden_channels=48,
    modes_height=12,
    modes_width=12,
    scale=0.12,
)
full_model = SILVATherINO(
    strain_components=3,
    update=full_update,
    config=SolverConfig(
        solver="anderson",
        max_iter=60,
        tol=1e-5,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
        return_best=True,
    ),
)
print(full_model.update)
print("trainable parameters:", sum(p.numel() for p in full_model.parameters()))
""",
            ),
            md(
                p,
                r"""
## Interpretation Boundary

The compact cell verifies the SILVA implementation route and exact mechanics
identities. It does not reproduce the source paper's three-dimensional
microstructure corpus, finite-element labels, architecture scale, training
budget, or reported aggregate metrics. The final preflight cell lists those
requirements explicitly so a sufficiently provisioned reader can proceed
without changing the public interface.
""",
            ),
        ]
    )


def fixed_point_diffusion_lab() -> dict[str, object]:
    p = "silva-fpdm"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Fixed-Point Diffusion Models

This lab derives a timestep-conditioned implicit denoiser, verifies variable
compute and equilibrium reuse, exercises stochastic Jacobian-free training, and
distinguishes per-timestep fixed points from a joint diffusion-restoration
trajectory. The first mechanism follows Fixed-Point Diffusion Models [74]; the
joint trajectory route follows DeqIR [49].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                PLOT_SETUP
                + """
from torch import nn
from silva_networks import (
    SILVADiffusionEquilibrium,
    SILVAFixedPointDenoiser,
    SILVAFixedPointDiffusionModel,
    SILVATimestepFixedPointBlock,
    SolverConfig,
    make_fixed_point_diffusion_dataset,
)
""",
            ),
            md(
                p,
                r"""
## 1. One Fixed Point at Every Diffusion Time

An explicit diffusion block first computes input features
$h_t=f_{pre}(x_t)$ and an injection $p_t=P(h_t)$. The implicit block solves

$$
z_t^\star=F_\theta(z_t^\star,p_t,e(t),c),
\qquad
\widehat\epsilon_t=f_{post}(z_t^\star),
$$

where $e(t)$ is the timestep embedding and $c$ is optional conditioning. A
reverse update then maps $(x_t,\widehat\epsilon_t)$ to $x_s$ for $s<t$.

The architecture has four independently replaceable regions:

$$
x_t\xrightarrow{f_{pre}}h_t\xrightarrow{P}p_t
\xrightarrow{\operatorname{root}(F_\theta)}z_t^\star
\xrightarrow{f_{post}}\widehat\epsilon_t.
$$

SILVA does not require the internal transition to be convolutional. It may be a
residual block, U-Net, attention block, transformer, Fourier operator, or any
shape-preserving module with the declared four-argument transition contract.
""",
            ),
            md(
                p,
                r"""
## 2. Exact Contractive Denoiser

For a transparent verification problem, define

$$
F(z,p,t)=\rho z+(1-\rho)(0.5p+0.1t),\qquad 0<\rho<1.
$$

The exact fixed point is $z_t^\star=0.5p+0.1t$. This lets us measure how the
iteration allocation changes numerical error without requiring a pretrained
image model.
""",
            ),
            code(
                p,
                """
class ContractiveTimestepTransition(nn.Module):
    def __init__(self, gain=0.35):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, state, injection, time, condition=None):
        del condition
        time_field = time.reshape(-1, 1, 1, 1).to(state)
        target = 0.5 * injection + 0.1 * time_field
        return self.gain * state + (1.0 - self.gain) * target

transition = ContractiveTimestepTransition()
denoiser = SILVAFixedPointDenoiser(
    channels=1,
    preprocessor=nn.Identity(),
    projection=nn.Identity(),
    transition=transition,
    postprocessor=nn.Identity(),
    config=SolverConfig(
        solver="picard",
        max_iter=20,
        tol=1e-8,
        backward_mode="unrolled",
        anderson_batch_dims=1,
        return_best=False,
    ),
)
compact_data = make_fixed_point_diffusion_dataset(
    samples=3, channels=1, size=18, seed=74
)
noise = compact_data.noise
time = compact_data.times
exact = compact_data.target
result = denoiser(noise, time, return_result=True)
assert torch.allclose(result.output, exact, atol=2e-6)
print("equilibrium shape:", tuple(result.equilibrium.shape))
print("iterations/residual:", result.solver_result.iterations, result.solver_result.residual)
""",
            ),
            code(
                p,
                """
budgets = (1, 2, 4, 8, 12)
errors = []
for budget in budgets:
    estimate = denoiser(noise, time, iterations=budget)
    errors.append(float((estimate - exact).square().mean().sqrt()))

figure, axis = plt.subplots(figsize=(4.8, 2.8))
axis.semilogy(budgets, errors, marker="o")
axis.set(xlabel="fixed-point block evaluations", ylabel="RMSE to exact fixed point")
axis.grid(alpha=0.25)
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 3. Reverse Schedule, Variable Compute, and State Reuse

For reverse times $t_0>t_1>\cdots>t_K$, the computation is a sequence of
related roots:

$$
z_{t_k}^\star=F_\theta(z_{t_k}^\star,P(x_{t_k}),t_k,c),
\qquad
x_{t_{k+1}}=R(x_{t_k},Q(z_{t_k}^\star),t_k,t_{k+1},\xi_k).
$$

The previous equilibrium is a warm start for the next time. An allocation
$(m_0,\ldots,m_{K-1})$ controls block evaluations independently at each reverse
step. The reverse operator remains replaceable, so the same wrapper can host a
declared DDIM, DDPM, ODE, or task-specific schedule.
""",
            ),
            code(
                p,
                """
def reverse_step(sample, prediction, timestep, next_timestep, condition, step_noise):
    del timestep, next_timestep, condition
    return 0.45 * sample + 0.55 * prediction + step_noise

process = SILVAFixedPointDiffusionModel(
    denoiser,
    timesteps=(8, 5, 3, 1, 0),
    allocations=(2, 3, 5, 8),
    step_operator=reverse_step,
    reuse_equilibria=True,
)
process_result = process(noise, return_result=True)
assert process_result.allocations == (2, 3, 5, 8)
print("reverse samples:", len(process_result.samples))
print("solver residuals:", [item.residual for item in process_result.solver_results])

figure, axes = plt.subplots(1, len(process_result.samples), figsize=(10.0, 2.1))
for index, (axis, sample) in enumerate(zip(axes, process_result.samples)):
    image = axis.imshow(sample[0, 0].detach(), cmap="coolwarm")
    axis.set_title(f"state {index}")
    axis.set_xticks([])
    axis.set_yticks([])
figure.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
plt.show()
""",
            ),
            md(
                p,
                r"""
## 4. Stochastic Jacobian-Free Training

A memory-limited training step can sample $n$ no-gradient transitions followed
by $m$ differentiable transitions:

$$
z_n=F_\theta^{\,n}(z_0,p,t),\quad n\sim\mathcal U\{0,\ldots,N\},
$$

$$
\widetilde z=F_\theta^{\,m}(\operatorname{stopgrad}(z_n),p,t),
\quad m\sim\mathcal U\{1,\ldots,M\}.
$$

This does not store the first $n$ states. The following deterministic choice
checks the same gradient route; omitting `no_grad_steps` and `grad_steps` samples
them from the configured ranges.
""",
            ),
            code(
                p,
                """
transition.gain.grad = None
prediction = denoiser.stochastic_jfb(
    noise,
    time,
    no_grad_steps=3,
    grad_steps=2,
    max_no_grad=8,
    max_grad=4,
)
training_loss = prediction.square().mean()
training_loss.backward()
assert transition.gain.grad is not None
print("training loss:", float(training_loss.detach()))
print("transition gradient:", float(transition.gain.grad.detach()))
""",
            ),
            md(
                p,
                r"""
## 5. Install a Full Spatial Transition

The built-in transition concatenates the current state, input injection, and a
scalar timestep field, applies a shape-preserving spatial network, and bounds
its correction. It is a usable default and a reference implementation of the
contract. Full source reproduction can replace it with the reported
transformer blocks while retaining the outer denoiser, solver, allocation, and
diagnostic interfaces.
""",
            ),
            code(
                p,
                """
spatial_transition = SILVATimestepFixedPointBlock(
    channels=4,
    hidden_channels=64,
    scale=0.15,
)
spatial_denoiser = SILVAFixedPointDenoiser(
    channels=4,
    preprocessor=nn.Conv2d(4, 4, 3, padding=1),
    projection=nn.Conv2d(4, 4, 1),
    transition=spatial_transition,
    postprocessor=nn.Conv2d(4, 4, 3, padding=1),
    config=SolverConfig(
        solver="anderson",
        max_iter=32,
        tol=1e-5,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
        return_best=True,
    ),
)
print(spatial_denoiser)
print("trainable parameters:", sum(p.numel() for p in spatial_denoiser.parameters()))
""",
            ),
            md(
                p,
                r"""
## 6. Distinguish the Joint DeqIR Route

The model above solves one implicit feature state at each diffusion timestep
[74]. `SILVADiffusionEquilibrium` instead places the complete selected reverse
trajectory in one triangular fixed point. Its `step_operator` replaces the
reverse rule and `data_consistency` projects every candidate against an
observation, which is the SILVA route for joint diffusion restoration [49].
These are related but different abstractions; both remain available.
""",
            ),
            code(
                p,
                """
class JointRestorationStep(nn.Module):
    def forward(self, state, timestep, next_timestep, condition, step_noise):
        del timestep, next_timestep, condition
        return 0.7 * state + step_noise

class MeasurementProjection(nn.Module):
    def forward(self, candidate, observation, next_timestep):
        del next_timestep
        return 0.8 * candidate + 0.2 * observation

joint = SILVADiffusionEquilibrium(
    denoiser=None,
    alphas_cumprod=torch.linspace(1.0, 0.2, 6),
    timesteps=(5, 3, 1, -1),
    step_operator=JointRestorationStep(),
    data_consistency=MeasurementProjection(),
    config=SolverConfig(
        solver="picard", max_iter=8, tol=1e-7,
        anderson_batch_dims=0, return_best=True
    ),
)
observation = torch.zeros_like(noise)
joint_result = joint(noise, observation=observation, return_result=True)
assert joint_result.trajectory.shape == (4, *noise.shape)
print("joint trajectory:", tuple(joint_result.trajectory.shape))
print("joint residual:", joint_result.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 7. Source-Scale Reproduction Checklist

To reproduce the fixed-point diffusion study [74], retain the published image
preprocessing, latent encoder, diffusion schedule, timestep conditioning,
architecture widths, iteration-allocation policy, stochastic backward ranges,
optimizer, precision, checkpoint selection, and FID-50K protocol. Report both
quality and the number of transformer-block evaluations, wall time, peak
memory, and equilibrium residuals.

To reproduce a DeqIR restoration result [49], additionally retain the exact
pretrained denoiser, degradation/SVD operator, observation noise, initialization,
reverse schedule, data-consistency rule, and restoration metrics. Compact
results on this page verify the public mechanism contracts; they are not
substitutes for either full source experiment.
""",
            ),
        ]
    )


NOTEBOOKS = {
    "28_silva_consistency_deq.ipynb": consistency_lab,
    "29_silva_psi_gnn.ipynb": psi_lab,
    "30_silva_ifno_materials.ipynb": ifno_lab,
    "31_silva_snarf_forward_skinning.ipynb": snarf_lab,
    "32_silva_mesh_inference.ipynb": mesh_lab,
    "33_silva_physics_guided_diffusion_pde.ipynb": diffusion_lab,
    "34_silva_therino_mechanics.ipynb": therino_lab,
    "35_silva_fixed_point_diffusion.ipynb": fixed_point_diffusion_lab,
}

FAMILIES = {
    "28_silva_consistency_deq.ipynb": "silva_consistency_deq",
    "29_silva_psi_gnn.ipynb": "silva_psi_gnn",
    "30_silva_ifno_materials.ipynb": "silva_ifno",
    "31_silva_snarf_forward_skinning.ipynb": "silva_snarf",
    "32_silva_mesh_inference.ipynb": "silva_mesh_inference",
    "33_silva_physics_guided_diffusion_pde.ipynb": ("silva_physics_guided_diffusion_pde"),
    "34_silva_therino_mechanics.ipynb": "silva_therino",
    "35_silva_fixed_point_diffusion.ipynb": "silva_fixed_point_diffusion",
}

SOURCE_PREFLIGHT = {
    "silva_consistency_deq": (
        "The reference route provides WikiText-103 [65], OGB node tasks [66], "
        "and registered ImageNet access [67]. Use 512-2,048 examples and a "
        "small deterministic trajectory cache before producing the complete "
        "teacher-state archive."
    ),
    "silva_psi_gnn": (
        "The benchmark is generated from the paper protocol [60]. Begin with "
        "32-128 first-order Gmsh meshes [72], retain sparse finite-element "
        "objects, and restore the 6,000/2,000/2,000 split only after the "
        "equation and boundary diagnostics pass."
    ),
    "silva_ifno": (
        "The source material simulations and DIC measurements [61] are not "
        "one redistributable archive. Start with 64-256 declared fields at "
        "32-by-32, then restore the chosen constitutive task, units, grid, "
        "modes, shared depth, split, and normalization."
    ),
    "silva_snarf": (
        "Use the unrestricted articulated-stick case first. Full experiments "
        "follow the source preprocessing [62] and separately licensed AMASS "
        "[68], D-FAUST [69], CAPE [70], and SMPL [71] assets. Preflight one "
        "subject and a few frames before the complete sequence protocol."
    ),
    "silva_mesh_inference": (
        "The paper cases [63] are generated from declared topology, typed "
        "evidence, precision, policy, and seed. Sweep a small carrier graph "
        "against the centralized solve before increasing node counts and "
        "retained policy traces."
    ),
    "silva_physics_guided_diffusion_pde": (
        "The source fields are procedurally generated [64]. Validate 64-256 "
        "small fields and a frozen prior first, then recreate the 4,000-field "
        "64-by-64 coefficient, boundary, normalization, and reverse-schedule "
        "protocol."
    ),
    "silva_therino": (
        "The source mechanics corpus [73] is generated from periodic "
        "microstructures and numerical localization solves. Validate 16-64 "
        "small two-dimensional cells first, then restore the three-dimensional "
        "geometry, constitutive contrast, loading, finite-element, split, "
        "normalization, Fourier-mode, and evaluation protocol."
    ),
    "silva_fixed_point_diffusion": (
        "The source image experiments [74] require their licensed or registered "
        "datasets, latent encoder, diffusion schedule, full architecture, "
        "training checkpoints, and FID-50K budget. Begin with 128-512 encoded "
        "samples and a short timestep schedule before restoring the complete "
        "allocation and generation protocol."
    ),
}


def source_preflight_cells(family: str) -> list[dict[str, object]]:
    prefix = f"{family}-source"
    return [
        md(
            prefix,
            f"""
## Source Data and Full Experiment Preflight

{SOURCE_PREFLIGHT[family]}

The executable record below distinguishes public, generated, and licensed
inputs and keeps storage and launch steps next to the model contract. Compact
results validate the implementation route; the cited benchmark additionally
requires every recorded source-scale step.
""",
        ),
        code(
            prefix,
            f"""
from silva_networks import silva_reproduction_spec

source_plan = silva_reproduction_spec({family!r})
print("data sources:")
for source in source_plan.data_sources:
    print(" -", source)
print("access:")
for item in source_plan.data_access:
    print(" -", item)
print("storage:")
for item in source_plan.storage_plan:
    print(" -", item)
print("source-scale steps:")
for index, item in enumerate(source_plan.source_scale_steps, start=1):
    print(f" {{index}}. {{item}}")
""",
        ),
    ]


def main() -> None:
    for name, builder in NOTEBOOKS.items():
        payload = builder()
        payload["cells"].extend(source_preflight_cells(FAMILIES[name]))
        for directory in OUT_DIRS:
            path = directory / name
            write_notebook(
                path,
                payload,
                replace_changed=True,
                preserve_unmatched=False,
            )
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
