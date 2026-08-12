"""Generate additive learned-solver, backward, quantum, and family-atlas labs."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (ROOT / "notebooks/package_api", ROOT / "docs/package-notebooks", ROOT / "colab")
COUNTERS: defaultdict[str, int] = defaultdict(int)


def _cell_id(prefix: str) -> str:
    COUNTERS[prefix] += 1
    return f"{prefix}-{COUNTERS[prefix]:04d}"


def md(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": _cell_id(prefix),
        "metadata": {},
        "source": value.splitlines(True),
    }


def code(prefix: str, source: str) -> dict[str, object]:
    value = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "id": _cell_id(prefix),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": value.splitlines(True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = r"""
from pathlib import Path
import sys

root = Path.cwd()
while root != root.parent and not (root / "src" / "silva_networks").exists():
    root = root.parent
if not (root / "src" / "silva_networks").exists():
    root = Path("/content/silva-networks")
sys.path.insert(0, str(root / "src"))

import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(91)
"""


def learned_solver_lab() -> dict[str, object]:
    prefix = "silva-learned-solvers"
    return notebook(
        [
            md(
                prefix,
                r"""
# SILVA Learned Equilibrium Solvers

This lab derives and trains a HyperDEQ-style learned Anderson solver inside
SILVA [[87]]. It keeps the transition, initializer, residual compressor,
controller, and readout independent, then replaces the vector transition with a
spatial module without changing the learned-solver contract.

[Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/48_silva_learned_solvers.ipynb)
""",
            ),
            code(prefix, BOOTSTRAP),
            md(
                prefix,
                r"""
## 1. Derive the Learned Anderson Update

For $z^\star=T_\theta(z^\star,x)$, define

$$
z_0=h_\phi(x),\qquad f_i=T_\theta(z_i,x),\qquad r_i=f_i-z_i.
$$

The controller predicts $a_k$ and $\beta_k$, where

$$
\mathbf 1^\top a_k=1,\qquad 0\leq\beta_k\leq1.
$$

The next state is

$$
z_{k+1}=\beta_k\sum_i a_{k,i}f_i+(1-\beta_k)\sum_i a_{k,i}z_i.
$$

Classical Anderson solves for the coefficients from residual least squares.
Here the controller learns them from compressed residual history and the
condition. This changes the solver, not the equilibrium equation.
""",
            ),
            code(
                prefix,
                """
from silva_networks import SILVAHyperDEQ, SolverConfig, silva_hyper_deq_loss

model = SILVAHyperDEQ(
    state_shape=8,
    condition_dim=4,
    learned_steps=5,
    history=4,
    teacher_config=SolverConfig(
        solver="broyden", max_iter=35, tol=1e-8, history=10
    ),
)
condition = torch.randn(24, 4)
teacher = model.teacher(condition)
initial_prediction = model(condition)
print("teacher residual:", teacher.residual)
print("initial learned residual:", float(initial_prediction.residual.mean()))
print("coefficient sums:", initial_prediction.coefficients[-1].sum(dim=1)[:5])
print("mixing range:", float(initial_prediction.mixing[-1].min()), float(initial_prediction.mixing[-1].max()))
""",
            ),
            md(
                prefix,
                r"""
## 2. Train the Solver Around a Fixed Transition

The high-precision state $\bar z$ supervises initialization, intermediate
states, projected residuals, and an optional task output:

$$
\mathcal L
=\lambda_i\|z_0-\bar z\|_2^2
+\lambda_t\sum_k\gamma^{K-k}\|z_k-\bar z\|_2^2
+\lambda_r\frac{1}{K+1}\sum_k\|T_\theta(z_k,x)-z_k\|_2^2
+\lambda_y\mathcal L_{\rm task}.
$$

We freeze the transition below so the experiment isolates solver learning.
Joint optimization is also available, but it asks a different question.
""",
            ),
            code(
                prefix,
                """
for parameter in model.transition.parameters():
    parameter.requires_grad_(False)

optimizer = torch.optim.Adam(
    [*model.initializer.parameters(), *model.controller.parameters()], lr=2e-2
)
loss_curve = []
residual_curve = []
for step in range(40):
    optimizer.zero_grad()
    prediction = model(condition)
    losses = silva_hyper_deq_loss(
        prediction,
        teacher.z,
        trajectory_weight=1.0,
        initializer_weight=0.5,
        residual_projection_weight=0.2,
    )
    losses.total.backward()
    optimizer.step()
    loss_curve.append(float(losses.total.detach()))
    residual_curve.append(float(prediction.residual.mean().detach()))

trained = model(condition)
print("final total loss:", loss_curve[-1])
print("final learned residual:", float(trained.residual.mean()))
print("final state RMSE:", float(torch.mean((trained.state - teacher.z).square()).sqrt()))
""",
            ),
            code(
                prefix,
                """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].plot(loss_curve, color="#2563eb")
axes[0].set(title="distillation objective", xlabel="optimizer step", ylabel="loss")
axes[0].set_yscale("log")
axes[1].plot(residual_curve, color="#d97706")
axes[1].set(title="learned-solver residual", xlabel="optimizer step", ylabel="mean residual")
axes[1].set_yscale("log")
fig.tight_layout()
plt.show()
""",
            ),
            md(
                prefix,
                r"""
## 3. Inspect the Learned Trajectory

A fast task metric is not enough. We also inspect every state error, residual,
coefficient vector, and mixing value.
""",
            ),
            code(
                prefix,
                """
for index, state in enumerate(trained.states):
    rmse = torch.mean((state - teacher.z).square()).sqrt()
    alpha = trained.coefficients[index]
    beta = trained.mixing[index]
    print(
        f"step {index + 1}: rmse={float(rmse):.6f} "
        f"residual={float(trained.residuals[index].mean()):.6f} "
        f"alpha_sum={float(alpha[0].sum()):.6f} beta={float(beta.mean()):.4f}"
    )
""",
            ),
            md(
                prefix,
                r"""
## 4. Replace the Vector Transition With a Field Architecture

The state contract is $B\times C\times H\times W$. The controller still sees
compressed residuals shaped $B\times d_r$, so no solver code changes.
""",
            ),
            code(
                prefix,
                """
from torch import nn

class FieldTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.source = nn.Conv2d(1, 2, 1)

    def forward(self, z, x):
        return torch.tanh(0.08 * self.state(z) + self.source(x))

class FieldInitializer(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)

field_model = SILVAHyperDEQ(
    state_shape=(2, 8, 8),
    condition_dim=1,
    transition=FieldTransition(),
    initializer=FieldInitializer(),
    learned_steps=3,
    history=3,
)
field_condition = torch.randn(3, 1, 8, 8)
field_result = field_model(field_condition)
field_result.state.square().mean().backward()
print("field state:", field_result.state.shape)
print("field output:", field_result.output.shape)
print("transition gradient norm:", float(field_model.transition.state.weight.grad.norm()))
""",
            ),
            md(
                prefix,
                r"""
## 5. Source-Scale Route

### Source-Scale Reproduction Route

The source study uses trained DEQ transitions on WikiText-103, ImageNet, and
Cityscapes. Reproduce the base checkpoint first, generate high-precision teacher
roots with the declared solver budget, train the initializer/controller on
cached trajectories, and report both task quality and wall-clock latency. Keep
sequence length, image resolution, batching, solver tolerance, and hardware
fixed across the classical and learned solvers [[65]] [[67]].
""",
            ),
            code(
                prefix,
                """
from silva_networks import silva_reproduction_spec

spec = silva_reproduction_spec("silva_hyper_deq")
print("equation:", spec.equation)
print("datasets:", spec.datasets)
print("repositories:", spec.repositories)
print("configurable parts:")
for item in spec.configurable_parts:
    print(" -", item)
print("benchmark obligations:")
for item in spec.benchmark_requirements:
    print(" -", item)
""",
            ),
            md(
                prefix,
                """
## Where to Go Next

| Question | Page |
| --- | --- |
| How is every loss term derived? | [Learned Solvers and Backward Approximations](https://jseluis.github.io/silva-networks/learn/solver-learning-and-gradients/) |
| How do JFB and SHINE differ? | [Backward Methods Lab](https://jseluis.github.io/silva-networks/package-notebooks/49_jfb_shine_backward_methods/) |
| Where is the public class surface? | [Learned Solver API](https://jseluis.github.io/silva-networks/api/solver_learning/) |
| Where are the full citations? | [References](https://jseluis.github.io/silva-networks/paper/references/) |
""",
            ),
        ]
    )


def backward_lab() -> dict[str, object]:
    prefix = "silva-backward-methods"
    return notebook(
        [
            md(
                prefix,
                r"""
# JFB, SHINE, and SILVA Backward Methods

This lab derives exact implicit differentiation, JFB [[88]], and SHINE [[89]],
then compares their gradients against an analytic fixed point.

[Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/49_jfb_shine_backward_methods.ipynb)
""",
            ),
            code(prefix, BOOTSTRAP),
            md(
                prefix,
                r"""
## 1. One Equation, Three Adjoint Choices

### Exact Implicit Differentiation

For $z^\star=T_\theta(z^\star,x)$,

$$
(I-J_T^\top)u=g,
\qquad
\frac{d\mathcal L}{d\theta}=u^\top\frac{\partial T_\theta}{\partial\theta}.
$$

The exact path solves for $u$ and retains the complete local inverse action.

### Jacobian-Free Backpropagation

JFB uses $u\approx g$, replacing the inverse adjoint factor by identity. This
removes the backward linear solve while preserving the converged forward root.

### Inverse Estimate Sharing

If forward Broyden retains $B\approx(J_T-I)^{-1}$, SHINE starts from

$$
u_0=-B^\top g
$$

and may refine the exact adjoint residual.

### Compare All Three Gradients

The next cell computes the analytic gradient and evaluates exact implicit, JFB,
raw SHINE, and refined SHINE routes under one fixed transition.
""",
            ),
            code(
                prefix,
                """
from silva_networks import SolverConfig, solve_equilibrium

def scalar_gradient(mode, refine_steps=0):
    bias = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
    weight = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    config = SolverConfig(
        solver="broyden" if mode == "shine" else "anderson",
        max_iter=60,
        tol=1e-11,
        history=8,
        anderson_batch_dims=0,
        backward_mode=mode,
        backward_solver="gmres",
        backward_max_iter=8,
        backward_tol=1e-11,
        shine_refine_steps=refine_steps,
    )
    result = solve_equilibrium(
        lambda z: bias + weight * z,
        torch.zeros(1, dtype=torch.float64),
        config,
        params=(bias, weight),
    )
    (0.5 * result.z.square().sum()).backward()
    return result, float(bias.grad), float(weight.grad)

exact_z = 0.4 / (1.0 - 0.2)
expected_bias = exact_z / (1.0 - 0.2)
expected_weight = exact_z**2 / (1.0 - 0.2)
print("analytic:", exact_z, expected_bias, expected_weight)
for mode, refinement in (("implicit", 0), ("jfb", 0), ("shine", 0), ("shine", 2)):
    result, bias_gradient, weight_gradient = scalar_gradient(mode, refinement)
    print(
        mode,
        "refine", refinement,
        "state", float(result.z),
        "bias grad", bias_gradient,
        "weight grad", weight_gradient,
        "backward residual", result.info.get("backward_residual"),
    )
""",
            ),
            md(
                prefix,
                r"""
## 2. Read the Difference

JFB intentionally differs from the exact gradient because it replaces the
inverse adjoint factor by identity. SHINE should approach the exact result when
the forward inverse estimate is accurate or refinement is sufficient. That is
a numerical claim we can test directly.
""",
            ),
            code(
                prefix,
                """
from silva_networks import fixed_point, shine_adjoint_solve

matrix = torch.tensor([[0.35, 0.12], [-0.08, 0.2]], dtype=torch.float64)
source = torch.tensor([0.3, -0.2], dtype=torch.float64)
forward = fixed_point(
    lambda z: source + matrix @ z,
    torch.zeros(2, dtype=torch.float64),
    SolverConfig(solver="broyden", max_iter=12, tol=1e-12, history=6),
)
gradient = torch.tensor([1.0, -0.4], dtype=torch.float64)
exact = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - matrix.T, gradient)
refinement_steps = list(range(6))
errors = []
residuals = []
for steps in refinement_steps:
    result = shine_adjoint_solve(
        lambda z: source + matrix @ z,
        forward.z,
        gradient,
        forward.inverse_estimate,
        refine_steps=steps,
        tol=1e-12,
    )
    errors.append(float(torch.linalg.vector_norm(result.x - exact)))
    residuals.append(result.residual)
    print(steps, "error", errors[-1], "residual", residuals[-1])
""",
            ),
            code(
                prefix,
                """
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.plot(refinement_steps, errors, marker="o", label="adjoint error")
ax.plot(refinement_steps, residuals, marker="s", label="linear residual")
ax.set(xlabel="SHINE refinement steps", ylabel="norm", title="shared inverse refinement")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
plt.show()
""",
            ),
            md(
                prefix,
                r"""
## 3. Broyden Inverse Factors Are Public

The forward solver represents

$$
B_k=-I+\sum_{j=1}^{r}u_jv_j^\top.
$$

`history` bounds $r$. The result can apply $B_k$, $B_k^\top$, or the
fixed-point adjoint approximation $-B_k^\top$ without a dense matrix.
""",
            ),
            code(
                prefix,
                """
estimate = forward.inverse_estimate
probe = torch.tensor([0.25, -0.5], dtype=torch.float64)
print("retained rank:", estimate.rank)
print("B probe:", estimate.apply_residual_inverse(probe))
print("B^T probe:", estimate.apply_residual_inverse_transpose(probe))
print("-(B^T) probe:", estimate.apply_fixed_point_adjoint_inverse(probe))
""",
            ),
            md(
                prefix,
                r"""
## 4. Selection Guidance

| Method | Additional backward solve | Reuses forward information | Typical reason to test it |
| --- | --- | --- | --- |
| implicit | yes | no | highest local adjoint accuracy |
| JFB | no | no | lowest backward solver cost |
| SHINE | optional | Broyden inverse factors | forward/backward numerical reuse |
| phantom | short state trajectory | final state | controllable inexact gradient |
| unrolled | no separate solve | complete finite graph | finite-depth reference |

Hold the transition, forward tolerance, optimizer, data order, and random seed
fixed when comparing these methods. Report task metric and gradient agreement
together with runtime and memory.
""",
            ),
            md(
                prefix,
                """
## Where to Go Next

| Question | Page |
| --- | --- |
| Where is the complete derivation? | [Learned Solvers and Backward Approximations](https://jseluis.github.io/silva-networks/learn/solver-learning-and-gradients/) |
| Where are all configuration fields? | [Solvers API](https://jseluis.github.io/silva-networks/api/solvers/) |
| How does this combine with other families? | [Equilibrium Expansion Atlas](https://jseluis.github.io/silva-networks/learn/equilibrium-expansion-atlas/) |
""",
            ),
        ]
    )


def quantum_lab() -> dict[str, object]:
    prefix = "silva-quantum-deq"
    return notebook(
        [
            md(
                prefix,
                r"""
# SILVA Quantum Deep Equilibrium Model

This lab reconstructs the QDEQ transition [[90]], executes a four-wire exact
statevector circuit, compares direct and implicit routes, trains a tiny task,
and lays out the ten-wire source-scale experiment.

[Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/50_silva_quantum_deq.ipynb)
""",
            ),
            code(prefix, BOOTSTRAP),
            md(
                prefix,
                r"""
## 1. Derive the Measured Equilibrium

### Derive the Circuit Transition

The input adapter produces $s=S_\psi(x)$. One tied transition is

$$
T_\theta(z,x)=\mathcal M\left(U_\theta\mathcal E(z+s)\right),
$$

and QDEQ solves

$$
z^\star=T_\theta(z^\star,x),\qquad \widehat y=Q_\omega(z^\star).
$$

For amplitude encoding,

$$
|\psi(v)\rangle=\sum_j\frac{v_j}{\|v\|_2}|j\rangle,
$$

and each measured feature is $m_j=\langle\psi|Z_j|\psi\rangle$.
""",
            ),
            code(
                prefix,
                """
from silva_networks import (
    SILVAQuantumDEQ,
    SILVAQuantumImageFilter,
    SILVAStatevectorQuantumCircuit,
    SolverConfig,
)

circuit = SILVAStatevectorQuantumCircuit(
    n_qubits=4,
    output_dim=4,
    fixed_depth=4,
    fixed_seed=1111,
)
directions = torch.randn(41, 16)
directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
measurements = circuit(directions)
print("measurement shape:", measurements.shape)
print("measurement range:", float(measurements.min()), float(measurements.max()))
""",
            ),
            code(
                prefix,
                """
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for wire in range(4):
    ax.plot(measurements[:, wire].detach(), label=f"wire {wire}")
ax.set(xlabel="encoded sample", ylabel="Pauli-Z expectation", title="four-wire measured features")
ax.legend(ncol=2)
fig.tight_layout()
plt.show()
""",
            ),
            md(
                prefix,
                r"""
## 2. Direct Warmup and Implicit Solving

Direct mode applies $z_{k+1}=T_\theta(z_k,x)$ for a finite tied depth.
Implicit mode finds a root of $T_\theta(z,x)-z$. A warmup boundary selects
the finite path before switching to the root solver.
""",
            ),
            code(
                prefix,
                """
model = SILVAQuantumDEQ(
    input_dim=6,
    output_dim=3,
    n_qubits=4,
    circuit=SILVAStatevectorQuantumCircuit(n_qubits=4, fixed_depth=2),
    direct_steps=3,
    warmup_steps=2,
    config=SolverConfig(
        solver="broyden",
        max_iter=7,
        tol=1e-4,
        history=5,
        backward_mode="jfb",
    ),
)
inputs = torch.randn(3, 6)
warmup = model(inputs, training_step=0, return_result=True)
implicit = model(inputs, training_step=3, compute_jacobian=True, return_result=True)
print("warmup:", warmup.state.shape, warmup.solver_result.solver, warmup.solver_result.residual)
print("implicit:", implicit.state.shape, implicit.solver_result.solver, implicit.solver_result.residual)
print("Jacobian penalty:", float(implicit.jacobian_penalty.detach()))
""",
            ),
            md(
                prefix,
                r"""
## 3. Tiny Classification Study

The compact task checks the complete path: injected features, repeated circuit,
measurement, readout, loss, and parameter gradients. It is a mechanism check,
not a claim about the article's dataset accuracy.
""",
            ),
            code(
                prefix,
                """
torch.manual_seed(7)
train_x = torch.randn(24, 6)
train_y = (train_x[:, :3].sum(dim=1) > train_x[:, 3:].sum(dim=1)).long()
tiny = SILVAQuantumDEQ(
    input_dim=6,
    output_dim=2,
    n_qubits=4,
    circuit=SILVAStatevectorQuantumCircuit(n_qubits=4, fixed_depth=1),
    mode="direct",
    direct_steps=2,
)
optimizer = torch.optim.Adam(tiny.parameters(), lr=2e-2)
losses = []
for step in range(18):
    optimizer.zero_grad()
    logits = tiny(train_x)
    loss = torch.nn.functional.cross_entropy(logits, train_y)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
accuracy = float((tiny(train_x).argmax(dim=1) == train_y).float().mean())
print("initial loss:", losses[0])
print("final loss:", losses[-1])
print("training accuracy:", accuracy)
""",
            ),
            code(
                prefix,
                """
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.plot(losses, color="#0f766e", marker="o", markersize=2.5)
ax.set(xlabel="optimizer step", ylabel="cross entropy", title="compact QDEQ training")
fig.tight_layout()
plt.show()
""",
            ),
            md(
                prefix,
                r"""
## 4. Source Image Contract

The four-wire path maps grayscale $28\times28$ images to 16 features. The
ten-wire path maps them to 100 features. The source study uses MNIST-4, MNIST,
Fashion-MNIST, and CIFAR-10 [[91]] [[92]] [[81]].
""",
            ),
            code(
                prefix,
                """
images = torch.randn(5, 1, 28, 28)
four_wire = SILVAQuantumImageFilter(4)(images)
ten_wire = SILVAQuantumImageFilter(10)(images)
print("four-wire image features:", four_wire.shape)
print("ten-wire image features:", ten_wire.shape)
print("four-wire statevector amplitudes:", 2**4)
print("ten-wire statevector amplitudes:", 2**10)
""",
            ),
            md(
                prefix,
                r"""
## 5. Source-Scale Route

For article-scale runs, match the official split, class subset, image
preprocessing, wire count, encoding, fixed circuit seed, trainable gate pattern,
direct/warmup/implicit schedule, forward and backward thresholds, Jacobian
weight/frequency, optimizer, update count, and evaluation metric. Use
`SILVAQuantumCircuitAdapter` to place the source circuit backend behind the same
measured-state contract.
""",
            ),
            code(
                prefix,
                """
from silva_networks import silva_reproduction_spec

spec = silva_reproduction_spec("silva_quantum_deq")
print("equation:", spec.equation)
print("datasets:", spec.datasets)
print("repositories:", spec.repositories)
print("metrics:", spec.metrics)
print("source-scale steps:")
for index, item in enumerate(spec.source_scale_steps, 1):
    print(index, item)
""",
            ),
            md(
                prefix,
                """
## Where to Go Next

| Question | Page |
| --- | --- |
| How are the circuit and fixed point derived? | [Quantum Equilibria](https://jseluis.github.io/silva-networks/learn/quantum-equilibria/) |
| Which classes are public? | [Quantum Equilibria API](https://jseluis.github.io/silva-networks/api/quantum_equilibria/) |
| How do backward choices change the experiment? | [Backward Methods Lab](https://jseluis.github.io/silva-networks/package-notebooks/49_jfb_shine_backward_methods/) |
| Where are the article and repository citations? | [References](https://jseluis.github.io/silva-networks/paper/references/) |
""",
            ),
        ]
    )


def expansion_atlas_lab() -> dict[str, object]:
    prefix = "silva-expansion-atlas"
    return notebook(
        [
            md(
                prefix,
                r"""
# SILVA Equilibrium Expansion Atlas

This lab places HyperDEQ, JFB, SHINE, monotone splitting, C-DEQ, diffusion
equilibria, PIDEQ, and QDEQ on separate architecture axes. It preserves the
existing family labs and adds a common experiment contract.

[Download this notebook](https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/51_equilibrium_expansion_atlas.ipynb)
""",
            ),
            code(prefix, BOOTSTRAP),
            md(
                prefix,
                r"""
## 1. The Complete Experiment Tuple

$$
\mathcal E=(\mathcal D,S,H,L,G,\mathcal S_f,\mathcal S_b,\mathcal L,\mathcal M).
$$

Two runs reproduce the same method only when data, transition, forward solver,
backward rule, objective, and metrics agree. A matching class name alone is not
enough.
""",
            ),
            code(
                prefix,
                """
from silva_networks import silva_reproduction_spec

families = [
    "silva_hyper_deq",
    "silva_consistency_deq",
    "diffusion_equilibrium",
    "silva_generative_equilibrium_transformer",
    "silva_fixed_point_diffusion",
    "silva_physics_guided_diffusion_pde",
    "silva_physics_informed_equilibrium",
    "silva_monotone_operator_equilibrium",
    "silva_quantum_deq",
]
for family in families:
    spec = silva_reproduction_spec(family)
    print(f"{family:43s} refs={spec.paper_refs} datasets={spec.datasets[0]}")
    print("  ", spec.equation)
""",
            ),
            md(
                prefix,
                r"""
## 2. Monotone Splitting Is a Transition Guarantee

With

$$
W=(1-m)I-A^\top A+B-B^\top,
$$

the equilibrium inclusion is

$$
0\in(I-W)z-Ux-b+\partial f(z).
$$

Forward-backward and Peaceman-Rachford are alternative splitting maps for this
same inclusion [[75]].
""",
            ),
            code(
                prefix,
                """
from silva_networks import SILVAMonotoneOperatorEquilibrium, make_monotone_operator_dataset

monotone_data = make_monotone_operator_dataset(samples=10)
for splitting in ("forward_backward", "peaceman_rachford"):
    monotone = SILVAMonotoneOperatorEquilibrium(
        4,
        6,
        2,
        splitting=splitting,
        step_size=0.5,
        margin=0.5,
    )
    result = monotone(monotone_data.inputs, return_result=True)
    print(
        splitting,
        "shape", tuple(result.output.shape),
        "certificate", float(result.monotonicity_certificate),
        "residual", result.solver_result.residual,
    )
""",
            ),
            md(
                prefix,
                r"""
## 3. PIDEQ Places Physics on the Implicit Prediction

For $z^\star(t)=T_\theta(z^\star(t),t)$,

$$
(I-J_zT_\theta)\frac{dz^\star}{dt}=\partial_tT_\theta.
$$

The physical residual compares the readout derivative to a declared dynamics
callable. The transition and the physical law are separate public objects
[[51]].
""",
            ),
            code(
                prefix,
                """
from silva_networks import SILVAPhysicsInformedEquilibrium, SolverConfig

times = torch.linspace(0, 1, 8)[:, None]
pideq = SILVAPhysicsInformedEquilibrium(
    3,
    1,
    config=SolverConfig(
        solver="picard",
        max_iter=12,
        tol=1e-6,
        anderson_batch_dims=1,
        backward_mode="jfb",
    ),
)
physics = pideq.physics_loss(
    times,
    lambda time, state: -0.5 * state,
    initial_time=times[:1],
    initial_state=torch.ones(1, 1),
    jacobian_weight=0.01,
)
print("prediction:", physics.prediction.shape)
print("time derivative:", physics.time_derivative.shape)
print("initial term:", float(physics.initial.detach()))
print("physics residual:", float(physics.residual.detach()))
print("Jacobian term:", float(physics.jacobian.detach()))
""",
            ),
            md(
                prefix,
                r"""
## 4. Diffusion Has Four Different Equilibrium Placements

| Family | Equilibrium variable |
| --- | --- |
| DEQ-DDIM | the complete deterministic diffusion trajectory [[38]] |
| GET | a one-time-injected generative token state [[48]] |
| fixed-point diffusion | a denoiser state at each timestep [[74]] |
| physics-guided diffusion PDE | reverse field state guided by residual energy [[64]] |

The placement determines the state shape, solver call count, loss, and metric.
""",
            ),
            code(
                prefix,
                """
from torch import nn
from silva_networks import SILVADiffusionEquilibrium

class ZeroDenoiser(nn.Module):
    def forward(self, value, timestep):
        return torch.zeros_like(value)

alphas = torch.linspace(0.95, 0.5, 10)
joint = SILVADiffusionEquilibrium(
    ZeroDenoiser(),
    alphas,
    (9, 6, 3, 0),
    eta=0.0,
    config=SolverConfig(max_iter=5, tol=1e-8),
)
noise = torch.randn(2, 1, 4, 4)
joint_result = joint(noise, return_result=True)
print("joint trajectory:", joint_result.trajectory.shape)
print("joint output:", joint_result.output.shape)
print("joint residual:", joint_result.solver_result.residual)
""",
            ),
            code(
                prefix,
                """
labels = ["monotone", "PIDEQ physics", "joint DDIM"]
values = [
    max(result.solver_result.residual, 1e-12),
    max(float(physics.residual.detach()), 1e-12),
    max(joint_result.solver_result.residual, 1e-12),
]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(labels, values, color=["#2563eb", "#d97706", "#0f766e"])
ax.set_yscale("log")
ax.set(ylabel="diagnostic magnitude", title="three equilibrium placements")
fig.tight_layout()
plt.show()
""",
            ),
            md(
                prefix,
                r"""
## 5. Acceleration and Backward Rules Can Cross Families

HyperDEQ and C-DEQ modify forward evaluation. JFB and SHINE modify backward
evaluation. They can be paired with Fourier, graph, physics-informed,
multiscale, diffusion, or circuit transitions after a compact convergence and
gradient check.
""",
            ),
            code(
                prefix,
                """
from silva_networks import SolverConfig

configs = {
    "exact": SolverConfig(backward_mode="implicit", backward_solver="gmres"),
    "JFB": SolverConfig(backward_mode="jfb"),
    "SHINE": SolverConfig(
        solver="broyden", backward_mode="shine", shine_refine_steps=2
    ),
    "phantom": SolverConfig(backward_mode="phantom", phantom_steps=3),
}
for name, config in configs.items():
    print(name, "forward", config.solver, "backward", config.backward_mode)
""",
            ),
            md(
                prefix,
                r"""
## 6. What a Complete Result Must Contain

Record the family and constructor, all replaceable modules, forward and
backward configurations, data source/split/preprocessing, objective terms,
optimizer and schedule, task metric, normalized residual, iteration counts,
runtime, memory, failure count, article, and research repository.

Compact results validate equations, shapes, gradients, and diagnostics. Article
reproduction additionally requires the source data, task-scale architecture,
training budget, and evaluation protocol.
""",
            ),
            md(
                prefix,
                """
## Where to Go Next

| Question | Page |
| --- | --- |
| How are all mechanisms derived? | [Equilibrium Expansion Atlas](https://jseluis.github.io/silva-networks/learn/equilibrium-expansion-atlas/) |
| How are learned solvers trained? | [Learned Solvers Lab](https://jseluis.github.io/silva-networks/package-notebooks/48_silva_learned_solvers/) |
| How is QDEQ built? | [QDEQ Lab](https://jseluis.github.io/silva-networks/package-notebooks/50_silva_quantum_deq/) |
| How are complete experiments planned? | [Reconstructing Paper Experiments](https://jseluis.github.io/silva-networks/learn/reconstructing-paper-experiments/) |
""",
            ),
        ]
    )


NOTEBOOKS = {
    "48_silva_learned_solvers.ipynb": learned_solver_lab,
    "49_jfb_shine_backward_methods.ipynb": backward_lab,
    "50_silva_quantum_deq.ipynb": quantum_lab,
    "51_equilibrium_expansion_atlas.ipynb": expansion_atlas_lab,
}


def main() -> None:
    for filename, builder in NOTEBOOKS.items():
        payload = builder()
        for directory in OUT_DIRS:
            write_notebook(
                directory / filename,
                payload,
                replace_changed=True,
                preserve_unmatched=True,
            )
        print(f"generated {filename}")


if __name__ == "__main__":
    main()
