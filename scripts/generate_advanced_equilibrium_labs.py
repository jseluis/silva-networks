"""Generate advanced SILVA equilibrium and physics-informed notebooks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

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


COMMON_IMPORTS = """
import torch
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
"""


def monotone_graph_lab() -> dict[str, object]:
    p = "silva-monotone-graph"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Monotone Graph Equilibrium

This lab derives the monotone operator parameterization, its forward-backward
step, the normalized graph operator, and the corresponding SILVA family. It
checks an exact graph-elliptic dataset, trains a compact node field, and tests
node relabeling. The mechanism follows monotone implicit graph networks [47]
and remains inside the canonical `silva_monotone_graph_equilibrium` family.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                COMMON_IMPORTS
                + """
from silva_networks import (
    SILVAMonotoneGraphEquilibrium,
    SolverConfig,
    make_monotone_chain_dataset,
    normalized_laplacian_field,
)

torch.manual_seed(210)
""",
            ),
            md(
                p,
                r"""
## 1. Graph Operator and Shape Contract

For node state $Z\in\mathbb R^{N\times d}$, define

$$
G=\frac12\left(I-D^{-1/2}AD^{-1/2}\right),
\qquad GZ\in\mathbb R^{N\times d}.
$$

The factor $1/2$ places the normalized-Laplacian spectrum in $[0,1]$. The
package accepts a directed edge list; bidirectional edges represent an
undirected graph.
""",
            ),
            code(
                p,
                """
data = make_monotone_chain_dataset(nodes=12, channels=1, diffusion=0.6, seed=21)
graph_field = normalized_laplacian_field(data.target, data.edge_index)
equation_error = data.equation_residual().abs().max()

assert data.source.shape == data.target.shape == (12, 1)
assert graph_field.shape == data.target.shape
assert equation_error < 1e-6
print("edges:", data.edge_index.shape[1])
print("maximum graph-equation residual:", float(equation_error))
""",
            ),
            md(
                p,
                r"""
## 2. Monotone Channel Parameterization

The channel operator is not an unconstrained matrix. It is formed as

$$
W=(1-m)I-CC^T+F-F^T,
\qquad m>0.
$$

Because the skew term vanishes in the symmetric part,

$$
I-\frac{W+W^T}{2}=mI+CC^T\succeq mI.
$$

The smallest eigenvalue is therefore a directly testable certificate. This is
the stability constraint represented by
`SILVAMonotoneGraphTransition.monotonicity_certificate()`.
""",
            ),
            md(
                p,
                r"""
## 3. Forward-Backward Step as a SILVA Transition

With source $B(X)$ and proximal activation, one operator-splitting step is

$$
Z^{k+1}=\operatorname{prox}_{\alpha f}
\left((1-\alpha)Z^k+\alpha(WGZ^k+B(X))\right).
$$

In SILVA, $B(X)$ is the source branch, $WGZ$ is the graph-local branch, and
the proximal map is the output nonlinearity. Reusing this transition until
convergence produces one implicit graph point rather than an explicit stack.
""",
            ),
            code(
                p,
                """
model = SILVAMonotoneGraphEquilibrium(
    in_dim=1,
    state_dim=6,
    out_dim=1,
    margin=0.15,
    step_size=0.7,
    config=SolverConfig(solver="picard", max_iter=25, tol=1e-6),
)
initial = model(data.source, data.edge_index, return_result=True)

assert initial.output.shape == data.target.shape
assert initial.monotonicity_certificate >= 0.15 - 1e-6
print("certificate:", float(initial.monotonicity_certificate))
print("equilibrium residual:", initial.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 4. Tiny Equation-Supervised Task

The target solves

$$
(I+\nu G)u=s.
$$

The loss below teaches the readout and equilibrium transition to approximate
that solution. This is a deterministic small-scale reproduction of the graph
mechanism, not a citation-network benchmark.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
losses = []
for _ in range(12):
    optimizer.zero_grad()
    prediction = model(data.source, data.edge_index)
    loss = torch.nn.functional.mse_loss(prediction, data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

trained = model(data.source, data.edge_index, return_result=True)
print("initial/final task loss:", losses[0], losses[-1])
print("final equilibrium residual:", trained.solver_result.residual)
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
axes[0].plot(losses, marker="o", markersize=2)
axes[0].set(xlabel="optimization step", ylabel="MSE", yscale="log")
axes[1].plot(data.target[:, 0], label="exact", linewidth=2)
axes[1].plot(trained.output.detach()[:, 0], "--", label="SILVA")
axes[1].set(xlabel="node", ylabel="field")
axes[1].legend()
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 5. Node Relabeling

For permutation matrix $P$, a graph equilibrium must satisfy

$$
F(PX,PEP^T)=PF(X,E).
$$

The edge list must be relabeled with the nodes. This is different from
permuting features while leaving graph topology unchanged.
""",
            ),
            code(
                p,
                """
permutation = torch.tensor([7, 1, 10, 3, 5, 9, 0, 11, 2, 8, 4, 6])
inverse = torch.empty_like(permutation)
inverse[permutation] = torch.arange(permutation.numel())
permuted_edges = inverse[data.edge_index]

with torch.no_grad():
    reference = model(data.source, data.edge_index)
    relabeled = model(data.source[permutation], permuted_edges)
equivariance_error = (relabeled - reference[permutation]).abs().max()
assert equivariance_error < 1e-5
print("relabeling error:", float(equivariance_error))
""",
            ),
            md(
                p,
                r"""
## 6. What to Report

Record the graph normalization, directed-edge convention, margin $m$, proximal
map, forward-backward step size, solver, fixed-point residual, and task metric.
The monotonicity certificate diagnoses the parameterization; it does not replace
the numerical residual or downstream accuracy.
""",
            ),
        ]
    )


def transformer_lab() -> dict[str, object]:
    p = "silva-equilibrium-transformer"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Generative Equilibrium Transformer

This lab separates one-time source encoding from the weight-tied transformer
equilibrium, derives QKV injection, trains a tiny teacher-matching task, and
checks optional class conditioning. The architecture mechanism follows the
Generative Equilibrium Transformer [48]; SILVA supplies the general source,
state, solver, and diagnostic contract.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                COMMON_IMPORTS
                + """
from silva_networks import (
    SILVAGenerativeEquilibriumTransformer,
    SolverConfig,
    make_teacher_image_pairs,
    silva_distillation_loss,
)

torch.manual_seed(220)
""",
            ),
            md(
                p,
                r"""
## 1. Patches Become Source Tokens

For patch size $p$, an image $x\in\mathbb R^{B\times C\times H\times W}$
becomes

$$
X_p\in\mathbb R^{B\times N\times d},
\qquad N=\frac Hp\frac Wp.
$$

A patch convolution performs extraction and projection together. Fixed 2D
sine/cosine positions are added before the injection transformer.
""",
            ),
            code(
                p,
                """
data = make_teacher_image_pairs(samples=4, channels=1, height=6, width=6, seed=22)
assert data.equation_residual().abs().max() == 0
print("noise/target:", tuple(data.noise.shape), tuple(data.target.shape))
""",
            ),
            md(
                p,
                r"""
## 2. Injection Is Computed Once

Let $I_\phi$ be a finite injection transformer. It computes

$$
U=I_\phi(X_p),
\qquad
(U_1,\ldots,U_L)=W_UU,
$$

where every $U_\ell\in\mathbb R^{B\times N\times3d}$ supplies query, key,
and value offsets for one internal equilibrium block. $I_\phi$ is outside the
root solve and is therefore evaluated once per model call.
""",
            ),
            md(
                p,
                r"""
## 3. QKV-Injected Equilibrium Block

At internal block $\ell$,

$$
(Q,K,V)=Z W_{qkv}^{(\ell)}+U_\ell+C_y,
$$

$$
A_\ell=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V,
$$

$$
\widetilde Z=Z+A_\ell,
\qquad
Z^+=\tanh\left(s[\widetilde Z+operatorname{FFN}(\widetilde Z)]\right).
$$

$C_y$ is optional class injection. The final bounded map is SILVA's compact
stability envelope; the one-time QKV-injection mechanism is unchanged.
""",
            ),
            code(
                p,
                """
config = SolverConfig(
    solver="picard",
    max_iter=12,
    tol=1e-5,
    anderson_batch_dims=1,
)
model = SILVAGenerativeEquilibriumTransformer(
    in_channels=1,
    patch_size=2,
    hidden_dim=8,
    heads=2,
    injection_depth=1,
    equilibrium_depth=2,
    state_scale=0.15,
    config=config,
)
result = model(data.noise, return_result=True)
assert result.output.shape == data.target.shape
assert result.state.shape == (4, 9, 8)
assert result.injection.shape == (4, 9, 48)
print("solver residual:", result.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 4. One-Step Teacher Matching

The architectural fixed point and the distillation objective answer different
questions. The equilibrium determines the hidden representation. The teaching
loss matches a supplied target:

$$
\mathcal L_{\mathrm{distill}}
=\frac1{BCHW}\|Q(Z^\star)-x_{\mathrm{teacher}}\|_2^2.
$$

The generated smoothing pairs make this pipeline executable without claiming a
diffusion benchmark or a pretrained teacher.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for _ in range(8):
    optimizer.zero_grad()
    prediction = model(data.noise)
    loss = silva_distillation_loss(prediction, data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

trained = model(data.noise, return_result=True)
print("initial/final distillation loss:", losses[0], losses[-1])
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 4, figsize=(7.6, 2.0))
axes[0].imshow(data.noise[0, 0], cmap="gray")
axes[0].set_title("source")
axes[1].imshow(data.target[0, 0], cmap="gray")
axes[1].set_title("teacher")
axes[2].imshow(trained.output.detach()[0, 0], cmap="gray")
axes[2].set_title("equilibrium")
axes[3].plot(losses, marker="o", markersize=2)
axes[3].set(xlabel="step", ylabel="MSE", yscale="log")
for axis in axes[:3]:
    axis.axis("off")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 5. Class Conditioning

When `classes` is configured, the class embedding produces another
$3dL$-dimensional source that is split across equilibrium blocks. It shifts
Q, K, and V without changing the token shape or adding class tokens.
""",
            ),
            code(
                p,
                """
conditioned = SILVAGenerativeEquilibriumTransformer(
    in_channels=1,
    patch_size=2,
    hidden_dim=8,
    heads=2,
    equilibrium_depth=1,
    classes=3,
    config=SolverConfig(max_iter=3, anderson_batch_dims=1),
)
conditioned_output = conditioned(data.noise[:2], labels=torch.tensor([0, 2]))
assert conditioned_output.shape == data.target[:2].shape
print("conditioned output:", tuple(conditioned_output.shape))
""",
            ),
            md(
                p,
                r"""
## 6. Scaling the Experiment

For a full distillation study, replace the generated pairs with the exact
teacher checkpoint, sampling schedule, image preprocessing, class protocol,
and evaluation metrics from the target experiment. Report both image quality
and equilibrium convergence; parameter count alone does not establish a
distillation result.
""",
            ),
        ]
    )


def poisson_lab() -> dict[str, object]:
    p = "silva-poisson-mirror"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Poisson Mirror Equilibrium

This lab derives the Poisson data term, the Burg mirror map, its closed-form
positive update, and the fixed-point reconstruction. It verifies a seeded
Poisson imaging problem and compares data fidelity before and after the
equilibrium. The mechanism follows DEQ-MD [50] inside SILVA.
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                COMMON_IMPORTS
                + """
from silva_networks import (
    SILVABurgMirrorTransition,
    SILVAPoissonMirrorEquilibrium,
    SolverConfig,
    make_poisson_inverse_dataset,
    poisson_kl,
)

torch.manual_seed(230)
""",
            ),
            md(
                p,
                r"""
## 1. Poisson Observation Model

For nonnegative image $x$ and forward operator $A$,

$$
y_i\sim\operatorname{Poisson}((Ax)_i).
$$

Ignoring terms independent of $x$, the data fidelity is the generalized KL
divergence

$$
D_{\mathrm{KL}}(y,Ax)
=\sum_i y_i\log\frac{y_i}{(Ax)_i}+(Ax)_i-y_i.
$$

Its gradient is

$$
\nabla_xD_{\mathrm{KL}}=A^T\left(1-\frac{y}{Ax}\right).
$$
""",
            ),
            code(
                p,
                """
data = make_poisson_inverse_dataset(samples=3, height=8, width=8, exposure=30, seed=23)
assert data.expected_equation_residual().abs().max() == 0
assert data.clean.min() > 0 and data.observation.min() >= 0
print("clean/observed:", tuple(data.clean.shape), tuple(data.observation.shape))
print("clean-image KL:", float(data.data_fidelity(data.clean)))
""",
            ),
            md(
                p,
                r"""
## 2. Why Euclidean Descent Is Not the Same Update

Ordinary gradient descent adds a vector in Euclidean coordinates. Burg entropy

$$
h(x)=-\sum_i\log x_i,
\qquad \nabla h(x)=-x^{-1},
$$

defines geometry on the positive orthant. A mirror step with learned
regularizer gradient $r_\theta(x)$ is

$$
x^+=\nabla h^*\left(
\nabla h(x)-\tau[\nabla D_{\mathrm{KL}}+r_\theta(x)]
\right).
$$
""",
            ),
            md(
                p,
                r"""
## 3. Closed-Form Burg Update

Substituting $\nabla h(x)=-1/x$ gives

$$
x^+
=\frac{x}{1+\tau x\odot
\left[A^T\left(1-\frac{y}{Ax}\right)+r_\theta(x)\right]}.
$$

`SILVABurgMirrorTransition` applies this expression and a positive box
projection. The denominator floor is a numerical safeguard for compact runs;
full experiments should report any line search or backtracking policy.
""",
            ),
            code(
                p,
                """
transition = SILVABurgMirrorTransition(
    forward_operator=data.forward_operator,
    adjoint_operator=data.adjoint_operator,
    step_size=0.05,
    minimum=1e-4,
    maximum=3.0,
)
initial = data.observation.clamp_min(1e-4)
one_step = transition(initial, data.observation)
assert one_step.min() > 0
print("KL before/after one mirror step:",
      float(poisson_kl(data.observation, data.forward_operator(initial))),
      float(poisson_kl(data.observation, data.forward_operator(one_step))))
""",
            ),
            md(
                p,
                r"""
## 4. Mirror Step as a SILVA Equilibrium

The source is the observed count field $y$. The local/global field is the
forward-adjoint physics pair $A,A^T$. The self branch is the optional learned
regularizer gradient. Reusing the mirror transition gives

$$
x^\star=T_{\mathrm{Burg}}(x^\star;y).
$$

At a positive interior fixed point, the combined data and regularizer gradient
vanishes.
""",
            ),
            code(
                p,
                """
model = SILVAPoissonMirrorEquilibrium(
    transition=transition,
    config=SolverConfig(
        solver="picard",
        max_iter=20,
        tol=1e-6,
        anderson_batch_dims=1,
    ),
)
result = model(data.observation, z0=initial, return_result=True)
final_kl = poisson_kl(data.observation, result.intensity)
assert result.output.min() > 0
print("equilibrium residual:", result.solver_result.residual)
print("final KL:", float(final_kl))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 4, figsize=(7.8, 2.0))
axes[0].imshow(data.clean[0, 0], cmap="magma")
axes[0].set_title("clean")
axes[1].imshow(data.observation[0, 0], cmap="magma")
axes[1].set_title("observed")
axes[2].imshow(result.output.detach()[0, 0], cmap="magma")
axes[2].set_title("mirror state")
axes[3].semilogy(result.solver_result.residuals)
axes[3].set(xlabel="iteration", ylabel="fixed-point residual")
for axis in axes[:3]:
    axis.axis("off")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 5. Learned Regularizer Contract

`regularizer_gradient` must preserve the image shape and return a gradient-like
field. A convolutional network, U-Net, or another SILVA point architecture can
occupy this branch. If the module is intended to be the gradient of a scalar
regularizer, that integrability claim requires a separate check; shape
preservation alone does not prove it.
""",
            ),
            md(
                p,
                r"""
## 6. Reproduction Checklist

Report the sensing operator and adjoint test, count scaling, exposure, box
bounds, mirror step, denominator or line-search safeguard, equilibrium solver,
residual, reconstruction metric, and learned regularizer architecture. This
small dataset validates the implementation path; it is not a medical or
astronomical benchmark.
""",
            ),
        ]
    )


def physics_informed_lab() -> dict[str, object]:
    p = "silva-physics-informed"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Physics-Informed Equilibrium

This lab derives a physics-informed equilibrium for an ODE initial-value
problem, computes time derivatives with the implicit function theorem, and
trains a tiny linear-decay task with boundary, equation, and Jacobian terms.
The construction follows Physics-Informed Deep Equilibrium Models [51].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                COMMON_IMPORTS
                + """
from silva_networks import (
    SILVAPhysicsInformedEquilibrium,
    SolverConfig,
    make_linear_ivp_dataset,
)

torch.manual_seed(240)
""",
            ),
            md(
                p,
                r"""
## 1. Initial-Value Problem

Consider

$$
\frac{dy}{dt}=N(t,y(t)),
\qquad y(t_0)=y_0.
$$

A physics-informed model is trained at collocation times without requiring a
target state at every point. The package's analytic batch supplies targets only
so this notebook can measure error after training.
""",
            ),
            code(
                p,
                """
data = make_linear_ivp_dataset(points=9, final_time=1.5, rate=-0.5)
assert data.equation_residual().abs().max() == 0
assert torch.allclose(data.target[:1], data.initial_state)
print("collocation points:", data.times.shape[0])
print("exact final state:", float(data.target[-1]))
""",
            ),
            md(
                p,
                r"""
## 2. State Is Defined Implicitly

Instead of evaluating a finite stack, define

$$
z^\star(t)=f_\theta(z^\star(t),t),
\qquad \widehat y(t)=Q_\psi(z^\star(t)).
$$

In SILVA, time enters through the source branch and the latent state enters
through the self-interaction branch. A standard output loss can use the
package's implicit adjoint, avoiding storage of every forward solver iterate.
""",
            ),
            md(
                p,
                r"""
## 3. Time Derivative from the Implicit Function Theorem

Differentiate the fixed-point equation:

$$
\frac{dz^\star}{dt}
=J_zf_\theta\frac{dz^\star}{dt}+J_tf_\theta.
$$

Therefore

$$
\frac{dz^\star}{dt}
=(I-J_zf_\theta)^{-1}J_tf_\theta,
\qquad
\frac{d\widehat y}{dt}=J_Q\frac{dz^\star}{dt}.
$$

`implicit_time_derivative` solves this system exactly with dense Jacobians. It
is transparent and differentiable for small latent dimensions; large systems
should replace the dense solve with matrix-free products.
""",
            ),
            code(
                p,
                """
model = SILVAPhysicsInformedEquilibrium(
    state_dim=4,
    output_dim=1,
    state_scale=0.15,
    config=SolverConfig(
        solver="picard",
        max_iter=15,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
initial = model(data.times, return_result=True)
derivative = model.implicit_time_derivative(data.times, initial.state)
assert derivative.shape == initial.output.shape == data.target.shape
print("equilibrium residual:", initial.solver_result.residual)
""",
            ),
            md(
                p,
                r"""
## 4. Three-Term Physics-Informed Objective

The decomposed objective is

$$
\mathcal J_b=\|\widehat y(t_0)-y_0\|_2^2,
$$

$$
\mathcal J_N
=\frac1M\sum_{i=1}^M
\left\|\frac{d\widehat y(t_i)}{dt}
-N(t_i,\widehat y(t_i))\right\|_2^2,
$$

$$
\mathcal J
=\mathcal J_b+\lambda\mathcal J_N
+\kappa\|J_zf_\theta\|_F^2.
$$

The Jacobian term is estimated with Rademacher probes [6, 14, 51]. It is a
solver-conditioning term, not the differential-equation residual.
""",
            ),
            code(
                p,
                """
loss = model.physics_loss(
    data.times,
    data.dynamics,
    initial_time=data.times[:1],
    initial_state=data.initial_state,
    physics_weight=1.0,
    jacobian_weight=1e-3,
    jacobian_samples=1,
)
print("boundary:", float(loss.initial))
print("ODE residual:", float(loss.residual))
print("Jacobian estimate:", float(loss.jacobian))
""",
            ),
            md(
                p,
                r"""
## 5. Tiny Physics-Only Training Run

No trajectory targets appear in the optimization objective below. They are
used afterward only to calculate an error curve. This separation is essential
when describing a physics-informed experiment.
""",
            ),
            code(
                p,
                """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
history = []
for _ in range(8):
    optimizer.zero_grad()
    terms = model.physics_loss(
        data.times,
        data.dynamics,
        initial_time=data.times[:1],
        initial_state=data.initial_state,
        jacobian_weight=1e-3,
    )
    terms.total.backward()
    optimizer.step()
    history.append((float(terms.initial.detach()), float(terms.residual.detach())))

trained = model(data.times, return_result=True)
trajectory_mse = torch.nn.functional.mse_loss(trained.output, data.target)
print("trajectory MSE used for evaluation:", float(trajectory_mse))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
axes[0].plot(data.times[:, 0], data.target[:, 0], label="exact", linewidth=2)
axes[0].plot(data.times[:, 0], trained.output.detach()[:, 0], "--", label="SILVA")
axes[0].set(xlabel="time", ylabel="state")
axes[0].legend()
axes[1].semilogy([item[0] for item in history], label="boundary")
axes[1].semilogy([item[1] for item in history], label="ODE residual")
axes[1].set(xlabel="optimization step", ylabel="loss")
axes[1].legend()
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 6. Stiff Systems and Scaling

An equilibrium layer does not automatically solve stiffness. Stability depends
on the transition, root solver, Jacobian spectrum, collocation distribution,
and loss balancing. For larger states, use matrix-free implicit products,
report forward and backward tolerances separately, and compare against a
trusted numerical integrator on the same time interval.
""",
            ),
        ]
    )


def dae_and_residual_lab() -> dict[str, object]:
    p = "silva-dae-residual"
    return notebook(
        [
            md(
                p,
                r"""
# SILVA Implicit DAE and Adversarial Residual Lab

This lab derives an implicit Runge-Kutta layer for index-1
differential-algebraic equations, validates one- and two-stage roots, and then
adds an optional adversarial residual objective. DAE-PINNs motivate the
implicit stage construction [52]. The similarly named DEQGAN uses “DEQ” to
mean “Differential Equation,” not “Deep Equilibrium” [53].
""",
            ),
            code(p, BOOTSTRAP),
            code(
                p,
                COMMON_IMPORTS
                + """
from silva_networks import (
    SILVAImplicitDAEStep,
    SILVAResidualDiscriminator,
    make_linear_dae_dataset,
    silva_adversarial_residual_loss,
)

torch.manual_seed(250)
""",
            ),
            md(
                p,
                r"""
## 1. Semi-Explicit Index-1 DAE

Separate differential and algebraic states:

$$
\dot y=f(y,z),
\qquad 0=g(y,z).
$$

The teaching system is

$$
\dot y=-y+z,
\qquad g(y,z)=z-\frac12y.
$$

Eliminating $z$ gives $\dot y=-y/2$, so
$y(t)=y_0e^{-t/2}$ and $z(t)=y(t)/2$.
""",
            ),
            code(
                p,
                """
data = make_linear_dae_dataset(steps=8, dimensions=1, step_size=0.1)
assert data.constraint_residual().abs().max() == 0
print("trajectory:", tuple(data.differential.shape))
print("exact final differential state:", float(data.differential[-1]))
""",
            ),
            md(
                p,
                r"""
## 2. Implicit Runge-Kutta Stage Equations

For $s$ stages and Butcher coefficients $(A,b,c)$,

$$
Y_j=y_n+h\sum_{i=1}^s a_{ji}f(Y_i,Z_i),
\qquad g(Y_j,Z_j)=0,
$$

$$
y_{n+1}=y_n+h\sum_{i=1}^s b_i f(Y_i,Z_i),
\qquad g(y_{n+1},z_{n+1})=0.
$$

All stage and endpoint algebraic variables form one root vector. A damped
Newton solve makes this root system an implicit SILVA layer. The dynamics may
be known, learned, or a composition of both.
""",
            ),
            md(
                p,
                r"""
## 3. Backward Euler Is the One-Stage Case

With $A=[1]$, $b=[1]$, and $c=[1]$,

$$
y_{n+1}=y_n+h(-y_{n+1}+z_{n+1}),
\qquad z_{n+1}=\frac12y_{n+1}.
$$

Therefore

$$
y_{n+1}=\frac{y_n}{1+h/2}.
$$
""",
            ),
            code(
                p,
                """
backward_euler = SILVAImplicitDAEStep(max_iter=6, tol=1e-8)
step = backward_euler(
    data.differential[:1],
    data.algebraic[:1],
    data.step_size,
    data.dynamics,
    data.constraint,
)
discrete_exact = data.differential[:1] / (1.0 + data.step_size / 2.0)
assert step.residual < 1e-7
assert torch.allclose(step.differential, discrete_exact, atol=1e-6)
print("root residual:", step.residual)
print("differential/algebraic:", float(step.differential), float(step.algebraic))
""",
            ),
            md(
                p,
                r"""
## 4. Two-Stage Gauss-Legendre Layer

The fourth-order two-stage tableau is

$$
A=\begin{bmatrix}
1/4 & 1/4-\sqrt3/6\\
1/4+\sqrt3/6 & 1/4
\end{bmatrix},
\quad
b=\begin{bmatrix}1/2&1/2\end{bmatrix}.
$$

Changing the tableau changes the discretization while preserving the public
DAE root contract.
""",
            ),
            code(
                p,
                """
root_three = 3.0**0.5
gauss = SILVAImplicitDAEStep(
    a=torch.tensor([[0.25, 0.25-root_three/6], [0.25+root_three/6, 0.25]]),
    b=torch.tensor([0.5, 0.5]),
    c=torch.tensor([0.5-root_three/6, 0.5+root_three/6]),
    max_iter=6,
    tol=1e-8,
)
gauss_step = gauss(
    data.differential[:1], data.algebraic[:1], data.step_size,
    data.dynamics, data.constraint,
)
assert gauss_step.stage_differential.shape == (1, 2, 1)
assert gauss_step.residual < 1e-6
print("two-stage result:", float(gauss_step.differential))
""",
            ),
            md(
                p,
                r"""
## 5. Roll Out the Implicit Layer

Each step solves a local root. A trajectory is a sequence of such implicit
points; it is not one globally weight-tied deep-equilibrium state. This
classification keeps DAE time stepping distinct from a Bai-style DEQ while
still placing the implicit layer inside SILVA.
""",
            ),
            code(
                p,
                """
y = data.differential[:1]
z = data.algebraic[:1]
trajectory = [y.detach().squeeze()]
root_residuals = []
for _ in range(8):
    result = gauss(y, z, data.step_size, data.dynamics, data.constraint)
    y, z = result.differential, result.algebraic
    trajectory.append(y.detach().squeeze())
    root_residuals.append(result.residual)
trajectory = torch.stack(trajectory)
print("maximum rollout root residual:", max(root_residuals))
""",
            ),
            code(
                p,
                """
figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
axes[0].plot(data.times[:, 0], data.differential[:, 0], label="continuous exact")
axes[0].plot(data.times[:, 0], trajectory, "--", label="implicit rollout")
axes[0].set(xlabel="time", ylabel="differential state")
axes[0].legend()
axes[1].semilogy(root_residuals, marker="o", markersize=3)
axes[1].set(xlabel="time step", ylabel="root residual")
figure.tight_layout()
plt.show()
""",
            ),
            md(
                p,
                r"""
## 6. Adversarial Residual Objective and Naming Boundary

An optional discriminator can compare equation residuals $r_\theta$ with a
near-zero reference distribution $r_0$. With discriminator $D_\omega$,

$$
\mathcal L_D
=-\mathbb E\log D_\omega(r_0)
-\mathbb E\log(1-D_\omega(r_\theta)),
$$

$$
\mathcal L_G=-\mathbb E\log D_\omega(r_\theta).
$$

This objective follows the differential-equation GAN work [53]. It is not an
equilibrium solver and is deliberately exposed as
`silva_adversarial_residual_loss`, not as a selectable DEQ family.
""",
            ),
            code(
                p,
                """
discriminator = SILVAResidualDiscriminator(residual_dim=1, hidden_dim=8, depth=1)
equation_residual = data.constraint(data.differential, data.algebraic)
equation_residual = equation_residual + 0.02 * torch.randn_like(equation_residual)
losses = silva_adversarial_residual_loss(
    discriminator,
    equation_residual,
    reference=torch.zeros_like(equation_residual),
    instance_noise=0.005,
)
assert torch.isfinite(losses.generator + losses.discriminator)
print("generator/discriminator:", float(losses.generator), float(losses.discriminator))
""",
            ),
            md(
                p,
                r"""
## 7. Choosing the Physics Construction

Use a physics-informed equilibrium when the solution representation itself is
a fixed point and implicit time derivatives are needed. Use an implicit DAE
step when every time advance must satisfy differential and algebraic stage
equations. Add adversarial residual matching only when distributional residual
training is part of the study; it supplements rather than replaces direct
constraint, root, and trajectory diagnostics.
""",
            ),
        ]
    )


NOTEBOOKS = {
    "21_silva_monotone_graph_equilibrium.ipynb": monotone_graph_lab,
    "22_silva_generative_equilibrium_transformer.ipynb": transformer_lab,
    "23_silva_poisson_mirror_equilibrium.ipynb": poisson_lab,
    "24_silva_physics_informed_equilibrium.ipynb": physics_informed_lab,
    "25_silva_implicit_dae_and_residuals.ipynb": dae_and_residual_lab,
}


def main() -> None:
    for name, builder in NOTEBOOKS.items():
        payload = json.dumps(builder(), indent=2) + "\n"
        for directory in OUT_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            path.write_text(payload, encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
