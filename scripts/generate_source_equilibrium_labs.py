"""Generate source-aligned equilibrium family labs 61 through 74."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = (ROOT / "notebooks/package_api", ROOT / "docs/package-notebooks", ROOT / "colab")
COUNTERS: defaultdict[str, int] = defaultdict(int)


@dataclass(frozen=True)
class LabSpec:
    filename: str
    family: str
    title: str
    ref: int
    summary: str
    derivation: str
    shape_contract: str
    setup: str
    plot: str
    replaceable: str
    source_scale: str


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
from torch import nn
from silva_networks import SolverConfig, silva_family_experiment_protocol

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
torch.manual_seed(121)

config = SolverConfig(
    solver="picard",
    max_iter=30,
    tol=1e-6,
    backward_mode="unrolled",
    anderson_batch_dims=1,
    return_best=True,
)
"""


SPECS = (
    LabSpec(
        "61_silva_lipschitz_mdeq.ipynb",
        "silva_lipschitz_mdeq",
        "SILVA Lipschitz Multiscale Equilibrium",
        99,
        "Couple several resolution states in one contractive equilibrium and inspect the bound that controls the complete cross-scale map.",
        r"""
Let $z=[z_1;\ldots;z_R]$ concatenate all resolution branches. The recurrent matrix is normalized once for the coupled system,

$$
\widehat W=\rho\frac{W}{\max(1,\lVert W\rVert_\infty)},\qquad
z^\star=\tanh\!\left(S_\theta(x)+\widehat Wz^\star+b\right).
$$

Because $\tanh$ is 1-Lipschitz and $\lVert\widehat W\rVert_\infty\leq\rho<1$,
$\lVert T(z)-T(v)\rVert_\infty\leq\rho\lVert z-v\rVert_\infty$. Banach's theorem therefore gives a unique fixed point. Splitting $z^\star$ after the solve recovers every scale without weakening the joint certificate.
""",
        "Input: `(batch, input_dim)`. Solved state: `(batch, sum(scale_dims))`. Each returned scale has its declared final width; the readout consumes the concatenated equilibrium.",
        """
from silva_networks import SILVALipschitzMultiscaleEquilibrium

x = torch.randn(12, 4, requires_grad=True)
model = SILVALipschitzMultiscaleEquilibrium(
    4, (8, 4, 2), 3, contraction=0.65, config=config
)
result = model(x, return_result=True)
parts = model.split_state(result.state)
loss = result.output.square().mean()
loss.backward()
scale_norms = torch.tensor([part.detach().norm(dim=-1).mean() for part in parts])
summary = {
    "state_shape": tuple(result.state.shape),
    "scale_shapes": [tuple(part.shape) for part in parts],
    "lipschitz_bound": float(model.lipschitz_bound().detach()),
    "residual": result.solver_result.residual,
    "iterations": result.solver_result.iterations,
    "input_grad_norm": float(x.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].bar(["fine", "middle", "coarse"], scale_norms, color=["#2563eb", "#d97706", "#059669"])
axes[0].set(title="equilibrium energy by scale", ylabel="mean state norm")
axes[1].semilogy(result.solver_result.residuals, marker="o", color="#7c3aed")
axes[1].set(title="coupled fixed-point solve", xlabel="iteration", ylabel="residual")
fig.tight_layout()
plt.show()
""",
        "Replace `injection` with a multiresolution stem and `readout` with a task head. The cross-scale recurrent parameter stays visible, and `split_state` exposes each branch for auxiliary losses, pyramidal decoders, or independent diagnostics.",
        "For ImageNet or Cityscapes-scale work, preserve the paper's simultaneous resolutions, normalization rule, augmentation, and evaluation protocol. Replace the compact vectors with convolutional feature pyramids and record both task metrics and the measured global bound.",
    ),
    LabSpec(
        "62_silva_subhomogeneous_equilibrium.ipynb",
        "silva_subhomogeneous_equilibrium",
        "SILVA Subhomogeneous Equilibrium",
        100,
        "Construct a positive normalized fixed point whose projective geometry provides a convergence route beyond ordinary Euclidean contraction tests.",
        r"""
The translated positive map is

$$
g_x(z)=\left[\tanh(Wz)+f_\theta(x)+a\right]^q,
\qquad a>1,\quad 0<q\leq1,
$$

and the equilibrium transition normalizes it on the positive $p$-sphere,

$$T_x(z)=\frac{g_x(z)}{\lVert g_x(z)\rVert_p},\qquad z^\star=T_x(z^\star).$$

Translation keeps every component strictly positive; the exponent controls subhomogeneity; normalization removes radial growth. The resulting state is interpreted projectively, so positivity and the norm constraint are primary numerical checks.
""",
        "Input: `(batch, input_dim)`. State: `(batch, state_dim)`, strictly positive with unit `p`-norm. Output: `(batch, output_dim)`.",
        """
from silva_networks import SILVASubhomogeneousEquilibrium

x = torch.randn(10, 3, requires_grad=True)
model = SILVASubhomogeneousEquilibrium(
    3, 9, 2, norm_p=4.0, power=0.75, config=config
)
result = model(x, return_result=True)
norms = torch.linalg.vector_norm(result.state, ord=4.0, dim=-1)
result.output.square().mean().backward()
summary = {
    "minimum_state": float(result.state.min().detach()),
    "maximum_norm_error": float((norms - 1).abs().max().detach()),
    "residual": result.solver_result.residual,
    "iterations": result.solver_result.iterations,
    "input_grad_norm": float(x.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].imshow(result.state.detach(), aspect="auto", cmap="viridis")
axes[0].set(title="positive equilibrium states", xlabel="state coordinate", ylabel="sample")
axes[1].semilogy(result.solver_result.residuals, marker="o", color="#dc2626")
axes[1].set(title="normalized fixed-point solve", xlabel="iteration", ylabel="residual")
fig.tight_layout()
plt.show()
""",
        "Replace `input_map`, `state_map`, or `readout`, while retaining strict positivity and explicit normalization when the theoretical guarantee is required. `translation`, `power`, and `norm_p` expose the projective construction directly.",
        "A source-scale study must preserve the article's positive architecture, data transformations, norm, optimizer, and stopping rule. Compare projective and Euclidean residuals and report any departure from the positivity assumptions.",
    ),
    LabSpec(
        "63_silva_algorithmic_reasoner.ipynb",
        "silva_algorithmic_reasoner",
        "SILVA Algorithmic Equilibrium Reasoner",
        101,
        "Solve a graph algorithm state as an equilibrium, then expose the processor and readout needed to move from a toy cycle to CLRS-style algorithmic tasks.",
        r"""
For directed edges $j\to i$, messages and their degree-normalized aggregate are

$$m_{ji}=P_\theta([z_j,z_i]),\qquad
\bar m_i=\frac{1}{\max(1,d_i)}\sum_{j\to i}m_{ji}.$$

SILVA solves

$$z_i^\star=\tanh\!\left(E_\theta(x_i)+\rho\bar m_i(z^\star)\right),$$

then decodes node, edge, or graph outputs. The processor is tied across the implicit depth; changing the root solver changes evaluation, not the algorithm represented by the fixed-point equation.
""",
        "Node inputs: `(nodes, input_dim)`. Edges: `(2, edges)`. Coupled state: `(nodes, state_dim)` with `anderson_batch_dims=0`. Node output: `(nodes, output_dim)`.",
        """
from silva_networks import SILVAAlgorithmicReasoner

node_x = torch.randn(8, 3, requires_grad=True)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
    [1, 2, 3, 4, 5, 6, 7, 0, 4, 6, 0, 2],
])
graph_config = SolverConfig(
    solver="picard", max_iter=30, tol=1e-6, backward_mode="unrolled",
    anderson_batch_dims=0, return_best=True,
)
model = SILVAAlgorithmicReasoner(3, 10, 2, contraction=0.45, config=graph_config)
result = model(node_x, edge_index, return_result=True)
result.output.square().mean().backward()
summary = {
    "state_shape": tuple(result.state.shape),
    "edge_count": edge_index.shape[1],
    "residual": result.solver_result.residual,
    "iterations": result.solver_result.iterations,
    "input_grad_norm": float(node_x.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].imshow(result.state.detach(), aspect="auto", cmap="coolwarm")
axes[0].set(title="solved node algorithm state", xlabel="state coordinate", ylabel="node")
axes[1].plot(result.output.detach()[:, 0], marker="o", label="channel 0")
axes[1].plot(result.output.detach()[:, 1], marker="s", label="channel 1")
axes[1].set(title="node readout", xlabel="node", ylabel="value")
axes[1].legend()
fig.tight_layout()
plt.show()
""",
        "Replace `processor` with a typed message-passing block and `readout` with task-specific node, edge, or graph heads. Add hint encoders and termination heads externally while keeping the graph equilibrium contract unchanged.",
        "Full CLRS reproduction requires the official generators, train/validation/test size regimes, per-algorithm encoders and decoders, hint supervision, and out-of-distribution graph sizes. Report task accuracy together with residuals and solver work across graph sizes.",
    ),
    LabSpec(
        "64_silva_hamiltonian_equilibrium.ipynb",
        "silva_hamiltonian_equilibrium",
        "SILVA Hamiltonian Equilibrium",
        102,
        "Build a symmetric, rotation-invariant self-consistent Hamiltonian and verify its geometric contract before replacing the compact radial interaction.",
        r"""
Pairwise geometry enters only through invariant distances $r_{ij}=\lVert p_i-p_j\rVert_2$. A replaceable equivariant backbone proposes $H_\theta(F,r)$, while self-consistency solves

$$
H^\star=\operatorname{sym}\!\left(H_\theta(F,r)+\gamma\tanh(H^\star)\right),
\qquad \operatorname{sym}(A)=\tfrac12(A+A^\top).
$$

Distance invariance gives $H(F,PR)=H(F,P)$ for orthogonal $R$; symmetrization gives a Hermitian real Hamiltonian. These are architecture checks, separate from energy or force accuracy.
""",
        "Features: `(batch, atoms, feature_dim)`. Positions: `(batch, atoms, spatial_dim)`. Hamiltonian state and output: `(batch, atoms, atoms)`.",
        """
from silva_networks import SILVAHamiltonianEquilibrium

features = torch.randn(2, 6, 4, requires_grad=True)
positions = torch.randn(2, 6, 3)
rotation, _ = torch.linalg.qr(torch.randn(3, 3))
model = SILVAHamiltonianEquilibrium(4, contraction=0.35, config=config)
result = model(features, positions, return_result=True)
rotated = model(features, positions @ rotation)
eigenvalues = torch.linalg.eigvalsh(result.output.detach())
result.output.square().mean().backward()
summary = {
    "symmetry_error": float((result.output - result.output.mT).abs().max().detach()),
    "rotation_error": float((result.output.detach() - rotated.detach()).abs().max()),
    "residual": result.solver_result.residual,
    "feature_grad_norm": float(features.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
image = axes[0].imshow(result.output[0].detach(), cmap="coolwarm")
axes[0].set(title="self-consistent Hamiltonian", xlabel="orbital", ylabel="orbital")
fig.colorbar(image, ax=axes[0], fraction=0.046)
for row in eigenvalues:
    axes[1].plot(row, marker="o", alpha=0.8)
axes[1].set(title="Hamiltonian spectrum", xlabel="eigenvalue index", ylabel="energy")
fig.tight_layout()
plt.show()
""",
        "Replace `interaction` with an E(3)-equivariant graph network, orbital-aware tensor product, or domain Hamiltonian. Keep symmetrization and explicitly test rotations, atom permutations, gradients, and spectral observables.",
        "Article-scale molecular experiments require the named molecular datasets, basis conventions, train/validation/test splits, energy units, geometry preprocessing, and all reported spectral and force metrics. The compact radial cell validates contracts, not chemical accuracy.",
    ),
    LabSpec(
        "65_silva_inverse_imaging_equilibrium.ipynb",
        "silva_inverse_imaging_equilibrium",
        "SILVA Inverse Imaging Equilibrium",
        103,
        "Separate the known sensing operator, its adjoint, and the learned prior so inverse-problem experiments can change physics without changing the equilibrium engine.",
        r"""
Given measurements $y=Ax+\varepsilon$, one data-consistent prior step is

$$
u_k=x_k-\eta A^\top(Ax_k-y),\qquad x_{k+1}=D_\theta(u_k).
$$

The reconstruction is the fixed point

$$x^\star=D_\theta\!\left(x^\star-\eta A^\top(Ax^\star-y)\right).$$

The forward operator $A$, adjoint $A^\top$, prior $D_\theta$, solver, and loss are independent choices. Correctness begins with an adjoint test and a measurement-consistency residual.
""",
        "Measurement and reconstruction use `(batch, channels, height, width)` when the adjoint returns an image. Custom operators may have another measurement shape if their adjoint reconstructs the declared image state.",
        """
from silva_networks import SILVAInverseImagingEquilibrium

class MaskOperator(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
    def forward(self, value):
        return self.mask * value

grid = torch.linspace(0, 1, 24)
truth = (torch.sin(3 * torch.pi * grid)[None, :] * torch.cos(2 * torch.pi * grid)[:, None])
truth = truth.unsqueeze(0).unsqueeze(0)
mask = torch.zeros_like(truth)
mask[..., ::2, :] = 1
operator = MaskOperator(mask)
measurement = operator(truth).requires_grad_()
model = SILVAInverseImagingEquilibrium(
    1, forward_operator=operator, adjoint_operator=operator,
    prior=nn.Identity(), step_size=0.8, config=config,
)
result = model(measurement, return_result=True)
consistency = (operator(result.output) - measurement).square().mean().sqrt()
result.output.square().mean().backward()
summary = {
    "measurement_rmse": float(consistency.detach()),
    "residual": result.solver_result.residual,
    "iterations": result.solver_result.iterations,
    "measurement_grad_norm": float(measurement.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
for axis, image, title in zip(
    axes,
    [truth[0, 0], measurement[0, 0].detach(), result.output[0, 0].detach()],
    ["compact truth", "masked measurement", "equilibrium reconstruction"],
):
    axis.imshow(image, cmap="viridis")
    axis.set_title(title)
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
        "Replace `forward_operator`, `adjoint_operator`, and `prior` independently. For MRI, tomography, deblurring, or coded imaging, implement the true sensing map and verify its adjoint numerically before training.",
        "Source-scale reproduction requires the article's sensing matrices or masks, noise model, learned prior architecture, solver tolerance, training pairs, and PSNR/SSIM protocol. Store the operator configuration and measurement realization with every result.",
    ),
    LabSpec(
        "66_silva_snapshot_compressive_equilibrium.ipynb",
        "silva_snapshot_compressive_equilibrium",
        "SILVA Snapshot Compressive Equilibrium",
        104,
        "Reconstruct a video cube from one coded snapshot while keeping the analytic projection and learned spatiotemporal prior independently replaceable.",
        r"""
Snapshot compressive imaging observes

$$y=\sum_{t=1}^{T}M_t\odot x_t+\varepsilon.$$

For $\mathcal A(x)=\sum_tM_t\odot x_t$, the data correction is

$$
\Pi_y(x)=x+\eta\,M\odot
\frac{y-\mathcal A(x)}{\sum_t M_t^2+\epsilon}.
$$

SILVA solves $x^\star=\Pi_y(x^\star)+\lambda D_\theta(\Pi_y(x^\star))$. With $\eta=1$ and $\lambda=0$, the compact check isolates exact measurement consistency.
""",
        "Snapshot: `(batch, height, width)`. Masks and video state: `(batch, frames, height, width)`. The static `measure` method implements the sensing equation.",
        """
from silva_networks import SILVASnapshotCompressiveEquilibrium

height = width = 20
frames = 4
yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
video = torch.stack([torch.exp(-12 * ((xx - 0.3 * t) ** 2 + yy ** 2)) for t in torch.linspace(-1, 1, frames)])
video = video.unsqueeze(0)
masks = (torch.rand_like(video) > 0.35).float().clamp_min(0.15)
measurement = SILVASnapshotCompressiveEquilibrium.measure(video, masks).requires_grad_()
model = SILVASnapshotCompressiveEquilibrium(
    frames, prior=nn.Identity(), step_size=1.0, prior_scale=0.0, config=config
)
result = model(measurement, masks, return_result=True)
remeasured = model.measure(result.output, masks)
result.output.square().mean().backward()
summary = {
    "video_shape": tuple(result.output.shape),
    "measurement_error": float((remeasured - measurement).abs().max().detach()),
    "residual": result.solver_result.residual,
    "measurement_grad_norm": float(measurement.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
axes[0].imshow(measurement[0].detach(), cmap="magma")
axes[0].set_title("coded snapshot")
axes[1].imshow(video[0, 1], cmap="magma")
axes[1].set_title("source frame")
axes[2].imshow(result.output[0, 1].detach(), cmap="magma")
axes[2].set_title("equilibrium frame")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
        "Replace `prior` with a 3-D convolutional, recurrent, transformer, or operator prior. Keep `measure` and the analytic correction tied to the calibrated masks; alter them only when the camera model changes.",
        "Full experiments require the article's coded-mask construction, benchmark videos, frame counts, noise assumptions, crop policy, training schedule, and PSNR/SSIM evaluation. The compact cell verifies sensing and gradients rather than video quality.",
    ),
    LabSpec(
        "67_silva_magnetic_particle_equilibrium.ipynb",
        "silva_magnetic_particle_equilibrium",
        "SILVA Magnetic Particle Equilibrium",
        105,
        "Expose every primal, split, and dual variable in an ADMM-style magnetic-particle reconstruction fixed point.",
        r"""
For system matrix $A$ and measurement $y$, split data and prior variables around image $x$. A compact iteration is

$$
z_d\leftarrow C_\theta(Ax-d_d,y),\quad z_p\leftarrow R_\theta(x-d_p),
$$
$$
x\leftarrow(A^\top A+I)^{-1}\!\left[A^\top(z_d+d_d)+z_p+d_p\right],
$$
$$d_d\leftarrow d_d+z_d-Ax,\qquad d_p\leftarrow d_p+z_p-x.$$

SILVA packs $(x,z_d,z_p,d_d,d_p)$ into one solved state. Learned consistency and regularization remain named modules rather than hidden inside the solver.
""",
        "Measurements: `(batch, measurement_dim)`. Matrix: `(measurement_dim, image_dim)`. Packed state width: `3*image_dim + 2*measurement_dim`; output width: `image_dim`.",
        """
from silva_networks import SILVAMagneticParticleEquilibrium

image_dim, measurement_dim = 16, 9
matrix = torch.randn(measurement_dim, image_dim) / image_dim ** 0.5
truth = torch.zeros(6, image_dim)
truth[:, 3:7] = torch.linspace(0.3, 1.0, 4)
measurement = (truth @ matrix.mT + 0.01 * torch.randn(6, measurement_dim)).requires_grad_()
model = SILVAMagneticParticleEquilibrium(
    image_dim, measurement_dim, mixing=0.3,
    config=SolverConfig(solver="picard", max_iter=12, tol=1e-6, backward_mode="unrolled", anderson_batch_dims=1, return_best=True),
)
result = model(measurement, matrix, return_result=True)
result.output.square().mean().backward()
summary = {
    "packed_state_shape": tuple(result.state.shape),
    "output_shape": tuple(result.output.shape),
    "residual": result.solver_result.residual,
    "iterations": result.solver_result.iterations,
    "measurement_grad_norm": float(measurement.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].bar(range(image_dim), truth[0], alpha=0.7, label="compact source")
axes[0].plot(result.output[0].detach(), marker="o", color="#dc2626", label="equilibrium")
axes[0].set(title="particle concentration vector", xlabel="voxel", ylabel="value")
axes[0].legend()
axes[1].semilogy(result.solver_result.residuals, marker="o")
axes[1].set(title="packed ADMM fixed point", xlabel="iteration", ylabel="residual")
fig.tight_layout()
plt.show()
""",
        "Replace `regularizer` and `learned_consistency`; supply the calibrated MPI system matrix at call time. `_unpack` makes all primal and dual blocks available for penalties and diagnostics.",
        "Full MPI studies require the scanner system matrix, acquisition protocol, calibration preprocessing, concentration phantoms, source splits, and reconstruction metrics. Record linear-solve conditioning, primal/dual residuals, runtime, and memory in addition to image error.",
    ),
    LabSpec(
        "68_silva_sparse_hyperspectral_equilibrium.ipynb",
        "silva_sparse_hyperspectral_equilibrium",
        "SILVA Sparse Hyperspectral Equilibrium",
        106,
        "Solve a sparse latent spectral code with explicit analysis, synthesis, shrinkage, and learned spatial-spectral prior components.",
        r"""
Let $D$ synthesize a cube from sparse code $c$, and let $E$ be the analysis map. One proximal transition is

$$
c_{k+1}=\operatorname{soft}_{\lambda}\!\left[
c_k-\eta E(Dc_k-y)+\gamma E(P_\theta(Dc_k))
\right].
$$

The equilibrium $c^\star=T_\theta(c^\star;y)$ is decoded as $\widehat x=Dc^\star$. Shrinkage supplies the sparse prior; the learned cube prior and linear spectral dictionary are separately inspectable.
""",
        "Noisy cube: `(batch, channels, height, width)`. Equilibrium code: `(batch, code_channels, height, width)`. Reconstruction returns the original channel count and spatial grid.",
        """
from silva_networks import SILVASparseHyperspectralEquilibrium

bands, height, width = 6, 14, 14
wavelength = torch.linspace(0, 1, bands)[:, None, None]
spatial = torch.exp(-5 * (torch.linspace(-1, 1, height)[:, None] ** 2 + torch.linspace(-1, 1, width)[None, :] ** 2))
clean = (0.3 + torch.sin(2 * torch.pi * wavelength).square()) * spatial
noisy = (clean + 0.04 * torch.randn_like(clean)).unsqueeze(0).requires_grad_()
model = SILVASparseHyperspectralEquilibrium(
    bands, 10, threshold=0.03, step_size=0.08, prior_scale=0.01,
    config=SolverConfig(solver="picard", max_iter=12, tol=1e-6, backward_mode="unrolled", anderson_batch_dims=1, return_best=True),
)
result = model(noisy, return_result=True)
sparsity = (result.state.detach().abs() < 1e-5).float().mean()
result.output.square().mean().backward()
summary = {
    "code_shape": tuple(result.state.shape),
    "cube_shape": tuple(result.output.shape),
    "near_zero_fraction": float(sparsity),
    "residual": result.solver_result.residual,
    "input_grad_norm": float(noisy.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
axes[0].imshow(clean[2], cmap="viridis")
axes[0].set_title("clean compact band")
axes[1].imshow(noisy[0, 2].detach(), cmap="viridis")
axes[1].set_title("noisy band")
axes[2].imshow(result.output[0, 2].detach(), cmap="viridis")
axes[2].set_title("equilibrium band")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
        "Replace `analysis`, `synthesis`, and `prior` to express learned dictionaries, spectral convolutions, low-rank blocks, or spatial operators. Keep the code and cube available when adding sparsity, spectral-angle, or data-consistency losses.",
        "Source-scale reproduction requires the article's hyperspectral datasets, band selection, noise levels, normalization, patch extraction, dictionary dimensions, and PSNR/SSIM/SAM metrics. Report results per noise level and data split.",
    ),
    LabSpec(
        "69_silva_serialized_smoothing_equilibrium.ipynb",
        "silva_serialized_smoothing_equilibrium",
        "SILVA Serialized Smoothing Equilibrium",
        107,
        "Warm-start successive noisy equilibrium solves and turn their class counts into an explicit randomized-smoothing certificate.",
        r"""
For Gaussian perturbations $\varepsilon_s\sim\mathcal N(0,\sigma^2I)$, each sample solves

$$z_s^\star=T_\theta(z_s^\star;x+\varepsilon_s),\qquad c_s=\arg\max Q_\psi(z_s^\star).$$

Serialized evaluation initializes solve $s+1$ from $z_s^\star$. If $\underline p_A$ is a confidence lower bound for the majority class, the isotropic certificate is

$$R=\sigma\Phi^{-1}(\underline p_A).$$

Warm starts affect computation, not the statistical count definition. Seeds, sample count, confidence construction, and abstention policy must therefore be recorded.
""",
        "Inputs: `(batch, input_dim)`. Predictions: `(samples, batch)`. Counts: `(batch, classes)`. Certificate radius and lower probability: `(batch,)`.",
        """
from silva_networks import SILVASerializedSmoothingEquilibrium

x = torch.randn(5, 4)
model = SILVASerializedSmoothingEquilibrium(4, 10, 3, sigma=0.25, config=config)
predictions, records = model.sample_predictions(x, samples=32, seed=7, serialized=True)
certificate = model.certify(x, samples=64, seed=7)
iterations = torch.tensor([record.iterations for record in records])
summary = {
    "prediction_shape": tuple(predictions.shape),
    "count_totals": certificate.counts.sum(dim=-1).tolist(),
    "mean_serial_iterations": float(iterations.float().mean()),
    "radii": certificate.radius.detach().tolist(),
    "classes": certificate.predicted_class.tolist(),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].bar(range(3), certificate.counts[0], color=["#2563eb", "#d97706", "#059669"])
axes[0].set(title="smoothed class counts", xlabel="class", ylabel="samples")
axes[1].bar(range(len(x)), certificate.radius.detach(), color="#7c3aed")
axes[1].set(title="certified radii", xlabel="input", ylabel="radius")
fig.tight_layout()
plt.show()
""",
        "Replace `input_map`, `state_map`, and `readout` with any bounded equilibrium classifier. `sample_predictions` exposes every solver record, and `serialized=False` provides the cold-start control.",
        "Full certification requires the source classifier, data augmentation and normalization, exact sample counts, confidence level, abstention rule, attack/evaluation radii, and certified-accuracy curve. Record wall time and iterations for serialized and independent solves.",
    ),
    LabSpec(
        "70_silva_diffusion_restoration_equilibrium.ipynb",
        "silva_diffusion_restoration_equilibrium",
        "SILVA Diffusion Restoration Equilibrium",
        108,
        "Solve all restoration-time variables jointly while projecting observed pixels at every equilibrium transition.",
        r"""
Stack the restoration trajectory as $Z=(x_0,\ldots,x_{T-1})$. For $t\geq1$,

$$
\widetilde x_t=(1-\eta)D_\theta(x_{t-1})+\eta x_t,
\qquad
x_t' = M\odot y+(1-M)\odot\widetilde x_t.
$$

The multivariate fixed point is $Z^\star=F_\theta(Z^\star;y,M,x_0)$. Hard projection makes observed-pixel consistency exact throughout the solved trajectory; the denoiser controls only unobserved content.
""",
        "Measurement, mask, and noise: `(batch, channels, height, width)`. Joint state: `(batch, timesteps, channels, height, width)`. Output: final trajectory slice.",
        """
from silva_networks import SILVADiffusionRestorationEquilibrium

grid = torch.linspace(-1, 1, 20)
truth = torch.exp(-7 * (grid[:, None] ** 2 + grid[None, :] ** 2)).unsqueeze(0).unsqueeze(0)
mask = torch.zeros_like(truth)
mask[..., ::3, :] = 1
measurement = (mask * truth).requires_grad_()
initial_noise = 0.15 * torch.randn_like(truth)
model = SILVADiffusionRestorationEquilibrium(1, 5, eta=0.15, config=config)
result = model(measurement, mask=mask, initial_noise=initial_noise, return_result=True)
observed_error = ((result.state[:, 1:] - measurement.unsqueeze(1)) * mask.unsqueeze(1)).abs().max()
result.output.square().mean().backward()
summary = {
    "trajectory_shape": tuple(result.state.shape),
    "observed_pixel_error": float(observed_error.detach()),
    "residual": result.solver_result.residual,
    "measurement_grad_norm": float(measurement.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
for axis, image, title in zip(
    axes,
    [truth[0, 0], measurement[0, 0].detach(), result.output[0, 0].detach()],
    ["compact truth", "observed pixels", "joint equilibrium output"],
):
    axis.imshow(image, cmap="magma")
    axis.set_title(title)
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
        "Replace `denoiser` with a trained diffusion prior or restoration network. Supply the true observation projection through the mask or subclass the transition for a general sensing operator while preserving the joint trajectory state.",
        "Full DeqIR-style studies require the pretrained diffusion model, degradation operators, noise schedule, datasets, sampling settings, baselines, FID/LPIPS/PSNR/SSIM protocol, and runtime comparison. The compact lab establishes joint-state and hard-projection contracts only.",
    ),
    LabSpec(
        "71_silva_recurrent_equilibrium_network.ipynb",
        "silva_recurrent_equilibrium_network",
        "SILVA Recurrent Equilibrium Network",
        109,
        "Combine an explicit stable dynamic state with an algebraic equilibrium solved at every time step.",
        r"""
At time $t$, solve the algebraic state

$$w_t^\star=\tanh(D_{11}w_t^\star+C_1x_t+D_{12}u_t),
\qquad \lVert D_{11}\rVert<1,$$

then update the explicit dynamic state and output,

$$x_{t+1}=\alpha x_t+(1-\alpha)(B_1w_t^\star+B_2u_t),$$
$$y_t=Q_\psi([x_{t+1},w_t^\star,u_t]).$$

This separates temporal memory from the instantaneous equilibrium nonlinearity. Stability controls, solver state reuse, and readout remain visible.
""",
        "Sequence input: `(batch, time, input_dim)`. Dynamic state: `(batch, time, state_dim)`. Algebraic equilibrium: `(batch, time, equilibrium_dim)`. Output: `(batch, time, output_dim)`.",
        """
from silva_networks import SILVARecurrentEquilibriumNetwork

time = torch.linspace(0, 2 * torch.pi, 32)
inputs = torch.stack([torch.sin(time), torch.cos(time), torch.sin(2 * time)], dim=-1)
inputs = inputs.unsqueeze(0).repeat(3, 1, 1).requires_grad_()
model = SILVARecurrentEquilibriumNetwork(3, 6, 8, 2, contraction=0.6, state_decay=0.8, config=config)
result = model(inputs)
result.output.square().mean().backward()
residuals = [record.residual for record in result.solver_results]
summary = {
    "output_shape": tuple(result.output.shape),
    "dynamic_state_shape": tuple(result.state.shape),
    "equilibrium_shape": tuple(result.equilibrium.shape),
    "maximum_step_residual": max(residuals),
    "input_grad_norm": float(inputs.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].plot(time, result.output[0, :, 0].detach(), label="output 0")
axes[0].plot(time, result.output[0, :, 1].detach(), label="output 1")
axes[0].set(title="recurrent equilibrium outputs", xlabel="time", ylabel="value")
axes[0].legend()
axes[1].plot(time, result.state[0].detach().norm(dim=-1), label="dynamic state")
axes[1].plot(time, result.equilibrium[0].detach().norm(dim=-1), label="algebraic state")
axes[1].set(title="state trajectories", xlabel="time", ylabel="norm")
axes[1].legend()
fig.tight_layout()
plt.show()
""",
        "Replace the input maps, state update, and readout by composing a subclass or wrapping the returned states. `algebraic_transition` is public for custom solvers, losses, and stability tests; initial dynamic state is supplied at call time.",
        "Full REN studies require the article's control datasets or simulated systems, horizon, stability parameterization, training objective, baselines, and long-horizon rollout metrics. Report solver work per step and closed-loop stability, not just one-step error.",
    ),
    LabSpec(
        "72_silva_lipschitz_robust_equilibrium.ipynb",
        "silva_lipschitz_robust_equilibrium",
        "SILVA Lipschitz Robust Equilibrium",
        110,
        "Compare four structure-preserving recurrent parameterizations and obtain a visible global input-output bound and margin radius.",
        r"""
The bounded state equation is

$$z^\star=\tanh(W_z z^\star+W_xx+b),
\qquad \lVert W_z\rVert\leq\rho<1.$$

If $\lVert W_x\rVert\leq L_x$ and $\lVert W_o\rVert\leq L_o$, implicit sensitivity obeys

$$\left\lVert\frac{\partial y}{\partial x}\right\rVert
\leq\frac{L_xL_o}{1-\rho}=:L.$$

For top-two logit margin $m$, a conservative radius is $R=m/(\sqrt2L)$. SILVA exposes LBEN-style normalization, orthogonal, sandwich, and Cayley-like (`cpl`) recurrent maps under one result contract.
""",
        "Input: `(batch, input_dim)`. State: `(batch, state_dim)`. Logits: `(batch, classes)`. Result also returns global bound, margins, radii, and solver diagnostics.",
        """
from silva_networks import SILVALipschitzRobustEquilibrium

x = torch.randn(10, 5)
parameterizations = ["lben", "orthogonal", "sandwich", "cpl"]
models = []
results = []
spectral_bounds = []
for name in parameterizations:
    candidate = SILVALipschitzRobustEquilibrium(
        5, 12, 4, parameterization=name, recurrent_bound=0.6, config=config
    )
    solved = candidate(x, return_result=True)
    models.append(candidate)
    results.append(solved)
    spectral_bounds.append(float(torch.linalg.matrix_norm(candidate.recurrent_weight(), ord=2).detach()))
model = models[0]
summary = {
    name: {
        "spectral_norm": bound,
        "global_bound": float(result.lipschitz_bound.detach()),
        "mean_radius": float(result.certified_radius.mean().detach()),
        "residual": result.solver_result.residual,
    }
    for name, bound, result in zip(parameterizations, spectral_bounds, results)
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].bar(parameterizations, spectral_bounds, color="#2563eb")
axes[0].axhline(0.6, color="#dc2626", linestyle="--", label="declared recurrent bound")
axes[0].set(title="recurrent map norms", ylabel="spectral norm")
axes[0].legend(fontsize=7)
axes[1].bar(parameterizations, [float(item.certified_radius.mean()) for item in results], color="#059669")
axes[1].set(title="compact margin certificates", ylabel="mean radius")
fig.tight_layout()
plt.show()
""",
        "Choose `parameterization` explicitly and inspect `recurrent_weight`. Replace the input and readout matrices in a subclass when using convolutional or operator states, while preserving the intended norm and certificate derivation.",
        "Full robust benchmarks require the source architecture, dataset and augmentation, adversarial or certified evaluation protocol, norm, radii, attack budget, and certified-accuracy reporting. Natural accuracy alone does not reproduce a robustness result.",
    ),
    LabSpec(
        "73_silva_image_matting_equilibrium.ipynb",
        "silva_image_matting_equilibrium",
        "SILVA Image Matting Equilibrium",
        111,
        "Solve an alpha matte while enforcing foreground and background trimap regions exactly at every transition.",
        r"""
An encoder produces image-trimap features $F=E_\theta([I,M])$. The unknown alpha region is refined by

$$\widetilde\alpha=\sigma\!\left(R_\theta([\rho\alpha,F])\right).$$

The transition projects known trimap regions,

$$
T(\alpha)_{ij}=
\begin{cases}
0,&M_{ij}\leq0.05,\\
1,&M_{ij}\geq0.95,\\
\widetilde\alpha_{ij},&\text{otherwise}.
\end{cases}
$$

Thus the solved matte cannot violate known foreground/background labels. The learned refiner acts only where the trimap is uncertain.
""",
        "Image: `(batch, image_channels, height, width)`. Trimap and alpha state: `(batch, 1, height, width)`. Alpha remains in `[0, 1]` and exact on known trimap pixels.",
        """
from silva_networks import SILVAImageMattingEquilibrium

height = width = 28
yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
soft_object = torch.sigmoid(18 * (0.55 - torch.sqrt(xx.square() + yy.square())))
image = torch.stack([soft_object, 0.3 + 0.5 * soft_object, 1 - 0.7 * soft_object]).unsqueeze(0).requires_grad_()
trimap = torch.full((1, 1, height, width), 0.5)
trimap[:, :, soft_object < 0.1] = 0.0
trimap[:, :, soft_object > 0.9] = 1.0
model = SILVAImageMattingEquilibrium(hidden_channels=10, contraction=0.45, config=config)
result = model(image, trimap, return_result=True)
known = (trimap <= 0.05) | (trimap >= 0.95)
constraint_error = (result.output[known] - trimap[known]).abs().max()
result.output.mean().backward()
summary = {
    "alpha_shape": tuple(result.output.shape),
    "known_region_error": float(constraint_error.detach()),
    "alpha_range": (float(result.output.min().detach()), float(result.output.max().detach())),
    "residual": result.solver_result.residual,
    "image_grad_norm": float(image.grad.norm()),
}
summary
""",
        """
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
axes[0].imshow(image[0].detach().permute(1, 2, 0).clamp(0, 1))
axes[0].set_title("compact image")
axes[1].imshow(trimap[0, 0], cmap="gray", vmin=0, vmax=1)
axes[1].set_title("trimap")
axes[2].imshow(result.output[0, 0].detach(), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("equilibrium alpha")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
        "Replace `encoder` and `refiner` with multiscale backbones, attention blocks, or pretrained image features. Retain the trimap projection when exact known-region consistency is part of the task contract.",
        "Full matting reproduction requires the article's compositing datasets, trimap-generation policy, crop and augmentation rules, backbone, losses, inference scale, and SAD/MSE/gradient/connectivity metrics. Evaluate on the exact benchmark split and trimaps.",
    ),
    LabSpec(
        "74_silva_dynamic_economic_equilibrium.ipynb",
        "silva_dynamic_economic_equilibrium",
        "SILVA Dynamic Economic Equilibrium",
        112,
        "Represent feasible policy functions directly and train them from Euler and resource residuals without requiring labeled optimal policies.",
        r"""
For capital $k$ and productivity $z$, available resources are

$$r(k,z)=e^z k^\alpha+(1-\delta)k.$$

Softmax policy shares guarantee feasibility,

$$c=s_c r,\qquad k'=s_k r,\qquad s_c+s_k=1.$$

The Euler residual for CRRA utility is

$$
\mathcal R_E=c^{-\gamma}-\beta(c')^{-\gamma}
\left[\alpha e^{z'}(k')^{\alpha-1}+1-\delta\right].
$$

Training minimizes sampled resource and Euler residuals. This family solves an economic equilibrium-function approximation; it is not a hidden-state root solve, and the distinction is explicit in the API.
""",
        "States: `(batch, state_dim)` with capital first and log productivity second. The result returns consumption, next capital, resource residual, and Euler residual, each `(batch,)`.",
        """
from silva_networks import SILVADynamicEconomicEquilibrium

capital = torch.linspace(0.3, 3.0, 80)
productivity = torch.zeros_like(capital)
states = torch.stack([capital, productivity], dim=-1).requires_grad_()
model = SILVADynamicEconomicEquilibrium(state_dim=2, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
loss_curve = []
for _ in range(60):
    optimizer.zero_grad()
    trained = model(states)
    loss = trained.euler_residual.square().mean() + 10 * trained.resource_residual.square().mean()
    loss.backward()
    optimizer.step()
    loss_curve.append(float(loss.detach()))
result = model(states)
summary = {
    "maximum_resource_error": float(result.resource_residual.abs().max().detach()),
    "euler_rmse": float(result.euler_residual.square().mean().sqrt().detach()),
    "minimum_consumption": float(result.consumption.min().detach()),
    "minimum_next_capital": float(result.next_capital.min().detach()),
    "final_compact_loss": loss_curve[-1],
}
summary
""",
        """
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
axes[0].plot(capital, result.consumption.detach(), label="consumption")
axes[0].plot(capital, result.next_capital.detach(), label="next capital")
axes[0].plot(capital, model.resources(states).detach(), linestyle="--", label="resources")
axes[0].set(title="feasible policy functions", xlabel="capital", ylabel="level")
axes[0].legend(fontsize=7)
axes[1].semilogy(loss_curve, color="#7c3aed")
axes[1].set(title="residual training", xlabel="optimizer step", ylabel="loss")
fig.tight_layout()
plt.show()
""",
        "Replace `policy` with wider, recurrent, sparse-grid, mixture, or domain-constrained policy maps. Extend `resources`, transition laws, and residuals in a subclass for multiple agents, assets, shocks, or equilibrium conditions.",
        "Full economic reproduction requires the article's model calibration, shock process and quadrature, simulation state distribution, residual weighting, optimization schedule, test grids, Euler-error convention, and economic welfare comparisons. Report feasibility and extrapolation separately.",
    ),
)


def build_lab(spec: LabSpec) -> dict[str, object]:
    prefix = spec.family.replace("_", "-")
    download = f"https://github.com/jseluis/silva-networks/raw/main/notebooks/package_api/{spec.filename}"
    cells = [
        md(
            prefix,
            f"""
# {spec.title}

{spec.summary} This lab adapts the cited mechanism into explicit SILVA
components [[{spec.ref}]], runs a deterministic compact check, and separates
that evidence from a source-scale reproduction claim.

[Download this notebook]({download})
""",
        ),
        code(prefix, BOOTSTRAP),
        md(prefix, f"## 1. Mechanism and Derivation\n\n{spec.derivation}"),
        md(
            prefix,
            f"""
## 2. SILVA State and Shape Contract

{spec.shape_contract}

The transition remains a named callable, the numerical method is selected by
`SolverConfig`, and the result exposes both the solved state and solver record.
This makes architecture equivalence, numerical equivalence, and task quality
three separate questions.
""",
        ),
        code(prefix, spec.setup),
        md(
            prefix,
            """
## 3. Read the Compact Evidence

The preceding output is a measured contract check: shapes, constraints,
residuals, and gradients were produced by this notebook. It does not imply
that the cited source benchmark has been reproduced. The figure below makes
one family-specific state or diagnostic visible.
""",
        ),
        code(prefix, spec.plot),
        md(
            prefix,
            f"""
## 4. Inspect and Replace the Internals

{spec.replaceable}

The following inventory is deliberately mechanical: an advanced experiment
can replace a child module without changing the solver or reporting contract.
""",
        ),
        code(
            prefix,
            """
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
for name, child in model.named_children():
    print(f"{name:24s} -> {child.__class__.__name__}")
""",
        ),
        md(
            prefix,
            f"""
## 5. Compact, Workstation, and Source Scale

{spec.source_scale}

SILVA stores all three execution routes in the family protocol. Resource
figures are planning ranges; measured hardware, runtime, peak memory, data
revision, split, seed, and deviations belong in the completed result record.
""",
        ),
        code(
            prefix,
            f"""
protocol = silva_family_experiment_protocol("{spec.family}")
for tier in protocol.tiers:
    print(f"{{tier.tier:11s}} | {{tier.dataset.name}} | {{tier.dataset.expected_storage}}")
    print("  source:", tier.dataset.source_url)
    print("  split: ", tier.dataset.split)
    print("  run:   ", tier.command)
""",
        ),
        md(
            prefix,
            """
## 6. Reproduction Checklist

Before labeling a result as source-scale reproduced, preserve the cited
equation and architecture choices, use the declared source data and split,
match preprocessing and evaluation, run the required seeds, and report task
metrics beside equilibrium residuals, iterations, failures, runtime, and peak
memory. Compact and subset runs remain valuable, but keep their evidence level
explicit.
""",
        ),
        md(
            prefix,
            """
## 7. Build the Next Variant

1. Replace one named component and keep its tensor contract fixed.
2. Verify the transition on a deterministic fixture before solving it.
3. Compare finite iteration and converged outputs at the same weights.
4. Add a task loss only after constraints, invariances, and gradients pass.
5. Scale the data and architecture independently so the cause of each change is visible.
6. Record the exact source relation: reproduced, adapted, or newly extended.
""",
        ),
        md(
            prefix,
            """
## Where to Go Next

| Question | Page |
| --- | --- |
| How are all 14 source-aligned mechanisms derived? | [Source-Aligned Equilibrium Families](https://jseluis.github.io/silva-networks/learn/source-equilibrium-families/) |
| Which classes and result fields are public? | [Source-Aligned Equilibria API](https://jseluis.github.io/silva-networks/api/source_equilibria/) |
| Where is the family-specific scale plan? | [Family Reproduction Dossiers](https://jseluis.github.io/silva-networks/families/) |
| How should source-scale evidence be reported? | [Evidence and Source-Scale Experiments](https://jseluis.github.io/silva-networks/learn/evidence-and-source-scale/) |
""",
        ),
    ]
    return notebook(cells)


def main() -> None:
    for spec in SPECS:
        payload = build_lab(spec)
        for directory in OUT_DIRS:
            write_notebook(
                directory / spec.filename,
                payload,
                replace_changed=True,
                preserve_unmatched=True,
            )
        print(f"generated {spec.filename}")


if __name__ == "__main__":
    main()
