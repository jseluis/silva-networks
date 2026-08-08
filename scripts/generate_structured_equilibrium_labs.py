"""Generate executable labs for six structured SILVA equilibrium families."""

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
torch.manual_seed(91)
"""


def source_preflight(prefix: str, family: str) -> list[dict[str, object]]:
    return [
        md(
            prefix,
            """
## Source-Scale Reproduction Contract

The compact run above verifies the defining mechanism, shapes, diagnostics,
and gradients. A published benchmark requires the source data, preprocessing,
architecture dimensions, optimization schedule, seeds, and evaluation budget.
The executable registry keeps those obligations beside the constructor.
""",
        ),
        code(
            prefix,
            f"""
from silva_networks import silva_reproduction_spec

spec = silva_reproduction_spec({family!r})
print("equation:", spec.equation)
print("datasets:", spec.datasets)
print("data sources:")
for value in spec.data_sources:
    print(" -", value)
print("source-scale steps:")
for index, value in enumerate(spec.source_scale_steps, start=1):
    print(f" {{index}}. {{value}}")
print("metrics:", spec.metrics)
print("preserved mechanisms:", spec.preserved_mechanisms)
print("SILVA extension points:", spec.silva_extensions)
print("benchmark obligations:", spec.benchmark_requirements)
print("constructor:", spec.constructor_signature)
""",
        ),
        md(
            prefix,
            """
## Where to Go Next

| Question | Page |
| --- | --- |
| How is this family derived in the documentation? | [Structured Equilibrium Families](https://jseluis.github.io/silva-networks/learn/structured-equilibrium-families/) |
| Which options are public? | [Structured Equilibria API](https://jseluis.github.io/silva-networks/api/structured_equilibria/) |
| How are complete source experiments planned? | [Reconstructing Paper Experiments](https://jseluis.github.io/silva-networks/learn/reconstructing-paper-experiments/) |
| Where are all citations? | [References](https://jseluis.github.io/silva-networks/paper/references/) |
""",
        ),
    ]


def _snapshot_loader(prefix: str, filename: str, variable: str = "source_sample") -> dict[str, object]:
    return code(
        prefix,
        f"""
from urllib.request import urlretrieve
from silva_networks import load_source_snapshot

snapshot_path = root / "docs/assets/source-data/{filename}"
if not snapshot_path.exists():
    snapshot_path = Path(".silva-source-data") / "{filename}"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_url = (
        "https://raw.githubusercontent.com/jseluis/silva-networks/main/"
        "docs/assets/source-data/{filename}"
    )
    urlretrieve(snapshot_url, snapshot_path)

{variable} = load_source_snapshot(snapshot_path)
print("dataset:", {variable}.receipt.dataset)
print("source indices:", {variable}.receipt.selected_indices)
print("content SHA-256:", {variable}.receipt.content_sha256)
print("preprocessing:")
for step in {variable}.receipt.preprocessing:
    print(" -", step)
""",
    )


def real_source_cells(prefix: str, family: str) -> list[dict[str, object]]:
    """Return family-specific real-data cells without changing known-solution cells."""

    if family == "monotone":
        return [
            md(
                prefix,
                r"""
## 4. Attributed CIFAR-10 Mechanism Check

The preceding known-solution problem answers whether the splitting is
implemented correctly. This section answers a different question: can the same
public constructor receive real image tensors and produce trainable logits?
The ten-image snapshot contains one source-indexed CIFAR-10 example per class
[81]. It is a data-path and gradient check, not a CIFAR-10 accuracy result.
""",
            ),
            _snapshot_loader(prefix, "cifar10-balanced-10.pt"),
            code(
                prefix,
                """
from torch.nn import functional as F

real_images = source_sample.tensors["images"]
real_labels = source_sample.tensors["labels"].long()
real_vectors = real_images.flatten(1)
real_model = SILVAMonotoneOperatorEquilibrium(
    real_vectors.shape[1],
    24,
    10,
    step_size=0.5,
    splitting="forward_backward",
    config=SolverConfig(
        solver="picard",
        max_iter=80,
        tol=1e-6,
        backward_mode="unrolled",
        anderson_batch_dims=1,
    ),
)
optimizer = torch.optim.Adam(real_model.parameters(), lr=5e-3)
real_losses = []
for _ in range(3):
    optimizer.zero_grad()
    real_result = real_model(real_vectors, return_result=True)
    real_loss = F.cross_entropy(real_result.output, real_labels)
    real_loss.backward()
    optimizer.step()
    real_losses.append(float(real_loss.detach()))

real_result = real_model(real_vectors, return_result=True)
print("loss trajectory:", real_losses)
print("final residual:", real_result.solver_result.residual)
print("monotonicity certificate:", float(real_result.monotonicity_certificate))
assert torch.isfinite(real_result.output).all()
assert real_result.monotonicity_certificate > 0
""",
            ),
            code(
                prefix,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
axes[0].imshow(real_images[0].permute(1, 2, 0))
axes[0].set_title(f"CIFAR-10 label {int(real_labels[0])}")
axes[0].axis("off")
axes[1].plot(range(1, len(real_losses) + 1), real_losses, marker="o")
axes[1].set(xlabel="optimizer step", ylabel="cross entropy")
axes[2].semilogy(real_result.solver_result.residuals)
axes[2].set(xlabel="equilibrium iteration", ylabel="residual")
figure.tight_layout()
plt.show()
""",
            ),
        ]
    if family == "positive_concave":
        return [
            md(
                prefix,
                r"""
## 5. Attributed Positive Image Check

CIFAR-10 pixels scaled to $[0,1]$ satisfy the input-side positivity required by
the convolutional branch [81]. The experiment keeps the spatial state
$z^\star\in\mathbb{R}^{B\times C_z\times H\times W}$, averages only the final
class field, and projects the recurrent parameter after each optimizer step.
The small snapshot verifies the complete trainable path rather than benchmark
accuracy.
""",
            ),
            _snapshot_loader(prefix, "cifar10-balanced-10.pt"),
            code(
                prefix,
                """
from torch.nn import functional as F

real_images = source_sample.tensors["images"]
real_labels = source_sample.tensors["labels"].long()
real_model = SILVAPositiveConcaveEquilibrium(
    3,
    8,
    10,
    operator="conv2d",
    variant=1,
    weight_parameterization="projected",
    config=SolverConfig(
        solver="picard",
        max_iter=80,
        tol=1e-6,
        backward_mode="unrolled",
        anderson_batch_dims=1,
    ),
)
optimizer = torch.optim.Adam(real_model.parameters(), lr=3e-3)
real_losses = []
for _ in range(3):
    optimizer.zero_grad()
    real_result = real_model(real_images, return_result=True)
    logits = real_result.output.mean(dim=(-2, -1))
    real_loss = F.cross_entropy(logits, real_labels)
    real_loss.backward()
    optimizer.step()
    real_model.project_nonnegative_()
    real_losses.append(float(real_loss.detach()))

real_result = real_model(real_images, return_result=True)
print("loss trajectory:", real_losses)
print("minimum recurrent weight:", float(real_result.minimum_weight))
print("minimum equilibrium state:", float(real_result.state.min()))
assert real_result.minimum_weight >= 0
assert real_result.state.min() >= 0
""",
            ),
            code(
                prefix,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
axes[0].imshow(real_images[1].permute(1, 2, 0))
axes[0].set_title(f"CIFAR-10 label {int(real_labels[1])}")
axes[0].axis("off")
axes[1].plot(range(1, len(real_losses) + 1), real_losses, marker="o")
axes[1].set(xlabel="optimizer step", ylabel="cross entropy")
axes[2].imshow(real_result.state[1, 0].detach(), cmap="viridis")
axes[2].set_title("positive equilibrium channel")
axes[2].axis("off")
figure.tight_layout()
plt.show()
""",
            ),
        ]
    if family == "non_euclidean":
        return [
            md(
                prefix,
                r"""
## 4. Real Images and the Sensitivity Contract

The weighted matrix measure constrains the latent fixed-point map; it does not
turn a ten-image run into a robustness benchmark. On source-indexed CIFAR-10
examples [81], we therefore report the clean loss, a bounded input
perturbation, the observed logit displacement, and the model's latent
input-Lipschitz bound as distinct quantities.
""",
            ),
            _snapshot_loader(prefix, "cifar10-balanced-10.pt"),
            code(
                prefix,
                """
from torch.nn import functional as F

real_images = source_sample.tensors["images"]
real_labels = source_sample.tensors["labels"].long()
real_vectors = real_images.flatten(1)
epsilon = 8.0 / 255.0
direction = torch.sign(torch.sin(torch.arange(real_vectors.numel()))).reshape_as(real_vectors)
perturbed_vectors = (real_vectors + epsilon * direction).clamp(0.0, 1.0)
real_model = SILVANonEuclideanEquilibrium(
    real_vectors.shape[1],
    24,
    10,
    one_sided_bound=0.05,
    config=SolverConfig(
        solver="picard",
        max_iter=120,
        tol=1e-6,
        backward_mode="unrolled",
        anderson_batch_dims=1,
    ),
)
clean = real_model(real_vectors, return_result=True)
perturbed_logits = real_model(perturbed_vectors)
clean_loss = F.cross_entropy(clean.output, real_labels)
clean_loss.backward()
per_sample_shift = torch.linalg.vector_norm(
    perturbed_logits.detach() - clean.output.detach(), dim=1
)
print("clean loss:", float(clean_loss.detach()))
print("weighted matrix measure:", float(clean.one_sided_lipschitz))
print("latent input-Lipschitz bound:", float(clean.latent_input_lipschitz_bound))
print("maximum observed logit shift:", float(per_sample_shift.max()))
assert clean.one_sided_lipschitz < 1
""",
            ),
            code(
                prefix,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.5, 2.5))
axes[0].imshow(real_images[2].permute(1, 2, 0))
axes[0].set_title(f"CIFAR-10 label {int(real_labels[2])}")
axes[0].axis("off")
axes[1].bar(torch.arange(len(per_sample_shift)), per_sample_shift)
axes[1].set(xlabel="source example", ylabel="logit displacement")
axes[2].semilogy(clean.solver_result.residuals)
axes[2].set(xlabel="equilibrium iteration", ylabel="residual")
figure.tight_layout()
plt.show()
""",
            ),
        ]
    if family == "eignn":
        return [
            md(
                prefix,
                r"""
## 4. Cora With Source Masks

The snapshot is a connected, source-indexed induced subgraph of Cora [82]. It
retains train, validation, and test node identities so the complete tensor path
can be exercised quickly. An induced graph changes the transductive problem;
published EIGNN comparisons [78] must use the full graph and fixed Planetoid
masks, which the same loader returns when `subset_nodes=None`.
""",
            ),
            _snapshot_loader(prefix, "cora-induced-96.pt"),
            code(
                prefix,
                """
from torch.nn import functional as F
from silva_networks import normalized_graph_operator

real_x = source_sample.tensors["x"]
real_edges = source_sample.tensors["edge_index"]
real_y = source_sample.tensors["y"].long()
real_train = source_sample.tensors["train_mask"].bool()
real_validation = source_sample.tensors["validation_mask"].bool()
real_test = source_sample.tensors["test_mask"].bool()
real_operator = normalized_graph_operator(real_edges, real_x.shape[0]).to(real_x)
real_model = SILVAEfficientInfiniteGraphEquilibrium(
    real_x.shape[1],
    16,
    int(real_y.max()) + 1,
    gamma=0.7,
    solve_mode="iterative",
    config=SolverConfig(
        solver="picard", max_iter=80, tol=1e-6, backward_mode="unrolled"
    ),
)
optimizer = torch.optim.Adam(real_model.parameters(), lr=1e-2)
real_losses = []
for _ in range(3):
    optimizer.zero_grad()
    real_result = real_model(real_x, real_operator, return_result=True)
    real_loss = F.cross_entropy(real_result.output[real_train], real_y[real_train])
    real_loss.backward()
    optimizer.step()
    real_losses.append(float(real_loss.detach()))

real_result = real_model(real_x, real_operator, return_result=True)
prediction = real_result.output.argmax(dim=1)
for name, mask in (
    ("train", real_train),
    ("validation", real_validation),
    ("test", real_test),
):
    accuracy = (prediction[mask] == real_y[mask]).float().mean()
    print(name, "nodes", int(mask.sum()), "compact accuracy", float(accuracy))
print("loss trajectory:", real_losses)
print("residual:", real_result.solver_result.residual)
""",
            ),
            code(
                prefix,
                """
eigenvalues, eigenvectors = torch.linalg.eigh(real_operator)
coordinates = eigenvectors[:, -3:-1].detach()
figure, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=real_y, s=14, cmap="tab10")
axes[0].set_title("source labels")
axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=prediction, s=14, cmap="tab10")
axes[1].set_title("current predictions")
axes[2].semilogy(real_result.solver_result.residuals)
axes[2].set(xlabel="equilibrium iteration", ylabel="residual")
for axis in axes[:2]:
    axis.set(xticks=[], yticks=[])
figure.tight_layout()
plt.show()
""",
            ),
        ]
    if family == "mgnni":
        return [
            md(
                prefix,
                r"""
## 5. Cora Across Three Graph Scales

For the same source-indexed Cora subgraph [82], the scale branches apply
$S$, $S^2$, and $S^4$ before nodewise fusion. The heat map below exposes the
learned scale allocation instead of hiding it behind a single prediction.
Full MGNNI comparisons [79] require the complete Planetoid graph and source
split; this compact induced graph validates construction, gradients, and
diagnostics only.
""",
            ),
            _snapshot_loader(prefix, "cora-induced-96.pt"),
            code(
                prefix,
                """
from torch.nn import functional as F
from silva_networks import normalized_graph_operator

real_x = source_sample.tensors["x"]
real_edges = source_sample.tensors["edge_index"]
real_y = source_sample.tensors["y"].long()
real_train = source_sample.tensors["train_mask"].bool()
real_operator = normalized_graph_operator(real_edges, real_x.shape[0]).to(real_x)
real_model = SILVAMultiscaleGraphImplicitNetwork(
    real_x.shape[1],
    16,
    int(real_y.max()) + 1,
    scales=(1, 2, 4),
    gamma=0.7,
    config=SolverConfig(
        solver="picard", max_iter=80, tol=1e-6, backward_mode="unrolled"
    ),
)
optimizer = torch.optim.Adam(real_model.parameters(), lr=1e-2)
real_losses = []
for _ in range(3):
    optimizer.zero_grad()
    real_result = real_model(real_x, real_operator, return_result=True)
    real_loss = F.cross_entropy(real_result.output[real_train], real_y[real_train])
    real_loss.backward()
    optimizer.step()
    real_losses.append(float(real_loss.detach()))

real_result = real_model(real_x, real_operator, return_result=True)
print("loss trajectory:", real_losses)
print("mean scale weights:", real_result.attention_weights.mean(dim=0).tolist())
print(
    "scale residuals:",
    [result.residual for result in real_result.solver_results],
)
assert torch.allclose(
    real_result.attention_weights.sum(dim=1),
    torch.ones(real_x.shape[0]),
    atol=1e-6,
)
""",
            ),
            code(
                prefix,
                """
figure, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
axes[0].plot(range(1, len(real_losses) + 1), real_losses, marker="o")
axes[0].set(xlabel="optimizer step", ylabel="masked cross entropy")
image = axes[1].imshow(
    real_result.attention_weights.detach().T,
    aspect="auto",
    cmap="magma",
)
axes[1].set(
    xlabel="Cora subset node",
    ylabel="scale branch",
    yticks=range(3),
    yticklabels=["1", "2", "4"],
)
figure.colorbar(image, ax=axes[1], fraction=0.046)
axes[2].bar(
    ["1", "2", "4"],
    [result.residual for result in real_result.solver_results],
)
axes[2].set(xlabel="graph scale", ylabel="final residual", yscale="log")
figure.tight_layout()
plt.show()
""",
            ),
        ]
    if family == "delta":
        return [
            md(
                prefix,
                r"""
## 6. Real-Image Delta Activity

The public motion snapshot contains consecutive frames 100 and 101 from the
real-video example used by the optical-flow tutorial [86]. It has no ground-truth
flow, so this section measures only cache activity and disagreement with the
exact convolution. Sintel [83], KITTI Flow [84], or FlyingChairs [85] must be
loaded through `load_optical_flow_source_subset` for supervised endpoint-error
experiments.
""",
            ),
            _snapshot_loader(prefix, "public-motion-frames-100-101.pt"),
            code(
                prefix,
                """
real_first = source_sample.tensors["frame1"]
real_second = source_sample.tensors["frame2"]
torch.manual_seed(98)
real_convolution = nn.Conv2d(3, 6, kernel_size=3, padding=1)
thresholds = [0.0, 1e-4, 1e-3, 1e-2]
real_activity = []
real_errors = []
real_max_errors = []
for threshold in thresholds:
    real_cache = SILVADeltaOperator(real_convolution, threshold=threshold)
    real_cache(real_first)
    cached_second = real_cache(real_second)
    exact_second = real_convolution(real_second)
    real_activity.append(real_cache.stats[-1].active_fraction)
    real_errors.append(
        float(torch.linalg.vector_norm(cached_second - exact_second).detach())
    )
    real_max_errors.append(
        float((cached_second - exact_second).abs().max().detach())
    )

for row in zip(thresholds, real_activity, real_errors, real_max_errors):
    print("threshold/activity/error norm/maximum error:", row)
assert real_max_errors[0] < 1e-6
""",
            ),
            code(
                prefix,
                """
difference = (real_second - real_first).abs().mean(dim=1)[0]
figure, axes = plt.subplots(1, 4, figsize=(10.2, 2.5))
axes[0].imshow(real_first[0].permute(1, 2, 0))
axes[0].set_title("frame 100")
axes[1].imshow(real_second[0].permute(1, 2, 0))
axes[1].set_title("frame 101")
axes[2].imshow(difference, cmap="inferno")
axes[2].set_title("absolute change")
axes[3].semilogx(thresholds[1:], real_activity[1:], marker="o", label="activity")
axes[3].set(xlabel="delta threshold", ylabel="active fraction")
for axis in axes[:3]:
    axis.axis("off")
figure.tight_layout()
plt.show()
""",
            ),
        ]
    raise KeyError(family)


def monotone_lab() -> dict[str, object]:
    p = "silva-mondeq"
    cells = [
        md(
            p,
            r"""
# SILVA Monotone Operator Equilibrium

This lab derives the monotone inclusion, runs forward-backward and
Peaceman-Rachford splitting on the same known solution, checks the certificate,
and shows how a custom structured operator enters SILVA. The defining mechanism
follows monDEQ [75].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from torch import nn
from silva_networks import (
    SILVAMonotoneOperatorEquilibrium,
    SolverConfig,
    make_monotone_operator_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. From Fixed Point to Monotone Inclusion

$$
z^\star=\operatorname{prox}_f(Wz^\star+Ux+b)
$$

is equivalent to

$$
0\in(I-W)z^\star-Ux-b+\partial f(z^\star).
$$

The source parameterization

$$
W=(1-m)I-A^\mathsf{T}A+B-B^\mathsf{T}
$$

gives

$$
\operatorname{Sym}(I-W)=mI+A^\mathsf{T}A\succeq mI.
$$
""",
        ),
        code(
            p,
            """
data = make_monotone_operator_dataset(samples=16, seed=91)

class KnownMonotoneOperator(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("weight", matrix)

    def forward(self, state):
        return state @ self.weight.T

    def resolvent(self, values, step_size):
        identity = torch.eye(self.weight.shape[0], device=values.device)
        system = (1 + step_size) * identity - step_size * self.weight
        return torch.linalg.solve(system, values.T).T

    def monotonicity_certificate(self):
        identity = torch.eye(self.weight.shape[0], device=self.weight.device)
        symmetric = identity - 0.5 * (self.weight + self.weight.T)
        return torch.linalg.eigvalsh(symmetric).min()

source = nn.Linear(4, 6)
readout = nn.Linear(6, 2, bias=False)
with torch.no_grad():
    source.weight.copy_(data.source)
    source.bias.copy_(data.bias)
    readout.weight.copy_(data.readout)

operator = KnownMonotoneOperator(data.recurrent)
print("certificate:", float(operator.monotonicity_certificate()))
print("known state shape:", data.equilibrium.shape)
""",
        ),
        md(
            p,
            r"""
## 2. Two Splittings, One Equilibrium

Forward-backward uses

$$
z_{k+1}=\operatorname{prox}_{af}
\left((1-a)z_k+a(Wz_k+Ux+b)\right).
$$

Peaceman-Rachford alternates a proximal reflection with the resolvent

$$
\left((1+a)I-aW\right)^{-1}.
$$
""",
        ),
        code(
            p,
            """
config = SolverConfig(
    solver="picard", max_iter=180, tol=1e-8, backward_mode="unrolled"
)
results = {}
for splitting in ("forward_backward", "peaceman_rachford"):
    model = SILVAMonotoneOperatorEquilibrium(
        4,
        6,
        2,
        operator=operator,
        source=source,
        prox=torch.relu,
        readout=readout,
        splitting=splitting,
        step_size=0.5,
        config=config,
    )
    results[splitting] = model(data.inputs, return_result=True)
    state_error = torch.linalg.vector_norm(
        results[splitting].state - data.equilibrium
    )
    print(splitting, "state error", float(state_error))
    assert state_error < 2e-4

agreement = torch.linalg.vector_norm(
    results["forward_backward"].state
    - results["peaceman_rachford"].state
)
assert agreement < 2e-4
print("splitter agreement:", float(agreement))
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
for name, result in results.items():
    axes[0].semilogy(result.solver_result.residuals, label=name.replace("_", " "))
axes[0].set(xlabel="iteration", ylabel="fixed-point residual")
axes[0].legend(fontsize=7)

target = data.target.detach().flatten()
prediction = results["peaceman_rachford"].output.detach().flatten()
axes[1].scatter(target, prediction, s=12)
limits = [float(min(target.min(), prediction.min())), float(max(target.max(), prediction.max()))]
axes[1].plot(limits, limits, color="black", linewidth=0.8)
axes[1].set(xlabel="known target", ylabel="SILVA readout")
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            """
## 3. Extension Boundary

A new monotone architecture can replace the dense operator when it implements
`forward`, `resolvent`, and `monotonicity_certificate`. Convolutional,
multiscale, diagonalizable, and matrix-free resolvents can therefore use the
same source, readout, splitting, result object, and training loop.
""",
        ),
    ]
    cells.extend(real_source_cells(p, "monotone"))
    cells.extend(source_preflight(p, "silva_monotone_operator_equilibrium"))
    return notebook(cells)


def positive_concave_lab() -> dict[str, object]:
    p = "silva-pcdeq"
    cells = [
        md(
            p,
            r"""
# SILVA Positive-Concave Equilibrium

This lab derives positive-concave order structure, trains the vector variant,
runs the convolutional variant, and checks state and weight positivity. The
mechanism follows pcDEQ [76].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from silva_networks import (
    SILVAPositiveConcaveEquilibrium,
    SolverConfig,
    make_positive_concave_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. Positive-Concave Map

$$
z^\star=\phi(W_+z^\star+s_+(x)),
\qquad W_+\geq0,
\qquad s_+(x)\geq0.
$$

SILVA supports a smooth positive parameterization

$$
W_+=\operatorname{softplus}(\widetilde W)+\epsilon_w.
$$

For source-repository alignment it also supports weight normalization,

$$
W_+=g_+\frac{v_+}{\lVert v_+\rVert}+\epsilon_w,
$$

where $v_+$ and $g_+$ are projected onto the nonnegative orthant after each
optimizer update. The direct projected parameterization is available as a
controlled ablation.

Variant 1 uses a strictly positive softplus source with tanh, softsign, or
ReLU6. Variant 2 uses a nonnegative ReLU source with sigmoid. Standard
fixed-point iteration retains the positive orthant.
""",
        ),
        code(
            p,
            """
data = make_positive_concave_dataset(samples=48, seed=92)
config = SolverConfig(
    solver="picard", max_iter=45, tol=1e-6, backward_mode="unrolled"
)
model = SILVAPositiveConcaveEquilibrium(
    3, 5, 1, variant=1, activation="tanh", config=config
)
initial = model(data.inputs, return_result=True)
assert initial.minimum_weight > 0
assert initial.state.min() >= 0
print("minimum recurrent weight:", float(initial.minimum_weight.detach()))
print("minimum state:", float(initial.state.min().detach()))
print("initial residual:", initial.solver_result.residual)
""",
        ),
        md(
            p,
            """
## 2. Compact Positive Regression

The known target is generated by a bounded nonnegative recurrent map. Training
checks the complete source-transition-readout gradient path rather than only a
standalone activation.
""",
        ),
        code(
            p,
            """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for _ in range(45):
    optimizer.zero_grad()
    prediction = model(data.inputs)
    loss = torch.nn.functional.mse_loss(prediction, data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

trained = model(data.inputs, return_result=True)
assert losses[-1] < losses[0]
assert trained.state.min() >= 0
print("initial/final loss:", losses[0], losses[-1])
print("trained residual:", trained.solver_result.residual)
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].semilogy(losses)
axes[0].set(xlabel="optimization step", ylabel="MSE")
axes[1].hist(trained.state.detach().flatten(), bins=18, color="#2f7d32")
axes[1].set(xlabel="equilibrium state", ylabel="count")
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            r"""
## 3. Spatial Positive-Concave Point

For a convolutional state,

$$
Z^\star=\phi(K_+*Z^\star+S_+(X)),
\qquad K_+\geq0.
$$

The state shape remains $(B,C,H,W)$, so this transition can sit inside a larger
image architecture without flattening the field.
""",
        ),
        code(
            p,
            """
spatial = SILVAPositiveConcaveEquilibrium(
    2,
    4,
    1,
    variant=2,
    operator="conv2d",
    kernel_size=3,
    config=SolverConfig(max_iter=8, tol=1e-5, backward_mode="unrolled"),
)
images = torch.rand(3, 2, 7, 9)
spatial_result = spatial(images, return_result=True)
assert spatial_result.output.shape == (3, 1, 7, 9)
assert torch.all((0 <= spatial_result.state) & (spatial_result.state <= 1))
print("spatial output:", spatial_result.output.shape)
print("spatial state range:", float(spatial_result.state.min()), float(spatial_result.state.max()))
""",
        ),
        md(
            p,
            """
## 4. Source-Aligned Weight Projection

The reference training loop applies weight normalization and clamps its
direction and magnitude parameters after every optimizer step. This cell uses
the same policy while retaining SILVA's solver and result object. The smooth
softplus mode above remains useful when a differentiable positivity
parameterization is preferred.
""",
        ),
        code(
            p,
            """
projected = SILVAPositiveConcaveEquilibrium(
    3,
    5,
    1,
    variant=1,
    activation="tanh",
    weight_parameterization="source_weight_norm",
    config=config,
)
projected_optimizer = torch.optim.Adam(projected.parameters(), lr=1e-2)
projected_optimizer.zero_grad()
projected_loss = torch.nn.functional.mse_loss(
    projected(data.inputs), data.target
)
projected_loss.backward()
projected_optimizer.step()
projected.project_nonnegative_()
projected_result = projected(data.inputs, return_result=True)
assert projected.transition.weight_scale is not None
assert projected.transition.raw_weight.min() >= 0
assert projected.transition.weight_scale.min() >= 0
assert projected_result.minimum_weight >= 0
print("projected loss:", float(projected_loss.detach()))
print("minimum direction:", float(projected.transition.raw_weight.min()))
print("minimum magnitude:", float(projected.transition.weight_scale.min()))
""",
        ),
    ]
    cells.extend(real_source_cells(p, "positive_concave"))
    cells.extend(source_preflight(p, "silva_positive_concave_equilibrium"))
    return notebook(cells)


def non_euclidean_lab() -> dict[str, object]:
    p = "silva-nemon"
    cells = [
        md(
            p,
            r"""
# SILVA Non-Euclidean Equilibrium

This lab derives the weighted-infinity certificate, reconstructs an exact
compact map, compares clean and perturbed equilibria, and evaluates the
analytic sensitivity bound. The mechanism follows NEMON [77].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from silva_networks import (
    SILVANonEuclideanEquilibrium,
    SolverConfig,
    make_non_euclidean_robustness_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. Weighted Infinity Geometry

$$
\lVert z\rVert_{\infty,D}=\lVert Dz\rVert_\infty,
$$

$$
\mu_{\infty,D}(W)
=\max_i\left((DWD^{-1})_{ii}
+\sum_{j\ne i}|(DWD^{-1})_{ij}|\right).
$$

With free $A$ and target $m<1$,

$$
W=mI+D^{-1}AD-\operatorname{diag}(|A|\mathbf1)
$$

ensures $\mu_{\infty,D}(W)\leq m$.
""",
        ),
        code(
            p,
            """
data = make_non_euclidean_robustness_dataset(samples=24, seed=93)
model = SILVANonEuclideanEquilibrium(
    4,
    6,
    2,
    one_sided_bound=0.05,
    averaging=0.5,
    config=SolverConfig(
        solver="picard", max_iter=180, tol=1e-7, backward_mode="unrolled"
    ),
)
with torch.no_grad():
    model.operator.free_weight.copy_(data.free_weight)
    model.operator.log_metric.copy_(data.metric.log())
    model.source.weight.copy_(data.source)
    model.source.bias.copy_(data.bias)

nominal = model(data.inputs, return_result=True)
perturbed = model(data.perturbed_inputs, return_result=True)
state_error = torch.linalg.vector_norm(nominal.state - data.equilibrium)
assert nominal.one_sided_lipschitz <= 0.050001
assert state_error < 3e-4
print("one-sided certificate:", float(nominal.one_sided_lipschitz))
print("known-state error:", float(state_error))
print("recommended/used averaging:", float(model.operator.recommended_averaging()), float(nominal.averaging))
""",
        ),
        md(
            p,
            r"""
## 2. Sensitivity Bound

For source matrix $U$,

$$
\operatorname{Lip}(x\mapsto z^\star(x))
\leq
\frac{\lVert DU\rVert_\infty}
{1-\mu_{\infty,D}(W)}.
$$

The compact perturbation is entrywise bounded. The empirical ratio should be
finite and can be compared with the returned analytic latent bound.
""",
        ),
        code(
            p,
            """
input_change = torch.linalg.vector_norm(
    data.perturbed_inputs - data.inputs, ord=float("inf"), dim=1
)
state_change = torch.linalg.vector_norm(
    perturbed.state - nominal.state, ord=float("inf"), dim=1
)
empirical = state_change / input_change
print("maximum empirical ratio:", float(empirical.max()))
print("analytic latent bound:", float(nominal.latent_input_lipschitz_bound))
assert torch.isfinite(empirical).all()
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
axes[0].semilogy(nominal.solver_result.residuals, label="clean")
axes[0].semilogy(perturbed.solver_result.residuals, label="perturbed")
axes[0].set(xlabel="iteration", ylabel="fixed-point residual")
axes[0].legend()
axes[1].scatter(input_change.detach(), state_change.detach(), s=14)
bound_x = torch.linspace(0, float(input_change.max()), 30)
axes[1].plot(
    bound_x,
    bound_x * float(nominal.latent_input_lipschitz_bound.detach()),
    color="black",
    label="analytic bound",
)
axes[1].set(xlabel="input change", ylabel="state change")
axes[1].legend(fontsize=7)
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            """
## 3. Extension Boundary

The operator, metric, source, activation, averaging coefficient, and readout
are replaceable. A custom operator must expose its metric weights, weighted
matrix measure, and recommended averaging rule when the corresponding
certificate fields are expected in the result.
""",
        ),
    ]
    cells.extend(real_source_cells(p, "non_euclidean"))
    cells.extend(source_preflight(p, "silva_non_euclidean_equilibrium"))
    return notebook(cells)


def eignn_lab() -> dict[str, object]:
    p = "silva-eignn"
    cells = [
        md(
            p,
            r"""
# SILVA Efficient Infinite-Depth Graph Equilibrium

This lab derives the EIGNN spectral solution, verifies it against the iterative
SILVA route, checks gradients, and explains when a sparse solve is required.
The defining mechanism follows EIGNN [78].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from silva_networks import (
    SILVAEfficientInfiniteGraphEquilibrium,
    SolverConfig,
    make_eignn_chain_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. Graph-Channel Equilibrium

$$
C=g(F)=\frac{F^\mathsf{T}F}{\lVert F^\mathsf{T}F\rVert_F+\epsilon_F},
$$

$$
Z^\star=\gamma S^\mathsf{T}Z^\star C^\mathsf{T}+X.
$$

For $S=Q\Lambda Q^\mathsf{T}$ and $C=V\Sigma V^\mathsf{T}$,

$$
(Q^\mathsf{T}Z^\star V)_{ij}
=\frac{(Q^\mathsf{T}XV)_{ij}}
{1-\gamma\lambda_i\sigma_j}.
$$
""",
        ),
        code(
            p,
            """
data = make_eignn_chain_dataset(nodes=21, state_dim=4, seed=94)
model = SILVAEfficientInfiniteGraphEquilibrium(
    3,
    4,
    1,
    gamma=data.gamma,
    solve_mode="closed_form",
    config=SolverConfig(
        solver="picard", max_iter=350, tol=1e-8, backward_mode="unrolled"
    ),
)
with torch.no_grad():
    model.factor.copy_(data.factor)
    model.source.weight.zero_()
    model.source.bias.zero_()
    diagonal = min(model.in_dim, model.state_dim)
    model.source.weight[:diagonal, :diagonal] = torch.eye(diagonal)
    model.readout.weight.zero_()
    model.readout.bias.zero_()
    model.readout.weight[0, 0] = 1

spectrum = model.precompute_spectrum(data.graph_operator)
closed = model(
    data.inputs, data.graph_operator, spectrum=spectrum, return_result=True
)
model.solve_mode = "iterative"
iterative = model(data.inputs, data.graph_operator, return_result=True)
agreement = torch.linalg.vector_norm(closed.state - iterative.state)
known_error = torch.linalg.vector_norm(closed.state - data.equilibrium)
assert agreement < 5e-5
assert known_error < 5e-5
print("closed/iterative agreement:", float(agreement))
print("known-state error:", float(known_error))
print("minimum spectral denominator:", float(closed.denominator_margin))
""",
        ),
        md(
            p,
            """
## 2. Direct and Iterative Differentiation

The closed-form route is differentiated directly through the channel
eigendecomposition. The iterative route uses the configured SILVA backward
mode. Both routes expose the same state and readout contract.
""",
        ),
        code(
            p,
            """
model.zero_grad(set_to_none=True)
closed.output.square().mean().backward()
assert model.factor.grad is not None
print("factor gradient norm:", float(model.factor.grad.norm()))
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
axes[0].plot(data.target[:, 0], label="known")
axes[0].plot(closed.output.detach()[:, 0], "--", label="closed form")
axes[0].set(xlabel="chain node", ylabel="long-range response")
axes[0].legend()
axes[1].semilogy(iterative.solver_result.residuals)
axes[1].set(xlabel="iteration", ylabel="iterative residual")
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            """
## 3. Dense Versus Sparse Route

The dense spectral cache stores an $N$ by $N$ eigenvector matrix. Use it only
when that matrix fits comfortably. Large graphs should retain a sparse graph
operator and select `solve_mode="iterative"`; no dense graph power or spectrum
is then required.
""",
        ),
    ]
    cells.extend(real_source_cells(p, "eignn"))
    cells.extend(source_preflight(p, "silva_efficient_infinite_graph"))
    return notebook(cells)


def mgnni_lab() -> dict[str, object]:
    p = "silva-mgnni"
    cells = [
        md(
            p,
            r"""
# SILVA Multiscale Graph Implicit Network

This lab derives one equilibrium per graph-power scale, verifies every known
scale state, trains nodewise attention, and inspects the fusion weights. The
defining mechanism follows MGNNI [79].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from silva_networks import (
    SILVAMultiscaleGraphImplicitNetwork,
    SolverConfig,
    make_mgnni_multiscale_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. Parallel Infinite Graph Scales

$$
Z_m^\star=\gamma(S^m)^\mathsf{T}Z_m^\star g(F_m)^\mathsf{T}+X.
$$

Nodewise fusion uses

$$
e_{im}=q^\mathsf{T}\tanh(W_a z_{im}^\star+b_a),
\qquad
\beta_{im}=\frac{\exp(e_{im})}{\sum_r\exp(e_{ir})},
$$

$$
z_i=\sum_m\beta_{im}z_{im}^\star.
$$
""",
        ),
        code(
            p,
            """
scales = (1, 2, 3)
data = make_mgnni_multiscale_dataset(
    nodes=24, state_dim=4, scales=scales, seed=95
)
model = SILVAMultiscaleGraphImplicitNetwork(
    3,
    4,
    1,
    scales=scales,
    gamma=data.gamma,
    config=SolverConfig(
        solver="picard", max_iter=220, tol=1e-7, backward_mode="unrolled"
    ),
)
with torch.no_grad():
    for factor in model.factors:
        factor.copy_(torch.eye(4))
    model.source.weight.zero_()
    model.source.bias.zero_()
    model.source.weight[:3, :3] = torch.eye(3)
    model.readout.weight.zero_()
    model.readout.bias.zero_()
    model.readout.weight[0, 0] = 1

result = model(data.inputs, data.graph_operator, return_result=True)
scale_errors = [
    float(torch.linalg.vector_norm(actual - expected))
    for actual, expected in zip(result.scale_states, data.scale_states)
]
assert max(scale_errors) < 8e-5
assert torch.allclose(result.attention_weights.sum(dim=1), torch.ones(24))
print("per-scale known-state errors:", scale_errors)
print("per-scale final residuals:", [value.residual for value in result.solver_results])
""",
        ),
        md(
            p,
            """
## 2. Learn the Nodewise Scale Preference

The compact target changes its preferred graph scale with node position. The
equilibrium states are held fixed here so the optimization isolates the
attention mechanism.
""",
        ),
        code(
            p,
            """
states = torch.stack([value.detach() for value in result.scale_states], dim=1)
optimizer = torch.optim.Adam(
    [*model.attention_projection.parameters(), model.attention_query], lr=3e-2
)
losses = []
for _ in range(100):
    optimizer.zero_grad()
    scores = torch.tanh(model.attention_projection(states))
    scores = torch.einsum("nka,a->nk", scores, model.attention_query)
    weights = torch.softmax(scores, dim=1)
    fused = (weights.unsqueeze(-1) * states).sum(dim=1)
    loss = torch.nn.functional.mse_loss(fused[:, :1], data.target)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))

trained = model(data.inputs, data.graph_operator, return_result=True)
assert losses[-1] < losses[0]
print("initial/final fusion loss:", losses[0], losses[-1])
print("attention normalization error:", float((trained.attention_weights.sum(1) - 1).abs().max()))
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
for scale, state in zip(scales, trained.scale_states):
    axes[0].plot(state.detach()[:, 0], label=f"scale {scale}")
axes[0].set(xlabel="chain node", ylabel="equilibrium channel 0")
axes[0].legend(fontsize=7)
image = axes[1].imshow(
    trained.attention_weights.detach().T,
    aspect="auto",
    origin="lower",
    cmap="viridis",
)
axes[1].set(xlabel="chain node", ylabel="scale index")
figure.colorbar(image, ax=axes[1], fraction=0.046)
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            """
## 3. Graph-Conditioned Source Construction

The packaged default applies a feature projection before every scale. A source
architecture that computes $f(X,G)$ can instead receive both the features and
graph operator. This keeps graph preprocessing outside the equilibrium while
making the injected state fully configurable.
""",
        ),
        code(
            p,
            """
from torch import nn

class GraphConditionedSource(nn.Module):
    def __init__(self, in_dim, state_dim):
        super().__init__()
        self.projection = nn.Linear(2 * in_dim, state_dim)

    def forward(self, features, graph_operator):
        one_hop = graph_operator @ features
        return self.projection(torch.cat((features, one_hop), dim=-1))

graph_source = GraphConditionedSource(3, 4)
graph_model = SILVAMultiscaleGraphImplicitNetwork(
    3,
    4,
    1,
    scales=(1, 2),
    graph_source=graph_source,
    config=SolverConfig(max_iter=80, tol=1e-6, backward_mode="unrolled"),
)
graph_output = graph_model(data.inputs, data.graph_operator)
graph_output.square().mean().backward()
assert graph_source.projection.weight.grad is not None
print("graph-conditioned output:", graph_output.shape)
print("source gradient norm:", float(graph_source.projection.weight.grad.norm()))
""",
        ),
        md(
            p,
            """
## 4. Extension Boundary

Every scale has its own channel factor and may receive its own `SolverConfig`.
The source map, scale list, attention module, mean-fusion ablation, and readout
are independently configurable. Sparse graph propagation applies the operator
repeatedly and does not require a dense graph power.
""",
        ),
    ]
    cells.extend(real_source_cells(p, "mgnni"))
    cells.extend(source_preflight(p, "silva_multiscale_graph_implicit"))
    return notebook(cells)


def delta_lab() -> dict[str, object]:
    p = "silva-deltadeq"
    cells = [
        md(
            p,
            r"""
# SILVA Delta-Cached Equilibrium

This lab derives cached linear updates, verifies zero-threshold equivalence,
sweeps the activity-error tradeoff, and tests a convolutional cache. The
defining mechanism follows DeltaDEQ [80].
""",
        ),
        code(p, BOOTSTRAP),
        code(
            p,
            PLOT_SETUP
            + """
from torch import nn
from silva_networks import (
    SILVADeltaEquilibrium,
    SILVADeltaOperator,
    SolverConfig,
    make_delta_heterogeneous_dataset,
)
""",
        ),
        md(
            p,
            r"""
## 1. Delta Identity

For $L(z)=Wz+b$,

$$
L(z_{k+1})=L(z_k)+W(z_{k+1}-z_k).
$$

With

$$
\Delta_\tau z_{k+1}
=\mathbf1(|z_{k+1}-z_k|>\tau)\odot(z_{k+1}-z_k),
$$

the cache update is

$$
c_{k+1}=c_k+W\Delta_\tau z_{k+1}.
$$

At $\tau=0$ this is algebraically equivalent to full linear evaluation. At
$\tau>0$ it is an approximation whose task error and exact residual must be
reported with activity.
""",
        ),
        code(
            p,
            """
data = make_delta_heterogeneous_dataset(samples=8, state_dim=6, seed=96)
recurrent = nn.Linear(6, 6, bias=False).double()
source = nn.Linear(3, 6).double()
readout = nn.Linear(6, 1, bias=False).double()
with torch.no_grad():
    recurrent.weight.copy_(torch.diag(data.rates.double()))
    source.weight.copy_(data.source.double())
    source.bias.copy_(data.bias.double())
    readout.weight.fill_(1 / 6)

model = SILVADeltaEquilibrium(
    3,
    6,
    1,
    recurrent=recurrent,
    source=source,
    activation=lambda value: value,
    readout=readout,
    delta_threshold=0.0,
    config=SolverConfig(
        solver="picard", max_iter=420, tol=1e-10, backward_mode="unrolled"
    ),
)
inputs = data.inputs.double()
model.train()
full = model(inputs, use_delta=False, return_result=True)
model.eval()
delta = model(inputs, use_delta=True, return_result=True)
agreement = torch.linalg.vector_norm(full.state - delta.state)
known_error = torch.linalg.vector_norm(delta.state - data.equilibrium.double())
assert agreement < 1e-7
assert known_error < 1e-6
print("zero-threshold full/delta agreement:", float(agreement))
print("known equilibrium error:", float(known_error))
print("exact full-map residual:", delta.exact_residual)
""",
        ),
        md(
            p,
            """
## 2. Threshold Sweep

The exact residual is always evaluated with the ordinary full recurrent map.
The active fraction records retained coordinate changes after the initial
cache fill. Neither quantity should be replaced by a hardware-independent
speed claim.
""",
        ),
        code(
            p,
            """
thresholds = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
active = []
state_errors = []
exact_residuals = []
for threshold in thresholds:
    model.delta_operator.threshold = threshold
    value = model(inputs, use_delta=True, return_result=True)
    active.append(value.mean_active_fraction)
    state_errors.append(float(torch.linalg.vector_norm(value.state - full.state)))
    exact_residuals.append(value.exact_residual)

for row in zip(thresholds, active, state_errors, exact_residuals):
    print("threshold/activity/state error/exact residual:", row)
assert active[-1] < active[0]
""",
        ),
        code(
            p,
            """
figure, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
axes[0].semilogx(thresholds[1:], active[1:], marker="o")
axes[0].set(xlabel="delta threshold", ylabel="mean active fraction")
axes[1].loglog(thresholds[1:], state_errors[1:], marker="o", label="state error")
axes[1].loglog(thresholds[1:], exact_residuals[1:], marker="s", label="exact residual")
axes[1].set(xlabel="delta threshold", ylabel="error")
axes[1].legend(fontsize=7)
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            r"""
## 3. Why Activity Becomes Sparse

For diagonal rate $r_j$, the affine Picard error obeys

$$
e_{k,j}=r_j^k e_{0,j}.
$$

Coordinates with small $r_j$ cross a fixed delta threshold first. This is the
compact analogue of heterogeneous convergence in larger recurrent blocks.
""",
        ),
        code(
            p,
            """
iterations = torch.arange(40)
figure, axis = plt.subplots(figsize=(5.0, 2.8))
for rate in data.rates:
    axis.semilogy(iterations, rate ** iterations, label=f"r={float(rate):.2f}")
axis.set(xlabel="iteration", ylabel="relative coordinate error")
axis.legend(ncol=2, fontsize=6)
figure.tight_layout()
plt.show()
""",
        ),
        md(
            p,
            """
## 4. Convolutional Cache

The same identity applies to convolution because convolution is linear in its
input. The bias is added once and removed from subsequent increments.
""",
        ),
        code(
            p,
            """
torch.manual_seed(97)
convolution = nn.Conv2d(2, 3, kernel_size=3, padding=1)
cached = SILVADeltaOperator(convolution, threshold=0.0)
first = torch.randn(2, 2, 8, 9)
second = torch.randn(2, 2, 8, 9)
cached(first)
cached_second = cached(second)
direct_second = convolution(second)
assert torch.allclose(cached_second, direct_second, atol=2e-6, rtol=2e-6)
print("convolution maximum difference:", float((cached_second - direct_second).abs().max()))
print("second-update active fraction:", cached.stats[-1].active_fraction)
""",
        ),
        md(
            p,
            """
## 5. Training Routes

The source-aligned experiment trains the ordinary full map and enables delta
caches only for evaluation. SILVA also permits a delta-cached forward solve
during training with implicit differentiation. In that extension, the forward
state uses the thresholded cache while the backward linear system uses the
exact full transition. Unrolled differentiation is not valid for the mutable
cache and is rejected explicitly.
""",
        ),
        code(
            p,
            """
train_model = SILVADeltaEquilibrium(
    3,
    6,
    1,
    delta_threshold=1e-4,
    config=SolverConfig(
        solver="picard",
        max_iter=40,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="picard",
        backward_max_iter=40,
    ),
)
train_inputs = data.inputs.clone().requires_grad_(True)
train_model.train()
train_result = train_model(train_inputs, use_delta=True, return_result=True)
train_result.output.square().mean().backward()
assert train_inputs.grad is not None
assert train_model.recurrent.weight.grad is not None
assert train_model.source.weight.grad is not None
print("delta-forward training residual:", train_result.exact_residual)
print("input gradient norm:", float(train_inputs.grad.norm()))
print("recurrent gradient norm:", float(train_model.recurrent.weight.grad.norm()))
""",
        ),
    ]
    cells.extend(real_source_cells(p, "delta"))
    cells.extend(source_preflight(p, "silva_delta_equilibrium"))
    return notebook(cells)


NOTEBOOKS = {
    "36_silva_monotone_operator_equilibrium.ipynb": monotone_lab,
    "37_silva_positive_concave_equilibrium.ipynb": positive_concave_lab,
    "38_silva_non_euclidean_equilibrium.ipynb": non_euclidean_lab,
    "39_silva_efficient_infinite_graph.ipynb": eignn_lab,
    "40_silva_multiscale_graph_implicit.ipynb": mgnni_lab,
    "41_silva_delta_equilibrium.ipynb": delta_lab,
}


def main() -> None:
    for name, builder in NOTEBOOKS.items():
        payload = builder()
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
