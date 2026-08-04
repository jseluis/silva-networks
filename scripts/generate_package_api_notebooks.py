from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/package_api"
DOCS_OUT = ROOT / "docs/package-notebooks"
COLAB_OUT = ROOT / "colab"
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE)


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


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"cell-{_CELL_COUNTER:04d}"


def _matplotlib_import_with_publication_dpi(match: re.Match[str]) -> str:
    indent = match.group("indent")
    return (
        f"{indent}import matplotlib.pyplot as plt\n\n"
        f'{indent}plt.rcParams.update({{"figure.dpi": 300, "savefig.dpi": 300}})'
    )


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


CITATION_SOURCE = r"""
## Citation and Sources

If this notebook or package is used, cite the software repository:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770099
```

When the work is connected to the SILVA Networks paper, cite the paper as well:

```text
Jose Luis Lima de Jesus Silva. SILVA Networks as Structured Implicit Layers and
Vector Attractors via Dynamic Interaction Fields. 2026. arXiv:2607.28989.
https://arxiv.org/abs/2607.28989
```

Background references used in the tutorial suite include:

- Deep Equilibrium Models, Bai, Kolter, and Koltun, NeurIPS 2019:
  https://arxiv.org/abs/1909.01377
- Multiscale Deep Equilibrium Models, Bai, Koltun, and Kolter, NeurIPS 2020:
  https://arxiv.org/abs/2006.08656
- Stabilizing Equilibrium Models by Jacobian Regularization, Bai, Koltun, and
  Kolter, ICML 2021: https://arxiv.org/abs/2106.14342
- Graph Attention Networks, Velickovic et al., ICLR 2018:
  https://arxiv.org/abs/1710.10903
- Attention Is All You Need, Vaswani et al., 2017:
  https://arxiv.org/abs/1706.03762
"""


STATIC_MIRRORED_NOTEBOOKS = ("07_research_citation_audit.ipynb",)


NOTEBOOKS = {
    "01_package_quickstart.ipynb": notebook(
        [
            md(
                r"""
# Package Quickstart

This notebook uses the public package API directly.

The basic SILVA equilibrium has the form

$$
z^\star = \sigma\{S_\theta(x)+L_\theta(z^\star)+G_\theta(z^\star)\}.
$$

The forward pass solves for \(z^\star\), then a PyTorch head maps the state to
outputs.
"""
            ),
            code(BOOTSTRAP),
            code(
                """
import torch
import matplotlib.pyplot as plt
from silva_networks import SILVAGraphNetwork, SolverConfig, resolve_device

torch.manual_seed(0)
device = resolve_device("auto")
device
"""
            ),
            md(
                r"""
Create a small graph. The edge tensor has shape \((2,E)\): the first row stores
sources and the second row stores destinations.
"""
            ),
            code(
                """
x = torch.randn(10, 5, device=device)
edge_index = torch.tensor(
    [list(range(9)), list(range(1, 10))],
    dtype=torch.long,
    device=device,
)
"""
            ),
            code(
                """
model = SILVAGraphNetwork(
    in_dim=5,
    hidden_dims=[16, 16],
    out_dim=3,
    task="node",
    local="graph",
    global_term="mean",
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.5, history=4),
).to(device)

result = model(x, edge_index=edge_index, return_results=True)
logits = result.output
logits.shape, [round(r.residual, 6) for r in result.solver_results]
"""
            ),
            md(
                r"""
The model is an ordinary PyTorch module. Parameters are visible to optimizers,
and gradients flow through the unrolled fixed-point solve used in the public
implementation.
"""
            ),
            code(
                """
y = torch.randint(0, 3, (x.shape[0],), device=device)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
sum(p.grad is not None for p in model.parameters()), float(loss.detach().cpu())
"""
            ),
            md(
                r"""
	The residual curve records
	
	$$
	\|f_\theta(z_k,x)-z_k\|_2
	$$
	
	at each solver step. The curve is a practical diagnostic for whether the
	chosen damping and solver budget are reasonable on this example.
	"""
            ),
            code(
                """
plt.figure(figsize=(5, 3))
for layer_index, solver_result in enumerate(result.solver_results):
    plt.plot(solver_result.residuals, marker="o", label=f"layer {layer_index}")
plt.yscale("log")
plt.xlabel("solver step")
plt.ylabel("residual")
plt.legend()
plt.tight_layout()
"""
            ),
        ]
    ),
    "02_solvers_and_jacobians.ipynb": notebook(
        [
            md(
                r"""
# Solvers and Jacobians

For a transition \(f\), the equilibrium solves

$$
r(z)=f(z)-z=0.
$$

Local stability is governed by the state Jacobian

$$
J_z=\frac{\partial f}{\partial z}(z^\star).
$$
"""
            ),
            code(BOOTSTRAP),
            code(
                """
import torch
import matplotlib.pyplot as plt
from silva_networks import SolverConfig, fixed_point, full_jacobian, jvp, stability_report, vjp

torch.manual_seed(1)
W = 0.25 * torch.randn(4, 4)
b = torch.linspace(-0.2, 0.2, 4)

def f(z):
    return torch.tanh(W @ z + b)

z0 = torch.zeros(4)
"""
            ),
            code(
                """
solver_runs = {}
for solver in ["picard", "anderson", "broyden"]:
    result = fixed_point(f, z0, SolverConfig(solver=solver, max_iter=20, alpha=0.6))
    solver_runs[solver] = result
    print(solver, result.iterations, result.residual)
"""
            ),
            code(
                """
plt.figure(figsize=(5, 3))
for solver, result in solver_runs.items():
    plt.plot(result.residuals, marker="o", label=solver)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("fixed-point residual")
plt.legend()
plt.tight_layout()
"""
            ),
            md(
                r"""
For small states, materialize \(J_z\). For larger states, use products
\(J_zv\) and \(J_z^\top v\).
"""
            ),
            code(
                """
result = fixed_point(f, z0, SolverConfig(solver="anderson", max_iter=20, alpha=0.6))
J = full_jacobian(f, result.z)
probe = torch.ones_like(result.z)
_, Jv = jvp(f, result.z, probe)
Jtv = vjp(f, result.z, probe)
print("J shape:", tuple(J.shape))
print("Jv:", Jv)
print("J^T v:", Jtv)
"""
            ),
            code(
                """
report = stability_report(f, result.z, samples=4, iters=8)
report
"""
            ),
        ]
    ),
    "03_datasets_to_silva.ipynb": notebook(
        [
            md(
                r"""
# Datasets to SILVA

This notebook downloads a public tabular dataset, standardizes the columns, and
builds a sample graph.

For each feature column,

$$
\tilde X_{ij}=\frac{X_{ij}-\mu_j}{\max(\sigma_j,\varepsilon)}.
$$

The graph connects each sample to its \(k\) nearest neighbors in standardized
feature space.
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
    available_datasets,
    load_tabular_dataset,
    resolve_device,
    tabular_to_silva_graph,
)

available_datasets()[:8]
"""
            ),
            md(
                r"""
The loader returns a small `TabularDataset` object. The adapter turns the table
into the package graph contract:

$$
(X,y)\longmapsto (x,E,b,y).
$$

Here \(x\) is the standardized feature matrix, \(E\) is the kNN edge set, and
`batch` is omitted because all rows form one graph.
"""
            ),
            code(
                """
device = resolve_device("auto")
dataset = load_tabular_dataset("iris", root="data", download=True, normalize=True)
graph = tabular_to_silva_graph(dataset, k=8, normalize=True, device=device)
x = graph.x
y = graph.y
edge_index = graph.edge_index
dataset.name, x.shape, y.shape, dataset.target_names
"""
            ),
            md(
                r"""
The edge tensor has shape \((2,E)\). The first row contains source indices and
the second row contains destination indices:

$$
\texttt{edge\_index}
=
\begin{bmatrix}
j_1&\cdots&j_E\\
i_1&\cdots&i_E
\end{bmatrix}.
$$
"""
            ),
            code("edge_index.shape"),
            md(
                r"""
The classifier treats each row as a node. The local branch is dynamic top-k
interaction and the global branch is mean-field context:

$$
z^\star
=
f_\theta(z^\star,x,E),
\qquad
\hat y_i=R_\phi(z_i^\star).
$$
"""
            ),
            code(
                """
model = SILVAGraphNetwork(
    in_dim=x.shape[1],
    hidden_dims=[16, 16],
    out_dim=len(dataset.target_names),
    task="node",
    local="topk",
    local_kwargs={"k": 8},
    global_term="mean",
    config=SolverConfig(solver="anderson", max_iter=8, alpha=0.5, history=4),
    head_hidden_dims=(16,),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for step in range(6):
    logits = model(x, edge_index=edge_index)
    loss = torch.nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    print(step, float(loss.detach().cpu()))
"""
            ),
            code(
                """
plt.figure(figsize=(5, 3))
plt.plot(losses, marker="o")
plt.xlabel("training step")
plt.ylabel("cross entropy")
plt.tight_layout()
"""
            ),
            code(
                """
accuracy = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
accuracy
"""
            ),
        ]
    ),
    "04_public_experiments.ipynb": notebook(
        [
            md(
                r"""
# Public Experiments

Public experiment configs are small package checks. They produce JSON metrics
that can be plotted later.

The residual reported by solver experiments is

$$
\|f_\theta(z)-z\|_2.
$$
"""
            ),
            code(BOOTSTRAP),
            code(
                """
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

runner_path = root / "experiments/public/run_experiment.py"
spec = spec_from_file_location("public_runner", runner_path)
runner = module_from_spec(spec)
spec.loader.exec_module(runner)
"""
            ),
            md(
                r"""
Experiment configs are plain JSON. A solver config names the transition,
solver, damping, and iteration budget. The runner returns numerical quantities
that can be compared across configurations without changing the package code.
"""
            ),
            code(
                """
config_path = root / "experiments/public/configs/solver_sweep.json"
config = json.loads(config_path.read_text())
metrics = runner.run_config(config)
metrics
"""
            ),
            md(
                r"""
Graph validation experiments use the same loss as a normal PyTorch classifier:

$$
\mathcal L(\theta,\phi)
=
\operatorname{CE}(R_\phi(z^\star),y).
$$

The metrics dictionary records the loss curve and final accuracy.
"""
            ),
            code(
                """
config_path = root / "experiments/public/configs/graph_silva_smoke.json"
config = json.loads(config_path.read_text())
metrics = runner.run_config(config)
metrics["losses"], metrics["accuracy"]
"""
            ),
            code(
                """
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(metrics["losses"], marker="o")
plt.xlabel("training step")
plt.ylabel("loss")
plt.tight_layout()
"""
            ),
            md(
                r"""
A fully configurable stack is still just JSON. The configuration below chooses
each local branch, global branch, self branch, and solver separately:

$$
z_\ell^\star=f_{\theta_\ell}(z_\ell^\star,h_{\ell-1}),
\qquad
h_\ell=z_\ell^\star.
$$

Layer \(\ell\) receives its own \(L_\ell\), \(G_\ell\), \(H_\ell\), and
`SolverConfig`.
"""
            ),
            code(
                """
config_path = root / "experiments/public/configs/fully_configurable_graph.json"
config = json.loads(config_path.read_text())
config["device"] = "cpu"
metrics = runner.run_config(config)
metrics["output_shape"], metrics["state_shape"], metrics["solver_residuals"]
"""
            ),
        ]
    ),
    "05_custom_operator_experiment.ipynb": notebook(
        [
            md(
                r"""
# Custom Operator Experiment

The package lets a custom branch replace \(L_\theta\) while preserving the
SILVA fixed-point structure:

$$
z^\star=\sigma\{S_\theta(x)+L_{\psi}(z^\star)+G_\theta(z^\star)\}.
$$
"""
            ),
            code(BOOTSTRAP),
            code(
                """
import torch
from silva_networks import SILVAGraphNetwork, SolverConfig

class SignedLocal(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, z, edge_index=None):
        messages = torch.tanh(self.proj(z))
        if edge_index is None:
            return messages
        src, dst = edge_index
        out = torch.zeros_like(messages)
        out.index_add_(0, dst, messages[src])
        return out
"""
            ),
            md(
                r"""
The custom module implements a local operator

$$
L_\psi:\mathbb R^{N\times d}\to\mathbb R^{N\times d}.
$$

The wrapper passes `edge_index` when it is available. The only hard contract is
that the returned tensor has the same shape as the recurrent state.
"""
            ),
            code(
                """
torch.manual_seed(4)
x = torch.randn(12, 6)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
edge_index = torch.tensor([list(range(11)), list(range(1, 12))], dtype=torch.long)

model = SILVAGraphNetwork(
    in_dim=6,
    hidden_dims=[16, 16],
    out_dim=2,
    task="node",
    local=lambda dim, index: SignedLocal(dim),
    global_term="mean",
    config=SolverConfig(solver="picard", max_iter=6, alpha=0.5),
)
"""
            ),
            md(
                r"""
Training does not need a special loop. The equilibrium layer is part of an
ordinary PyTorch module, so `loss.backward()` differentiates through the solver
steps used in the forward pass.
"""
            ),
            code(
                """
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for step in range(5):
    logits = model(x, edge_index=edge_index)
    loss = torch.nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
    print(step, float(loss.detach()))
"""
            ),
            code(
                """
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.plot(losses, marker="o")
plt.xlabel("training step")
plt.ylabel("custom-operator loss")
plt.tight_layout()
"""
            ),
            ]
        ),
        "06_silva_operator_options.ipynb": notebook(
            [
                md(
                    r"""
    # SILVA Operator Options
    
    The reference graph model keeps the Figure 1 decomposition explicit:
    
    $$
    z_{k+1}
    =
    (1-\alpha)z_k
    +
    \alpha\Phi\{S_\theta(x)+L_\theta(\tanh z_k,E)+G_\theta(\tanh z_k,b)\}.
    $$
    
    `hidden_dim`, `stack_alphas`, `graph_mode`, `attention_mode`, solver settings,
    and readout dimensions are constructor arguments.
    """
                ),
                code(BOOTSTRAP),
                code(
                    """
    import torch
    import matplotlib.pyplot as plt
    from silva_networks import (
        SILVAGraphPresetNetwork,
        SILVAMolecularRegressor,
        SILVAVisionVectorClassifier,
        SolverConfig,
        damped_spectral_radius,
        molecular_to_silva_graph,
        tabular_to_silva_graph,
    )
    
    torch.manual_seed(6)
    x = torch.randn(9, 5)
    y = (x[:, 0] > 0).long()
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8],
         [1, 2, 3, 4, 5, 6, 7, 8, 0]],
        dtype=torch.long,
    )
    """
                ),
                md(
                    r"""
    The SILVA ablations are available through operator names. Each row below
    uses the same input graph and changes only the local or global interaction.
    The ablation value is the string `"none"`: it removes that branch from
    \(L_\theta\) or \(G_\theta\). Solver self-persistence
    \((1-\alpha)z_k\) is still present, so the no-local/no-global case is
    stimulus plus damped recurrent persistence.
    """
                ),
                code(
                    """
    cases = [
        ("full", "GAT", "simple"),
        ("local_only", "GAT", "none"),
        ("global_only", "none", "simple"),
        ("static_global", "GAT", "static"),
        ("topk_global", "GAT", "topk"),
    ]
    
    case_rows = []
    for name, graph_mode, attention_mode in cases:
        model = SILVAGraphPresetNetwork(
            in_dim=5,
            hidden_dim=[12, 8],
            out_dim=2,
            task="node",
            graph_mode=graph_mode,
            attention_mode=attention_mode,
            num_heads=2,
            k_neighbors=3,
            stack_alphas=[0.5, 0.2],
            max_iter=3,
        )
        result = model(x, edge_index=edge_index, return_results=True)
        residuals = [float(r.residual) for r in result.solver_results]
        case_rows.append({"case": name, "shape": tuple(result.output.shape), "residuals": residuals})
        print(name, tuple(result.output.shape), [round(value, 5) for value in residuals])
    case_rows
    """
                ),
                md(
                    r"""
    The bar plot compares the largest final residual across the ablation stack.
    It is a quick numerical check that changing an operator still produces a
    finite solved state on the same input.
    """
                ),
                code(
                    """
    plt.figure(figsize=(6, 3))
    plt.bar([row["case"] for row in case_rows], [max(row["residuals"]) for row in case_rows])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("max final residual")
    plt.tight_layout()
    """
                ),
                md(
                    r"""
    The generic API accepts per-layer solvers and custom branches. A custom branch
    only has to return a tensor shaped like the recurrent state.
    """
                ),
                code(
                    """
    class BiasLocal(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(dim))
    
        def forward(self, z, edge_index=None, edge_attr=None, batch=None):
            return torch.tanh(z + self.bias)
    
    from silva_networks import SILVAGraphNetwork
    
    custom = SILVAGraphNetwork(
        in_dim=5,
        hidden_dims=[12, 8, 8],
        out_dim=2,
        task="node",
        local=lambda dim, index: BiasLocal(dim),
        global_term=["mean", "simple", "topk_attention"],
        global_kwargs=[None, None, {"k": 4}],
        self_term=[None, "linear", None],
        config=[
            SolverConfig(solver="picard", max_iter=3, alpha=0.5),
            SolverConfig(solver="anderson", max_iter=3, alpha=0.35, history=3),
            SolverConfig(solver="broyden", max_iter=3, alpha=0.25),
        ],
    )
    custom(x, edge_index=edge_index).shape
    """
                ),
                md(
                    r"""
    Molecular SILVA accepts categorical atom/bond ids or continuous feature
    vectors. The continuous path uses explicit input projectors.
    """
                ),
                code(
                    """
    mol = molecular_to_silva_graph(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 0, 4, 3, 0]]),
        edge_attr=torch.randn(6, 2),
        batch=torch.tensor([0, 0, 0, 1, 1]),
        y=torch.tensor([0.3, -0.1]),
    )
    molecular_model = SILVAMolecularRegressor(
        hidden_dim=[9, 6],
        atom_feature_dim=3,
        bond_feature_dim=2,
        num_heads=3,
        alphas=(0.5, 0.2),
        max_iter=2,
        dropout=0.0,
        spectral_norm=False,
    )
    molecular_model(**mol.model_kwargs(), return_results=True).output
    """
                ),
                md(
                    r"""
    Lyapunov-style and Jacobian diagnostics can be run on any transition. Here the
    spectral-radius estimate is applied to the first SILVA layer transition.
    """
                ),
                code(
                    """
    layer = SILVAGraphPresetNetwork(
        in_dim=5,
        hidden_dim=8,
        out_dim=2,
        stack_alphas=[0.5],
        num_heads=2,
        max_iter=3,
    ).layers[0]
    z0 = torch.zeros(x.shape[0], 8)
    z_star = layer(x, edge_index=edge_index)
    rho = damped_spectral_radius(lambda z: layer.f(z, x, edge_index=edge_index), z_star, alpha=0.5, iters=3)
    rho
    """
                ),
            ]
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    COLAB_OUT.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        nb_with_citation = {
            **nb,
            "cells": [*nb["cells"], md(CITATION_SOURCE)],
        }
        payload = json.dumps(nb_with_citation, indent=2) + "\n"
        for target in (OUT, DOCS_OUT, COLAB_OUT):
            (target / name).write_text(payload)
    for name in STATIC_MIRRORED_NOTEBOOKS:
        payload = (DOCS_OUT / name).read_text(encoding="utf-8")
        for target in (OUT, COLAB_OUT):
            (target / name).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
