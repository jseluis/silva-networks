from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "notebooks/package_api", ROOT / "docs/package-notebooks", ROOT / "colab"]
NAME = "10_training_helpers_smoke.ipynb"
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")
_MATPLOTLIB_IMPORT_RE = re.compile(r"^(?P<indent>[ \t]*)import matplotlib\.pyplot as plt$", re.MULTILINE)


def _next_cell_id() -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"train-{_CELL_COUNTER:04d}"


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
# Training Helpers Validation Tutorial

This notebook checks the optional supervised training helpers:

$$
\text{model},\ \text{loader},\ \text{loss},\ \text{optimizer}
\quad\longrightarrow\quad
\text{history},\ \text{metric},\ \text{checkpoint}.
$$

The helpers do not define a SILVA architecture by themselves. They provide the
repeatable training loop around any PyTorch model, including models built from
`silva_networks`.
"""
        ),
        code(BOOTSTRAP),
        code(
            """
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from silva_networks import (
    TrainConfig,
    evaluate,
    fit_supervised,
    resolve_device,
    seed_everything,
)

device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(10)
device
"""
        ),
        md(
            r"""
## Synthetic Classification Data

Use a tiny linearly separable dataset. The batch contract here is the ordinary
PyTorch `(x, y)` tuple:

$$
x\in\mathbb R^{N\times d},
\qquad
y\in\{0,1\}^N.
$$
"""
        ),
        code(
            """
x_pos = torch.randn(24, 3) + torch.tensor([1.5, 0.0, 0.0])
x_neg = torch.randn(24, 3) + torch.tensor([-1.5, 0.0, 0.0])
x = torch.cat([x_pos, x_neg], dim=0)
y = torch.cat([torch.ones(24, dtype=torch.long), torch.zeros(24, dtype=torch.long)])

loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)
val_loader = DataLoader(TensorDataset(x, y), batch_size=16)
model = torch.nn.Sequential(
    torch.nn.Linear(3, 12),
    torch.nn.Tanh(),
    torch.nn.Linear(12, 2),
)
"""
        ),
        md(
            r"""
## Fit and Evaluate

For classification, `loss="auto"` becomes cross entropy and `metric="auto"`
becomes accuracy. The training helper moves the model and batches to the
requested device.
"""
        ),
        code(
            """
with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "training.pt"
    result = fit_supervised(
        model,
        loader,
        val_loader,
        config=TrainConfig(
            task="classification",
            epochs=3,
            lr=0.05,
            optimizer="adam",
            gradient_clipping=1.0,
            device=device,
            seed=10,
            checkpoint_path=checkpoint,
        ),
    )
    evaluation = evaluate(model, val_loader, device=device)
    print("epochs:", len(result.history))
    print("best epoch:", result.best_epoch)
    print("metric:", evaluation.metric_name, round(evaluation.metric, 4))
    print("checkpoint exists:", checkpoint.exists())
"""
        ),
        md(
            r"""
## Resume

When `resume=True`, the helper reloads the model, optimizer, scheduler if
present, and history from the checkpoint path.
"""
        ),
        code(
            """
with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "resume.pt"
    _ = fit_supervised(
        model,
        loader,
        config=TrainConfig(epochs=1, lr=0.01, device=device, checkpoint_path=checkpoint),
    )
    resumed = fit_supervised(
        model,
        loader,
        config=TrainConfig(epochs=2, lr=0.01, device=device, checkpoint_path=checkpoint, resume=True),
    )
    print("history after resume:", len(resumed.history))
"""
        ),
        md(
            r"""
## Citation

If this notebook or package is used, cite:

```text
Dr. Jose Luis Silva. SILVA Networks. Version 1.0.0. MIT License.
https://github.com/jseluis/silva-networks
https://doi.org/10.5281/zenodo.21770099
```

When training SILVA models in connection with the SILVA Networks paper, cite
the paper as well.
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
