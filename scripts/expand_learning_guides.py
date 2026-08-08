"""Add complete executable evidence bridges to compact learning pages."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- silva-learning-study:start -->"
END = "<!-- silva-learning-study:end -->"


@dataclass(frozen=True)
class LearningGuide:
    state: str
    condition: str
    equation: str
    script: str
    evidence: str
    scale: str


GUIDES = {
    "fixed-points.md": LearningGuide(
        "one scalar equilibrium state",
        "a constant injected source",
        r"z^\star=az^\star+b,\qquad z^\star=\frac{b}{1-a}",
        "scalar_deq.py",
        "closed-form agreement, a zero final residual, and the local Jacobian",
        "replace the scalar coefficient with a matrix or nonlinear transition and sweep its spectral radius",
    ),
    "custom-layers.md": LearningGuide(
        "a user-selected latent tensor",
        "the adapted input and any graph, spatial, or physical context",
        r"z^\star=T_\theta(z^\star;c),\qquad \operatorname{shape}(T_\theta(z;c))=\operatorname{shape}(z)",
        "custom_layers.py",
        "the custom state shape and the transition's measured equilibrium residual",
        "replace one internal module at a time, then increase state width and source-data size",
    ),
    "interactive-diagnostics-lab.md": LearningGuide(
        "one state vector per graph node",
        "features, edges, and graph context",
        r"r_K=\frac{\|T_\theta(z_K;c)-z_K\|_2}{\|z_K\|_2+\varepsilon},\qquad \rho=\rho(J_T(z_K;c))",
        "graph_silva.py",
        "task loss, normalized solver evidence, and a spectral-radius estimate",
        "hold model and graph fixed while sweeping solver, damping, tolerance, and maximum iterations",
    ),
    "advanced-equilibrium-datasets.md": LearningGuide(
        "the family-specific graph, token, inverse, trajectory, or algebraic state",
        "a generated batch that retains the equation coefficients and constraints",
        r"r_{\mathrm{data}}=\|\mathcal A(x,y,c)\|_2,\qquad r_{\mathrm{model}}=\|T_\theta(z;c)-z\|_2",
        "advanced_equilibria.py",
        "separate compact outputs for monotone, generative, inverse, physics-informed, DAE, and residual mechanisms",
        "substitute an official dataset adapter with the same named fields and preserve the original split and metric",
    ),
    "stacking-and-devices.md": LearningGuide(
        "three linked equilibrium states with independent solver policies",
        "the output of the preceding point plus the original task input",
        r"z_i^\star=T_{\theta_i}(z_i^\star;c_i),\qquad c_{i+1}=A_i(z_i^\star,x)",
        "stacked_architecture.py",
        "the final logit shape, all three selected solvers, and a differentiable task loss",
        "profile each point separately before increasing point count, state width, batch size, or device count",
    ),
}


def _capture(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}:\n{output}")
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def _block(page: Path, guide: LearningGuide, output: str) -> str:
    return f"""
{START}
## Worked Evidence Bridge

The derivation above becomes a complete SILVA study when the state, condition,
solver result, task result, and gradient path are kept separate. Here the state
is **{guide.state}** and the condition is **{guide.condition}**. The compact
relation is

$$
{guide.equation}
$$

The following is the complete executable program used by the repository tests:

```python
--8<-- "examples/{guide.script}"
```

Run it from the project root:

```bash
python examples/{guide.script}
```

### Measured Output

```text
{output}
```

### What This Result Establishes

This run records {guide.evidence}. It establishes that the compact mechanism is
executable with finite outputs and that its stated shape or structural contract
can be inspected. It does not establish source-scale accuracy by itself.

For the next controlled study, {guide.scale}. Keep the compact run as a
regression case. For every larger run, archive the resolved data source and
split, preprocessing, seed, constructor arguments, forward and backward solver
settings, task metric, normalized residual, iteration count, gradient norm,
runtime, peak memory, and convergence failures. This keeps task quality,
numerical convergence, and computational cost from being collapsed into one
number.

{END}
"""


def _expand(path: Path, guide: LearningGuide) -> None:
    source = path.read_text(encoding="utf-8")
    if START in source:
        prefix, remainder = source.split(START, 1)
        _, suffix = remainder.split(END, 1)
        source = prefix.rstrip() + "\n\n" + suffix.lstrip()
    marker = "## Where to Go Next"
    if marker not in source:
        raise RuntimeError(f"missing next-step section in {path}")
    output = _capture(guide.script)
    source = source.replace(marker, _block(path, guide, output).strip() + "\n\n" + marker, 1)
    path.write_text(source, encoding="utf-8")


def main() -> int:
    for name, guide in GUIDES.items():
        _expand(ROOT / "docs/learn" / name, guide)
    print(f"expanded {len(GUIDES)} compact learning guides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
