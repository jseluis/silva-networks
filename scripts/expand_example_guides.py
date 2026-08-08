"""Add complete programs, measured outputs, and scale guidance to example pages."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from expand_notebook_curriculum import Curriculum, _curriculum_for

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/examples"
START = "<!-- silva-worked-example:start -->"
END = "<!-- silva-worked-example:end -->"


@dataclass(frozen=True)
class ExampleRoute:
    script: str
    curriculum_hint: str
    task_evidence: str
    full_scale_target: str


ROUTES: dict[str, ExampleRoute] = {
    "advanced-equilibria.md": ExampleRoute(
        "advanced_equilibria.py",
        "monotone_graph_poisson_physics_informed",
        "one result for each advanced equilibrium family, with family-specific residuals",
        "the selected graph, inverse-problem, transformer, ODE, or DAE benchmark",
    ),
    "citation-aware-reporting.md": ExampleRoute(
        "reproduction_registry.py",
        "dataset_reproduction_training",
        "a complete machine-readable source and verification record",
        "a source-conforming multi-seed study with archived configuration and receipts",
    ),
    "constrained-optimization.md": ExampleRoute(
        "constrained_optimization.py",
        "projected_qp_optimization",
        "simplex feasibility, energy, solver residual, and parameter gradients",
        "the complete constrained task at its original variable and constraint count",
    ),
    "cortex-hierarchy.md": ExampleRoute(
        "cortex_hierarchy.py",
        "cortex_multiscale_vision",
        "per-point state shapes, solver choices, logits, loss, and gradients",
        "a heterogeneous linked-point architecture on the target dataset",
    ),
    "custom-layers.md": ExampleRoute(
        "custom_layers.py",
        "fixed_point_custom_transition",
        "the custom state shape, final residual, and differentiable loss path",
        "the user-defined transition at the intended width and data scale",
    ),
    "datasets-quickstart.md": ExampleRoute(
        "datasets_quickstart.py",
        "dataset_training_quickstart",
        "dataset identity, tensor shape, and measured classification accuracy",
        "the official split with preprocessing fitted only on training data",
    ),
    "deq-engine-bridge.md": ExampleRoute(
        "deq_engine_bridge.py",
        "solver_implicit_gradient",
        "state shape, iterations, residual ratio, and gradient availability",
        "the chosen implicit model with forward and backward solver sweeps",
    ),
    "emerging-equilibria.md": ExampleRoute(
        "emerging_equilibria.py",
        "consistency_deq_psi_gnn_ifno_snarf_diffusion",
        "family-specific exact-solution, boundary, reconstruction, or trajectory checks",
        "the cited dataset and complete architecture for the selected family",
    ),
    "frontier-equilibria.md": ExampleRoute(
        "frontier_equilibria.py",
        "fno_pde_graph_homotopy_distributional",
        "task, equation, invariance, and fixed-point residuals for four operator classes",
        "the complete PDE, graph, homotopy, or measure benchmark",
    ),
    "full-cortex-operators.md": ExampleRoute(
        "full_cortex_operators.py",
        "cortex_architecture_catalog",
        "branch activations, solver history, state shape, loss, and gradients",
        "the complete multi-operator point architecture with task data",
    ),
    "full-scale-training.md": ExampleRoute(
        "add_layers_on_top.py",
        "training_scale_dataset",
        "a measured training loss from the complete optimization path",
        "the configured sharded or distributed training run with resume checks",
    ),
    "graph-silva.md": ExampleRoute(
        "graph_silva.py",
        "graph_message_passing",
        "node-state shape, task loss, equilibrium residual, and gradients",
        "the complete graph split with sparse operators and task metrics",
    ),
    "molecules.md": ExampleRoute(
        "molecules.py",
        "molecular_zinc_graph",
        "atom and molecule tensor shapes, residual, prediction loss, and gradients",
        "the official molecular split with the complete feature and metric protocol",
    ),
    "optical-flow-silva.md": ExampleRoute(
        "optical_flow_silva.py",
        "optical_flow_raft",
        "flow shape, endpoint error, iterations, residual, and gradients",
        "Sintel, KITTI Flow, or FlyingChairs with the source preprocessing protocol",
    ),
    "paper-family-cases.md": ExampleRoute(
        "paper_family_cases.py",
        "sequence_mdeq_graph_diffusion",
        "shape and residual checks across sequence, vision, graph, and diffusion cases",
        "the complete source task for one selected generalized family",
    ),
    "point-architecture-catalog.md": ExampleRoute(
        "point_architecture_catalog.py",
        "architecture_catalog_cortex_vision",
        "parameters, loss, residual trajectory, and gradient norm for every architecture",
        "the selected internal architecture at production width and resolution",
    ),
    "raft-deq-flow.md": ExampleRoute(
        "raft_deq_flow.py",
        "raft_flow",
        "flow shape, correction trajectory, solver residual, loss, and gradients",
        "the complete optical-flow protocol with correlation and learned upsampling",
    ),
    "reproduction-registry.md": ExampleRoute(
        "reproduction_registry.py",
        "dataset_reproduction_training",
        "verification levels, preserved mechanisms, scale tiers, and source obligations",
        "a complete source-conforming run with archived deviations and evidence",
    ),
    "scalar-deq.md": ExampleRoute(
        "scalar_deq.py",
        "solver_contraction_fixed_point",
        "closed-form agreement, final residual, iteration count, and implicit gradient",
        "a higher-dimensional transition with the same solver and gradient report",
    ),
    "scientific-operators.md": ExampleRoute(
        "scientific_operators.py",
        "ode_pde_neural_operator_fno",
        "ODE error plus PDE, boundary, and equilibrium residuals",
        "the target mesh, time horizon, forcing distribution, and physical metric suite",
    ),
    "source-data.md": ExampleRoute(
        "source_data_families.py",
        "dataset_graph_vision_flow",
        "losses, certificates, residuals, scale allocation, and cache activity on source data",
        "the official complete splits with every source receipt retained",
    ),
    "spatial-cortex.md": ExampleRoute(
        "spatial_cortex.py",
        "spatial_cortex_multiscale_vision",
        "spatial and vector state shapes, per-point solvers, loss, and gradients",
        "the complete image task with the intended spatial resolution and links",
    ),
    "stacked-architecture.md": ExampleRoute(
        "stacked_architecture.py",
        "stacked_cortex_architecture",
        "logit shape, pointwise solvers, loss, and gradient flow across the stack",
        "the complete stacked architecture with independently budgeted points",
    ),
    "structured-equilibria.md": ExampleRoute(
        "structured_equilibria.py",
        "monotone_positive_non_euclidean_graph_delta",
        "certificates, positivity, one-sided bounds, scale weights, and cache activity",
        "the source benchmark associated with the selected structured family",
    ),
    "vision-channels.md": ExampleRoute(
        "vision_channels.py",
        "vision_multiscale_cortex",
        "image-state shape, iteration count, residual, loss, and gradients",
        "the full image split with production channels and resolution",
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


def _full_program(script: str, existing: str) -> str:
    snippet = f'--8<-- "examples/{script}"'
    if snippet in existing:
        return ""
    return f"""
### Complete Program

The complete executable source is included here so the example can be studied
without reconstructing omitted setup, solver, loss, or gradient steps.

```python
{snippet}
```
"""


def _worked_block(
    page: Path,
    route: ExampleRoute,
    curriculum: Curriculum,
    output: str,
    existing: str,
) -> str:
    program = _full_program(route.script, existing)
    return rf"""
{START}
## Complete Worked Study

The short construction above identifies the main API. A complete study must
also distinguish the state equation, task objective, numerical residual,
gradient path, and scale transfer. In this example, the equilibrium state is
**{curriculum.state}**, the condition is **{curriculum.condition}**, and the
repeated map is **{curriculum.repeated}**.

### Derivation From Transition to Reported Result

The forward solve is defined by

$$
z^\star = T_\theta(z^\star,x).
$$

The task output and task objective are separate from convergence:

$$
\widehat y = R_\phi(z^\star),
\qquad
\mathcal L_{{\mathrm{{task}}}}=\ell(\widehat y,y).
$$

For a computed state $z_K$, the normalized fixed-point residual is

$$
r_K =
\frac{{\lVert T_\theta(z_K,x)-z_K\rVert_2}}
{{\lVert z_K\rVert_2+\varepsilon}}.
$$

A small task loss does not imply a small $r_K$, and a small $r_K$ does not
establish task quality. Both belong in the result. For implicit training, the
parameter sensitivity follows

$$
\frac{{\mathrm d z^\star}}{{\mathrm d\theta}}
=
\left(I-\partial_z T_\theta(z^\star,x)\right)^{{-1}}
\partial_\theta T_\theta(z^\star,x).
$$

This is why the example checks gradients in addition to forward convergence.
The reader-facing evidence for this route is **{route.task_evidence}**. The
invariants that must remain true are **{curriculum.invariants}**.

{program}
### Run the Complete Example

```bash
python examples/{route.script}
```

### Measured Compact Output

The following output was produced by the executable program in the current
repository. Floating-point values may vary slightly across devices and library
builds, while shapes, finite values, invariants, and declared tolerances must
remain stable.

```text
{output}
```

### Interpret the Output

| Evidence | What it answers | What would require investigation |
| --- | --- | --- |
| Tensor shapes | Did every source, state, branch, and readout preserve its declared contract? | A changed entity, channel, token, or spatial dimension |
| Task metric | Did the compact task execute and produce finite evidence? | Non-finite loss, a missing mask, or a metric computed on the wrong split |
| Fixed-point residual | Did the returned state satisfy the repeated transition to the requested tolerance? | A residual plateau, rising trajectory, or convergence flag inconsistent with the value |
| Iteration or trajectory data | How much numerical work was required? | Solver effort that grows sharply under a small input or resolution change |
| Gradient evidence | Can the loss reach every trainable component through the selected backward mode? | Missing, non-finite, or implausibly large gradients |
| Domain invariant | Did the method retain positivity, feasibility, boundary values, permutation behavior, or another structural requirement? | A task metric that looks acceptable while the structural contract fails |

The compact output is a mechanism check, not a paper-scale benchmark claim. It
shows that data enter the intended construction, the transition executes, the
solver returns diagnostics, and differentiation reaches trainable parameters.

### Add a Solver and Scale Sweep

The next run should hold model parameters and data fixed while changing one
numerical control at a time. A complete experiment record can use this schema:

```yaml
experiment:
  example: {page.stem}
  state: {curriculum.state}
  condition: {curriculum.condition}
  repeated_transition: {curriculum.repeated}
  invariant_checks: {curriculum.invariants}
  compact_evidence: {route.task_evidence}
  scale_axes: {curriculum.scale_axis}
solver_sweep:
  methods: [picard, anderson, broyden]
  tolerances: [1.0e-4, 1.0e-6, 1.0e-8]
  maximum_iterations: [25, 50, 100]
report:
  - task_metric
  - fixed_point_residual
  - backward_linear_residual
  - iterations
  - wall_time
  - peak_memory
  - gradient_norm
```

At full scale, move toward **{route.full_scale_target}**. Increase only one of
**{curriculum.scale_axis}** at a time. Retain this compact run as a regression
test, preserve the source split and preprocessing receipt, archive the resolved
configuration and checkpoint, and report convergence failures rather than
discarding them.
{END}
"""


def _replace_generated(text: str, block: str) -> str:
    pattern = re.compile(
        rf"\n?{re.escape(START)}.*?{re.escape(END)}\n?",
        re.DOTALL,
    )
    text = pattern.sub("\n", text)
    marker = "\n## Where to Go Next\n"
    if marker not in text:
        raise ValueError("example page is missing its Where to Go Next section")
    return text.replace(marker, "\n" + block.strip() + "\n" + marker, 1)


def _expand_index() -> None:
    path = DOCS / "index.md"
    text = path.read_text(encoding="utf-8")
    rows = "\n".join(
        f"| [{page.removesuffix('.md').replace('-', ' ').title()}]({page}) | "
        f"`examples/{route.script}` | {route.task_evidence.capitalize()} |"
        for page, route in ROUTES.items()
    )
    block = rf"""
{START}
## Executable Evidence Map

Every worked page now retains its introductory route and adds the complete
program, measured compact output, mathematical result contract, interpretation,
and full-scale transfer record.

| Worked page | Executable program | Compact evidence |
| --- | --- | --- |
{rows}

Across the collection, a reported result is treated as the tuple

$$
\mathcal E =
(\text{{task metric}},\text{{fixed-point residual}},\text{{iterations}},
\text{{gradient evidence}},\text{{domain invariants}}).
$$

Keeping these entries separate prevents task quality, solver convergence, and
structural validity from being collapsed into one number.
{END}
"""
    path.write_text(_replace_generated(text, block), encoding="utf-8")


def main() -> int:
    missing = sorted({path.name for path in DOCS.glob("*.md")} - set(ROUTES) - {"index.md"})
    if missing:
        raise RuntimeError(f"example pages are missing expansion routes: {missing}")
    for name, route in ROUTES.items():
        page = DOCS / name
        text = page.read_text(encoding="utf-8")
        curriculum = _curriculum_for(Path(route.curriculum_hint))
        output = _capture(route.script)
        block = _worked_block(page, route, curriculum, output, text)
        page.write_text(_replace_generated(text, block), encoding="utf-8")
    _expand_index()
    print(f"expanded {len(ROUTES)} worked example pages and the example index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
