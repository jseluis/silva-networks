"""Add executable, result-bearing studies to compact API reference pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- silva-api-study:start -->"
END = "<!-- silva-api-study:end -->"


@dataclass(frozen=True)
class Guide:
    subject: str
    equation: str
    state: str
    condition: str
    diagnostic: str
    replacement: str
    scale_axes: str
    citations: str
    example: str
    output: str
    interpretation: str


ADVANCED_OUTPUT = """monotone graph: (8, 1) 0.023554455488920212
equilibrium transformer: 0.18536624312400818
Poisson mirror: 0.005979819223284721
physics-informed loss: 0.8003759384155273
implicit DAE step: [0.4761904776096344] 1.862645149230957e-09
adversarial residual objective: 0.7888258695602417 1.3886094093322754"""
EMERGING_OUTPUT = """consistency: {'shape': (4, 3), 'teacher_error': 5.960464477539063e-08}
psi_gnn: {'shape': (25, 1), 'boundary_error': 0.0}
ifno: {'shape': (2, 1, 4, 8), 'final_increment': 3.803837776184082}
snarf: {'shape': (7, 1), 'root_residual': 8.068445911391109e-10}
mesh: {'shape': (5, 2), 'centralized_error': 1.8730469264482963e-07}
physics_diffusion: {'shape': (1, 1, 8, 8), 'final_energy': 40.260433197021484}
therino: {'shape': (2, 3, 6, 6), 'strain_error': 1.4901161193847656e-08}
fixed_point_diffusion: {'shape': (2, 1, 6, 6), 'reverse_steps': 3}"""
FRONTIER_OUTPUT = """SILVA Fourier equilibrium: {'shape': (1, 1, 8, 8), 'residual': 1.6093609644940443e-07, 'dataset_equation_residual': 5.960464477539062e-07}
SILVA physics graph equilibrium: {'shape': (6, 1), 'residual': 5.127419058226224e-07, 'dataset_equation_residual': 5.960464477539063e-08}
SILVA homotopy equilibrium: {'shape': (2, 1), 'terminal_residual': 0.00807332992553711, 'analytic_error': 0.01614689826965332}
SILVA distributional equilibrium: {'shape': (1, 5, 4), 'initial_discrepancy': 0.48991382122039795, 'final_discrepancy': 0.453036904335022}"""
STRUCTURED_OUTPUT = """monotone operator torch.Size([8, 2]) certificate 0.5005146265029907
positive concave torch.Size([8, 1]) minimum state 0.042785972356796265
non-Euclidean torch.Size([8, 2]) one-sided bound 0.04999999701976776
efficient infinite graph torch.Size([12, 1]) spectral margin 0.44062745571136475
multiscale graph torch.Size([12, 1]) attention sums tensor([1.0000, 1.0000, 1.0000])
delta equilibrium torch.Size([8, 1]) mean active fraction 0.2036637931034483 exact residual 0.0014585574390366673"""
SCALE_OUTPUT = """family fno_deq
public objects 12
verification compact-verified
benchmark tasks 2
solver anderson
max iterations 12
runtime auto none
loader 4 0"""


GUIDES = {
    "advanced_data.md": Guide(
        "advanced-family data contracts",
        r"\mathcal D=\{(x_i,y_i,c_i)\}_{i=1}^N,\qquad y_i=\mathcal G(x_i;c_i)",
        "batch fields consumed by the monotone, generative, inverse, ODE, and DAE families",
        "every generated target must satisfy the same discrete equation used by its verifier",
        "target-equation residual and tensor shape",
        "the analytic generator, boundary sampler, noise model, or public-data adapter",
        "sample count, graph or grid resolution, trajectory length, and noise level",
        "[[47]](../paper/references.md#ref-47) through [[52]](../paper/references.md#ref-52)",
        "advanced_equilibria.py",
        ADVANCED_OUTPUT,
        "The DAE residual is near machine precision, while the other values are task losses or fixed-point diagnostics with different units. They must be compared only to the matching equation and tolerance.",
    ),
    "advanced_equilibria.md": Guide(
        "advanced equilibrium transitions",
        r"z^\star=\Phi\!\left(S_\theta(x)+H_\theta(z^\star)+L_\theta(z^\star)+G_\theta(z^\star)\right)",
        "the converged graph, token, image, physical, or algebraic state",
        "the transition must preserve the declared state shape, device, and floating dtype",
        "forward residual, task loss, and backward linear-solve residual",
        "the stimulus, internal transition, interaction operator, solver, or readout",
        "state width, token or node count, grid size, solver tolerance, and maximum iterations",
        "[[47]](../paper/references.md#ref-47) through [[52]](../paper/references.md#ref-52)",
        "advanced_equilibria.py",
        ADVANCED_OUTPUT,
        "The output demonstrates six distinct mechanisms through one package surface. The small values verify equations or compact objectives; they are not source-scale benchmark scores.",
    ),
    "emerging_data.md": Guide(
        "emerging-family data generators",
        r"r_{\mathrm{data}}=\|\mathcal A(x,y,c)\|,\qquad r_{\mathrm{data}}\rightarrow 0",
        "boundary graphs, coefficient fields, canonical points, mesh observations, and diffusion trajectories",
        "each batch keeps the coordinates, masks, conditions, and targets required to recompute its governing residual",
        "boundary error, constitutive error, root residual, or energy",
        "the compact generator with a source-dataset adapter that returns the same named fields",
        "mesh density, spatial resolution, number of poses, diffusion steps, and stored trajectory count",
        "[[59]](../paper/references.md#ref-59) through [[64]](../paper/references.md#ref-64), [[73]](../paper/references.md#ref-73), and [[74]](../paper/references.md#ref-74)",
        "emerging_equilibria.py",
        EMERGING_OUTPUT,
        "The exact checks expose different invariants: zero boundary error, a small deformation root residual, a small strain error, and the declared reverse-step count. The diffusion energy is an objective value and is not expected to be zero after this compact run.",
    ),
    "emerging_equilibria.md": Guide(
        "emerging equilibrium mechanisms",
        r"F_\theta(z;c)=T_\theta(z;c)-z=0",
        "a family-specific implicit state with an explicit condition bundle",
        "calling the transition again at the returned state must reproduce that state within the solver tolerance",
        "family invariant plus normalized fixed-point residual",
        "every default backbone, processor, increment, deformation, energy, constitutive map, or denoiser",
        "state dimension, discretization size, trajectory depth, solver policy, and checkpoint schedule",
        "[[59]](../paper/references.md#ref-59) through [[64]](../paper/references.md#ref-64), [[73]](../paper/references.md#ref-73), and [[74]](../paper/references.md#ref-74)",
        "emerging_equilibria.py",
        EMERGING_OUTPUT,
        "The program validates all eight mechanisms independently. Each reported quantity has a family-specific meaning, so a scale study must retain both the shared fixed-point residual and the named physical or structural check.",
    ),
    "frontier_data.md": Guide(
        "operator, graph, homotopy, and measure data",
        r"r_{\mathrm{PDE}}(u;a,f)=-\nabla\!\cdot(a\nabla u)-f",
        "regular-grid fields, graph states, continuation pairs, or variable-cardinality samples",
        "the batch must retain enough source information to recompute its equation or discrepancy",
        "equation residual, analytic continuation error, or measure discrepancy",
        "the exact compact generator with an official split and preprocessing adapter",
        "resolution, graph size, continuation steps, particle count, and batch size",
        "[[31]](../paper/references.md#ref-31) and [[43]](../paper/references.md#ref-43) through [[46]](../paper/references.md#ref-46)",
        "frontier_equilibria.py",
        FRONTIER_OUTPUT,
        "The Fourier and graph batches satisfy their generating equations to about single-precision tolerance. The homotopy and distributional rows report finite-discretization behavior and therefore require a step or particle-count sweep.",
    ),
    "structured_data.md": Guide(
        "structured equilibrium data contracts",
        r"z^\star=\sigma(Az^\star+Bx),\qquad \|A\|\ \text{or a structural certificate controls the update}",
        "positive vectors, transformed coordinates, graph signals, multiscale states, or cached deltas",
        "the generated sample carries the operator or perturbation needed to verify its certificate",
        "certificate margin, one-sided perturbation, attention normalization, or cache error",
        "the synthetic source with an attributed image, graph, or geometric adapter",
        "sample count, node count, feature width, number of graph scales, and perturbation radius",
        "[[75]](../paper/references.md#ref-75) through [[80]](../paper/references.md#ref-80)",
        "structured_equilibria.py",
        STRUCTURED_OUTPUT,
        "The compact run checks one defining property per family. Positivity, spectral margin, normalized attention, and delta activity are different contracts and should remain separate columns in a larger report.",
    ),
    "structured_equilibria.md": Guide(
        "structured equilibrium operators",
        r"F_\theta(z;x)=z-\mathcal T_\theta(z;x)=0,\qquad C_\theta(z,x)\geq 0",
        "the equilibrium state together with the family certificate or operator statistics",
        "the returned certificate must be recomputable from public state, operator, and configuration fields",
        "exact residual and the named structural certificate",
        "the dense or factorized operator, activation, graph spectrum, scale mixer, or delta policy",
        "operator rank, state width, graph scale, solver tolerance, and cache threshold",
        "[[75]](../paper/references.md#ref-75) through [[80]](../paper/references.md#ref-80)",
        "structured_equilibria.py",
        STRUCTURED_OUTPUT,
        "All six outputs retain their own certificate. This prevents a low task loss from hiding a failed positivity, monotonicity, spectral, multiscale, or cache contract.",
    ),
    "physics_informed.md": Guide(
        "physics-informed and algebraic equilibrium layers",
        r"\mathcal L=\lambda_{\mathrm{data}}\mathcal L_{\mathrm{data}}+\lambda_{\mathrm{phys}}\|\partial_t\hat y-\mathcal N(t,\hat y)\|_2^2+\lambda_J\mathcal R_J",
        "the implicit representation and the decoded physical state",
        "the physical residual callable is evaluated on the decoded state with declared boundary or algebraic constraints",
        "physics residual, boundary residual, equilibrium residual, and adjoint residual",
        "the dynamics, residual, boundary projector, regularizer, stage equation, or discriminator",
        "collocation points, temporal horizon, stiffness, stage count, tolerance, and precision",
        "[[50]](../paper/references.md#ref-50) through [[52]](../paper/references.md#ref-52)",
        "advanced_equilibria.py",
        ADVANCED_OUTPUT,
        "The DAE row verifies its stage equation directly, while the physics-informed row is a weighted training objective. A full study must print the individual objective components rather than only their sum.",
    ),
    "devices.md": Guide(
        "device and dtype propagation",
        r"T_\theta:\mathbb R^{B\times N\times D}_{(d,q)}\rightarrow\mathbb R^{B\times N\times D}_{(d,q)}",
        "the same state layout on the selected device and floating dtype",
        "inputs, parameters, temporary tensors, solver history, and outputs must agree on device and dtype",
        "shape, finite values, gradient availability, and residual",
        "the automatic device selection with an explicit device passed by the experiment runner",
        "batch size, precision, device count, and data-loader workers",
        "the SILVA construction [[1]](../paper/references.md#ref-1) and implicit-layer foundation [[4]](../paper/references.md#ref-4)",
        "graph_silva.py",
        """state_shape (8, 12)
loss 0.7801069021224976
residual 0.07725001126527786
spectral_radius 0.7778381109237671""",
        "The printed shape confirms the graph state contract, and the finite loss, residual, and spectral-radius estimate are computed on the same selected device. Device equivalence still requires a separate CPU/accelerator comparison with fixed seeds.",
    ),
    "extensibility.md": Guide(
        "custom transition validation",
        r"T_\theta(z,c)\in\mathbb R^{B\times\cdots\times D}=\operatorname{shape}(z)",
        "a caller-declared tensor state and condition bundle",
        "one transition call preserves shape, device, dtype, finiteness, and a usable derivative path",
        "transition report followed by equilibrium residual and task gradient",
        "the initializer, transition, readout, or complete conditioned equilibrium module",
        "state shape, parameter count, solver, tolerance, backward mode, and condition size",
        "the SILVA construction [[1]](../paper/references.md#ref-1) and implicit-layer foundation [[4]](../paper/references.md#ref-4)",
        "reproduction_registry.py",
        """silva_fno_deq paper-adaptation compact-verified
silva_monotone_graph_equilibrium paper-adaptation compact-verified
silva_physics_informed_equilibrium paper-adaptation compact-verified
diffusion_equilibrium paper-adaptation compact-verified
transition report SILVATransitionReport(state_shape=(5, 4), output_shape=(5, 4), preserves_shape=True, preserves_device=True, preserves_dtype=True, finite=True, differentiable=True, parameter_count=28)
equilibrium residual 1.095007249318769e-07""",
        "The transition report verifies the mechanical contract before a solver is involved. The subsequent residual then verifies the numerical fixed point, keeping module validity and solver convergence as separate checks.",
    ),
}


SCALE_GUIDE = Guide(
    "coverage, reproduction, data, and scale configuration",
    r"F_\theta(z;x)=0,\qquad \widehat F_{\theta,s}(z;x)=0\ \text{uses the same mathematical contract at scale tier }s",
    "the selected family, constructor contract, runtime tier, and data-loader configuration",
    "changing a runtime tier may change numerical budgets and resource use but must not silently change the family equation",
    "coverage record, verification level, solver settings, effective batch size, and source-scale metrics",
    "compact defaults with family-specific modules, official data adapters, and an archived experiment configuration",
    "solver iterations, tolerance, model width, batch size, precision, workers, process count, and checkpoint interval",
    "the SILVA construction [[1]](../paper/references.md#ref-1) and the selected family's primary references",
    "api_scale_workflow.py",
    SCALE_OUTPUT,
    "The family resolves through four independent registries: public coverage, source relation, scale guidance, and runtime/data configuration. The compact-verified label describes repository evidence; it does not convert the two listed benchmark tasks into claimed benchmark results.",
)

for name in (
    "coverage.md",
    "reproducibility.md",
    "scale_cli.md",
    "scaling.md",
    "scaling_data.md",
):
    GUIDES[name] = SCALE_GUIDE


def _block(guide: Guide) -> str:
    command = f"python examples/{guide.example}"
    return f"""
{START}
## Operational Contract

This API surface connects {guide.subject} to the same SILVA experiment
contract used by the learning pages and notebooks. Its central relation is

$$
{guide.equation}
$$

| Part | What must remain inspectable |
| --- | --- |
| State | {guide.state}. |
| Condition | {guide.condition}. |
| Diagnostic | {guide.diagnostic}. |
| Replacement point | {guide.replacement}. |
| Scale axes | {guide.scale_axes}. |

The relevant method lineage is recorded in {guide.citations}. Those references
define the source mechanisms; this API exposes them through SILVA objects so a
reader can inspect, replace, solve, differentiate, and scale the construction.

## Complete Compact Study

Run the complete repository program below from the project root. The page uses
the same file that is exercised by the test suite, so the displayed call is not
an isolated fragment.

```python
--8<-- "examples/{guide.example}"
```

```bash
{command}
```

### Measured Compact Output

```text
{guide.output}
```

### Interpret the Output

{guide.interpretation}

For a controlled experiment, retain the compact call as a regression case and
change one scale axis at a time. Record the resolved constructor, data source
and split, preprocessing, seed, forward and backward solver settings, task
metric, normalized residual, iteration count, runtime, peak memory, and any
failed convergence case. A larger run becomes evidence only when its own
resolved configuration and outputs are archived; the compact output above is
evidence for the executable mechanism and its stated invariants.

{END}
"""


def _expand(path: Path, guide: Guide) -> None:
    source = path.read_text(encoding="utf-8")
    if START in source:
        prefix, remainder = source.split(START, 1)
        _, suffix = remainder.split(END, 1)
        source = prefix.rstrip() + "\n\n" + suffix.lstrip()
    marker = "::: silva_networks"
    if marker not in source:
        raise RuntimeError(f"missing API reference marker in {path}")
    source = source.replace(marker, _block(guide).strip() + "\n\n" + marker, 1)
    path.write_text(source, encoding="utf-8")


def main() -> int:
    for name, guide in GUIDES.items():
        _expand(ROOT / "docs/api" / name, guide)
    print(f"expanded {len(GUIDES)} compact API guides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
