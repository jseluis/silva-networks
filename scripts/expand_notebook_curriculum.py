"""Add extension, equivalence, and reproduction depth to every SILVA notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "silva-extension-curriculum"
READER_TEXT_REPLACEMENTS = {
    "Upstream references to compare with:": "Primary implementation and tutorial references:",
    "- Recent-paper cache: references/papers/recent_deq/": (
        "- Numbered research references: "
        "https://jseluis.github.io/silva-networks/paper/references/"
    ),
    "- `references/papers/recent_deq/`": (
        "- Numbered research references: "
        "https://jseluis.github.io/silva-networks/paper/references/"
    ),
}
CODE_REPLACEMENTS = {
    "    picard,\n    anderson,": "    SolverConfig,\n    picard,\n    anderson,",
    "trace = picard(f_torch, z0, alpha=0.7, max_iter=80, tol=1e-10)": (
        "trace = picard(\n"
        "    f_torch,\n"
        "    z0,\n"
        "    SolverConfig(alpha=0.7, max_iter=80, tol=1e-10),\n"
        ")"
    ),
}


@dataclass(frozen=True)
class Curriculum:
    state: str
    condition: str
    repeated: str
    invariants: str
    compact_metric: str
    scale_axis: str
    components: str


GENERIC = Curriculum(
    state="the tensor solved to equilibrium",
    condition="the observed input or source tensor",
    repeated="the state-preserving transition evaluated by the root solver",
    invariants="shape, device, dtype, finiteness, and differentiability",
    compact_metric="fixed-point residual and task error against a deterministic target",
    scale_axis="state width, batch size, and data volume",
    components="initializer, source encoder, transition, readout, and solver",
)

RULES: tuple[tuple[tuple[str, ...], Curriculum], ...] = (
    (("solver", "contraction", "fixed_point", "original_deq", "diagnostic"), Curriculum(
        "the latent vector or tensor z",
        "the injected observation x",
        "the tied map f_theta(z, x)",
        "state shape and a decreasing or bounded residual",
        "distance to an analytic fixed point and final relative residual",
        "latent width, solver tolerance, and iteration budget",
        "transition, damping, stopping rule, backward solver, and readout",
    )),
    (("jacobian", "implicit_gradient", "implicit_autodiff", "recent_theory"), Curriculum(
        "the converged latent state z_star",
        "differentiable parameters and external inputs",
        "the map whose Jacobian defines I - J_z f",
        "agreement of dense, JVP, VJP, and matrix-free products",
        "adjoint residual and gradient error against explicit differentiation",
        "state dimension and matrix-free linear-solver iterations",
        "transition, Jacobian product, linear solver, regularizer, and loss",
    )),
    (("ode", "homotopy"), Curriculum(
        "the evolving or terminal physical state",
        "time, initial condition, and external forcing",
        "an explicit flow step or residual field T(z, x) - z",
        "time-step shape, initial condition, and integration consistency",
        "trajectory error, terminal fixed-point residual, and conservation error",
        "time horizon, step count, state dimension, and stiffness",
        "vector field, integrator, equilibrium transition, readout, and tolerances",
    )),
    (("pde", "neural_operator", "fno", "scientific"), Curriculum(
        "a sampled solution field on a grid or mesh",
        "coefficient, forcing, boundary, and coordinate fields",
        "a tied local/spectral/operator field with source reinjection",
        "spatial shape, boundary conditions, and resolution semantics",
        "solution error, PDE residual, boundary error, and fixed-point residual",
        "resolution, retained modes, channels, domain size, and dataset size",
        "lifting map, spectral/local operator, physics field, readout, and solver",
    )),
    (("message_passing", "graph", "node_benchmark", "molecular", "zinc"), Curriculum(
        "one latent vector per node or entity",
        "node features, edges, edge attributes, and graph batches",
        "a source-injected graph message or monotone graph transition",
        "node relabeling equivariance, graph boundaries, and state shape",
        "node/graph error, physical graph residual, and fixed-point residual",
        "node count, edge count, feature width, and number of graphs",
        "input projection, message field, global field, transition, pooling, and head",
    )),
    (("mdeq", "vision", "multiscale", "cortex", "architecture_catalog"), Curriculum(
        "one image tensor per resolution or linked SILVA point",
        "image features and per-scale source injections",
        "shape-preserving convolutional, U-Net, attention, or multiscale fusion blocks",
        "channel/spatial shape at every scale and deterministic fusion",
        "task error, per-scale residuals, and gradient agreement",
        "image resolution, channels, scales, internal depth, and batch size",
        "stem, per-scale injections, transition blocks, links, task head, and solvers",
    )),
    (("sequence", "language", "music"), Curriculum(
        "one latent vector per sequence position",
        "tokens, embeddings, masks, and optional memory",
        "a tied recurrent, convolutional, or attention transition",
        "sequence length, causal masking, padding, and state width",
        "token loss, perplexity or sequence metric, and fixed-point residual",
        "vocabulary, context length, hidden width, heads, and memory length",
        "embedding, positional source, transition, memory, readout, and solver",
    )),
    (("flow", "raft"), Curriculum(
        "the flow field, optionally coupled to a recurrent hidden state",
        "image features, correlation volumes, context, and initial flow",
        "the tied correlation-conditioned refinement update",
        "flow shape, coordinate convention, image resolution, and warping domain",
        "endpoint error, warp error, correction loss, and fixed-point residual",
        "image resolution, correlation radius/levels, hidden width, and solver budget",
        "feature/context encoders, correlation, update block, transition, upsampler, and solver",
    )),
    (("distribution", "measure"), Curriculum(
        "a masked set of latent particles",
        "an empirical input measure and validity mask",
        "a permutation-compatible particle transition and discrepancy descent",
        "permutation equivariance, masks, variable cardinality, and finite particles",
        "measure discrepancy, moment error, task error, and descent residual",
        "particle count, latent width, pair chunk size, and batch cardinality",
        "particle initializer, attention transition, discrepancy, descent rule, and readout",
    )),
    (("diffusion", "generative", "transformer"), Curriculum(
        "a joint trajectory or equilibrium token/image state",
        "noise, timestep, condition, one-time injection, or teacher target",
        "a tied denoising or injected-attention transition",
        "trajectory ordering, token/image shape, conditioning, and deterministic noise",
        "teacher error, reconstruction/generation metric, and equilibrium residual",
        "image size, token count, hidden width, heads, and sampling schedule",
        "patch/noise source, injection blocks, transition, decoder, schedule, and solver",
    )),
    (("optimization", "projected_qp", "differentiable_optimization"), Curriculum(
        "the primal variable and any dual or auxiliary state",
        "objective coefficients and constraints",
        "a projected, proximal, or primal-dual update",
        "feasibility, domain projection, state shape, and optimality conditions",
        "objective gap, feasibility residual, KKT residual, and gradient error",
        "variable count, constraint count, conditioning, and linear-solver budget",
        "objective operator, projector/proximal map, transition, solver, and readout",
    )),
    (("inr", "representation", "coordinate"), Curriculum(
        "one latent feature vector per coordinate",
        "spatial, temporal, or spatiotemporal coordinates",
        "a coordinate-injected recurrent field",
        "coordinate shape, output domain, differentiability, and state width",
        "signal error, coordinate-derivative error, and fixed-point residual",
        "sample count, coordinate dimension, frequency scale, and hidden width",
        "coordinate lift, recurrent transition, activation, readout, and solver",
    )),
    (("poisson", "mirror", "inverse"), Curriculum(
        "a positive reconstruction",
        "observed counts and forward/adjoint measurement operators",
        "a Burg-geometry mirror transition with a learned or known regularizer",
        "positivity, adjoint consistency, finite intensity, and box constraints",
        "Poisson divergence, reconstruction error, data residual, and fixed-point residual",
        "measurement count, image size, operator cost, and regularizer width",
        "initializer, forward/adjoint operators, regularizer, mirror transition, and solver",
    )),
    (("physics_informed", "dae", "residual"), Curriculum(
        "an implicit latent state or coupled differential/algebraic stage state",
        "time, initial/boundary values, dynamics, and algebraic constraints",
        "a time-conditioned fixed point or implicit Runge-Kutta root map",
        "initial/boundary conditions, equation shape, and constraint consistency",
        "equation residual, boundary error, trajectory error, and solver residual",
        "collocation count, latent dimension, stages, stiffness, and time horizon",
        "time/source lift, transition, readout, dynamics, constraints, losses, and solvers",
    )),
    (("dataset", "preliminaries", "quickstart", "public_experiment", "training"), GENERIC),
)


def _source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _cell(kind: str, source: str, suffix: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "cell_type": kind,
        "id": f"silva-extension-{suffix}",
        "metadata": {"tags": [TAG]},
        "source": _source_lines(source.strip() + "\n"),
    }
    if kind == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def _normalize_reader_text(notebook: dict[str, object]) -> None:
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        for old, new in READER_TEXT_REPLACEMENTS.items():
            source = source.replace(old, new)
        if cell.get("cell_type") == "code":
            for old, new in CODE_REPLACEMENTS.items():
                source = source.replace(old, new)
        cell["source"] = _source_lines(source)


def _curriculum_for(path: Path) -> Curriculum:
    stem = path.stem.lower()
    for terms, curriculum in RULES:
        if any(term in stem for term in terms):
            return curriculum
    return GENERIC


def _family_recipe(path: Path) -> str:
    recipes = {
        "12_paper_family_architectures.ipynb": """
### Replace the Internals of the Generalized Cases

`SILVASequenceDEQ` accepts custom embedding, transition, and readout modules.
`SILVAMultiscaleDEQ` accepts a custom stem or per-scale injection modules and a
complete multiscale transition. `SILVAImplicitGraphNetwork` accepts source,
recurrent, complete-transition, and readout replacements.

```python
sequence = SILVASequenceDEQ(
    dim=width,
    input_dim=input_width,
    output_dim=classes,
    transition_module=my_sequence_transition,
    embedding_module=my_embedding,
    readout_module=my_readout,
)

multiscale = SILVAMultiscaleDEQ(
    in_channels=3,
    channels=(32, 64, 128),
    injection_mode="all",
    injection_modules=my_scale_sources,
    transition_module=my_multiscale_transition,
)
```
""",
        "13_raft_deq_flow.ipynb": """
### Replace the Flow Transition

The complete refinement can be supplied as
`transition_module(flow, fmap1, fmap2, correlation)`. Feature and context
encoders and the update block are independently replaceable.

```python
flow_model = SILVAOpticalFlowDEQ(
    feature_dim=feature_dim,
    encoder_module=my_feature_encoder,
    update_block=my_update_block,
    transition_module=my_flow_transition,
    config=solver_config,
)
```
""",
        "14_point_architecture_catalog.ipynb": """
### Define Every Operator Inside One Point

```python
point = SILVACortexLayer(
    state_dim=channels,
    input_encoder=my_source,
    state_network=[my_unet, my_spectral_block],
    self_terms=[my_self_field],
    local_terms=[my_graph_or_convolution],
    global_terms=[my_global_context],
    interaction_terms=[my_known_physics],
    output_network=my_transition_head,
    normalizer=my_normalizer,
    normalize=False,
    config=solver_config,
)
```

Modules in `state_network` are sequential. Branch lists are independent fields
summed into the completed transition.
""",
        "15_neural_operators_ode_pde.ipynb": """
### Replace the Scientific Operator

```python
operator = SILVAOperatorModel(
    in_channels=input_fields,
    state_channels=width,
    out_channels=solution_fields,
    input_encoder=my_lift,
    architecture=my_shape_preserving_operator,
    local_terms=[my_discretized_pde],
    global_terms=[my_integral_operator],
    readout=my_projection,
    config=solver_config,
)
```

The internal architecture may be spectral, convolutional, graph based, kernel
integral, wavelet based, or a composition, provided it preserves the sampled
state field.
""",
        "17_silva_fno_equilibrium_lab.ipynb": """
### Replace the Lift, Tied Field, and Readout

```python
class MyTiedOperator(nn.Module):
    def forward(self, state, lifted_forcing):
        spectral = self.spectral_operator(state)
        local = self.local_operator(state)
        return torch.tanh(lifted_forcing + spectral + local)


model = SILVAFNODEQ(
    in_channels=forcing_channels,
    state_channels=width,
    out_channels=solution_channels,
    forcing_lift=my_lift,
    block=MyTiedOperator(),
    readout=my_readout,
    config=solver_config,
)
```
""",
        "18_silva_graph_transport_lab.ipynb": """
### Replace the Graph Physics Transition

```python
class MyGraphPhysics(nn.Module):
    def forward(
        self, state, inputs, edge_index, *, edge_weight=None, edge_velocity=None
    ):
        source = self.source(inputs)
        transport = self.transport(state, edge_index, edge_velocity)
        diffusion = self.diffusion(state, edge_index, edge_weight)
        return torch.tanh(source + transport + diffusion)


model = SILVAPhysicsGuidedGraphDEQ(
    in_dim=input_dim,
    state_dim=width,
    out_dim=output_dim,
    transition=MyGraphPhysics(),
    readout=my_readout,
    config=solver_config,
)
```
""",
        "19_silva_homotopy_equilibrium_lab.ipynb": """
### Replace the Equilibrium Map Along the Homotopy

```python
model = SILVAHomotopyEquilibrium(
    in_dim=input_dim,
    state_dim=width,
    out_dim=output_dim,
    transition=my_conditioned_transition,
    readout=my_readout,
    integrator="rk4",
    steps=steps,
    horizon=horizon,
)
```

The custom transition implements `transition(state, condition)`. The homotopy
integrates `transition(state, condition) - state` without changing that map.
""",
        "20_silva_distributional_equilibrium_lab.ipynb": """
### Replace the Particle Transition

```python
class MyMeasureTransition(nn.Module):
    def forward(self, latent, context, *, latent_mask=None, context_mask=None):
        return self.permutation_compatible_update(
            latent, context, latent_mask, context_mask
        )


model = SILVADistributionalDEQ(
    input_dim=input_dim,
    latent_dim=latent_dim,
    transition=MyMeasureTransition(),
    kernel="energy",
    pairwise_chunk_size=pairwise_chunk_size,
)
```
""",
        "21_silva_monotone_graph_equilibrium.ipynb": """
### Replace the Monotone Transition and Certificate

```python
model = SILVAMonotoneGraphEquilibrium(
    in_dim=input_dim,
    state_dim=width,
    out_dim=output_dim,
    transition=my_monotone_transition,
    certificate=my_monotonicity_certificate,
    readout=my_readout,
    config=solver_config,
)
```

The transition has signature `(state, inputs, edge_index, edge_weight)` and
must preserve node count and state width. A custom certificate remains separate
from numerical convergence diagnostics.
""",
        "22_silva_generative_equilibrium_transformer.ipynb": """
### Replace Every Transformer Stage

```python
model = SILVAGenerativeEquilibriumTransformer(
    in_channels=image_channels,
    patch_size=patch_size,
    hidden_dim=width,
    injection_depth=len(my_injection_blocks),
    equilibrium_depth=len(my_equilibrium_blocks),
    patch_embed=my_patch_lift,
    injection_blocks=my_injection_blocks,
    injection_projection=my_qkv_projection,
    equilibrium_blocks=my_equilibrium_blocks,
    decoder=my_decoder,
    config=solver_config,
)
```

Each equilibrium block receives `(state, qkv_injection, class_injection)` and
must return the same token-state shape.
""",
        "23_silva_poisson_mirror_equilibrium.ipynb": """
### Replace the Inverse Operator and Mirror Transition

```python
transition = SILVABurgMirrorTransition(
    forward_operator=my_forward_operator,
    adjoint_operator=my_adjoint_operator,
    regularizer_gradient=my_regularizer_gradient,
    step_size=step_size,
)
model = SILVAPoissonMirrorEquilibrium(
    transition=transition,
    initializer=my_positive_initializer,
    intensity_operator=my_forward_operator,
    config=solver_config,
)
```

For a completely different positive-domain update, replace `transition`
directly with a module implementing `(state, observation) -> next_state`.
""",
        "24_silva_physics_informed_equilibrium.ipynb": """
### Build the Physics-Informed Transition Explicitly

```python
class MyPhysicsTransition(nn.Module):
    time_dim = 1
    state_dim = 64

    def __init__(self):
        super().__init__()
        self.time_source = MyTimeEncoder()
        self.state_operator = MyResidualOrOperatorNetwork()

    def forward(self, state, times):
        source = self.time_source(times)
        return torch.tanh(source + 0.2 * self.state_operator(state))


model = SILVAPhysicsInformedEquilibrium(
    state_dim=64,
    output_dim=physical_dimension,
    transition=MyPhysicsTransition(),
    readout=my_physical_readout,
    derivative_mode="matrix_free",
    derivative_max_iter=100,
    derivative_tol=1e-7,
    config=solver_config,
)
```

`physics_loss` then accepts the user-supplied dynamics. The transition defines
the implicit representation; the dynamics define the differential-equation
residual. They are related by the implicit time derivative but are not the same
module.
""",
        "25_silva_implicit_dae_and_residuals.ipynb": """
### Supply a New DAE and Implicit Tableau

```python
step = SILVAImplicitDAEStep(
    a=runge_kutta_a,
    b=runge_kutta_b,
    c=runge_kutta_c,
    linear_solver="gmres",
    linear_max_iter=100,
    linear_tol=1e-7,
)
result = step(
    differential_state,
    algebraic_state,
    step_size,
    dynamics=my_differential_field,
    constraint=my_algebraic_constraint,
)
```

The dimensions of the differential and algebraic states may change between
applications, while each supplied dynamics and constraint function must
preserve its declared equation shape.
""",
    }
    return recipes.get(path.name, "")


def _markdown_derivation(path: Path, curriculum: Curriculum) -> str:
    title = path.stem.replace("_", " ").title()
    recipe = _family_recipe(path)
    return rf"""
## From {title} to a Custom SILVA Family

The construction in this notebook can be separated into the universal
conditioned-equilibrium contract

$$
z_0=I_\eta(x),\qquad
z^\star=T_\theta(z^\star,x),\qquad
\widehat y=Q_\psi(z^\star).
$$

For this topic:

| Part | Concrete interpretation |
| --- | --- |
| Equilibrium state | {curriculum.state} |
| Condition | {curriculum.condition} |
| Repeated computation | {curriculum.repeated} |
| Required invariants | {curriculum.invariants} |
| Replaceable components | {curriculum.components} |

The initializer and source path are evaluated outside or alongside the root
solve. Only the state-preserving transition is repeated. Replacing an internal
architecture does not change this equation, provided the transition still maps
the same state space into itself.

{recipe}
"""


def _code_contract() -> str:
    return """
import torch as silva_extension_torch
from torch import nn as silva_extension_nn

from silva_networks import (
    SILVAConditionedEquilibrium,
    SILVAZeroInitializer,
    SolverConfig,
    validate_silva_transition,
)


class NotebookExtensionTransition(silva_extension_nn.Module):
    def __init__(self, condition_dim=2, state_dim=3):
        super().__init__()
        self.source = silva_extension_nn.Linear(condition_dim, state_dim)
        self.state_field = silva_extension_nn.Sequential(
            silva_extension_nn.Linear(state_dim, 2 * state_dim),
            silva_extension_nn.Tanh(),
            silva_extension_nn.Linear(2 * state_dim, state_dim),
        )

    def forward(self, state, condition):
        return silva_extension_torch.tanh(
            self.source(condition) + 0.15 * self.state_field(state)
        )


silva_extension_torch.manual_seed(610)
notebook_condition = silva_extension_torch.linspace(-1.0, 1.0, 8).reshape(4, 2)
notebook_state0 = silva_extension_torch.zeros(4, 3)
notebook_transition = NotebookExtensionTransition()

notebook_report = validate_silva_transition(
    notebook_transition,
    notebook_state0,
    notebook_condition,
)
assert notebook_report.valid

with silva_extension_torch.no_grad():
    notebook_reference_step = silva_extension_torch.tanh(
        notebook_transition.source(notebook_condition)
        + 0.15 * notebook_transition.state_field(notebook_state0)
    )
silva_extension_torch.testing.assert_close(
    notebook_transition(notebook_state0, notebook_condition),
    notebook_reference_step,
)

notebook_custom_model = SILVAConditionedEquilibrium(
    notebook_transition,
    SILVAZeroInitializer(3),
    readout=silva_extension_nn.Linear(3, 1),
    config=SolverConfig(
        solver="picard",
        max_iter=40,
        tol=1e-7,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
notebook_custom_result = notebook_custom_model(
    notebook_condition,
    return_result=True,
)
assert notebook_custom_result.output.shape == (4, 1)
assert notebook_custom_result.solver_result.residual < 1e-5

notebook_custom_result.output.square().mean().backward()
assert all(
    parameter.grad is not None and silva_extension_torch.isfinite(parameter.grad).all()
    for parameter in notebook_custom_model.parameters()
)
print("custom transition:", notebook_report)
print("equilibrium residual:", notebook_custom_result.solver_result.residual)
"""


def _markdown_reproduction(curriculum: Curriculum) -> str:
    return rf"""
## Numerical Equivalence, Compact Reproduction, and Scale

Before training, compare one packaged transition with an independently written
update:

$$
e_{{\mathrm{{step}}}}
=\frac{{\|T_\theta(z,x)-T_{{\mathrm{{ref}}}}(z,x)\|_2}}
{{\|T_{{\mathrm{{ref}}}}(z,x)\|_2+\varepsilon}}.
$$

After solving, report the fixed-point residual separately:

$$
e_{{\mathrm{{fp}}}}
=\frac{{\|T_\theta(z^\star,x)-z^\star\|_2}}
{{\|z^\star\|_2+\varepsilon}}.
$$

For this notebook, a compact reproduction must declare and assert
**{curriculum.compact_metric}**. A full experiment must additionally record the
source dataset version and split, preprocessing, architecture widths, solver
and optimizer schedules, random seeds, baseline configuration, checkpoints,
and every deviation from the cited protocol.

The principal scaling axes are **{curriculum.scale_axis}**. Increase one axis at
a time, retain the compact deterministic case as a regression test, and record
task error, domain-specific residual, forward residual, backward linear
residual, memory use, and runtime independently.

### Extension Exercises

1. Replace one component from this notebook while preserving its state and
   domain invariants.
2. Write the replacement first as an independent reference function, then as
   a module, and assert one-step equivalence.
3. Compare two solver configurations on the identical trained transition.
4. Add a compact baseline and a predeclared metric threshold.
5. Create a full-scale configuration without weakening the compact tests.

The complete authoring protocol is documented in
[Extending SILVA](https://jseluis.github.io/silva-networks/learn/extending-silva/).
"""


def _code_record(path: Path, curriculum: Curriculum) -> str:
    return f"""
notebook_reproduction_record = {{
    "notebook": {path.name!r},
    "state": {curriculum.state!r},
    "condition": {curriculum.condition!r},
    "transition": {curriculum.repeated!r},
    "invariants": {curriculum.invariants!r},
    "compact_metric": {curriculum.compact_metric!r},
    "scale_axis": {curriculum.scale_axis!r},
}}
assert all(notebook_reproduction_record.values())
notebook_reproduction_record
"""


def expand_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    _normalize_reader_text(notebook)
    cells = [
        cell
        for cell in notebook.get("cells", [])
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    curriculum = _curriculum_for(path)
    cells.extend(
        [
            _cell("markdown", _markdown_derivation(path, curriculum), "derivation"),
            _cell("code", _code_contract(), "contract"),
            _cell("markdown", _markdown_reproduction(curriculum), "reproduction"),
            _cell("code", _code_record(path, curriculum), "record"),
        ]
    )
    notebook["cells"] = cells
    return notebook


def _write(path: Path, notebook: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical_notebooks() -> list[Path]:
    package = sorted((ROOT / "notebooks/package_api").glob("*.ipynb"))
    bridge = sorted((ROOT / "notebooks/implicit_bridge").glob("*.ipynb"))
    book = sorted((ROOT / "notebooks").glob("*.ipynb"))
    notebooks = package + bridge + book
    if len(notebooks) != 62:
        raise RuntimeError(f"expected 62 canonical notebooks, found {len(notebooks)}")
    return notebooks


def main() -> int:
    for path in _canonical_notebooks():
        notebook = expand_notebook(path)
        _write(path, notebook)
        if path.parent.name == "package_api":
            _write(ROOT / "docs/package-notebooks" / path.name, notebook)
            _write(ROOT / "colab" / path.name, notebook)
        elif path.parent.name == "implicit_bridge":
            _write(ROOT / "docs/implicit-bridge-notebooks" / path.name, notebook)
            _write(ROOT / "colab/implicit_bridge" / path.name, notebook)
    print("expanded 62 canonical notebooks and synchronized publication mirrors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
