"""Generate the source-aware SILVA reproduction registry notebook."""

from __future__ import annotations

from pathlib import Path

from notebook_generation import write_notebook

ROOT = Path(__file__).resolve().parents[1]
NAME = "27_reproducing_silva_and_source_methods.ipynb"


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _markdown(text: str, identifier: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": _source(text + "\n"),
    }


def _code(text: str, identifier: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": identifier,
        "metadata": {},
        "outputs": [],
        "source": _source(text + "\n"),
    }


def notebook() -> dict[str, object]:
    cells = [
        _markdown(
            r"""# Reproducing SILVA and Source Methods

This lab makes the reproduction boundary executable. It inspects all canonical
families, resolves aliases to their real constructor signatures, builds a
custom conditioned equilibrium, adapts a joint diffusion trajectory to an
observation-conditioned restoration step, and emits a structured run record.

The universal equation is

$$
\begin{aligned}
z_0 &= I_\eta(x), \\
z^\star &= T_\theta(z^\star,x), \\
\widehat y &= Q_\psi(z^\star).
\end{aligned}
$$

A source method is reproduced only when its equation, data release,
preprocessing, dimensions, numerical settings, training schedule, checkpoints,
seeds, and metrics are all declared. Compact checks verify mechanisms; they do
not stand in for an unexecuted published benchmark.""",
            "reproduction-title",
        ),
        _code(
            """from dataclasses import asdict

import torch
from torch import nn

from silva_networks import (
    SILVAConditionedEquilibrium,
    SILVADiffusionEquilibrium,
    SILVAZeroInitializer,
    SolverConfig,
    all_silva_reproduction_specs,
    audit_silva_reproduction_specs,
    silva_reproduction_spec,
    validate_silva_transition,
)

torch.manual_seed(27)
assert audit_silva_reproduction_specs() == ()
specs = all_silva_reproduction_specs()
assert len(specs) == 30
[(spec.family, spec.source_relation, spec.verification_level) for spec in specs]""",
            "registry-audit",
        ),
        _markdown(
            r"""## Inspect the Complete Contract

The record separates scientific and numerical responsibilities:

| Field | Question answered |
| --- | --- |
| `equation` | What state-preserving map is solved? |
| `source_relation` | Is this native SILVA or a cited mechanism adaptation? |
| `datasets` and `preprocessing` | What observations enter the experiment? |
| `metrics` | What must be measured besides solver residual? |
| `notebooks` and `tests` | What executable evidence exists locally? |
| `configurable_parts` | Which operators and scale axes may be changed? |
| `preserved_mechanisms` | Which source mechanisms remain present? |
| `silva_extensions` | Which components may be replaced or enlarged inside SILVA? |
| `benchmark_requirements` | Which source-protocol obligations remain for benchmark equivalence? |
| `constructor_signature` | Which exact public arguments are accepted? |

The residual

$$
r(z^\star,x)=T_\theta(z^\star,x)-z^\star
$$

checks the equilibrium equation. It does not measure classification accuracy,
field error, physical residual, FID, endpoint error, or reconstruction quality.""",
            "inspect-contract",
        ),
        _code(
            """for family in ("fno_deq", "mignn", "pideq", "deq_ddim"):
    spec = silva_reproduction_spec(family)
    print("\\n", spec.family)
    print(" equation:", spec.equation)
    print(" preserves:", spec.preserved_mechanisms)
    print(" SILVA extensions:", spec.silva_extensions)
    print(" benchmark requires:", spec.benchmark_requirements)
    print(" data:", spec.datasets)
    print(" metrics:", spec.metrics)
    print(" signature:", spec.constructor_signature)""",
            "inspect-families",
        ),
        _markdown(
            r"""## Audit All 30 Source-Conformance Records

Every family has a distinct governing equation and three additional records:
what is retained from the source mechanism, what SILVA allows the user to
replace or scale, and what must still be reproduced before comparing with the
source benchmark. This avoids treating a compact mechanism check as a full
paper result while keeping the architecture open for new experiments.""",
            "all-source-contracts",
        ),
        _code(
            """assert all(spec.equation for spec in specs)
assert len({spec.preserved_mechanisms for spec in specs}) == len(specs)
assert len({spec.silva_extensions for spec in specs}) == len(specs)
assert len({spec.benchmark_requirements for spec in specs}) == len(specs)

for spec in specs:
    print(f"\\n{spec.family}")
    print(" equation:", spec.equation)
    print(" preserves:", *spec.preserved_mechanisms)
    print(" extends:", *spec.silva_extensions)
    print(" benchmark requires:", *spec.benchmark_requirements)
    print(" references:", *spec.paper_refs)
    print(" repositories:", *spec.repositories)""",
            "all-source-contracts-code",
        ),
        _markdown(
            r"""## Build a New Transition From Its Equation

For the compact transition

$$
T_\theta(z,x)=\tanh\!\left(W_xx+0.15\,h_\theta(z)\right),
$$

the source projection and recurrent field remain independently replaceable.
The validator checks shape, device, dtype, finiteness, and state-gradient
compatibility before the solver is introduced.""",
            "custom-derivation",
        ),
        _code(
            """class CustomTransition(nn.Module):
    def __init__(self, input_dim=2, state_dim=4):
        super().__init__()
        self.source = nn.Linear(input_dim, state_dim)
        self.recurrent = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.Tanh(),
            nn.Linear(8, state_dim),
        )

    def forward(self, state, inputs):
        return torch.tanh(self.source(inputs) + 0.15 * self.recurrent(state))


inputs = torch.linspace(-1.0, 1.0, 12).reshape(6, 2)
transition = CustomTransition()
report = validate_silva_transition(transition, torch.zeros(6, 4), inputs)
assert report.valid

custom = SILVAConditionedEquilibrium(
    transition,
    SILVAZeroInitializer(4),
    readout=nn.Linear(4, 1),
    config=SolverConfig(
        solver="anderson",
        max_iter=30,
        tol=1e-6,
        backward_mode="implicit",
        backward_solver="gmres",
        anderson_batch_dims=1,
    ),
)
custom_result = custom(inputs, return_result=True)
custom_result.output.square().mean().backward()
assert custom_result.output.shape == (6, 1)
assert all(parameter.grad is not None for parameter in custom.parameters())
print("custom residual:", custom_result.solver_result.residual)""",
            "custom-family",
        ),
        _markdown(
            r"""## Joint Diffusion Restoration Inside SILVA

Let $X=(x_{t_0},\ldots,x_{t_K})$ be one joint trajectory. A restoration
adaptation uses

$$
x_{t_{k+1}}^+
=P_{y,t_{k+1}}\!\left(D_{t_k\rightarrow t_{k+1}}
(x_{t_k};c,\xi_k)\right),
$$

where $D$ is a complete reverse step and $P$ is a declared measurement or
data-consistency operator. The triangular transition updates all trajectory
positions from the previous solver state. The step, observation operator,
schedule, stochastic terms, condition, and initial trajectory are separately
controllable.""",
            "restoration-derivation",
        ),
        _code(
            """class CompleteReverseStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, state, timestep, next_timestep, condition, noise):
        del timestep, next_timestep, noise
        return self.scale * state + condition


class ObservationOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, candidate, observation, next_timestep):
        del next_timestep
        weight = self.logit.sigmoid()
        return weight * candidate + (1.0 - weight) * observation


restoration = SILVADiffusionEquilibrium(
    denoiser=None,
    alphas_cumprod=torch.tensor([0.95, 0.80, 0.60]),
    timesteps=(2, 1, 0),
    step_operator=CompleteReverseStep(),
    data_consistency=ObservationOperator(),
    config=SolverConfig(
        solver="picard",
        max_iter=5,
        tol=1e-7,
        backward_mode="unrolled",
        anderson_batch_dims=0,
    ),
)
noise = torch.randn(2, 1, 4, 4, requires_grad=True)
condition = torch.full_like(noise, 0.1, requires_grad=True)
observation = torch.zeros_like(noise, requires_grad=True)
restoration_result = restoration(
    noise,
    condition=condition,
    observation=observation,
    return_result=True,
)
restoration_result.output.square().mean().backward()
assert restoration_result.trajectory.shape == (3, *noise.shape)
assert observation.grad is not None
print("restoration residual:", restoration_result.solver_result.residual)""",
            "restoration-code",
        ),
        _markdown(
            r"""## Record What Was Actually Run

A benchmark record must preserve the source relationship and every deviation
from the cited protocol. This prevents a compact mechanism check from being
mistaken for a published-scale result and makes controlled extensions possible.""",
            "record-explanation",
        ),
        _code(
            """selected = silva_reproduction_spec("deq_ddim")
run_record = {
    "family": selected.family,
    "paper_refs": selected.paper_refs,
    "source_relation": selected.source_relation,
    "verification_level": selected.verification_level,
    "dataset": "deterministic compact tensors",
    "split": "single checked batch",
    "model_options": {
        "trajectory_steps": 3,
        "complete_step": "CompleteReverseStep",
        "observation_operator": "ObservationOperator",
    },
    "solver": asdict(restoration.config),
    "metrics": {
        "fixed_point_residual": restoration_result.solver_result.residual,
        "output_norm": float(restoration_result.output.detach().norm()),
    },
    "seed": 27,
    "deviations": "compact mechanism check; no published image benchmark claimed",
}
assert run_record["metrics"]["fixed_point_residual"] < 1e-5
run_record""",
            "run-record",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook_payload = notebook()
    for path in (
        ROOT / "notebooks/package_api" / NAME,
        ROOT / "docs/package-notebooks" / NAME,
        ROOT / "colab" / NAME,
    ):
        write_notebook(path, notebook_payload)
    print(f"generated {NAME} and publication mirrors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
