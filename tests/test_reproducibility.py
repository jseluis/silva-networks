from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from silva_networks import (
    SILVADiffusionEquilibrium,
    SILVAPhysicsInformedEquilibrium,
    SolverConfig,
    all_silva_reproduction_specs,
    audit_silva_reproduction_specs,
    available_silva_families,
    build_silva_reproduction,
    silva_family_constructor,
    silva_family_signature,
    silva_reproduction_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_registry_covers_every_canonical_family() -> None:
    specs = all_silva_reproduction_specs()

    assert tuple(spec.family for spec in specs) == available_silva_families()
    assert audit_silva_reproduction_specs() == ()
    assert all(spec.verification_level == "compact-verified" for spec in specs)
    assert all(spec.constructor_signature.startswith("(") for spec in specs)
    assert all("0x" not in spec.constructor_signature for spec in specs)
    assert all(spec.preserved_mechanisms for spec in specs)
    assert all(spec.silva_extensions for spec in specs)
    assert all(spec.benchmark_requirements for spec in specs)
    assert all(not spec.equation.startswith("z_star = T_theta") for spec in specs)


def test_reproduction_evidence_paths_and_reference_anchors_exist() -> None:
    references = (ROOT / "docs/paper/references.md").read_text(encoding="utf-8")

    for spec in all_silva_reproduction_specs():
        for path in (*spec.notebooks, *spec.tests):
            assert (ROOT / path).exists(), (spec.family, path)
        for reference in spec.paper_refs:
            assert f'id="ref-{reference}"' in references, (spec.family, reference)


def test_every_family_has_a_distinct_source_conformance_contract() -> None:
    specs = all_silva_reproduction_specs()

    assert len({spec.preserved_mechanisms for spec in specs}) == len(specs)
    assert len({spec.silva_extensions for spec in specs}) == len(specs)
    assert len({spec.benchmark_requirements for spec in specs}) == len(specs)


def test_aliases_expose_real_constructor_and_signature() -> None:
    constructor = silva_family_constructor("pideq")
    signature = silva_family_signature("pideq")
    spec = silva_reproduction_spec("pideq")

    assert constructor is not None
    assert "transition" in signature.parameters
    assert "readout" in signature.parameters
    assert spec.family == "silva_physics_informed_equilibrium"
    assert "Van der Pol" in spec.datasets[0]


def test_reproduction_builder_forwards_granular_family_options() -> None:
    model = build_silva_reproduction(
        "pideq",
        tier="smoke",
        state_dim=5,
        output_dim=2,
        derivative_mode="dense",
    )

    assert isinstance(model, SILVAPhysicsInformedEquilibrium)
    assert model.state_dim == 5
    assert model.output_dim == 2
    assert model.derivative_mode == "dense"


class _CompleteDiffusionStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(
        self,
        state: torch.Tensor,
        timestep: int,
        next_timestep: int,
        condition: torch.Tensor | None,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        del timestep, next_timestep, noise
        source = 0.0 if condition is None else condition
        return self.scale * state + source


class _ObservationConsistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        candidate: torch.Tensor,
        observation: torch.Tensor,
        next_timestep: int,
    ) -> torch.Tensor:
        del next_timestep
        mix = self.mix_logit.sigmoid()
        return mix * candidate + (1.0 - mix) * observation


def test_joint_diffusion_accepts_complete_step_and_observation_operator() -> None:
    step = _CompleteDiffusionStep()
    consistency = _ObservationConsistency()
    model = SILVADiffusionEquilibrium(
        None,
        torch.tensor([0.95, 0.8, 0.6]),
        (2, 1, 0),
        step_operator=step,
        data_consistency=consistency,
        config=SolverConfig(
            solver="picard",
            max_iter=5,
            tol=1e-8,
            backward_mode="unrolled",
            anderson_batch_dims=0,
        ),
    )
    noise = torch.randn(2, 1, 3, 3, requires_grad=True)
    condition = torch.full_like(noise, 0.1, requires_grad=True)
    observation = torch.zeros_like(noise, requires_grad=True)

    result = model(
        noise,
        condition=condition,
        observation=observation,
        return_result=True,
    )
    result.output.square().mean().backward()

    assert result.output.shape == noise.shape
    assert result.trajectory.shape == (3, *noise.shape)
    assert step.scale.grad is not None
    assert consistency.mix_logit.grad is not None
    assert observation.grad is not None
