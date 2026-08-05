from __future__ import annotations

from torch import nn

from silva_networks import (
    SILVAImplicitDAEStep,
    SILVAMonotoneGraphEquilibrium,
    SILVAPhysicsInformedEquilibrium,
    all_silva_family_guides,
    audit_silva_family_guides,
    available_silva_families,
    build_scaled_silva,
    canonical_silva_family,
    full_scale_solver_config,
    prepare_silva_model,
    runtime_for_tier,
    silva_family_guide,
)


def test_scale_guides_cover_every_canonical_family() -> None:
    guides = all_silva_family_guides()

    assert tuple(guide.family for guide in guides) == available_silva_families()
    assert audit_silva_family_guides() == ()
    assert all(guide.paper_refs for guide in guides)
    assert all(guide.reference_repositories for guide in guides)
    assert all(guide.scale_controls and guide.extension_points for guide in guides)
    assert canonical_silva_family("fno-deq") == "silva_fno_deq"
    assert silva_family_guide("pideq").family == "silva_physics_informed_equilibrium"


def test_scaled_builders_enable_family_specific_numerical_paths() -> None:
    physics = build_scaled_silva(
        "pideq",
        state_dim=4,
        output_dim=2,
        tier="workstation",
    )
    dae = build_scaled_silva("dae_pinn", tier="workstation")
    monotone = build_scaled_silva(
        "mignn",
        in_dim=3,
        state_dim=8,
        out_dim=2,
        tier="workstation",
    )

    assert isinstance(physics, SILVAPhysicsInformedEquilibrium)
    assert physics.derivative_mode == "matrix_free"
    assert isinstance(dae, SILVAImplicitDAEStep)
    assert dae.linear_solver == "gmres"
    assert isinstance(monotone, SILVAMonotoneGraphEquilibrium)
    assert monotone.transition.operator_rank == 8
    assert monotone.config.backward_mode == "implicit"


def test_runtime_templates_connect_training_data_and_model_preparation() -> None:
    runtime = runtime_for_tier(
        "smoke",
        device="cpu",
        per_device_batch_size=3,
        gradient_accumulation_steps=2,
    )
    train = runtime.train_config(task="regression", epochs=2)
    data = runtime.data_config(shuffle=False)
    model = prepare_silva_model(nn.Linear(2, 1), runtime)

    assert runtime.effective_batch_size(world_size=4) == 24
    assert runtime.mixed_precision == "none"
    assert train.gradient_accumulation_steps == 2
    assert data.batch_size == 3
    assert next(model.parameters()).device.type == "cpu"
    solver = full_scale_solver_config(batch_dims=0, tier="smoke")
    assert solver.backward_mode == "implicit"
    assert solver.stop_mode == solver.backward_stop_mode == "relative"


def test_sequence_scale_defaults_do_not_leak_transformer_options_into_trellis() -> None:
    model = build_scaled_silva(
        "sequence_deq",
        tier="smoke",
        dim=8,
        input_dim=3,
        output_dim=2,
        mode="trellis",
    )

    assert model.transition.mode == "trellis"
    assert runtime_for_tier("workstation").mixed_precision == "none"
    assert runtime_for_tier("full").mixed_precision == "bfloat16"
