from __future__ import annotations

import pytest
import torch
from torch import nn

from silva_networks import (
    SILVAAlgorithmicReasoner,
    SILVADiffusionRestorationEquilibrium,
    SILVADynamicEconomicEquilibrium,
    SILVAHamiltonianEquilibrium,
    SILVAImageMattingEquilibrium,
    SILVAInverseImagingEquilibrium,
    SILVALipschitzMultiscaleEquilibrium,
    SILVALipschitzRobustEquilibrium,
    SILVAMagneticParticleEquilibrium,
    SILVARecurrentEquilibriumNetwork,
    SILVASerializedSmoothingEquilibrium,
    SILVASnapshotCompressiveEquilibrium,
    SILVASparseHyperspectralEquilibrium,
    SILVASubhomogeneousEquilibrium,
    SolverConfig,
    available_silva_families,
    silva_equilibrium_model,
)


def _config(*, batch_dims: int = 1, max_iter: int = 20) -> SolverConfig:
    return SolverConfig(
        solver="picard",
        max_iter=max_iter,
        tol=1e-6,
        backward_mode="unrolled",
        anderson_batch_dims=batch_dims,
        return_best=True,
    )


def _finite_backward(value: torch.Tensor) -> None:
    value.square().mean().backward()
    assert torch.isfinite(value).all()


def test_source_equilibrium_families_are_public_and_constructible() -> None:
    expected = {
        "silva_lipschitz_mdeq",
        "silva_subhomogeneous_equilibrium",
        "silva_algorithmic_reasoner",
        "silva_hamiltonian_equilibrium",
        "silva_inverse_imaging_equilibrium",
        "silva_snapshot_compressive_equilibrium",
        "silva_magnetic_particle_equilibrium",
        "silva_sparse_hyperspectral_equilibrium",
        "silva_serialized_smoothing_equilibrium",
        "silva_diffusion_restoration_equilibrium",
        "silva_recurrent_equilibrium_network",
        "silva_lipschitz_robust_equilibrium",
        "silva_image_matting_equilibrium",
        "silva_dynamic_economic_equilibrium",
    }
    assert expected <= set(available_silva_families())
    model = silva_equilibrium_model(
        "lipschitz_mdeq",
        input_dim=3,
        scale_dims=(4, 2),
        output_dim=2,
        config=_config(),
    )
    assert isinstance(model, SILVALipschitzMultiscaleEquilibrium)


def test_lipschitz_mdeq_exposes_scales_bound_and_gradients() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(3, 4, requires_grad=True)
    model = SILVALipschitzMultiscaleEquilibrium(
        4, (5, 3, 2), 2, contraction=0.65, config=_config()
    )
    result = model(inputs, return_result=True)
    assert result.state.shape == (3, 10)
    assert tuple(part.shape[-1] for part in model.split_state(result.state)) == (5, 3, 2)
    assert model.lipschitz_bound() <= 0.65001
    _finite_backward(result.output)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_subhomogeneous_equilibrium_is_positive_and_normalized() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(2, 3, requires_grad=True)
    model = SILVASubhomogeneousEquilibrium(
        3, 6, 2, norm_p=4.0, power=0.75, config=_config(max_iter=30)
    )
    result = model(inputs, return_result=True)
    norms = torch.linalg.vector_norm(result.state, ord=4.0, dim=-1)
    assert torch.all(result.state > 0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=2e-5)
    _finite_backward(result.output)


def test_algorithmic_reasoner_solves_graph_state_and_backpropagates() -> None:
    torch.manual_seed(2)
    inputs = torch.randn(5, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    model = SILVAAlgorithmicReasoner(3, 7, 2, config=_config(batch_dims=0))
    result = model(inputs, edge_index, return_result=True)
    assert result.state.shape == (5, 7)
    assert result.output.shape == (5, 2)
    _finite_backward(result.output)
    assert inputs.grad is not None


def test_hamiltonian_equilibrium_is_symmetric_and_rotation_invariant() -> None:
    torch.manual_seed(3)
    features = torch.randn(2, 4, 3, requires_grad=True)
    positions = torch.randn(2, 4, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    model = SILVAHamiltonianEquilibrium(3, config=_config())
    hamiltonian = model(features, positions)
    rotated = model(features, positions @ rotation)
    assert hamiltonian.shape == (2, 4, 4)
    assert torch.allclose(hamiltonian, hamiltonian.mT, atol=1e-6)
    assert torch.allclose(hamiltonian, rotated, atol=2e-5)
    _finite_backward(hamiltonian)


def test_inverse_imaging_accepts_replaceable_known_operators() -> None:
    measurement = torch.randn(2, 1, 6, 6, requires_grad=True)
    model = SILVAInverseImagingEquilibrium(
        1,
        forward_operator=nn.Identity(),
        adjoint_operator=nn.Identity(),
        prior=nn.Identity(),
        step_size=0.5,
        config=_config(),
    )
    result = model(measurement, return_result=True)
    assert torch.allclose(result.output, measurement, atol=1e-6)
    _finite_backward(result.output)


def test_snapshot_compressive_equilibrium_preserves_measurement_contract() -> None:
    torch.manual_seed(4)
    video = torch.rand(2, 3, 5, 4)
    masks = torch.rand(2, 3, 5, 4).clamp_min(0.2)
    measurement = SILVASnapshotCompressiveEquilibrium.measure(video, masks).requires_grad_()
    model = SILVASnapshotCompressiveEquilibrium(
        3, prior=nn.Identity(), prior_scale=0.0, config=_config()
    )
    reconstruction = model(measurement, masks)
    assert reconstruction.shape == video.shape
    assert torch.allclose(model.measure(reconstruction, masks), measurement, atol=2e-5)
    _finite_backward(reconstruction)


def test_magnetic_particle_equilibrium_packs_admm_state_and_gradients() -> None:
    torch.manual_seed(5)
    matrix = torch.randn(4, 6) / 3.0
    measurement = torch.randn(3, 4, requires_grad=True)
    model = SILVAMagneticParticleEquilibrium(6, 4, mixing=0.35, config=_config(max_iter=8))
    result = model(measurement, matrix, return_result=True)
    assert result.state.shape == (3, 26)
    assert result.output.shape == (3, 6)
    _finite_backward(result.output)
    assert measurement.grad is not None


def test_sparse_hyperspectral_equilibrium_outputs_cube_and_code() -> None:
    torch.manual_seed(6)
    noisy = torch.randn(2, 4, 5, 5, requires_grad=True)
    model = SILVASparseHyperspectralEquilibrium(
        4, 6, step_size=0.1, prior_scale=0.01, config=_config(max_iter=8)
    )
    result = model(noisy, return_result=True)
    assert result.state.shape == (2, 6, 5, 5)
    assert result.output.shape == noisy.shape
    _finite_backward(result.output)


def test_serialized_smoothing_returns_samples_counts_and_radii() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 3)
    model = SILVASerializedSmoothingEquilibrium(3, 5, 3, sigma=0.2, config=_config())
    predictions, records = model.sample_predictions(inputs, samples=12, seed=9)
    certificate = model.certify(inputs, samples=24, seed=9)
    assert predictions.shape == (12, 2)
    assert len(records) == 12
    assert certificate.counts.shape == (2, 3)
    assert torch.equal(certificate.counts.sum(dim=-1), torch.full((2,), 24))
    assert torch.all(certificate.radius >= 0)


def test_diffusion_restoration_enforces_observed_pixels_across_trajectory() -> None:
    torch.manual_seed(8)
    measurement = torch.randn(2, 1, 5, 5, requires_grad=True)
    mask = torch.zeros_like(measurement)
    mask[..., ::2, ::2] = 1.0
    initial_noise = torch.randn_like(measurement)
    model = SILVADiffusionRestorationEquilibrium(1, 4, eta=0.1, config=_config())
    result = model(
        measurement, mask=mask, initial_noise=initial_noise, return_result=True
    )
    assert result.state.shape == (2, 4, 1, 5, 5)
    observed = result.state[:, 1:] * mask.unsqueeze(1)
    expected = measurement.unsqueeze(1) * mask.unsqueeze(1)
    assert torch.allclose(observed, expected, atol=1e-6)
    _finite_backward(result.output)


def test_recurrent_equilibrium_network_returns_all_trajectories() -> None:
    torch.manual_seed(9)
    inputs = torch.randn(2, 5, 3, requires_grad=True)
    model = SILVARecurrentEquilibriumNetwork(3, 4, 6, 2, config=_config())
    result = model(inputs)
    assert result.output.shape == (2, 5, 2)
    assert result.state.shape == (2, 5, 4)
    assert result.equilibrium.shape == (2, 5, 6)
    assert len(result.solver_results) == 5
    _finite_backward(result.output)


@pytest.mark.parametrize("parameterization", ["lben", "orthogonal", "sandwich", "cpl"])
def test_lipschitz_robust_parameterizations_bound_state_map(parameterization: str) -> None:
    torch.manual_seed(10)
    inputs = torch.randn(3, 4, requires_grad=True)
    model = SILVALipschitzRobustEquilibrium(
        4,
        6,
        3,
        parameterization=parameterization,
        recurrent_bound=0.6,
        config=_config(),
    )
    result = model(inputs, return_result=True)
    spectral = torch.linalg.matrix_norm(model.recurrent_weight(), ord=2)
    if parameterization != "lben":
        assert spectral <= 0.6001
    assert result.output.shape == (3, 3)
    assert torch.all(result.certified_radius >= 0)
    _finite_backward(result.output)


def test_image_matting_equilibrium_holds_known_trimap_regions() -> None:
    torch.manual_seed(11)
    image = torch.rand(2, 3, 6, 6, requires_grad=True)
    trimap = torch.full((2, 1, 6, 6), 0.5)
    trimap[..., :2, :] = 0.0
    trimap[..., -2:, :] = 1.0
    model = SILVAImageMattingEquilibrium(hidden_channels=5, config=_config())
    alpha = model(image, trimap)
    assert alpha.shape == trimap.shape
    assert torch.equal(alpha[..., :2, :], torch.zeros_like(alpha[..., :2, :]))
    assert torch.equal(alpha[..., -2:, :], torch.ones_like(alpha[..., -2:, :]))
    assert torch.all((alpha >= 0) & (alpha <= 1))
    _finite_backward(alpha)


def test_dynamic_economic_equilibrium_is_resource_feasible_and_differentiable() -> None:
    torch.manual_seed(12)
    states = torch.tensor([[1.0, 0.0], [1.5, 0.1], [0.8, -0.1]], requires_grad=True)
    model = SILVADynamicEconomicEquilibrium(state_dim=2, hidden_dim=12)
    result = model(states)
    assert torch.all(result.consumption > 0)
    assert torch.all(result.next_capital > 0)
    assert torch.allclose(result.resource_residual, torch.zeros(3), atol=2e-6)
    loss = result.euler_residual.square().mean()
    loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
