from __future__ import annotations

import torch

from silva_networks import (
    SILVABayesianDEQ,
    SILVACertifiedEquilibrium,
    SILVAImplicitSpatiotemporalEquilibrium,
    SILVAJointInferenceEquilibrium,
    SILVAPeriodicDiffusion1D,
    SolverConfig,
    canonical_silva_family,
)


def _unrolled_config(*, max_iter: int = 50, tol: float = 1e-8) -> SolverConfig:
    return SolverConfig(
        solver="picard",
        max_iter=max_iter,
        tol=tol,
        anderson_batch_dims=1,
        backward_mode="unrolled",
    )


def test_bayesian_deq_reports_predictive_uncertainty_and_gradients() -> None:
    torch.manual_seed(301)
    model = SILVABayesianDEQ(
        3,
        5,
        2,
        posterior_samples=3,
        config=_unrolled_config(max_iter=40),
    )
    inputs = torch.randn(4, 3, requires_grad=True)

    result = model(inputs, seed=17, return_result=True)
    loss = result.output.square().mean() + 1e-5 * model.kl_divergence()
    loss.backward()

    assert result.state.shape == (4, 5)
    assert result.output.shape == (4, 2)
    assert result.sample_states.shape == (3, 4, 5)
    assert result.sample_outputs.shape == (3, 4, 2)
    assert result.predictive_variance.shape == (4, 2)
    assert torch.all(result.predictive_variance >= 0)
    assert torch.any(result.predictive_variance > 0)
    assert len(result.solver_results) == 3
    assert all(item.residual < 1e-5 for item in result.solver_results)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.transition.state_mean.grad is not None
    assert canonical_silva_family("probabilistic_deq") == "silva_bayesian_deq"


def test_joint_inference_solves_packed_state_and_backpropagates() -> None:
    torch.manual_seed(302)
    model = SILVAJointInferenceEquilibrium(
        observation_dim=4,
        state_dim=6,
        optimized_input_dim=3,
        output_dim=2,
        config=_unrolled_config(max_iter=60),
    )
    observation = torch.randn(5, 4, requires_grad=True)

    result = model(observation, return_result=True)
    result.output.square().mean().backward()

    assert result.state.shape == (5, 6)
    assert result.optimized_input.shape == (5, 3)
    assert result.output.shape == (5, 2)
    assert result.solver_result.z.shape == (5, 9)
    assert result.solver_result.residual < 1e-5
    assert observation.grad is not None and torch.isfinite(observation.grad).all()
    assert model.representation_transition.state.weight.grad is not None
    assert canonical_silva_family("jiio") == "silva_joint_inference_equilibrium"


def test_implicit_spatiotemporal_diffusion_is_stable_and_differentiable() -> None:
    torch.manual_seed(303)
    diffusivity = torch.nn.Parameter(torch.tensor(0.08))

    class LearnedDiffusion(torch.nn.Module):
        def forward(self, state: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
            del context
            laplacian = torch.roll(state, 1, dims=-1) - 2.0 * state + torch.roll(state, -1, dims=-1)
            return diffusivity * laplacian

    model = SILVAImplicitSpatiotemporalEquilibrium(
        known_dynamics=SILVAPeriodicDiffusion1D(diffusivity=0.05),
        learned_dynamics=LearnedDiffusion(),
        dt=0.2,
        theta=1.0,
        steps=5,
        config=_unrolled_config(max_iter=40),
    )
    initial = torch.randn(3, 24, requires_grad=True)

    result = model(initial, return_result=True)
    result.output[:, -1].square().mean().backward()

    assert result.state.shape == (3, 24)
    assert result.trajectory.shape == (3, 6, 24)
    assert len(result.solver_results) == 5
    assert all(item.residual < 1e-5 for item in result.solver_results)
    assert result.state.square().mean() <= initial.detach().square().mean() + 1e-6
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert diffusivity.grad is not None and torch.isfinite(diffusivity.grad)
    assert canonical_silva_family("im_pindiff") == "silva_implicit_spatiotemporal"


def test_certified_equilibrium_bounds_outputs_and_exports_system() -> None:
    torch.manual_seed(304)
    model = SILVACertifiedEquilibrium(
        2,
        5,
        3,
        contraction=0.65,
        config=_unrolled_config(max_iter=100, tol=1e-9),
    )
    inputs = torch.randn(4, 2)
    radius = 0.04

    output, point_result = model(inputs, return_result=True)
    bounds = model.interval_bounds(inputs - radius, inputs + radius)
    labels = output.argmax(dim=-1)
    certificate = model.certify(inputs, radius, labels)
    system = model.semialgebraic_system()

    assert point_result.residual < 1e-6
    assert torch.all(bounds.state_lower <= bounds.state_upper)
    assert torch.all(bounds.output_lower <= output + 1e-6)
    assert torch.all(output <= bounds.output_upper + 1e-6)
    assert certificate.certified.shape == (4,)
    assert certificate.margin.shape == (4,)
    assert system.activation == "relu"
    assert system.state_weight.shape == (5, 5)
    assert float(system.state_weight.abs().sum(dim=-1).max()) <= 0.65 + 1e-6
    assert canonical_silva_family("ibp_mondeq") == "silva_certified_equilibrium"
