from __future__ import annotations

import torch

from silva_networks import (
    DEQMLPTransition,
    ExplicitEulerODEBlock,
    QuadraticOptimizationLayer,
    SILVAEulerFlowBlock,
    SILVAFixedPointBlock,
    SILVAFixedPointClassifier,
    SILVAImplicitTransition,
    SILVAMultiscaleDEQBlock,
    SILVAQuadraticOptimizationLayer,
    SolverConfig,
    TanhFixedPointBlock,
    TanhFixedPointClassifier,
    ToyMultiscaleDEQBlock,
    fixed_point,
    full_jacobian,
    implicit_adjoint_solve,
    jacobian_regularization_loss,
    jvp,
    residual_ratio,
    silva_euler_flow_block,
    silva_fixed_point_block,
    silva_fixed_point_classifier,
    silva_implicit_transition,
    silva_jacobian_regularization_loss,
    silva_multiscale_deq_block,
    silva_quadratic_optimization_layer,
    silva_residual_ratio,
    vjp,
)


def test_tanh_fixed_point_block_supports_anderson_and_gradients() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    block = TanhFixedPointBlock(
        3,
        4,
        config=SolverConfig(solver="anderson", max_iter=5, alpha=0.7, history=3),
    )

    result = block(x, return_result=True)
    loss = result.z.pow(2).mean()
    loss.backward()

    assert result.z.shape == (5, 4)
    assert result.iterations >= 1
    assert block.transition.input_proj.weight.grad is not None
    assert residual_ratio(result.residuals) == residual_ratio(result.residuals)


def test_silva_named_factories_construct_public_bridge_modules() -> None:
    torch.manual_seed(10)
    transition = silva_implicit_transition(3, 4, spectral_scale=0.5)
    block = silva_fixed_point_block(3, 4, config=SolverConfig(max_iter=3, alpha=0.6))
    classifier = silva_fixed_point_classifier(3, 4, 2, config=SolverConfig(max_iter=3, alpha=0.6))
    ode = silva_euler_flow_block(4, steps=2, step_size=0.05)
    opt = silva_quadratic_optimization_layer(3, 2, config=SolverConfig(max_iter=3), reengage=False)
    mdeq = silva_multiscale_deq_block(3, 2, 2, config=SolverConfig(max_iter=3))

    x = torch.randn(5, 3)
    z = block(x)
    logits = classifier(x)
    ode_out = ode(torch.randn(5, 4))
    opt_out = opt(x)
    mdeq_out = mdeq(x, return_result=True)
    penalty = silva_jacobian_regularization_loss(lambda state: mdeq.transition(state, x), mdeq_out.z)

    assert isinstance(transition, SILVAImplicitTransition)
    assert isinstance(block, SILVAFixedPointBlock)
    assert isinstance(classifier, SILVAFixedPointClassifier)
    assert isinstance(ode, SILVAEulerFlowBlock)
    assert isinstance(opt, SILVAQuadraticOptimizationLayer)
    assert isinstance(mdeq, SILVAMultiscaleDEQBlock)
    assert z.shape == (5, 4)
    assert logits.shape == (5, 2)
    assert ode_out.shape == (5, 4)
    assert opt_out.shape == (5, 2)
    assert mdeq_out.z.shape == (5, 4)
    assert penalty.ndim == 0
    assert silva_residual_ratio(mdeq_out.residuals) == silva_residual_ratio(mdeq_out.residuals)


def test_tanh_fixed_point_classifier_trains_like_a_torch_module() -> None:
    torch.manual_seed(1)
    x = torch.randn(6, 1, 2, 3)
    y = torch.tensor([0, 1, 2, 1, 0, 2])
    model = TanhFixedPointClassifier(
        in_features=6,
        state_dim=5,
        num_classes=3,
        config=SolverConfig(max_iter=4, alpha=0.6),
    )

    result = model(x, return_result=True)
    loss = torch.nn.functional.cross_entropy(result.output, y)
    loss.backward()

    assert result.output.shape == (6, 3)
    assert result.state.shape == (6, 5)
    assert result.solver_result is not None
    assert model.readout.weight.grad is not None
    assert model.block.transition.state_proj.weight.grad is not None


def test_jacobian_materialized_vjp_jvp_and_adjoint_agree_on_small_deq() -> None:
    torch.manual_seed(2)
    transition = DEQMLPTransition(2, 2, spectral_scale=0.4)
    x = torch.randn(1, 2)
    z0 = torch.zeros(1, 2)

    def f(z: torch.Tensor) -> torch.Tensor:
        return transition(z, x)

    result = fixed_point(f, z0, SolverConfig(max_iter=12, alpha=0.7))
    J = full_jacobian(f, result.z)
    probe = torch.randn_like(result.z)
    _, jvp_value = jvp(f, result.z, probe)
    vjp_value = vjp(f, result.z, probe)
    adjoint = implicit_adjoint_solve(f, result.z, probe, max_iter=4, tol=1e-5)

    assert J.shape == (2, 2)
    assert torch.allclose(J @ probe.reshape(-1), jvp_value.reshape(-1), atol=1e-5)
    assert torch.allclose(J.T @ probe.reshape(-1), vjp_value.reshape(-1), atol=1e-5)
    assert adjoint.x.shape == result.z.shape


def test_quadratic_optimization_layer_matches_closed_form() -> None:
    torch.manual_seed(3)
    x = torch.randn(4, 3)
    layer = QuadraticOptimizationLayer(
        3,
        2,
        ridge=1.5,
        step_size=0.2,
        config=SolverConfig(max_iter=80, alpha=1.0, tol=1e-7),
        reengage=False,
    )

    iterative = layer(x)
    exact = layer.exact_solution(x)
    energy_before = layer.energy(torch.zeros_like(iterative), x).mean()
    energy_after = layer.energy(iterative, x).mean()

    assert iterative.shape == exact.shape
    assert torch.allclose(iterative, exact, atol=1e-4, rtol=1e-4)
    assert energy_after <= energy_before


def test_explicit_euler_ode_block_and_multiscale_deq_are_differentiable() -> None:
    torch.manual_seed(4)
    x = torch.randn(3, 4)
    ode = ExplicitEulerODEBlock(4, hidden_dim=6, steps=3, step_size=0.05)
    terminal, trajectory = ode(x, return_trajectory=True)
    loss = terminal.square().mean()
    loss.backward()

    assert terminal.shape == x.shape
    assert trajectory.shape == (4, 3, 4)
    assert next(ode.parameters()).grad is not None

    deq = ToyMultiscaleDEQBlock(
        4,
        low_dim=3,
        high_dim=5,
        config=SolverConfig(max_iter=4, alpha=0.6),
    )
    result = deq(x, return_result=True)
    penalty = jacobian_regularization_loss(lambda z: deq.transition(z, x), result.z, samples=1)
    total = result.z.square().mean() + 0.01 * penalty
    total.backward()

    low, high = deq.split_state(result.z)
    assert result.z.shape == (3, 8)
    assert low.shape == (3, 3)
    assert high.shape == (3, 5)
    assert deq.input_low.weight.grad is not None


def test_implicit_bridge_selected_device_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, device=device)
    model = TanhFixedPointClassifier(
        in_features=3,
        state_dim=6,
        num_classes=2,
        config=SolverConfig(max_iter=3, alpha=0.6),
    ).to(device)

    out = model(x)
    loss = out.square().mean()
    loss.backward()

    assert out.device.type == device.type
    assert model.readout.weight.grad is not None
