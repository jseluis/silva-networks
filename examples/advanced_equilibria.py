"""Run compact advanced equilibrium and physics-informed SILVA mechanisms."""

from __future__ import annotations

import torch

from silva_networks import (
    SILVABurgMirrorTransition,
    SILVAGenerativeEquilibriumTransformer,
    SILVAImplicitDAEStep,
    SILVAMonotoneGraphEquilibrium,
    SILVAPhysicsInformedEquilibrium,
    SILVAPoissonMirrorEquilibrium,
    SILVAResidualDiscriminator,
    SolverConfig,
    make_linear_dae_dataset,
    make_linear_ivp_dataset,
    make_monotone_chain_dataset,
    make_poisson_inverse_dataset,
    make_teacher_image_pairs,
    silva_adversarial_residual_loss,
    silva_distillation_loss,
)


def main() -> None:
    torch.manual_seed(25)

    chain = make_monotone_chain_dataset(nodes=8, seed=25)
    graph = SILVAMonotoneGraphEquilibrium(
        1,
        4,
        1,
        config=SolverConfig(solver="picard", max_iter=12, tol=1e-5),
    )
    graph_result = graph(chain.source, chain.edge_index, return_result=True)
    print("monotone graph:", tuple(graph_result.output.shape), graph_result.solver_result.residual)

    teacher = make_teacher_image_pairs(samples=2, height=4, width=4, seed=25)
    transformer = SILVAGenerativeEquilibriumTransformer(
        in_channels=1,
        patch_size=2,
        hidden_dim=8,
        heads=2,
        equilibrium_depth=1,
        config=SolverConfig(solver="picard", max_iter=8, tol=1e-5, anderson_batch_dims=1),
    )
    generated = transformer(teacher.noise, return_result=True)
    print(
        "equilibrium transformer:", float(silva_distillation_loss(generated.output, teacher.target))
    )

    poisson = make_poisson_inverse_dataset(samples=2, height=4, width=4, seed=25)
    mirror = SILVAPoissonMirrorEquilibrium(
        transition=SILVABurgMirrorTransition(
            forward_operator=poisson.forward_operator,
            adjoint_operator=poisson.adjoint_operator,
            step_size=0.05,
        ),
        config=SolverConfig(max_iter=8, tol=1e-5, anderson_batch_dims=1),
    )
    reconstruction = mirror(poisson.observation, return_result=True)
    print("Poisson mirror:", float(poisson.data_fidelity(reconstruction.output)))

    ivp = make_linear_ivp_dataset(points=5, rate=-0.5)
    physics_model = SILVAPhysicsInformedEquilibrium(
        3,
        1,
        config=SolverConfig(
            solver="picard",
            max_iter=8,
            tol=1e-5,
            backward_mode="implicit",
            anderson_batch_dims=1,
        ),
    )
    physics = physics_model.physics_loss(
        ivp.times,
        ivp.dynamics,
        initial_time=ivp.times[:1],
        initial_state=ivp.initial_state,
        jacobian_weight=0.01,
    )
    print("physics-informed loss:", float(physics.total))

    dae = make_linear_dae_dataset(steps=2, step_size=0.1)
    dae_result = SILVAImplicitDAEStep()(
        dae.differential[:1],
        dae.algebraic[:1],
        dae.step_size,
        dae.dynamics,
        dae.constraint,
    )
    print("implicit DAE step:", dae_result.differential.flatten().tolist(), dae_result.residual)

    discriminator = SILVAResidualDiscriminator(1, hidden_dim=8, depth=1)
    residual_losses = silva_adversarial_residual_loss(
        discriminator,
        physics.time_derivative - ivp.dynamics(ivp.times, physics.prediction),
    )
    print(
        "adversarial residual objective:",
        float(residual_losses.generator),
        float(residual_losses.discriminator),
    )


if __name__ == "__main__":
    main()
