"""Run the Bayesian, joint-inference, dynamic, and certified SILVA families."""

import torch

from silva_networks import (
    SILVABayesianDEQ,
    SILVACertifiedEquilibrium,
    SILVAImplicitSpatiotemporalEquilibrium,
    SILVAJointInferenceEquilibrium,
    SILVAPeriodicDiffusion1D,
    SolverConfig,
)


def main() -> None:
    torch.manual_seed(610)
    config = SolverConfig(max_iter=40, tol=1e-7, backward_mode="unrolled")

    bayesian = SILVABayesianDEQ(3, 6, 2, posterior_samples=3, config=config)
    bayesian_result = bayesian(torch.randn(4, 3), seed=11, return_result=True)
    print("bayesian variance", float(bayesian_result.predictive_variance.mean().detach()))

    joint = SILVAJointInferenceEquilibrium(4, 6, 3, 2, config=config)
    joint_result = joint(torch.randn(4, 4), return_result=True)
    print("joint residual", joint_result.solver_result.residual)

    dynamics = SILVAImplicitSpatiotemporalEquilibrium(
        known_dynamics=SILVAPeriodicDiffusion1D(0.1),
        dt=0.2,
        steps=4,
        config=config,
    )
    dynamic_result = dynamics(torch.randn(3, 24), return_result=True)
    print("trajectory", tuple(dynamic_result.trajectory.shape))

    certified = SILVACertifiedEquilibrium(2, 6, 3, config=config)
    inputs = torch.randn(4, 2)
    logits = certified(inputs)
    certificate = certified.certify(inputs, 0.02, logits.argmax(dim=-1))
    print("certified examples", int(certificate.certified.sum()))


if __name__ == "__main__":
    main()
