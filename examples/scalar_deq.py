from __future__ import annotations

import torch

from silva_networks import SolverConfig, fixed_point, full_jacobian, stability_report


def main() -> None:
    torch.manual_seed(7)
    a = torch.tensor(0.55)
    b = torch.tensor(1.0)

    def f(z: torch.Tensor) -> torch.Tensor:
        return a * z + b

    result = fixed_point(f, torch.zeros(()), SolverConfig(max_iter=80, alpha=0.7, tol=1e-9))
    closed_form = b / (1.0 - a)
    report = stability_report(f, result.z, samples=2, iters=10)
    J = full_jacobian(f, result.z)

    print("z_star", float(result.z))
    print("closed_form", float(closed_form))
    print("final_residual", result.residual)
    print("jacobian", J.reshape(-1).tolist())
    print("spectral_radius", report.spectral_radius)


if __name__ == "__main__":
    main()

