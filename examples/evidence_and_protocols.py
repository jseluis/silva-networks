"""Create a repeated-measurement record and inspect a scale protocol."""

from silva_networks import run_silva_evidence, silva_family_experiment_protocol


def compact_trial(seed: int) -> dict[str, object]:
    return {
        "metrics": {"absolute_error": 0.04 + 0.005 * seed},
        "residual": 1e-7 * (seed + 1),
        "evaluations": 7 + seed,
        "converged": True,
    }


def main() -> None:
    report = run_silva_evidence(
        "silva_implicit_spatiotemporal",
        "analytic diffusion",
        compact_trial,
        seeds=(0, 1, 2),
        configuration={"dt": 0.2, "steps": 4},
        data_receipt={"generator": "periodic diffusion", "samples": 16},
        bootstrap_samples=200,
    )
    print("mean error", report.summaries[0].mean)

    protocol = silva_family_experiment_protocol("im_pindiff")
    for tier in protocol.tiers:
        print(tier.tier, tier.dataset.name, tier.resources.accelerator_count)


if __name__ == "__main__":
    main()
