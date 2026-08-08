"""Run six structured SILVA families on attributed compact source snapshots."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from silva_networks import (
    SILVADeltaOperator,
    SILVAEfficientInfiniteGraphEquilibrium,
    SILVAMonotoneOperatorEquilibrium,
    SILVAMultiscaleGraphImplicitNetwork,
    SILVANonEuclideanEquilibrium,
    SILVAPositiveConcaveEquilibrium,
    SolverConfig,
    load_source_snapshot,
    normalized_graph_operator,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "docs" / "assets" / "source-data"


def _config(max_iter: int = 80) -> SolverConfig:
    return SolverConfig(
        solver="picard",
        max_iter=max_iter,
        tol=1e-6,
        backward_mode="unrolled",
        anderson_batch_dims=1,
    )


def run_source_subset_suite() -> dict[str, float]:
    """Execute one forward/backward or equivalence check for every family."""

    torch.manual_seed(91)
    vision = load_source_snapshot(SNAPSHOT_ROOT / "cifar10-balanced-10.pt")
    graph = load_source_snapshot(SNAPSHOT_ROOT / "cora-induced-96.pt")
    motion = load_source_snapshot(
        SNAPSHOT_ROOT / "public-motion-frames-100-101.pt"
    )

    images = vision.tensors["images"]
    labels = vision.tensors["labels"]
    vectors = images.flatten(1)

    monotone = SILVAMonotoneOperatorEquilibrium(
        vectors.shape[1],
        24,
        10,
        step_size=0.5,
        config=_config(),
    )
    monotone_result = monotone(vectors, return_result=True)
    monotone_loss = F.cross_entropy(monotone_result.output, labels)
    monotone_loss.backward()

    positive = SILVAPositiveConcaveEquilibrium(
        3,
        8,
        10,
        operator="conv2d",
        config=_config(),
    )
    positive_result = positive(images, return_result=True)
    positive_logits = positive_result.output.mean(dim=(-2, -1))
    positive_loss = F.cross_entropy(positive_logits, labels)
    positive_loss.backward()

    non_euclidean = SILVANonEuclideanEquilibrium(
        vectors.shape[1],
        24,
        10,
        config=_config(120),
    )
    robust_result = non_euclidean(vectors, return_result=True)
    perturbed = (vectors + 8.0 / 255.0).clamp(0.0, 1.0)
    perturbation_shift = torch.linalg.vector_norm(
        non_euclidean(perturbed) - robust_result.output
    )
    F.cross_entropy(robust_result.output, labels).backward()

    x = graph.tensors["x"]
    edge_index = graph.tensors["edge_index"]
    y = graph.tensors["y"].long()
    train_mask = graph.tensors["train_mask"].bool()
    operator = normalized_graph_operator(edge_index, x.shape[0]).to(x)

    eignn = SILVAEfficientInfiniteGraphEquilibrium(
        x.shape[1],
        16,
        int(y.max()) + 1,
        gamma=0.7,
        solve_mode="iterative",
        config=_config(),
    )
    eignn_result = eignn(x, operator, return_result=True)
    eignn_loss = F.cross_entropy(eignn_result.output[train_mask], y[train_mask])
    eignn_loss.backward()

    mgnni = SILVAMultiscaleGraphImplicitNetwork(
        x.shape[1],
        16,
        int(y.max()) + 1,
        scales=(1, 2, 4),
        gamma=0.7,
        config=_config(),
    )
    mgnni_result = mgnni(x, operator, return_result=True)
    mgnni_loss = F.cross_entropy(mgnni_result.output[train_mask], y[train_mask])
    mgnni_loss.backward()

    convolution = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    cached = SILVADeltaOperator(convolution, threshold=1e-3)
    cached(motion.tensors["frame1"])
    cached_second = cached(motion.tensors["frame2"])
    exact_second = convolution(motion.tensors["frame2"])
    delta_error = torch.linalg.vector_norm(cached_second - exact_second)

    return {
        "monotone_loss": float(monotone_loss.detach()),
        "monotone_residual": monotone_result.solver_result.residual,
        "positive_loss": float(positive_loss.detach()),
        "positive_minimum_weight": float(positive_result.minimum_weight.detach()),
        "non_euclidean_logit_shift": float(perturbation_shift.detach()),
        "non_euclidean_measure": float(robust_result.one_sided_lipschitz.detach()),
        "eignn_loss": float(eignn_loss.detach()),
        "eignn_residual": eignn_result.solver_result.residual,
        "mgnni_loss": float(mgnni_loss.detach()),
        "mgnni_mean_scale_entropy": float(
            (
                -mgnni_result.attention_weights
                * mgnni_result.attention_weights.clamp_min(1e-12).log()
            )
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "delta_cache_error": float(delta_error.detach()),
        "delta_active_fraction": cached.stats[-1].active_fraction,
    }


if __name__ == "__main__":
    for metric, value in run_source_subset_suite().items():
        print(f"{metric}: {value:.8g}")
