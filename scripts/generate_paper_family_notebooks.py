from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "notebooks/package_api", ROOT / "docs/package-notebooks", ROOT / "colab"]
_CELL_COUNTER = 0
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)")


def _next_cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:04d}"


def md(source: str, *, prefix: str) -> dict:
    source = _INLINE_MATH_RE.sub(r"$\1$", textwrap.dedent(source).strip())
    return {
        "cell_type": "markdown",
        "id": _next_cell_id(prefix),
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source: str, *, prefix: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_cell_id(prefix),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = """
from pathlib import Path
import importlib.util
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/jseluis/silva-networks.git"

def find_local_silva_root():
    candidates = [Path.cwd(), Path("/content/silva-networks"), Path("/content/drive/MyDrive/silva-networks")]
    root = Path.cwd()
    while root != root.parent:
        candidates.append(root)
        root = root.parent
    for candidate in candidates:
        if (candidate / "src" / "silva_networks").exists():
            return candidate
    return None

root = find_local_silva_root()
if root is not None:
    sys.path.insert(0, str(root / "src"))
elif IN_COLAB and importlib.util.find_spec("silva_networks") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{REPO_URL}"])
    root = Path.cwd()
else:
    root = Path.cwd()
"""


def architecture_notebook() -> dict:
    prefix = "families"
    return notebook(
        [
            md(
                r"""
                # Paper Families as Configurable SILVA Architectures

                This notebook checks the package mechanisms used to construct DEQ,
                MDEQ, Jacobian-regularized, IGNN, implicit-representation, and
                diffusion cases. It is a CPU smoke tutorial, not a claim that the
                source papers' datasets, schedules, checkpoints, or metrics were run.

                All cases share

                $$z^\star=f_\theta(z^\star,x).$$

                The user selects the state, transition operators, solver, gradient
                estimator, dimensions, data, optimizer, and evaluation protocol.
                """,
                prefix=prefix,
            ),
            code(BOOTSTRAP, prefix=prefix),
            code(
                """
                import torch
                from torch import nn

                from silva_networks import (
                    SILVADEQEngine,
                    SILVADiffusionEquilibrium,
                    SILVAImplicitGraphNetwork,
                    SILVAImplicitNeuralRepresentation,
                    SILVAMultiscaleClassifier,
                    SILVAMultiscaleSegmenter,
                    SILVASequenceDEQ,
                    SolverConfig,
                    jacobian_regularization_loss,
                )

                torch.manual_seed(12)
                fast = SolverConfig(solver="picard", max_iter=3, alpha=0.5)
                exact = SolverConfig(
                    solver="picard",
                    max_iter=3,
                    alpha=0.5,
                    backward_mode="implicit",
                    backward_solver="gmres",
                    backward_max_iter=8,
                    backward_tol=1e-5,
                    backward_stop_mode="relative",
                )
                """,
                prefix=prefix,
            ),
            md(
                r"""
                ## Sequence DEQ

                The transition can be relative-attention Transformer or causal
                trellis. Memory, local attention, banded adaptive input and output,
                weight/projection tying, dropout, and every solver option are
                constructor parameters. Custom transition, embedding, and readout
                modules are supported as well.
                """,
                prefix=prefix,
            ),
            code(
                """
                sequence = SILVASequenceDEQ(
                    8,
                    vocab_size=24,
                    heads=2,
                    inner_dim=16,
                    memory_length=3,
                    local_window=4,
                    adaptive_cutoffs=(8, 16),
                    adaptive_div_value=2.0,
                    embedding_dim=8,
                    config=exact,
                )
                seq = sequence(torch.randint(0, 24, (2, 5)), return_result=True)
                targets = torch.randint(0, 24, (2, 5))
                sequence.adaptive_loss(seq.state, targets).backward()
                print(seq.output.shape, seq.memory.shape, seq.solver_result.solver)
                print("normalized probabilities", seq.output.exp().sum(dim=-1))

                trellis = SILVASequenceDEQ(
                    8,
                    input_dim=5,
                    output_dim=3,
                    mode="trellis",
                    tie_embeddings=False,
                    config=fast,
                )
                print("trellis", trellis(torch.randn(2, 5, 5)).shape)
                """,
                prefix=prefix,
            ),
            md(
                r"""
                ## Multiscale DEQ and Jacobian Regularization

                Every resolution is part of one coupled state. Every-to-every
                projections and resampling implement cross-scale interaction. The
                built-in MDEQ mode uses learned downsampling chains and projected
                upsampling; stimulus injection, convolution weight normalization,
                and classification/segmentation heads remain selectable.

                A paper-selected Jacobian term estimates
                $\|J_f(z^\star)\|_F^2$ with Hutchinson probes.
                """,
                prefix=prefix,
            ),
            code(
                """
                image = torch.randn(1, 3, 8, 8)
                classifier = SILVAMultiscaleClassifier(
                    3,
                    (4, 6),
                    3,
                    expansion=1.0,
                    groups=2,
                    weight_norm=True,
                    fusion_mode="mdeq",
                    injection_mode="highest",
                    config=fast,
                )
                segmenter = SILVAMultiscaleSegmenter(
                    3, (4, 6), 2, expansion=1.0, groups=2, config=fast
                )
                cls = classifier(image, return_result=True)
                print("classification", cls.output.shape, [z.shape for z in cls.states])
                print("segmentation", segmenter(image).shape)

                matrix = nn.Parameter(0.2 * torch.eye(4))
                state = torch.randn(2, 4)
                transition = lambda z: torch.tanh(z @ matrix)
                penalty = jacobian_regularization_loss(transition, state, samples=2, weight=0.01)
                print("Jacobian penalty", float(penalty.detach()))
                """,
                prefix=prefix,
            ),
            md(
                """
                ## IGNN, Implicit Representations, and DEQ-DDIM

                These applications change the domain-specific transition while
                retaining the same equilibrium and gradient contracts.
                """,
                prefix=prefix,
            ),
            code(
                """
                edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
                graph = SILVAImplicitGraphNetwork(3, 5, 2, config=fast)
                graph.project_recurrent_norm(0.9)
                print("IGNN", graph(torch.randn(4, 3), edges).shape)

                coordinates = torch.rand(1, 7, 2, requires_grad=True)
                inr = SILVAImplicitNeuralRepresentation(
                    2, 6, 1, injection="fourier", activation="tanh", config=fast
                )
                print("INR", inr(coordinates).shape, inr.coordinate_gradient(coordinates).shape)

                class ZeroDenoiser(nn.Module):
                    def forward(self, x, timestep):
                        del timestep
                        return torch.zeros_like(x)

                ddim = SILVADiffusionEquilibrium(
                    ZeroDenoiser(),
                    torch.linspace(0.99, 0.5, 10),
                    (9, 6, 3, 0, -1),
                    config=SolverConfig(solver="picard", max_iter=6, alpha=1.0),
                )
                diffusion = ddim(torch.randn(1, 1, 3, 3), return_result=True)
                print("DDIM", diffusion.output.shape, diffusion.trajectory.shape)
                """,
                prefix=prefix,
            ),
            md(
                r"""
                ## Beyond the Referenced Cases

                `SILVADEQEngine` accepts a user transition over one tensor or a
                nested tuple/list state. This is the extension point for a new
                architecture; no package family registration is required.
                """,
                prefix=prefix,
            ),
            code(
                """
                stimulus = torch.randn(2, 4)
                custom_transition = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 4))
                engine = SILVADEQEngine(fast)
                custom = engine(lambda z: 0.2 * custom_transition(z) + stimulus, torch.zeros_like(stimulus))
                print("custom equilibrium", custom.shape)
                """,
                prefix=prefix,
            ),
            md(
                """
                ## Reproduction Boundary

                The successful cells establish executable architecture, solver, and
                gradient mechanisms. Reproducing a paper's reported numbers still
                requires its exact data, preprocessing, training schedule, evaluation,
                random-seed policy, and pretrained components.

                Cite the SILVA package and the source paper whose architecture or
                experimental protocol you instantiate.
                """,
                prefix=prefix,
            ),
        ]
    )


def raft_notebook() -> dict:
    prefix = "raft"
    return notebook(
        [
            md(
                r"""
                # Coupled RAFT and DEQ-Flow in SILVA

                The equilibrium state is $(h,u)$:

                $$
                h^+=\operatorname{ConvGRU}(h,c,m(u,C(u))),\qquad
                u^+=u+\Delta_\theta(h^+).
                $$

                This package-native case exposes residual-encoder stages and stride,
                correlation pyramid levels and radius, motion/GRU widths, global
                aggregation, solver and gradient rules, sparse correction indices,
                learned convex upsampling, fixed-point reuse, and custom encoder or
                update modules.
                """,
                prefix=prefix,
            ),
            code(BOOTSTRAP, prefix=prefix),
            code(
                """
                import torch

                from silva_networks import (
                    SILVARAFTDEQ,
                    SolverConfig,
                    make_silva_translation_flow_batch,
                    silva_flow_fixed_point_correction_loss,
                )

                torch.manual_seed(13)
                batch = make_silva_translation_flow_batch(
                    batch_size=1, channels=1, height=8, width=8, shift=(1.0, 0.0)
                )
                config = SolverConfig(
                    solver="picard",
                    max_iter=3,
                    alpha=0.5,
                    indexing=(1, 2),
                    backward_mode="implicit",
                    backward_solver="gmres",
                    backward_max_iter=8,
                )
                model = SILVARAFTDEQ(
                    in_channels=1,
                    feature_dim=8,
                    hidden_dim=4,
                    context_dim=4,
                    encoder_channels=(4,),
                    encoder_residual_blocks=1,
                    encoder_dropout=0.0,
                    output_stride=2,
                    corr_levels=2,
                    corr_radius=1,
                    motion_dim=8,
                    flow_head_dim=8,
                    gru_kernel_size=3,
                    correlation_hidden_dims=(8, 8),
                    flow_hidden_dims=(8, 4),
                    correction_steps=1,
                    config=config,
                )
                """,
                prefix=prefix,
            ),
            md(
                """
                ## Solve, Sparse Corrections, and Exact Gradient

                `indexing` stores selected numerical states. Short differentiable
                corrections turn them into auxiliary flow predictions without
                retaining the entire forward solver graph.
                """,
                prefix=prefix,
            ),
            code(
                """
                result = model(batch.image1, batch.image2, return_result=True)
                predictions = result.flow_sequence or [result.flow]
                loss = silva_flow_fixed_point_correction_loss(
                    predictions, batch.flow, valid=batch.valid, gamma=0.8
                )
                loss.backward()
                finite_gradients = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
                print("flow", result.flow.shape)
                print("low-resolution state", result.low_resolution_flow.shape)
                print("correction predictions", len(predictions))
                print("finite gradients", bool(finite_gradients))
                """,
                prefix=prefix,
            ),
            md(
                """
                ## Reuse the Fixed Point

                A previous hidden/flow equilibrium can initialize a related image
                pair. Whether this is appropriate across frames or augmentations is
                an experiment choice.
                """,
                prefix=prefix,
            ),
            code(
                """
                reused = model(batch.image1, batch.image2, cached_state=result.cached_state)
                print("reused flow", reused.shape)
                """,
                prefix=prefix,
            ),
            md(
                """
                ## Reproduction Boundary and Citations

                This smoke run validates the coupled state, correlation/GRU update,
                learned upsampling, correction loss, implicit backward path, and
                reuse contract. Paper metrics require the source dataset mixtures,
                augmentations, schedules, evaluation code, resolution, and model
                dimensions.

                Cite RAFT for all-pairs correlation and recurrent refinement,
                DEQ-Flow for the equilibrium optical-flow formulation and sparse
                correction/reuse strategy, and SILVA for this generalized package API.
                """,
                prefix=prefix,
            ),
        ]
    )


def main() -> None:
    generated = {
        "12_paper_family_architectures.ipynb": architecture_notebook(),
        "13_raft_deq_flow.ipynb": raft_notebook(),
    }
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in generated.items():
            (directory / name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
            print(directory / name)


if __name__ == "__main__":
    main()
