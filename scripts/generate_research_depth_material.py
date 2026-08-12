"""Generate additive family dossiers, scale plans, and research-depth labs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from textwrap import dedent

import nbformat
from notebook_generation import write_notebook

from silva_networks import all_silva_experiment_dossiers

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "experiments/reproduction/outputs/compact_comparisons.json"
FAMILY_DOCS = ROOT / "docs/families"
CONFIGS = ROOT / "experiments/reproduction/configs"
CANONICAL_NOTEBOOKS = ROOT / "notebooks/package_api"
DOC_NOTEBOOKS = ROOT / "docs/package-notebooks"
PORTABLE_NOTEBOOKS = ROOT / "colab"

EXTENSION_BLOCK = dedent(
    """

    <!-- silva-extension-path:start -->
    --8<-- "includes/extension/learn.md"
    <!-- silva-extension-path:end -->
    """
).strip()

DOMAIN_EQUATIONS = {
    "SILVA composition": r"z^\star=\sigma\!\left(S_\theta(x)+H_\theta(z^\star)+L_\theta(z^\star;E)+G_\theta(z^\star;b)\right)",
    "core equilibrium": r"z^\star=T_\theta(z^\star;x),\qquad y=Q_\psi(z^\star)",
    "scientific operators": r"u^\star=\sigma\!\left(S_\theta(a)+\mathcal K_\theta[u^\star]+\mathcal C(u^\star)\right)",
    "vision and generation": r"z^\star=T_\theta(z^\star;\mathcal E(x),c),\qquad \widehat y=\mathcal D_\psi(z^\star)",
    "graphs and distributed systems": r"Z^\star=T_\theta(Z^\star;X,A,E,b),\qquad \widehat Y=Q_\psi(Z^\star)",
    "physics and differential systems": r"u^\star=T_\theta(u^\star;c),\qquad \mathcal R_{\mathrm{phys}}(u^\star;c)=0",
    "geometry and distributions": r"\mu^\star=\Phi_\theta(\mu^\star;\nu),\qquad \widehat y=Q_\psi(\mu^\star)",
    "optimization and certified equilibria": r"0\in\mathcal A_\theta(z^\star;x)+\mathcal B(z^\star),\qquad y=Q_\psi(z^\star)",
}


def _remove_margin(source: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(
        line.removeprefix(prefix)
        for line in source.splitlines()
    ).strip()


def _write_text(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(source)


def _next_steps(rows: tuple[tuple[str, str, str], ...]) -> str:
    body = "\n".join(
        f"| {question} | [{label}]({target}) |" for question, label, target in rows
    )
    return (
        "## Where to Go Next\n\n"
        "| Question | Page |\n"
        "| --- | --- |\n"
        f"{body}"
    )


def _bullets(values: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _table_rows(values: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"| `{name}` | `{value}` |" for name, value in values)


def _citations(refs: tuple[int, ...]) -> str:
    return ", ".join(
        f"[[{ref}]](../paper/references.md#ref-{ref}){{ .silva-cite }}" for ref in refs
    )


def _repositories(values: tuple[str, ...]) -> str:
    return "<br>".join(
        f'<a href="{url}" target="_blank" rel="noopener">{url}</a>' for url in values
    )


def _benchmark_lookup() -> dict[str, dict[str, object]]:
    if not RESULTS_PATH.exists():
        return {}
    record = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        result["family"]: {"suite": suite["name"], **result}
        for suite in record["suites"]
        for result in suite["results"]
    }


def _measured_evidence(family: str, results: dict[str, dict[str, object]]) -> str:
    if family not in results:
        return dedent(
            """
            This family is verified through its listed mechanism tests and executed notebook.
            It is not included in a same-task comparison when another family does not share
            its input, state, output, and loss contract. The absence of a comparison row is
            therefore a scope decision, not missing implementation evidence.
            """
        ).strip()
    result = results[family]
    return dedent(
        f"""
        The `{result['suite']}` compact suite ran this family on the same task and data as
        the other compatible families in that suite.

        | Measure | Recorded value |
        | --- | ---: |
        | Initial loss | `{result['initial_loss']:.6g}` |
        | Final loss | `{result['final_loss']:.6g}` |
        | Fractional loss reduction | `{result['loss_reduction']:.3f}` |
        | Residual or final increment norm | `{result['residual']:.6g}` |
        | Iterations or tied increments | `{result['iterations']}` |
        | Parameter count | `{result['parameter_count']}` |
        | Final gradient norm | `{result['gradient_norm']:.6g}` |

        These values are **compact-verified** evidence. They establish finite optimization,
        gradient flow, and diagnostic reporting; they are not a publication ranking.
        """
    ).strip()


def _family_page(dossier, results: dict[str, dict[str, object]]) -> str:
    stages: list[str] = []
    for index, stage in enumerate(dossier.stages, start=1):
        stages.append(
            _remove_margin(
                f"""
                ### {index}. {stage.name}

                **Objective:** {stage.objective}

                Procedure:

                {_bullets(stage.procedure)}

                Acceptance checks:

                {_bullets(stage.acceptance_checks)}

                Evidence target: `{stage.evidence_status}`.
                """,
                16,
            )
        )
    config_path = f"experiments/reproduction/configs/{dossier.family}.json"
    stage_text = "\n\n".join(stages)
    return (
        _remove_margin(
            rf"""
            # `{dossier.family}` Reproduction Dossier

            **{dossier.title}.** This dossier connects the source mechanism to its SILVA
            implementation, compact evidence, replaceable components, and source-scale route.
            Existing tests and notebooks remain the executable authority.

            !!! info "Evidence boundary"
                The mechanism is `{dossier.stages[1].evidence_status}` in the package suite.
                The final source-scale stage remains `planned` until the cited data, complete
                optimization budget, checkpoints, and evaluation protocol have actually run.

            ## Identity and Sources

            | Field | Value |
            | --- | --- |
            | Domain | {dossier.domain} |
            | Task contract | {dossier.task_contract} |
            | Source relation | `{dossier.source_relation}` |
            | References | {_citations(dossier.paper_refs)} |
            | Repositories | {_repositories(dossier.repositories)} |
            | Editable scale plan | `{config_path}` |

            ## Governing Equation

            The domain-level state contract is

            $$
            {DOMAIN_EQUATIONS[dossier.domain]}.
            $$

            The implementation registry specializes it operationally as

            ```text
            {dossier.equation}
            ```

            Define the root residual

            $$
            R_\theta(z;x)=z-T_\theta(z;x).
            $$

            At a regular equilibrium, differentiating
            \(R_\theta(z^\star;x)=0\) gives

            $$
            \frac{{\partial z^\star}}{{\partial x}}
            =
            \left(I-\frac{{\partial T_\theta}}{{\partial z}}\right)^{{-1}}
            \frac{{\partial T_\theta}}{{\partial x}}.
            $$

            This identity explains why the forward residual, the conditioning derivative,
            and the adjoint linear solve must be diagnosed separately from the task metric.

            ## What Is Preserved

            {_bullets(dossier.preserved_mechanisms)}

            ## What Can Be Replaced

            Each item below is an explicit control rather than an undocumented modification:

            {_bullets(dossier.configurable_parts)}

            ## Constructor and Shape Contract

            ```python
            {dossier.family}{dossier.constructor_signature}
            ```

            The transition must preserve the declared equilibrium-state shape even when the
            encoder, branch operators, constraints, solver, and readout are replaced. Test the
            transition by itself before testing the complete root solve.

            ## Progressive Experiment Ladder

            {stage_text}

            ## Data, Access, and Storage

            Candidate datasets:

            {_bullets(dossier.datasets)}

            Authoritative routes:

            {_bullets(dossier.data_sources)}

            Access obligations:

            {_bullets(dossier.data_access)}

            Storage planning:

            {_bullets(dossier.storage_plan)}

            Preprocessing record:

            {_bullets(dossier.preprocessing)}

            ## Metrics and Current Evidence

            Required metrics:

            {_bullets(dossier.metrics)}

            {_measured_evidence(dossier.family, results)}

            Executed notebook paths:

            {_bullets(dossier.notebooks)}

            Mechanism tests:

            {_bullets(dossier.tests)}

            ## Compact Defaults

            | Option | Value |
            | --- | --- |
            {_table_rows(dossier.compact_defaults)}

            ## Full Defaults

            | Option | Value |
            | --- | --- |
            {_table_rows(dossier.full_defaults)}

            Defaults establish a starting budget; the cited source protocol takes precedence
            whenever reproduction is the claim.

            ## Source-Scale Checklist

            {_bullets(dossier.source_scale_steps)}

            Benchmark-specific requirements:

            {_bullets(dossier.benchmark_requirements)}

            Required archived artifacts:

            {_bullets(dossier.required_artifacts)}

            ## Reporting Rule

            Report the achieved evidence status, not the intended one. A compact or subset run
            may validate the implementation and data path, but only a completed cited protocol
            supports a source-scale reproduction statement. Modified operators are valuable
            SILVA extensions when every deviation is named and measured.
            """,
            12,
        )
        + "\n\n"
        + EXTENSION_BLOCK
        + "\n\n"
        + _next_steps(
            (
                ("Where are all family dossiers?", "Family Dossier Index", "index.md"),
                ("How is a custom family assembled?", "Advanced Extension Handbook", "../learn/advanced-extension-handbook.md"),
                ("How are experiment stages represented in the API?", "Research-Depth API", "../api/research_depth.md"),
                ("Which lab inspects every dossier?", "Family Dossier Lab", "../package-notebooks/42_family_reproduction_dossiers.ipynb"),
            )
        )
        + "\n"
    )


def _write_family_material() -> None:
    FAMILY_DOCS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    results = _benchmark_lookup()
    dossiers = all_silva_experiment_dossiers()
    domain_counts = Counter(dossier.domain for dossier in dossiers)
    rows = []
    for dossier in dossiers:
        _write_text(FAMILY_DOCS / f"{dossier.family}.md", _family_page(dossier, results))
        plan = dossier.as_dict()
        plan["schema_version"] = 1
        plan["requested_tier"] = "full"
        plan["achieved_evidence_status"] = "compact-verified"
        plan["source_scale_status"] = "planned"
        _write_text(CONFIGS / f"{dossier.family}.json", json.dumps(plan, indent=2) + "\n")
        rows.append(
            f"| [`{dossier.family}`]({dossier.family}.md) | {dossier.domain} | "
            f"`compact-verified` | `{dossier.stages[-1].evidence_status}` |"
        )
    counts = "\n".join(f"| {domain} | {count} |" for domain, count in sorted(domain_counts.items()))
    index = _remove_margin(
        f"""
        # Family Reproduction Dossiers

        These {len(dossiers)} dossiers expose the complete path from a governing equation to source-scale
        experimentation. Each page records what the implementation preserves, what SILVA makes
        replaceable, which compact checks have run, and what remains before a publication-scale
        reproduction can be claimed.

        ## Coverage

        | Domain | Families |
        | --- | ---: |
        {counts}

        ## Evidence Vocabulary

        | Status | Meaning |
        | --- | --- |
        | `contract-verified` | Equation, tensor shape, constructor, and required metadata are checked. |
        | `compact-verified` | A deterministic mechanism run includes forward, diagnostics, and gradients. |
        | `subset-verified` | The official data path and evaluation code have run on a recorded subset. |
        | `source-scale-reproduced` | The complete cited protocol and metric have run with archived artifacts. |
        | `planned` | The route is specified but has not been reported as completed. |

        ## All Families

        | Family | Domain | Current package evidence | Source-scale stage |
        | --- | --- | --- | --- |
        {chr(10).join(rows)}

        ## How to Use a Dossier

        1. Run the compact stage and keep its result as a regression fixture.
        2. Replace one configurable part at a time and record the deviation.
        3. Validate the official data loader on a small subset before increasing scale.
        4. Archive configuration, data receipt, checkpoint, metrics, runtime, and diagnostics.
        5. Promote the evidence status only after every requirement for that level has run.
        """,
        8,
    )
    _write_text(
        FAMILY_DOCS / "index.md",
        index
        + "\n\n"
        + EXTENSION_BLOCK
        + "\n\n"
        + _next_steps(
            (
                ("How do I build a new family?", "Advanced Extension Handbook", "../learn/advanced-extension-handbook.md"),
                ("How do compatible families compare?", "Cross-Family Comparisons", "../experiments/cross-family-comparisons.md"),
                ("Which objects expose these contracts?", "Research-Depth API", "../api/research_depth.md"),
                ("Where is the executable dossier audit?", "Family Dossier Lab", "../package-notebooks/42_family_reproduction_dossiers.ipynb"),
            )
        )
        + "\n",
    )


def _notebook(title: str, introduction: str, cells: list[dict]) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(f"# {title}\n\n{introduction}"),
        *cells,
        nbformat.v4.new_markdown_cell(
            "## Interpretation and Scale Boundary\n\n"
            "The stored outputs are compact evidence. Preserve the configuration and result "
            "record when changing scale; publication-scale claims require the cited data, "
            "training budget, checkpoints, and evaluation protocol."
        ),
    ]
    return notebook


def _code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def _markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def _comparison_notebook(suite: str, function: str, title: str) -> nbformat.NotebookNode:
    return _notebook(
        title,
        "This lab runs compatible SILVA families on one shared deterministic task. It measures "
        "optimization and equilibrium diagnostics without presenting the compact task as a ranking.",
        [
            _markdown(
                r"""
                ## Common Objective

                For family $m$, the compact task minimizes

                $$
                \mathcal L_m(\theta_m)=\frac{1}{N}\sum_{i=1}^{N}
                \left\|Q_m(z_{m,i}^\star)-y_i\right\|_2^2,
                \qquad z_{m,i}^\star=T_m(z_{m,i}^\star;x_i).
                $$

                The input, target, seed, number of optimizer steps, and reported metric are shared.
                Family-specific well-posedness constraints remain active.
                """
            ),
            _code(
                f"""
                from silva_networks import {function}

                suite = {function}()
                print("suite:", suite.name)
                print("task:", suite.task)
                print("metric:", suite.metric)
                for result in suite.results:
                    print(
                        result.family,
                        "initial=", f"{{result.initial_loss:.6f}}",
                        "final=", f"{{result.final_loss:.6f}}",
                        "residual=", f"{{result.residual:.6g}}",
                        "parameters=", result.parameter_count,
                    )
                """
            ),
            _markdown(
                """
                ## Read the Result

                Loss reduction checks that the complete forward/backward path is useful on the
                shared task. Residual and iteration columns diagnose numerical work. Parameter and
                runtime columns describe this compact configuration only.
                """
            ),
            _code(
                """
                import matplotlib.pyplot as plt

                plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
                labels = [result.family.replace("silva_", "") for result in suite.results]
                initial = [result.initial_loss for result in suite.results]
                final = [result.final_loss for result in suite.results]
                residual = [max(result.residual, 1e-12) for result in suite.results]

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
                positions = range(len(labels))
                axes[0].bar([p - 0.18 for p in positions], initial, width=0.36, label="initial")
                axes[0].bar([p + 0.18 for p in positions], final, width=0.36, label="final")
                axes[0].set_xticks(list(positions), labels, rotation=35, ha="right")
                axes[0].set_ylabel("mean squared error")
                axes[0].set_title("Shared-task optimization")
                axes[0].legend()
                axes[1].bar(labels, residual, color="#2a9d8f")
                axes[1].set_yscale("log")
                axes[1].tick_params(axis="x", rotation=35)
                axes[1].set_ylabel("residual or increment norm")
                axes[1].set_title("Numerical diagnostic")
                plt.show()
                """
            ),
            _code(
                """
                for result in suite.results:
                    assert result.final_loss < result.initial_loss
                    assert result.gradient_norm > 0.0
                    assert result.parameter_count > 0
                print("all family paths reduced loss and produced finite gradients")
                """
            ),
        ],
    )


def _dossier_notebook() -> nbformat.NotebookNode:
    return _notebook(
        "SILVA Family Reproduction Dossiers",
        "Inspect every family as a six-stage experiment contract, from equations and primitive "
        "mechanisms through compact evidence, official-data subsets, and source-scale work.",
        [
            _markdown(
                r"""
                ## Evidence Ladder

                A result advances through an ordered ladder

                $$
                \mathcal E_0\subset\mathcal E_1\subset\cdots\subset\mathcal E_5,
                $$

                where each stage adds acceptance checks and archived artifacts. A planned final
                stage does not inherit a stronger status from a successful compact stage.
                """
            ),
            _code(
                """
                from collections import Counter
                from silva_networks import (
                    all_silva_experiment_dossiers,
                    audit_silva_experiment_dossiers,
                )

                dossiers = all_silva_experiment_dossiers()
                assert audit_silva_experiment_dossiers() == ()
                print("families:", len(dossiers))
                print("domains:", Counter(dossier.domain for dossier in dossiers))
                print("stages per family:", sorted({len(dossier.stages) for dossier in dossiers}))
                """
            ),
            _code(
                """
                example = next(item for item in dossiers if item.family == "silva_fno_deq")
                print("family:", example.family)
                print("equation:", example.equation)
                print("datasets:", *example.datasets, sep="\\n- ")
                print("metrics:", *example.metrics, sep="\\n- ")
                for stage in example.stages:
                    print(stage.name, "->", stage.evidence_status)
                """
            ),
            _code(
                """
                import matplotlib.pyplot as plt

                plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
                counts = Counter(dossier.domain for dossier in dossiers)
                fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
                ax.barh(list(counts), list(counts.values()), color="#3a86ff")
                ax.set_xlabel("canonical families")
                ax.set_title("Experiment-dossier coverage by domain")
                for index, value in enumerate(counts.values()):
                    ax.text(value + 0.08, index, str(value), va="center")
                plt.show()
                """
            ),
            _code(
                """
                required = {
                    "machine-readable model and solver configuration",
                    "dataset receipt with source revision, split, license, and checksum",
                    "task metrics and equilibrium diagnostics in a machine-readable result",
                }
                for dossier in dossiers:
                    assert required.issubset(dossier.required_artifacts)
                    assert dossier.stages[-1].evidence_status == "planned"
                print("all dossiers retain the claim boundary and required result records")
                """
            ),
        ],
    )


def _extension_notebook() -> nbformat.NotebookNode:
    return _notebook(
        "Building and Verifying a New SILVA Abstraction",
        "Construct a custom branch, verify its primitive transition against the public layer, "
        "solve the equilibrium, train a readout, and record the extension boundary.",
        [
            _markdown(
                r"""
                ## Derive Before Assembling

                We define

                $$
                T_\theta(z;x)=\tanh\!\left(W_xx+b_x+\tanh(W_hz)\right).
                $$

                The custom self branch is $H_\theta(z)=\tanh(W_hz)$. The stimulus and outer
                activation remain public SILVA components. Numerical equality of the primitive
                expression and `layer.f` is checked before root solving.
                """
            ),
            _code(
                """
                import torch
                from torch import nn
                from silva_networks import SolverConfig, silva_generalized_layer

                torch.manual_seed(146)

                class CustomSelfBranch(nn.Module):
                    def __init__(self, width):
                        super().__init__()
                        self.weight = nn.Parameter(0.08 * torch.randn(width, width))

                    def forward(self, state):
                        return torch.tanh(state @ self.weight.T)

                branch = CustomSelfBranch(6)
                layer = silva_generalized_layer(
                    3,
                    6,
                    local="none",
                    global_term="none",
                    self_term=branch,
                    activation=nn.Identity(),
                    output_activation=torch.tanh,
                    normalize=False,
                    config=SolverConfig(max_iter=24, tol=1e-7, alpha=0.8),
                )
                inputs = torch.randn(12, 3)
                state = torch.zeros(12, 6)
                primitive = torch.tanh(layer.stimulus(inputs) + branch(state))
                assembled = layer.f(state, inputs)
                print("transition max error:", float((primitive - assembled).abs().max()))
                assert torch.allclose(primitive, assembled, atol=1e-7, rtol=1e-7)
                """
            ),
            _code(
                """
                result = layer(inputs, return_result=True)
                print("iterations:", result.iterations)
                print("residual:", result.residual)
                print("converged:", result.converged)
                """
            ),
            _code(
                """
                head = nn.Linear(6, 1)
                optimizer = torch.optim.Adam([*layer.parameters(), *head.parameters()], lr=0.02)
                target = torch.sin(inputs[:, :1]) + 0.2 * inputs[:, 1:2]
                losses = []
                for _ in range(18):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = head(layer(inputs))
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                print("loss:", losses[0], "->", losses[-1])
                assert losses[-1] < losses[0]
                """
            ),
            _code(
                """
                import matplotlib.pyplot as plt

                plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
                axes[0].semilogy(result.residuals, marker="o")
                axes[0].set_title("Equilibrium residual")
                axes[0].set_xlabel("iteration")
                axes[1].plot(losses, marker="o", markersize=3)
                axes[1].set_title("Tiny-task training")
                axes[1].set_xlabel("optimizer step")
                plt.show()
                """
            ),
            _markdown(
                """
                ## Turn It Into a Reusable Family

                Keep the branch constructor explicit, add a canonical family key only when the
                mechanism has a stable tensor contract, expose every replacement point, add a
                deterministic fixture and primitive-equivalence test, then register datasets,
                metrics, source requirements, scale defaults, and a dossier. An application-specific
                composition can remain a configured SILVA layer without becoming a new family.
                """
            ),
        ],
    )


def _failure_notebook() -> nbformat.NotebookNode:
    return _notebook(
        "SILVA Failure Diagnostics Workshop",
        "Create stable, slow, oscillatory, and damped scalar equilibria, then connect residual "
        "curves to the local iteration factor and practical solver decisions.",
        [
            _markdown(
                r"""
                ## Scalar Diagnostic Model

                For

                $$
                z_{k+1}=\rho z_k+u,
                \qquad z^\star=\frac{u}{1-\rho},
                $$

                the undamped error obeys $e_{k+1}=\rho e_k$. With Picard damping $\alpha$,
                the effective factor is

                $$
                \rho_{\mathrm{eff}}=(1-\alpha)+\alpha\rho.
                $$

                Damping can stabilize an oscillatory negative factor even though it cannot repair
                every expansive transition.
                """
            ),
            _code(
                """
                import torch
                from silva_networks import SolverConfig, solve_equilibrium

                cases = {
                    "stable": (0.70, 1.0),
                    "near critical": (0.98, 1.0),
                    "oscillatory": (-1.20, 1.0),
                    "oscillatory + damping": (-1.20, 0.40),
                }
                records = {}
                for name, (rho, alpha) in cases.items():
                    transition = lambda state, rho=rho: rho * state + 1.0
                    result = solve_equilibrium(
                        transition,
                        torch.zeros(1),
                        SolverConfig(max_iter=40, tol=1e-7, alpha=alpha, return_best=True),
                    )
                    records[name] = result
                    effective = (1.0 - alpha) + alpha * rho
                    print(
                        name,
                        "rho_eff=", round(effective, 3),
                        "iterations=", result.iterations,
                        "residual=", f"{result.residual:.4g}",
                        "converged=", result.converged,
                    )
                """
            ),
            _code(
                """
                import matplotlib.pyplot as plt

                plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
                fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
                for name, result in records.items():
                    ax.semilogy(result.residuals, marker="o", markersize=3, label=name)
                ax.set_xlabel("iteration")
                ax.set_ylabel("absolute fixed-point residual")
                ax.set_title("Failure signatures and damping")
                ax.legend()
                plt.show()
                """
            ),
            _markdown(
                """
                ## Diagnose in This Order

                1. Validate the transition shape and finite values before invoking a solver.
                2. Plot absolute and relative residuals; a final scalar hides oscillation and stalls.
                3. Compare undamped and damped Picard to identify a local stability problem.
                4. Inspect branch norms and a Jacobian-radius estimate near the returned state.
                5. Compare acceleration methods only after the transition itself is understood.
                6. Validate implicit gradients against unrolling or finite differences on a tiny case.
                """
            ),
            _code(
                """
                assert records["stable"].residual < records["near critical"].residual
                assert not records["oscillatory"].converged
                assert records["oscillatory + damping"].converged
                print("diagnostic assertions distinguish slow, divergent, and stabilized behavior")
                """
            ),
        ],
    )


def _write_notebooks() -> None:
    notebooks = {
        "42_family_reproduction_dossiers.ipynb": _dossier_notebook(),
        "43_cross_family_vector_benchmark.ipynb": _comparison_notebook(
            "vector", "run_vector_comparison", "Cross-Family Vector Equilibrium Benchmark"
        ),
        "44_cross_family_graph_benchmark.ipynb": _comparison_notebook(
            "graph", "run_graph_comparison", "Cross-Family Graph Equilibrium Benchmark"
        ),
        "45_cross_family_field_benchmark.ipynb": _comparison_notebook(
            "field", "run_field_comparison", "Cross-Family Field Operator Benchmark"
        ),
        "46_extension_builder_workshop.ipynb": _extension_notebook(),
        "47_failure_diagnostics_workshop.ipynb": _failure_notebook(),
    }
    for directory in (CANONICAL_NOTEBOOKS, DOC_NOTEBOOKS, PORTABLE_NOTEBOOKS):
        directory.mkdir(parents=True, exist_ok=True)
        for name, notebook in notebooks.items():
            write_notebook(
                directory / name,
                notebook,
                replace_changed=True,
                preserve_unmatched=True,
            )


def _write_guides() -> None:
    record = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    suite_sections = []
    for suite in record["suites"]:
        rows = "\n".join(
            f"| `{item['family']}` | {item['parameter_count']} | {item['initial_loss']:.5f} | "
            f"{item['final_loss']:.5f} | {item['loss_reduction']:.3f} | {item['residual']:.4g} | "
            f"{item['iterations']} |"
            for item in suite["results"]
        )
        suite_sections.append(
            _remove_margin(
                f"""
                ## {suite['name'].title()} Suite

                **Task:** {suite['task']}

                **Metric:** {suite['metric']}

                | Family | Parameters | Initial loss | Final loss | Reduction | Residual/increment | Iterations |
                | --- | ---: | ---: | ---: | ---: | ---: | ---: |
                {rows}

                Interpretation limits:

                {_bullets(suite['limitations'])}
                """,
                16,
            )
        )
    comparisons = _remove_margin(
        f"""
        # Cross-Family Compact Comparisons

        These suites answer a narrow question: can compatible SILVA families execute, optimize,
        differentiate, and report numerical diagnostics on exactly the same compact task? They do
        not rank source methods or replace their publication datasets.

        The machine-readable record is
        `experiments/reproduction/outputs/compact_comparisons.json`. Rerun it with:

        ```bash
        python experiments/reproduction/run_compact_comparisons.py
        ```

        {chr(10).join(suite_sections)}

        ## What to Compare at Larger Scale

        Keep task data, split, metric, optimizer budget, seed policy, and stopping rule fixed.
        Report task quality together with residual histories, operator evaluations, wall time,
        peak memory, parameter count, and failed seeds. Architecture-specific certificates remain
        separate columns rather than being collapsed into one score.
        """,
        8,
    )
    _write_text(
        ROOT / "docs/experiments/cross-family-comparisons.md",
        comparisons
        + "\n\n"
        + EXTENSION_BLOCK.replace("learn.md", "experiments.md")
        + "\n\n"
        + _next_steps(
            (
                ("Which families have complete experiment dossiers?", "Family Dossiers", "../families/index.md"),
                ("How are these suites called from Python?", "Compact Benchmark API", "../api/compact_benchmarks.md"),
                ("Where are the executable comparison labs?", "Notebook Library", "../notebooks.md"),
            )
        )
        + "\n",
    )

    extension = dedent(
        r"""
        # Advanced Extension Handbook

        A SILVA extension begins with a state contract, not a family label. Define the state,
        conditioning variables, branch decomposition, constraints, and readout before adding a
        constructor.

        ## Derivation Contract

        Start from

        $$
        z^\star=T_\theta(z^\star;x,E,b,c),\qquad
        T_\theta=\Pi_{\mathcal C}\circ\sigma\circ(S+H+L+G+P),
        $$

        where $P$ collects optional physics or proximal terms and
        $\Pi_{\mathcal C}$ enforces hard constraints. Every term must return the same state shape.

        ## Six Construction Steps

        1. Implement and test each primitive branch independently.
        2. Evaluate the assembled transition directly and compare it with the written equation.
        3. Establish contraction, monotonicity, positivity, projection, or another well-posedness route.
        4. Solve a deterministic analytic case and validate gradients independently.
        5. Train a tiny task, reload its checkpoint, and retain measured diagnostics.
        6. Register source data, metrics, scale defaults, tests, notebooks, and a reproduction dossier.

        ## Granularity Rules

        Keep encoders, stimulus, self-interaction, local interaction, global context, constraints,
        solver, backward method, and readout independently replaceable when their contracts differ.
        A configured composition does not need a new canonical family. Add a family when a stable
        mechanism, constructor, evidence path, and source relationship recur across experiments.

        ## Equivalence Test

        For a primitive implementation $T_{\mathrm{primitive}}$ and a public composition
        $T_{\mathrm{public}}$, check

        $$
        \epsilon_T=\max_i\left\|T_{\mathrm{primitive}}(z_i,x_i)-
        T_{\mathrm{public}}(z_i,x_i)\right\|_\infty,
        $$

        followed by equilibrium, readout, and gradient agreement. Notebook 46 executes the complete
        process with a custom self branch and a trained compact task.

        ## Executable Contract Check

        The SILVA construction [[1]](../paper/references.md#ref-1){ .silva-cite }
        requires the transition to preserve its state space. Validate that property before
        selecting a solver:

        ```python
        import torch
        from torch import nn
        from silva_networks import validate_silva_transition

        class ContractiveBranch(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.linear = nn.Linear(width, width, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(0.1 * torch.eye(width))

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                return torch.tanh(self.linear(state))

        state = torch.zeros(8, 6, requires_grad=True)
        report = validate_silva_transition(ContractiveBranch(6), state)
        assert report.preserves_shape
        assert report.finite
        assert report.differentiable
        print(report)
        ```

        The report checks the generic state contract. Add family-specific tests for graph
        equivariance, boundary projection, positivity, conservation, or multiscale coupling
        before the module is treated as a reusable family.

        ## Registration Checklist

        - Public constructor and complete signature
        - Canonical family key and aliases
        - Compact and full scale defaults
        - Equation, source relation, references, datasets, and metrics
        - Primitive and assembled equivalence tests
        - Solver, gradient, shape, device, and serialization tests
        - Executed notebook with results and figures
        - Family dossier and editable source-scale configuration
        """
    ).strip()
    _write_text(
        ROOT / "docs/learn/advanced-extension-handbook.md",
        extension
        + "\n\n"
        + EXTENSION_BLOCK
        + "\n\n"
        + _next_steps(
            (
                ("Where are the replaceable branch contracts introduced?", "Custom Layers", "custom-layers.md"),
                ("Which lab builds and verifies a custom family?", "Extension Builder Workshop", "../package-notebooks/46_extension_builder_workshop.ipynb"),
                ("Where are all family-scale experiment plans?", "Family Dossiers", "../families/index.md"),
            )
        )
        + "\n",
    )

    failure = dedent(
        r"""
        # Failure Diagnostics and Recovery

        Failure examples are part of the experiment contract. A solver that returns a tensor has
        not necessarily found a useful equilibrium.

        ## Local Error Dynamics

        Near a fixed point,

        $$
        e_{k+1}\approx J_T(z^\star)e_k.
        $$

        Slow decay indicates a radius near one; alternating residuals suggest a negative dominant
        mode; growth indicates an expansive direction; a plateau may indicate scaling, precision,
        or an incompatible tolerance. The fixed-point formulation follows the equilibrium-model
        foundation [[4]](../paper/references.md#ref-4){ .silva-cite }; Jacobian control and
        diagnostic motivation are developed further in
        [[6]](../paper/references.md#ref-6){ .silva-cite }.

        ## Diagnostic Matrix

        | Symptom | Measure | Controlled response |
        | --- | --- | --- |
        | Residual grows | Branch norms, Jacobian radius, finite values | Reduce recurrent scale, normalize inputs, or enforce a certificate |
        | Residual alternates | Signed state differences and damped Picard | Reduce damping before adding acceleration |
        | Residual stalls | Absolute/relative curves and precision | Rescale states, revise tolerance, or increase precision |
        | Forward works but gradients fail | Adjoint residual and finite differences | Tighten backward solve or revise the local Jacobian |
        | Constraint drifts | Boundary, positivity, feasibility, or conservation error | Project inside every transition rather than only at readout |
        | Training is unstable | Per-branch gradients and spectral diagnostics | Isolate the offending branch and add it back incrementally |

        ## Executable Residual Check

        This compact program compares a stable map with a near-critical one while retaining the
        complete residual history:

        ```python
        import torch
        from silva_networks import SolverConfig, solve_equilibrium

        def run_case(factor: float):
            transition = lambda state: factor * state + 1.0
            return solve_equilibrium(
                transition,
                torch.zeros(1),
                SolverConfig(
                    solver="picard",
                    max_iter=40,
                    tol=1e-7,
                    alpha=1.0,
                    return_best=True,
                ),
            )

        stable = run_case(0.70)
        slow = run_case(0.98)
        assert stable.residual < slow.residual
        print("stable:", stable.iterations, stable.residual)
        print("near critical:", slow.iterations, slow.residual)
        ```

        Plot \`stable.residuals\` and \`slow.residuals\`, not only their final values. The curve
        distinguishes geometric decay from a budget-limited near-critical solve.

        ## Required Failure Record

        Retain the input and seed, complete solver configuration, residual history, returned-versus-
        best state choice, branch norms, certificate or radius estimate, backward residual, task loss,
        and the exact recovery change. Notebook 47 provides executable stable, slow, oscillatory, and
        damped cases with a 300-dpi diagnostic figure.

        ## Recovery Is an Experiment

        Change one factor at a time. Damping, normalization, recurrent scaling, solver choice,
        tolerance, and backward method answer different questions. Record each as an ablation rather
        than silently changing several controls until the run succeeds.
        """
    ).strip()
    _write_text(
        ROOT / "docs/learn/failure-diagnostics-and-recovery.md",
        failure
        + "\n\n"
        + EXTENSION_BLOCK
        + "\n\n"
        + _next_steps(
            (
                ("How are solver residuals defined?", "Solvers", "../api/solvers.md"),
                ("Which diagnostics are available programmatically?", "Diagnostics", "../api/diagnostics.md"),
                ("Where can I run the failure cases?", "Failure Diagnostics Workshop", "../package-notebooks/47_failure_diagnostics_workshop.ipynb"),
            )
        )
        + "\n",
    )

    result_records = dedent(
        """
        # Result Records and Evidence Levels

        Every table and figure should be traceable to data, configuration, code revision, seed,
        hardware, and an explicit evidence level. This prevents a compact mechanism check from
        being presented as a source-scale reproduction.

        ## Required Record

        ```python
        from silva_networks import SILVAResultRecord

        record = SILVAResultRecord(
            family="silva_fno_deq",
            evidence_status="compact-verified",
            dataset="analytic periodic field",
            dataset_version="v1",
            split="seeded four-sample fixture",
            configuration="field-comparison-v1",
            seed=122,
            metrics=(("mse", 0.145), ("residual", 0.02)),
            data_fingerprint="sha256:...",
            code_revision="commit-or-working-tree-revision",
            hardware="CPU; precision=float32",
            deviations=("compact 8 by 8 grid",),
        )
        assert record.validate() == ()
        ```

        ## Promotion Rule

        Evidence moves forward only when the next stage's acceptance checks and artifacts exist.
        A successful subset run remains `subset-verified`; it does not become
        `source-scale-reproduced` because its learning curve looks promising.

        ## Figure and Table Captions

        State the family, dataset and split, metric, number of seeds, scale tier, evidence level,
        and whether uncertainty is across seeds, samples, or batches. Link the machine-readable
        result and configuration beside the caption whenever the publication surface permits it.
        """
    ).strip()
    _write_text(
        ROOT / "docs/experiments/result-records.md",
        result_records
        + "\n\n"
        + EXTENSION_BLOCK.replace("learn.md", "experiments.md")
        + "\n\n"
        + _next_steps(
            (
                ("Where are the family-specific acceptance stages?", "Family Dossiers", "../families/index.md"),
                ("How are result records represented in Python?", "Research-Depth API", "../api/research_depth.md"),
                ("Which comparisons emit measured records?", "Cross-Family Comparisons", "cross-family-comparisons.md"),
            )
        )
        + "\n",
    )


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "run experiments/reproduction/run_compact_comparisons.py before generating material"
        )
    _write_family_material()
    _write_guides()
    _write_notebooks()
    print(f"generated {len(all_silva_experiment_dossiers())} family dossiers and scale plans")
    print("generated 4 research-depth guides")
    print("generated 6 research-depth notebooks in canonical, docs, and portable locations")


if __name__ == "__main__":
    main()
