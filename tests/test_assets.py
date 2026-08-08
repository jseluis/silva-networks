from __future__ import annotations

import ast
import base64
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PUBLICATION_DPI = 299.0
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _markdown_units() -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        units.append((str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    for folder in [ROOT / "docs", ROOT / "notebooks", ROOT / "colab"]:
        for path in sorted(folder.rglob("*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "markdown":
                    units.append(
                        (
                            f"{path.relative_to(ROOT)}:cell{index}",
                            "".join(cell.get("source", [])),
                        )
                    )
    return units


def _without_code(text: str) -> str:
    text = CODE_FENCE_RE.sub("", text)
    return INLINE_CODE_RE.sub("", text)


def _single_dollar_count(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        char = text[index]
        previous = text[index - 1] if index else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char == "$" and previous != "\\" and next_char != "$":
            count += 1
        if char == "$" and next_char == "$":
            index += 2
        else:
            index += 1
    return count


def _png_dimensions_and_dpi(data: bytes) -> tuple[int, int, tuple[float, float] | None]:
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs" and len(chunk) >= 9:
            xppm, yppm, unit = struct.unpack(">IIB", chunk[:9])
            if unit == 1:
                return width, height, (xppm / 39.37007874015748, yppm / 39.37007874015748)
            return width, height, None
        offset += 12 + length
    return width, height, None


def test_companion_book_and_manual_are_planned_without_direct_pdf_links() -> None:
    book_page = (ROOT / "docs/book.md").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    sources = (ROOT / "src/silva_networks.egg-info/SOURCES.txt").read_text(encoding="utf-8")
    assert "Planned" in book_page
    assert "planned learning resources" in book_page
    assert "assets/pdfs/silva_deq_companion_book.pdf" not in book_page
    assert "assets/pdfs/silva_deq_solutions_manual.pdf" not in book_page
    assert "*.pdf" not in manifest
    assert "prune docs/assets/pdfs" in manifest
    assert "assets/pdfs/silva_deq_companion_book.pdf" not in sources
    assert "assets/pdfs/silva_deq_solutions_manual.pdf" not in sources


def test_bibtex_asset_is_packaged() -> None:
    assert (ROOT / "docs/assets/bib/silva-networks.bib").stat().st_size > 1_000
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for marker in [
        "CITATION.cff",
        "mkdocs.yml",
        "*.bib",
        "*.html",
        "src/silva_networks/cases.py",
    ]:
        assert marker in manifest


def test_method_interaction_visual_is_on_homepage() -> None:
    visual = ROOT / "docs/assets/images/silva-method-cinematic.png"
    home = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    assert visual.stat().st_size > 100_000
    assert visual.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "assets/images/silva-method-cinematic.png" in home
    assert "assets/images/silva-method-interactions.svg" not in home


def test_notebook_markdown_uses_portable_inline_math_delimiters() -> None:
    offenders: list[str] = []
    for folder in [ROOT / "docs", ROOT / "notebooks", ROOT / "colab"]:
        for path in sorted(folder.rglob("*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                source = "".join(cell.get("source", []))
                if r"\(" in source or r"\)" in source:
                    offenders.append(f"{path.relative_to(ROOT)}:cell{index}")
    assert offenders == []


def test_all_markdown_math_delimiters_are_balanced() -> None:
    offenders: list[str] = []
    for label, raw_source in _markdown_units():
        source = _without_code(raw_source)
        if source.count("$$") % 2:
            offenders.append(f"{label}: unbalanced $$")
        if source.count(r"\[") != source.count(r"\]"):
            offenders.append(f"{label}: unbalanced \\[...\\]")
        if source.count(r"\(") != source.count(r"\)"):
            offenders.append(f"{label}: unbalanced \\(...\\)")
        if _single_dollar_count(source) % 2:
            offenders.append(f"{label}: unbalanced $...$")
    assert offenders == []


def test_markdown_math_has_no_escaped_control_character_damage() -> None:
    offenders: list[str] = []
    for label, raw_source in _markdown_units():
        source = _without_code(raw_source)
        if "\t" in source:
            offenders.append(f"{label}: tab in mathematical prose")
        if re.search(r"\^star\b|_heta\b", source):
            offenders.append(f"{label}: malformed LaTeX command")
    assert offenders == []


def test_mathjax_loader_supports_docs_and_notebook_math() -> None:
    loader = (ROOT / "docs/javascripts/mathjax.js").read_text(encoding="utf-8")
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    font_dir = ROOT / "docs/javascripts/vendor/mathjax/output/chtml/fonts/woff-v2"
    required_fonts = sorted(
        set(
            re.findall(
                r"MathJax_[A-Za-z0-9_-]+\.woff",
                (ROOT / "docs/javascripts/vendor/mathjax/tex-mml-chtml.js").read_text(
                    encoding="utf-8"
                ),
            )
        )
    )
    for marker in [
        '["\\\\(", "\\\\)"]',
        '["$", "$"]',
        "md-typeset",
        "jp-RenderedHTMLCommon",
        "jp-MarkdownOutput",
        "data-silva-mathjax-loader",
        "vendor/mathjax/tex-mml-chtml.js",
        "cdn.jsdelivr.net/npm/mathjax@3",
        "cdnjs.cloudflare.com/ajax/libs/mathjax",
    ]:
        assert marker in loader
    assert (ROOT / "docs/javascripts/vendor/mathjax/tex-mml-chtml.js").stat().st_size > 1_000_000
    assert "Apache License" in (ROOT / "docs/javascripts/vendor/mathjax/LICENSE.txt").read_text(
        encoding="utf-8"
    )
    assert len(required_fonts) >= 20
    missing_fonts = [font for font in required_fonts if not (font_dir / font).exists()]
    assert missing_fonts == []
    assert "javascripts/vendor/mathjax/tex-mml-chtml.js" in config
    assert "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" not in config


def test_docs_notebook_embedded_png_outputs_are_publication_dpi() -> None:
    png_outputs: list[tuple[Path, int, int, int, int, tuple[float, float] | None]] = []
    for path in sorted((ROOT / "docs").rglob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(nb.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs", []) or []):
                encoded = (output.get("data") or {}).get("image/png")
                if encoded is None:
                    continue
                width, height, dpi = _png_dimensions_and_dpi(base64.b64decode(encoded))
                png_outputs.append((path, cell_index, output_index, width, height, dpi))

    assert len(png_outputs) >= 20
    assert any(path.name == "04_deq_and_silva.ipynb" for path, *_ in png_outputs)
    low_resolution = [
        f"{path.relative_to(ROOT)}:cell{cell_index}:output{output_index} {width}x{height} dpi={dpi}"
        for path, cell_index, output_index, width, height, dpi in png_outputs
        if dpi is None or min(dpi) < MIN_PUBLICATION_DPI
    ]
    assert low_resolution == []


def test_static_png_assets_are_publication_dpi() -> None:
    low_resolution: list[str] = []
    for path in sorted((ROOT / "docs/assets/images").glob("*.png")):
        width, height, dpi = _png_dimensions_and_dpi(path.read_bytes())
        if dpi is None or min(dpi) < MIN_PUBLICATION_DPI:
            low_resolution.append(f"{path.relative_to(ROOT)} {width}x{height} dpi={dpi}")
    assert low_resolution == []


def test_notebook_json_and_code_cells_parse() -> None:
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    assert len(notebooks) >= 25
    code_cells = 0
    for path in notebooks:
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code_cells += 1
                ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell{i}")
    assert code_cells > 100


def test_all_canonical_notebooks_include_extension_and_reproduction_depth() -> None:
    canonical = [
        *sorted((ROOT / "notebooks/package_api").glob("*.ipynb")),
        *sorted((ROOT / "notebooks/implicit_bridge").glob("*.ipynb")),
        *sorted((ROOT / "notebooks").glob("*.ipynb")),
    ]
    assert len(canonical) == 82
    for path in canonical:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        curriculum = [
            cell
            for cell in notebook["cells"]
            if "silva-extension-curriculum" in cell.get("metadata", {}).get("tags", [])
        ]
        assert len(curriculum) == 8, path
        source = "\n".join("".join(cell.get("source", [])) for cell in curriculum)
        for marker in (
            "to a Custom SILVA Family",
            "validate_silva_transition",
            "Numerical Equivalence, Compact Reproduction, and Scale",
            "notebook_reproduction_record",
            "Worked Convergence and Sensitivity Study",
            "Reading and Extending the Result",
        ):
            assert marker in source, (path, marker)

        identifiers = {cell.get("id") for cell in curriculum}
        for identifier in (
            "silva-extension-diagnostic-derivation",
            "silva-extension-diagnostic-table",
            "silva-extension-diagnostic-figure",
            "silva-extension-diagnostic-interpretation",
        ):
            assert identifier in identifiers, (path, identifier)

        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in code_cells), path
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ), path
        assert any(
            "image/png" in output.get("data", {})
            for cell in code_cells
            for output in cell.get("outputs", [])
        ), path


def test_worked_example_pages_include_derivation_code_results_and_scale_route() -> None:
    pages = sorted((ROOT / "docs/examples").glob("*.md"))
    expanded = []
    for path in pages:
        source = path.read_text(encoding="utf-8")
        if "<!-- silva-worked-example:start -->" not in source:
            continue
        expanded.append(path)
        assert "<!-- silva-worked-example:end -->" in source, path
        if path.name != "index.md":
            assert "Measured Compact Output" in source, path
            assert "Interpret the Output" in source, path
            assert "Add a Solver and Scale Sweep" in source, path
            assert source.count("$$") >= 8, path
            assert source.count("```") >= 8, path
            assert len(source.split()) >= 700, path

    assert len(expanded) == 26


def test_notebook_mirror_sets_are_complete() -> None:
    package_docs = sorted(path.name for path in (ROOT / "docs/package-notebooks").glob("*.ipynb"))
    bridge_docs = sorted(
        path.name for path in (ROOT / "docs/implicit-bridge-notebooks").glob("*.ipynb")
    )

    assert len(package_docs) >= 13
    assert len(bridge_docs) >= 9
    assert (
        sorted(path.name for path in (ROOT / "notebooks/package_api").glob("*.ipynb"))
        == package_docs
    )
    assert sorted(path.name for path in (ROOT / "colab").glob("*.ipynb")) == package_docs
    assert (
        sorted(path.name for path in (ROOT / "notebooks/implicit_bridge").glob("*.ipynb"))
        == bridge_docs
    )
    assert (
        sorted(path.name for path in (ROOT / "colab/implicit_bridge").glob("*.ipynb"))
        == bridge_docs
    )


def test_notebook_mirrors_retain_canonical_execution_results() -> None:
    groups = [
        (
            ROOT / "notebooks/package_api",
            (ROOT / "docs/package-notebooks", ROOT / "colab"),
        ),
        (
            ROOT / "notebooks/implicit_bridge",
            (ROOT / "docs/implicit-bridge-notebooks", ROOT / "colab/implicit_bridge"),
        ),
    ]
    compared = 0
    for canonical_dir, mirror_dirs in groups:
        for canonical_path in sorted(canonical_dir.glob("*.ipynb")):
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            expected = [
                (cell.get("execution_count"), cell.get("outputs", []))
                for cell in canonical["cells"]
                if cell["cell_type"] == "code"
            ]
            for mirror_dir in mirror_dirs:
                mirror = json.loads(
                    (mirror_dir / canonical_path.name).read_text(encoding="utf-8")
                )
                actual = [
                    (cell.get("execution_count"), cell.get("outputs", []))
                    for cell in mirror["cells"]
                    if cell["cell_type"] == "code"
                ]
                assert actual == expected, mirror_dir / canonical_path.name
                compared += 1

    assert compared == 112


def test_expanded_notebooks_are_synchronized_and_substantive() -> None:
    requirements = {
        "14_point_architecture_catalog.ipynb": (
            30,
            (
                "One Fully Populated SILVA Point",
                "Multiple Architectures Inside Every Linked Point",
                "Train a Tiny End-to-End Task",
                "Fourier",
            ),
        ),
        "15_neural_operators_ode_pde.ipynb": (
            44,
            (
                "One Implicit PDE Step as One SILVA Point",
                "PeriodicDiffusionField",
                "poisson_residual",
                "fourier_operator",
                "fixed-point residual",
                "Reaction-Diffusion as an Implicit SILVA Step",
                "Viscous Burgers Equation",
                "Coefficient-to-Solution Learning",
                "SILVAFourierNeuralOperator",
                "Irregular Domains as Graph PDEs",
            ),
        ),
        "16_frontier_equilibrium_families.ipynb": (
            34,
            (
                "Input-Injected Fourier Equilibrium",
                "Physics-Guided Graph Convection-Diffusion Equilibrium",
                "Continuous Residual Path",
                "Distributional SILVA Equilibrium",
                "Connect All Four Mechanisms",
                "Tiny Trained Task",
                "Reproduction Boundary",
            ),
        ),
        "17_silva_fno_equilibrium_lab.ipynb": (
            15,
            (
                "Exact Periodic Elliptic Data",
                "Three Errors, Three Questions",
                "make_periodic_elliptic_dataset",
                "Resolution Transfer Is a Testable Contract",
            ),
        ),
        "18_silva_graph_transport_lab.ipynb": (
            15,
            (
                "Continuous and Discrete Transport",
                "make_graph_transport_dataset",
                "Train a Small Node Field",
                "Node Relabeling",
            ),
        ),
        "19_silva_homotopy_equilibrium_lab.ipynb": (
            15,
            (
                "Fixed-Point Residual and Classical Homotopy",
                "Analytic Affine Path",
                "make_affine_homotopy_dataset",
                "Choosing the Horizon and Integrator",
            ),
        ),
        "20_silva_distributional_equilibrium_lab.ipynb": (
            16,
            (
                "Matrices Represent Empirical Measures",
                "Gaussian MMD and Energy Distance",
                "make_variable_measure_dataset",
                "Train a Readout from the Equilibrium Measure",
            ),
        ),
        "21_silva_monotone_graph_equilibrium.ipynb": (
            14,
            (
                "Monotone Channel Parameterization",
                "Forward-Backward Step as a SILVA Transition",
                "make_monotone_chain_dataset",
                "Node Relabeling",
            ),
        ),
        "22_silva_generative_equilibrium_transformer.ipynb": (
            14,
            (
                "Injection Is Computed Once",
                "QKV-Injected Equilibrium Block",
                "One-Step Teacher Matching",
                "Class Conditioning",
            ),
        ),
        "23_silva_poisson_mirror_equilibrium.ipynb": (
            14,
            (
                "Poisson Observation Model",
                "Closed-Form Burg Update",
                "Mirror Step as a SILVA Equilibrium",
                "Learned Regularizer Contract",
            ),
        ),
        "24_silva_physics_informed_equilibrium.ipynb": (
            14,
            (
                "Time Derivative from the Implicit Function Theorem",
                "Three-Term Physics-Informed Objective",
                "Tiny Physics-Only Training Run",
                "Stiff Systems and Scaling",
            ),
        ),
        "25_silva_implicit_dae_and_residuals.ipynb": (
            16,
            (
                "Implicit Runge-Kutta Stage Equations",
                "Two-Stage Gauss-Legendre Layer",
                "Adversarial Residual Objective and Naming Boundary",
                "Choosing the Physics Construction",
            ),
        ),
        "26_full_scale_silva.ipynb": (
            30,
            (
                "Every Canonical Family Has an Actionable Route",
                "Factorized Monotone Graph Operator",
                "Distributional Equilibria and Exact Pair Chunking",
                "Physics-Informed Equilibrium Derivative",
                "Implicit DAE Stage as Newton-Krylov SILVA",
                "Train a Fourier Equilibrium and Resume",
                "Reproduce, Then Go Beyond",
            ),
        ),
        "28_silva_consistency_deq.ipynb": (
            16,
            (
                "Teacher Equilibrium and Solver-Time Path",
                "Terminally Anchored Consistency Map",
                "Global and Local Distillation",
                "Replace the Teacher and Few-Step Refiner",
            ),
        ),
        "29_silva_psi_gnn.ipynb": (
            17,
            (
                "PDE, Discretization, and Directed Boundary Graph",
                "Typed Processor",
                "Complete Training Objective",
                "Replace the Boundary-Aware Graph Processor",
            ),
        ),
        "30_silva_ifno_materials.ipynb": (
            17,
            (
                "Material Operator Contract",
                "Tied Depth and Deep Limit",
                "Compact Coefficient-to-Displacement Training",
                "Replace the Tied Material Increment",
            ),
        ),
        "31_silva_snarf_forward_skinning.ipynb": (
            16,
            (
                "Forward Skinning Is the Model",
                "Canonical Correspondences as Roots",
                "What the Advanced User Replaces",
                "Replace Every Forward-Deformation Component",
            ),
        ),
        "32_silva_mesh_inference.ipynb": (
            15,
            (
                "Typed Anchors, Evidence, and Policy",
                "Why the Iteration Converges",
                "Directed Admission Sweep",
                "Replace Local Observation and Relaxation Maps",
            ),
        ),
        "33_silva_physics_guided_diffusion_pde.ipynb": (
            15,
            (
                "Residual Energy",
                "Reverse Step",
                "Prior Independence and Stochasticity",
                "Replace Prior, Physics Energy, Smoother, and Boundary Projection",
            ),
        ),
        "34_silva_therino_mechanics.ipynb": (
            17,
            (
                "Physical State, Constitutive Map, and Equilibrium",
                "Exact Periodic Verification Cell",
                "Trainable Finite-Iteration Objective",
                "Move from the Exact Cell to Full Mechanics",
            ),
        ),
        "35_silva_fixed_point_diffusion.ipynb": (
            17,
            (
                "One Fixed Point at Every Diffusion Time",
                "Reverse Schedule, Variable Compute, and State Reuse",
                "Stochastic Jacobian-Free Training",
                "Distinguish the Joint DeqIR Route",
            ),
        ),
    }

    for name, (minimum_cells, markers) in requirements.items():
        notebooks = [
            json.loads((ROOT / folder / name).read_text(encoding="utf-8"))
            for folder in ("notebooks/package_api", "docs/package-notebooks", "colab")
        ]
        signatures = [
            [(cell["cell_type"], "".join(cell.get("source", []))) for cell in notebook["cells"]]
            for notebook in notebooks
        ]
        assert signatures[0] == signatures[1] == signatures[2]
        assert len(signatures[0]) >= minimum_cells
        all_source = "\n".join(source for _, source in signatures[0])
        for marker in markers:
            assert marker in all_source


def test_emerging_labs_retain_executed_outputs_and_publication_plots() -> None:
    expected_pngs = {
        "28_silva_consistency_deq.ipynb": 2,
        "29_silva_psi_gnn.ipynb": 2,
        "30_silva_ifno_materials.ipynb": 2,
        "31_silva_snarf_forward_skinning.ipynb": 2,
        "32_silva_mesh_inference.ipynb": 2,
        "33_silva_physics_guided_diffusion_pde.ipynb": 1,
        "34_silva_therino_mechanics.ipynb": 2,
        "35_silva_fixed_point_diffusion.ipynb": 2,
    }
    for name, minimum_pngs in expected_pngs.items():
        notebook = json.loads(
            (ROOT / "docs/package-notebooks" / name).read_text(encoding="utf-8")
        )
        outputs = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        ]
        pngs = [output for output in outputs if "image/png" in output.get("data", {})]
        assert outputs, name
        assert len(pngs) >= minimum_pngs, name


def test_structured_labs_retain_real_source_outputs_and_publication_plots() -> None:
    expected = {
        "36_silva_monotone_operator_equilibrium.ipynb": (
            2,
            "Attributed CIFAR-10 Mechanism Check",
        ),
        "37_silva_positive_concave_equilibrium.ipynb": (
            2,
            "Attributed Positive Image Check",
        ),
        "38_silva_non_euclidean_equilibrium.ipynb": (
            2,
            "Real Images and the Sensitivity Contract",
        ),
        "39_silva_efficient_infinite_graph.ipynb": (
            2,
            "Cora With Source Masks",
        ),
        "40_silva_multiscale_graph_implicit.ipynb": (
            2,
            "Cora Across Three Graph Scales",
        ),
        "41_silva_delta_equilibrium.ipynb": (
            3,
            "Real-Image Delta Activity",
        ),
    }
    for name, (minimum_pngs, marker) in expected.items():
        notebooks = [
            json.loads((ROOT / folder / name).read_text(encoding="utf-8"))
            for folder in ("notebooks/package_api", "docs/package-notebooks", "colab")
        ]
        signatures = [
            [(cell["cell_type"], "".join(cell.get("source", []))) for cell in notebook["cells"]]
            for notebook in notebooks
        ]
        assert signatures[0] == signatures[1] == signatures[2]
        assert marker in "\n".join(source for _, source in signatures[0])
        for notebook in notebooks:
            outputs = [
                output
                for cell in notebook["cells"]
                for output in cell.get("outputs", [])
            ]
            pngs = [output for output in outputs if "image/png" in output.get("data", {})]
            assert outputs, name
            assert len(pngs) >= minimum_pngs, name


def test_rendered_notebooks_have_a_shared_download_control() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    override = (ROOT / "docs/overrides/main.html").read_text(encoding="utf-8")

    assert "custom_dir: docs/overrides" in config
    assert "include_source: true" in config
    assert "page.nb_url" in override
    assert "download" in override
    assert "Download notebook" in override


def test_no_external_paper_cache() -> None:
    assert not (ROOT / "references/papers").exists()
    assert not (ROOT / "references/upstream").exists()
