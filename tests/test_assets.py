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
    assert "Apache License" in (
        ROOT / "docs/javascripts/vendor/mathjax/LICENSE.txt"
    ).read_text(encoding="utf-8")
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
        f"{path.relative_to(ROOT)}:cell{cell_index}:output{output_index} "
        f"{width}x{height} dpi={dpi}"
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


def test_notebook_mirror_sets_are_complete() -> None:
    package_docs = sorted(path.name for path in (ROOT / "docs/package-notebooks").glob("*.ipynb"))
    bridge_docs = sorted(
        path.name for path in (ROOT / "docs/implicit-bridge-notebooks").glob("*.ipynb")
    )

    assert len(package_docs) >= 13
    assert len(bridge_docs) >= 9
    assert sorted(path.name for path in (ROOT / "notebooks/package_api").glob("*.ipynb")) == package_docs
    assert sorted(path.name for path in (ROOT / "colab").glob("*.ipynb")) == package_docs
    assert sorted(path.name for path in (ROOT / "notebooks/implicit_bridge").glob("*.ipynb")) == bridge_docs
    assert sorted(path.name for path in (ROOT / "colab/implicit_bridge").glob("*.ipynb")) == bridge_docs


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
    }

    for name, (minimum_cells, markers) in requirements.items():
        notebooks = [
            json.loads((ROOT / folder / name).read_text(encoding="utf-8"))
            for folder in ("notebooks/package_api", "docs/package-notebooks", "colab")
        ]
        signatures = [
            [
                (cell["cell_type"], "".join(cell.get("source", [])))
                for cell in notebook["cells"]
            ]
            for notebook in notebooks
        ]
        assert signatures[0] == signatures[1] == signatures[2]
        assert len(signatures[0]) >= minimum_cells
        all_source = "\n".join(source for _, source in signatures[0])
        for marker in markers:
            assert marker in all_source


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
