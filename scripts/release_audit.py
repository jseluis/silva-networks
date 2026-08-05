"""Release-readiness checks for the SILVA Networks package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import struct
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

from content_preservation_audit import run_content_preservation_audit
from documentation_audit import run_documentation_audit

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
ARXIV_ID = "2607.28989"
MIN_PUBLICATION_DPI = 299.0
EXPECTED_WORKFLOW_ACTIONS = {
    "actions/checkout": "v7",
    "actions/setup-python": "v7",
    "actions/upload-pages-artifact": "v5",
    "actions/deploy-pages": "v5",
    "actions/upload-artifact": "v7",
    "actions/download-artifact": "v8",
}
WORKFLOW_USE_RE = re.compile(r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s#]+)")

REQUIRED_FILES = (
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    "mkdocs.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/pages.yml",
    ".github/workflows/release.yml",
    "docs/index.md",
    "docs/cli.md",
    "docs/paper/references.md",
    "docs/research-citation-audit.md",
    "docs/equation-and-pdf-audit.md",
    "docs/release-readiness.md",
    "docs/javascripts/default-theme.js",
    "docs/api/public-api.md",
    "docs/api/families.md",
    "docs/api/scientific.md",
    "docs/api/extensibility.md",
    "docs/api/reproducibility.md",
    "docs/assets/papers/silva-networks-arxiv-2607.28989.pdf",
    "docs/assets/bib/silva-networks.bib",
    "docs/learn/solver-derivation-lab.md",
    "docs/learn/extending-silva.md",
    "docs/learn/reproducing-silva-and-papers.md",
    "docs/learn/implicit-backward-guide.md",
    "docs/learn/interactive-diagnostics-lab.md",
    "docs/learn/cortex-hierarchy.md",
    "docs/learn/point-architecture-catalog.md",
    "docs/learn/neural-operators-ode-pde.md",
    "docs/learn/frontier-equilibrium-families.md",
    "docs/learn/frontier-dataset-labs.md",
    "docs/examples/spatial-cortex.md",
    "docs/examples/point-architecture-catalog.md",
    "docs/examples/full-cortex-operators.md",
    "docs/examples/scientific-operators.md",
    "docs/overrides/main.html",
    "docs/learn/paper-family-adaptations.md",
    "docs/package-notebooks/11_cortex_hierarchy.ipynb",
    "docs/package-notebooks/12_paper_family_architectures.ipynb",
    "docs/package-notebooks/13_raft_deq_flow.ipynb",
    "docs/package-notebooks/14_point_architecture_catalog.ipynb",
    "docs/package-notebooks/15_neural_operators_ode_pde.ipynb",
    "docs/package-notebooks/16_frontier_equilibrium_families.ipynb",
    "docs/package-notebooks/17_silva_fno_equilibrium_lab.ipynb",
    "docs/package-notebooks/18_silva_graph_transport_lab.ipynb",
    "docs/package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb",
    "docs/package-notebooks/20_silva_distributional_equilibrium_lab.ipynb",
    "docs/package-notebooks/21_silva_monotone_graph_equilibrium.ipynb",
    "docs/package-notebooks/22_silva_generative_equilibrium_transformer.ipynb",
    "docs/package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb",
    "docs/package-notebooks/24_silva_physics_informed_equilibrium.ipynb",
    "docs/package-notebooks/25_silva_implicit_dae_and_residuals.ipynb",
    "docs/package-notebooks/26_full_scale_silva.ipynb",
    "docs/package-notebooks/27_reproducing_silva_and_source_methods.ipynb",
    "docs/api/frontier_data.md",
    "docs/learn/advanced-equilibrium-families.md",
    "docs/learn/physics-informed-equilibria.md",
    "docs/learn/advanced-equilibrium-datasets.md",
    "docs/api/advanced_equilibria.md",
    "docs/api/physics_informed.md",
    "docs/api/advanced_data.md",
    "docs/examples/advanced-equilibria.md",
    "docs/learn/full-scale-silva.md",
    "docs/examples/full-scale-training.md",
    "docs/examples/reproduction-registry.md",
    "docs/api/scaling.md",
    "docs/api/scaling_data.md",
    "docs/api/scale_cli.md",
    "docs/experiments/benchmark-cards.md",
    "src/silva_networks/dataset_cli.py",
    "src/silva_networks/scaling.py",
    "src/silva_networks/reproducibility.py",
    "src/silva_networks/scaling_data.py",
    "src/silva_networks/scale_cli.py",
    "src/silva_networks/public_experiments.py",
    "src/silva_networks/configs/solver_sweep.json",
    "experiments/public/configs/cifar10_cortex_smoke.json",
    "experiments/public/configs/cifar10_vector_smoke.json",
    "experiments/public/configs/torchvision_dataset_suite.json",
    "scripts/smoke_test.sh",
    "scripts/run_notebook_smoke.py",
    "scripts/notebook_navigation.py",
    "scripts/notebook_citations.py",
    "scripts/expand_notebook_curriculum.py",
    "scripts/expand_documentation_paths.py",
    "scripts/generate_reproduction_registry_lab.py",
)

REQUIRED_NAV_MARKERS = (
    "Equation and PDF Audit: equation-and-pdf-audit.md",
    "CLI Guide: cli.md",
    "Release Readiness: release-readiness.md",
    "Solver Derivation Lab: learn/solver-derivation-lab.md",
    "Implicit Backward Guide: learn/implicit-backward-guide.md",
    "Interactive Diagnostics Lab: learn/interactive-diagnostics-lab.md",
    "Cortex Hierarchies: learn/cortex-hierarchy.md",
    "Point Architecture Catalog: learn/point-architecture-catalog.md",
    "Neural Operators, ODEs, PDEs, and SILVA: learn/neural-operators-ode-pde.md",
    "Recent Equilibrium Families: learn/frontier-equilibrium-families.md",
    "Dataset-Backed Equilibrium Labs: learn/frontier-dataset-labs.md",
    "Spatial SILVA Cortex: examples/spatial-cortex.md",
    "Point Architecture Catalog: examples/point-architecture-catalog.md",
    "Full Cortex Operators: examples/full-cortex-operators.md",
    "Scientific Operators: examples/scientific-operators.md",
    "Paper Family Adaptations: learn/paper-family-adaptations.md",
    "Cortex Hierarchy: package-notebooks/11_cortex_hierarchy.ipynb",
    "Paper Family Architectures: package-notebooks/12_paper_family_architectures.ipynb",
    "RAFT and DEQ-Flow: package-notebooks/13_raft_deq_flow.ipynb",
    "Point Architecture Catalog: package-notebooks/14_point_architecture_catalog.ipynb",
    "Neural Operators, ODEs, and PDEs: package-notebooks/15_neural_operators_ode_pde.ipynb",
    "Recent Equilibrium Families: package-notebooks/16_frontier_equilibrium_families.ipynb",
    "SILVA Fourier Equilibrium Lab: package-notebooks/17_silva_fno_equilibrium_lab.ipynb",
    "SILVA Graph Transport Lab: package-notebooks/18_silva_graph_transport_lab.ipynb",
    "SILVA Homotopy Equilibrium Lab: package-notebooks/19_silva_homotopy_equilibrium_lab.ipynb",
    "SILVA Distributional Equilibrium Lab: package-notebooks/20_silva_distributional_equilibrium_lab.ipynb",
    "SILVA Monotone Graph Equilibrium: package-notebooks/21_silva_monotone_graph_equilibrium.ipynb",
    "SILVA Generative Equilibrium Transformer: package-notebooks/22_silva_generative_equilibrium_transformer.ipynb",
    "SILVA Poisson Mirror Equilibrium: package-notebooks/23_silva_poisson_mirror_equilibrium.ipynb",
    "SILVA Physics-Informed Equilibrium: package-notebooks/24_silva_physics_informed_equilibrium.ipynb",
    "SILVA Implicit DAE and Residuals: package-notebooks/25_silva_implicit_dae_and_residuals.ipynb",
    "Full-Scale SILVA Families: package-notebooks/26_full_scale_silva.ipynb",
    "Reproducing SILVA and Source Methods: package-notebooks/27_reproducing_silva_and_source_methods.ipynb",
    "Full-Scale SILVA: learn/full-scale-silva.md",
    "Full-Scale Training: examples/full-scale-training.md",
    "Source-Aware Reproduction: examples/reproduction-registry.md",
    "Scaling and Family Guides: api/scaling.md",
    "Scaling Data: api/scaling_data.md",
    "Scale CLI: api/scale_cli.md",
    "Advanced Equilibrium Families: learn/advanced-equilibrium-families.md",
    "Physics-Informed Equilibria and DAEs: learn/physics-informed-equilibria.md",
    "Advanced Equilibrium Datasets: learn/advanced-equilibrium-datasets.md",
    "Advanced Equilibria: api/advanced_equilibria.md",
    "Physics-Informed and DAE: api/physics_informed.md",
    "Advanced Equilibrium Data: api/advanced_data.md",
    "Generalized Cases: api/cases.md",
    "Public API: api/public-api.md",
    "Family Selection: api/families.md",
    "Point Architectures: api/point_architectures.md",
    "Scientific Operators: api/scientific.md",
    "Extensibility: api/extensibility.md",
    "Reproducibility: api/reproducibility.md",
    "Extending SILVA: learn/extending-silva.md",
    "Reproducing SILVA and Source Methods: learn/reproducing-silva-and-papers.md",
    "Benchmark Cards: experiments/benchmark-cards.md",
)

API_PAGE_NAMES = {
    "deq_engine": "deq-engine",
    "device": "devices",
    "jacobian": "jacobians",
}

STALE_PATTERNS = (
    "arXiv identifier pending",
    "identifier pending",
    "pending identifier",
    "once the public identifier",
    "well once the public",
    "Replace the public",
)

TEXT_SUFFIXES = {".md", ".cff", ".yml", ".yaml", ".toml", ".py", ".bib", ".ipynb"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "site", "dist", "build"}
PROVENANCE_TEXT_SUFFIXES = TEXT_SUFFIXES | {
    ".cfg",
    ".css",
    ".html",
    ".in",
    ".ini",
    ".js",
    ".json",
    ".rst",
    ".sh",
    ".txt",
}
PROVENANCE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
PROVENANCE_TOKEN_FINGERPRINTS = frozenset(
    {
        "57de4cf40144bdf7d00010f2f5557a7d642c2b9705309bfade167dd313e2ca93",
        "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
        "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
        "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
        "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
        "5d72436256ada53828b51895a94bb8489e9f1ac4fe937a8024ef1594e7045ff6",
        "3ea125d0bff386e6754b3782b300016fc79a9cf8f8669c0a5c3db64467ddb681",
    }
)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
VISIBLE_MATH_RE = re.compile(r"(\\\(|\\\)|\\\[|\\\]|\$\$|(?<!\\)\$(?=[^$]*(?:\\[A-Za-z]|[_^])))")
MATHJAX_PROCESSED_CLASSES = {
    "md-typeset",
    "arithmatex",
    "jp-RenderedHTMLCommon",
    "jp-RenderedMarkdown",
    "jp-MarkdownOutput",
    "mathjax",
}
IGNORED_HTML_TAGS = {"script", "style", "code", "pre", "textarea"}
TEST_SKIP_MARKERS = (
    "pytest.skip(",
    "pytest.mark.skip",
    "pytest.importorskip(",
    "unittest.skip",
)


def run_audit(root: Path = ROOT) -> dict[str, list[str]]:
    """Return release audit errors and warnings."""

    errors: list[str] = []
    warnings: list[str] = []

    _check_required_files(root, errors)
    _check_arxiv_and_bibtex(root, errors)
    _check_workflow_action_versions(root, errors)
    _check_test_skip_markers(root, errors)
    _check_notebook_curriculum(root, errors)
    _check_versions(root, errors)
    _check_nav(root, errors)
    _check_api_reference(root, errors)
    _check_stale_patterns(root, errors)
    _check_repository_provenance(root, errors)
    _check_coverage_registry(root, errors)
    _check_companion_asset_policy(root, errors, warnings)
    _check_docs_rendering_assets(root, errors)
    documentation_report = run_documentation_audit(root)
    errors.extend(documentation_report["errors"])
    warnings.extend(documentation_report["warnings"])
    preservation_report = run_content_preservation_audit(root)
    errors.extend(preservation_report["errors"])
    warnings.extend(preservation_report["warnings"])

    return {"errors": errors, "warnings": warnings}


def _check_required_files(root: Path, errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (root / relative).exists():
            errors.append(f"missing required file: {relative}")


def _check_workflow_action_versions(root: Path, errors: list[str]) -> None:
    found: set[str] = set()
    workflow_dir = root / ".github/workflows"
    for path in sorted(workflow_dir.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for action, version in WORKFLOW_USE_RE.findall(text):
            expected = EXPECTED_WORKFLOW_ACTIONS.get(action)
            if expected is None:
                continue
            found.add(action)
            if version != expected:
                errors.append(
                    f"{path.relative_to(root)} uses {action}@{version}; expected {action}@{expected}"
                )
    for action in sorted(EXPECTED_WORKFLOW_ACTIONS.keys() - found):
        errors.append(f"workflow action is not used: {action}@{EXPECTED_WORKFLOW_ACTIONS[action]}")


def _check_test_skip_markers(root: Path, errors: list[str]) -> None:
    for test_root in (root / "tests", root / "tests_extended"):
        if not test_root.exists():
            continue
        for path in sorted(test_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for marker in TEST_SKIP_MARKERS:
                if marker in text:
                    errors.append(f"{path.relative_to(root)} contains disallowed test skip marker: {marker}")


def _check_notebook_curriculum(root: Path, errors: list[str]) -> None:
    canonical = [
        *sorted((root / "notebooks/package_api").glob("*.ipynb")),
        *sorted((root / "notebooks/implicit_bridge").glob("*.ipynb")),
        *sorted((root / "notebooks").glob("*.ipynb")),
    ]
    if len(canonical) != 62:
        errors.append(f"expected 62 canonical notebooks; found {len(canonical)}")
        return
    for path in canonical:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        for stale in (
            "Upstream references to compare with:",
            "references/papers/recent_deq",
        ):
            if stale in source:
                errors.append(
                    f"{path.relative_to(root)} contains stale reader-facing text: {stale}"
                )
        cells = [
            cell
            for cell in notebook.get("cells", [])
            if "silva-extension-curriculum"
            in cell.get("metadata", {}).get("tags", [])
        ]
        if len(cells) != 4:
            errors.append(
                f"{path.relative_to(root)} has {len(cells)} extension curriculum cells; expected 4"
            )


def _check_arxiv_and_bibtex(root: Path, errors: list[str]) -> None:
    required_text = {
        "README.md": (ARXIV_ID, "silva2026silvanetworksstructuredimplicit"),
        "CITATION.cff": (ARXIV_ID, "https://arxiv.org/abs/2607.28989"),
        "MANIFEST.in": (
            ".zenodo.json",
            "CITATION.cff",
            "mkdocs.yml",
            "*.bib",
            "*.html",
            "scripts/smoke_test.sh",
            "docs/assets/papers/silva-networks-arxiv-2607.28989.pdf",
            "src/silva_networks/configs",
            "src/silva_networks/cases.py",
        ),
        "pyproject.toml": (
            "silva-experiment",
            "silva-download-datasets",
            "configs/*.json",
            "https://pypi.org/project/silva-networks/",
            "10.5281/zenodo.21770098",
        ),
        ".zenodo.json": (
            "Silva, Jose Luis",
            "10.48550/arXiv.2607.28989",
            "https://github.com/jseluis/silva-networks",
            "mit",
        ),
        ".github/workflows/release.yml": (
            "pypa/gh-action-pypi-publish@release/v1",
            "environment:",
            "name: pypi",
            "id-token: write",
            "python -m build",
            "twine check dist/*",
        ),
        "docs/publishing.md": (
            "PyPI Trusted Publishing",
            "silva-networks",
            "Workflow filename",
            "release.yml",
            ".zenodo.json",
            "10.5281/zenodo.21770098",
            "10.5281/zenodo.21770099",
        ),
        "docs/paper/references.md": (ARXIV_ID, "docs/assets/bib/silva-networks.bib"),
        "docs/assets/bib/silva-networks.bib": (
            "silva2026silvanetworksstructuredimplicit",
            "silva2026silvanetworkssoftware",
            "kolter2020deepimplicitlayers",
            "locuslab2019deq",
            "princeton2020raft",
            "locuslab2022deqflow",
            "gu2020implicit",
            "huang2021implicit2",
            "pokle2022deqddim",
            "rumelhart1986learning",
            "he2016deepresidual",
            "ronneberger2015unet",
            "huang2017densely",
            "vaswani2017attention",
            "sandler2018mobilenetv2",
            "li2021fourier",
            "kovachki2023neuraloperator",
            "tolstikhin2021mlpmixer",
            "woo2023convnextv2",
            "marwah2023fnodeq",
            "rodrigobonet2024pgcndeq",
            "geuter2025ddeq",
            "ding2023homoode",
            "baker2023mignn",
            "geng2023get",
            "cao2024deqir",
            "daniele2025deqmd",
        ),
        "docs/book.md": ("Planned", "planned learning resources"),
    }
    for relative, markers in required_text.items():
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(f"{relative} is missing marker: {marker}")


def _check_versions(root: Path, errors: list[str]) -> None:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]
    init_text = (root / "src/silva_networks/__init__.py").read_text(encoding="utf-8")
    expected = f'__version__ = "{package_version}"'
    if expected not in init_text:
        errors.append(f"package version mismatch: expected {expected}")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if f'version: "{package_version}"' not in citation:
        errors.append(f"CITATION.cff does not identify version {package_version}")
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    if zenodo.get("version") != package_version:
        errors.append(
            ".zenodo.json version does not match package version: "
            f"{zenodo.get('version')!r} != {package_version!r}"
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {package_version}" not in changelog:
        errors.append(f"CHANGELOG.md has no section for version {package_version}")
    mkdocs = (root / "mkdocs.yml").read_text(encoding="utf-8")
    expected_source_tag = f"source_release_tag: v{package_version}"
    if expected_source_tag not in mkdocs:
        errors.append(f"mkdocs.yml must identify the source widget tag: {expected_source_tag}")
    override = (root / "docs/overrides/main.html").read_text(encoding="utf-8")
    for marker in ("source_release_tag", "__source", "__source_cached_at", "expectedSourceVersion"):
        if marker not in override:
            errors.append(f"docs/overrides/main.html is missing source cache marker: {marker}")
    for relative in (
        "README.md",
        "docs/paper/references.md",
        "docs/assets/bib/silva-networks.bib",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        if package_version not in text:
            errors.append(f"{relative} does not identify version {package_version}")


def _check_nav(root: Path, errors: list[str]) -> None:
    nav = (root / "mkdocs.yml").read_text(encoding="utf-8")
    for marker in REQUIRED_NAV_MARKERS:
        if marker not in nav:
            errors.append(f"mkdocs navigation is missing: {marker}")


def _check_api_reference(root: Path, errors: list[str]) -> None:
    nav = (root / "mkdocs.yml").read_text(encoding="utf-8")
    api_dir = root / "docs/api"
    public_api = api_dir / "public-api.md"
    public_text = public_api.read_text(encoding="utf-8") if public_api.exists() else ""
    if "::: silva_networks" not in public_text:
        errors.append("docs/api/public-api.md should generate the top-level silva_networks API")
    for path in sorted((root / "src/silva_networks").glob("*.py")):
        module = path.stem
        if module == "__init__":
            continue
        page_name = API_PAGE_NAMES.get(module, module)
        page = api_dir / f"{page_name}.md"
        if not page.exists():
            errors.append(
                f"missing API reference page for silva_networks.{module}: {page.relative_to(root)}"
            )
            continue
        text = page.read_text(encoding="utf-8")
        if f"silva_networks.{module}" not in text:
            errors.append(f"{page.relative_to(root)} does not include silva_networks.{module}")
        if f"api/{page_name}.md" not in nav:
            errors.append(f"mkdocs navigation is missing API page: api/{page_name}.md")


def _check_stale_patterns(root: Path, errors: list[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in STALE_PATTERNS:
            if pattern in text:
                errors.append(f"stale citation text in {path.relative_to(root)}: {pattern}")


def _contains_blocked_provenance_token(text: str) -> bool:
    for match in PROVENANCE_TOKEN_RE.finditer(text):
        token = match.group(0).lower().encode("utf-8")
        if hashlib.sha256(token).hexdigest() in PROVENANCE_TOKEN_FINGERPRINTS:
            return True
    return False


def _tracked_paths(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
    ]


def _provenance_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        chunks = [json.dumps(notebook.get("metadata", {}), sort_keys=True)]
        chunks.extend("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        return "\n".join(chunks)
    return path.read_text(encoding="utf-8", errors="ignore")


def _check_repository_provenance(root: Path, errors: list[str]) -> None:
    for path in _tracked_paths(root):
        relative = path.relative_to(root)
        if _contains_blocked_provenance_token(relative.as_posix()):
            errors.append(f"blocked provenance reference in repository path: {relative}")
        if path.suffix.lower() not in PROVENANCE_TEXT_SUFFIXES:
            continue
        if _contains_blocked_provenance_token(_provenance_text(path)):
            errors.append(f"blocked provenance reference in repository text: {relative}")

    if not (root / ".git").exists():
        return
    result = subprocess.run(
        ["git", "log", "--all", "--format=%B%x00"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and _contains_blocked_provenance_token(result.stdout):
        errors.append("blocked provenance reference in commit history")


def _check_coverage_registry(root: Path, errors: list[str]) -> None:
    sys.path.insert(0, str(root / "src"))
    try:
        import silva_networks as sn
        from silva_networks.coverage import implementation_cases
    except Exception as exc:  # noqa: BLE001 - an audit should report every import failure
        errors.append(f"could not import package coverage registry: {exc}")
        return

    for case in implementation_cases():
        for relative in (case.tutorial, *case.notebooks, *case.smoke_tests, *case.examples):
            if not (root / relative).exists():
                errors.append(f"coverage case {case.key} points to missing file: {relative}")
        for public_object in case.public_objects:
            if not hasattr(sn, public_object):
                errors.append(
                    f"coverage case {case.key} names missing public object: {public_object}"
                )


def _check_companion_asset_policy(root: Path, errors: list[str], warnings: list[str]) -> None:
    article_pdfs = [
        p
        for p in root.rglob("*.pdf")
        if "article" in p.name.lower() or "paper" in p.name.lower() or ARXIV_ID in p.name
    ]
    if not article_pdfs:
        warnings.append("no separate article PDF found inside the package; arXiv is linked instead")

    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    if "*.pdf" in manifest:
        errors.append("MANIFEST.in should include PDFs explicitly rather than through a wildcard")
    if "prune docs/assets/pdfs" not in manifest:
        errors.append("MANIFEST.in should prune the companion PDF workspace from release artifacts")

    direct_pdf_markers = (
        "assets/pdfs/silva_deq_companion_book.pdf",
        "assets/pdfs/silva_deq_solutions_manual.pdf",
    )
    for path in [root / "README.md", *(root / "docs").rglob("*.md")]:
        text = path.read_text(encoding="utf-8")
        for marker in direct_pdf_markers:
            if marker in text:
                errors.append(f"{path.relative_to(root)} links a planned companion PDF directly")

    sources = root / "src/silva_networks.egg-info/SOURCES.txt"
    if sources.exists():
        sources_text = sources.read_text(encoding="utf-8")
        for marker in direct_pdf_markers:
            if marker in sources_text:
                errors.append(f"{sources.relative_to(root)} lists a planned companion PDF")

    local_drafts = [
        root / "docs/assets/pdfs/silva_deq_companion_book.pdf",
        root / "docs/assets/pdfs/silva_deq_solutions_manual.pdf",
    ]
    if any(pdf.exists() for pdf in local_drafts):
        warnings.append(
            "companion book/manual PDF workspace exists outside public documentation links"
        )


def _png_dimensions_and_dpi(data: bytes) -> tuple[int, int, tuple[float, float] | None]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG")
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


def _check_docs_rendering_assets(root: Path, errors: list[str]) -> None:
    mkdocs_config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    for marker in ("custom_dir: docs/overrides", "include_source: true"):
        if marker not in mkdocs_config:
            errors.append(f"mkdocs.yml is missing notebook download setting: {marker}")

    notebook_override = root / "docs/overrides/main.html"
    if notebook_override.exists():
        override_text = notebook_override.read_text(encoding="utf-8")
        for marker in ("page.nb_url", "download", "Download notebook"):
            if marker not in override_text:
                errors.append(f"docs/overrides/main.html is missing marker: {marker}")

    mathjax = root / "docs/javascripts/mathjax.js"
    if mathjax.exists():
        config = mathjax.read_text(encoding="utf-8")
        for marker in (
            '["$", "$"]',
            "md-typeset",
            "jp-RenderedHTMLCommon",
            "jp-MarkdownOutput",
            "data-silva-mathjax-loader",
            "vendor/mathjax/tex-mml-chtml.js",
            "cdn.jsdelivr.net/npm/mathjax@3",
            "cdnjs.cloudflare.com/ajax/libs/mathjax",
        ):
            if marker not in config:
                errors.append(f"MathJax config is missing notebook rendering marker: {marker}")
    local_mathjax = root / "docs/javascripts/vendor/mathjax/tex-mml-chtml.js"
    local_mathjax_license = root / "docs/javascripts/vendor/mathjax/LICENSE.txt"
    if not local_mathjax.exists() or local_mathjax.stat().st_size < 1_000_000:
        errors.append("local MathJax browser bundle is missing or unexpectedly small")
    else:
        required_fonts = sorted(
            set(
                re.findall(
                    r"MathJax_[A-Za-z0-9_-]+\.woff",
                    local_mathjax.read_text(encoding="utf-8"),
                )
            )
        )
        font_dir = root / "docs/javascripts/vendor/mathjax/output/chtml/fonts/woff-v2"
        missing_fonts = [font for font in required_fonts if not (font_dir / font).exists()]
        if len(required_fonts) < 20:
            errors.append("local MathJax bundle did not expose the expected webfont list")
        if missing_fonts:
            errors.append("local MathJax webfonts are missing: " + ", ".join(missing_fonts))
    if (
        not local_mathjax_license.exists()
        or "Apache License" not in local_mathjax_license.read_text(encoding="utf-8")
    ):
        errors.append("local MathJax license is missing")
    mkdocs = root / "mkdocs.yml"
    if mkdocs.exists():
        mkdocs_text = mkdocs.read_text(encoding="utf-8")
        if "javascripts/vendor/mathjax/tex-mml-chtml.js" not in mkdocs_text:
            errors.append("mkdocs.yml should load the local MathJax bundle")
        if "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" in mkdocs_text:
            errors.append("mkdocs.yml should let docs/javascripts/mathjax.js own MathJax loading")

    _check_math_source_delimiters(root, errors)
    _check_built_html_math_context(root, errors)

    for folder in [root / "docs", root / "notebooks", root / "colab"]:
        for path in sorted(folder.rglob("*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                source = "".join(cell.get("source", []))
                if r"\(" in source or r"\)" in source:
                    errors.append(
                        f"{path.relative_to(root)}:cell{index} uses notebook-unsafe inline math"
                    )

    png_outputs = []
    for path in sorted((root / "docs").rglob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(nb.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs", []) or []):
                encoded = (output.get("data") or {}).get("image/png")
                if encoded is None:
                    continue
                width, height, dpi = _png_dimensions_and_dpi(base64.b64decode(encoded))
                png_outputs.append((path, cell_index, output_index, width, height, dpi))
                if dpi is None or min(dpi) < MIN_PUBLICATION_DPI:
                    errors.append(
                        f"{path.relative_to(root)}:cell{cell_index}:output{output_index} "
                        f"is below 300 dpi: {width}x{height}, dpi={dpi}"
                    )
    if len(png_outputs) < 20:
        errors.append("docs notebooks should retain executed high-resolution PNG outputs")
    if not any(path.name == "04_deq_and_silva.ipynb" for path, *_ in png_outputs):
        errors.append("DEQ and SILVA notebook is missing its residual diagnostic PNG output")

    for path in sorted((root / "docs/assets/images").glob("*.png")):
        width, height, dpi = _png_dimensions_and_dpi(path.read_bytes())
        if dpi is None or min(dpi) < MIN_PUBLICATION_DPI:
            errors.append(f"{path.relative_to(root)} is below 300 dpi: {width}x{height}, dpi={dpi}")


def _iter_markdown_units(root: Path) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for path in sorted((root / "docs").rglob("*.md")):
        units.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    for folder in [root / "docs", root / "notebooks", root / "colab"]:
        for path in sorted(folder.rglob("*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "markdown":
                    units.append(
                        (
                            f"{path.relative_to(root)}:cell{index}",
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


def _check_math_source_delimiters(root: Path, errors: list[str]) -> None:
    for label, raw_source in _iter_markdown_units(root):
        source = _without_code(raw_source)
        if source.count("$$") % 2:
            errors.append(f"{label} has unbalanced $$ math delimiters")
        display_parts = source.split("$$")
        if any(not display_parts[index].strip() for index in range(1, len(display_parts), 2)):
            errors.append(f"{label} has an empty $$ display-math block")
        if source.count(r"\[") != source.count(r"\]"):
            errors.append(f"{label} has unbalanced \\[...\\] math delimiters")
        if source.count(r"\(") != source.count(r"\)"):
            errors.append(f"{label} has unbalanced \\(...\\) math delimiters")
        if _single_dollar_count(source) % 2:
            errors.append(f"{label} has unbalanced $...$ math delimiters")


class _MathContextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.offenders: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        class_attr = ""
        for key, value in attrs:
            if key == "class":
                class_attr = value or ""
                break
        self.stack.append((tag, set(class_attr.split())))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if not VISIBLE_MATH_RE.search(data):
            return
        if any(tag in IGNORED_HTML_TAGS for tag, _classes in self.stack):
            return
        if any(classes & MATHJAX_PROCESSED_CLASSES for _tag, classes in self.stack):
            return
        snippet = " ".join(data.split())[:120]
        self.offenders.append(snippet)


def _check_built_html_math_context(root: Path, errors: list[str]) -> None:
    site = root / "site"
    if not site.exists():
        return

    for path in sorted(site.rglob("*.html")):
        parser = _MathContextParser()
        parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
        if parser.offenders:
            errors.append(
                f"{path.relative_to(root)} has raw math outside MathJax-processed containers: "
                + "; ".join(parser.offenders[:3])
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = run_audit(ROOT)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("SILVA Networks release audit")
        print(f"errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(f"warnings: {len(report['warnings'])}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
