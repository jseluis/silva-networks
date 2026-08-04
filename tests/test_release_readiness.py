from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

API_PAGE_NAMES = {
    "deq_engine": "deq-engine",
    "device": "devices",
    "jacobian": "jacobians",
}


def test_release_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/release_audit.py", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["errors"] == []


def test_bibtex_contains_silva_article_and_related_sources() -> None:
    bibtex = (ROOT / "docs/assets/bib/silva-networks.bib").read_text(encoding="utf-8")
    for key in [
        "silva2026silvanetworksstructuredimplicit",
        "kolter2020deepimplicitlayersdeq",
        "locuslab2019deq",
        "torchdeq2023",
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
    ]:
        assert f"{{{key}," in bibtex
    assert "2607.28989" in bibtex


def test_notebook_smoke_script_lists_defaults() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_notebook_smoke.py", "--list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "docs/package-notebooks/07_research_citation_audit.ipynb" in result.stdout
    assert "docs/implicit-bridge-notebooks/09_method_adaptation_atlas.ipynb" in result.stdout
    assert "docs/package-notebooks/12_paper_family_architectures.ipynb" in result.stdout
    assert "docs/package-notebooks/13_raft_deq_flow.ipynb" in result.stdout
    assert "docs/package-notebooks/14_point_architecture_catalog.ipynb" in result.stdout
    assert "docs/package-notebooks/15_neural_operators_ode_pde.ipynb" in result.stdout


def test_api_reference_covers_every_source_module() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    public_api = (ROOT / "docs/api/public-api.md").read_text(encoding="utf-8")
    assert "::: silva_networks" in public_api
    assert "api/public-api.md" in nav
    for path in sorted((ROOT / "src/silva_networks").glob("*.py")):
        module = path.stem
        if module == "__init__":
            continue
        page_name = API_PAGE_NAMES.get(module, module)
        page = ROOT / "docs/api" / f"{page_name}.md"
        assert page.exists(), f"missing API page for silva_networks.{module}"
        assert f"silva_networks.{module}" in page.read_text(encoding="utf-8")
        assert f"api/{page_name}.md" in nav
