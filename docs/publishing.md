# Publishing

The documentation is designed to run directly on GitHub Pages through the
`Pages` workflow in `.github/workflows/pages.yml`.

## Recommended Public Setup

Use GitHub Pages as the canonical technical documentation host:

```text
https://jseluis.github.io/silva-networks/
```

GitHub Pages is the best first home for this project because the package,
issues, examples, notebooks, API reference, release checks, and source history
live in the same repository. The site is versioned with the code and rebuilt
from the same MkDocs configuration used locally.

The personal website can still be the front door. Link from `https://jsluis.com`
to the GitHub Pages site, or mirror the built `site/` directory later if a
single-domain presentation becomes preferable.

## Repository Settings

After pushing the repository to GitHub:

1. Open **Settings**.
2. Open **Pages**.
3. Set **Build and deployment** to **GitHub Actions**.
4. Do not select the suggested **GitHub Pages Jekyll** or **Static HTML**
   workflows. This repository already provides the custom MkDocs workflow in
   `.github/workflows/pages.yml`.
5. Push to `main` or run the **Pages** workflow manually.

The workflow installs the documentation extras and runs:

```bash
mkdocs build --strict
```

The published artifact includes the rendered MkDocs site, rendered notebooks,
MathJax assets, figures, and the local SILVA article PDF.

## Local Preview

For a local preview:

```bash
python -m pip install -e ".[docs,notebooks,examples]"
mkdocs serve
```

For a release-style local check:

```bash
mkdocs build --strict
python scripts/release_audit.py
```

## Website Mirror

If the site is mirrored to `jsluis.com`, keep GitHub Pages enabled as the
package-native documentation target. That keeps all repository links,
notebook paths, GitHub Actions checks, and release documentation stable.
