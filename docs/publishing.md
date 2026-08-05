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

## PyPI

The package name is:

```text
silva-networks
```

Published package:
[https://pypi.org/project/silva-networks/](https://pypi.org/project/silva-networks/)

The import name is:

```python
import silva_networks
```

Publishing is configured through the `Release` workflow in
`.github/workflows/release.yml`. Pushing a validated `v*` tag builds and checks
the distributions, publishes them to PyPI after environment approval, and then
creates the GitHub Release with the same artifacts. It uses PyPI Trusted
Publishing, so no PyPI API token is stored in GitHub. PyPI's trusted-publisher
flow exchanges a GitHub Actions identity token for a short-lived publishing
credential during the release job.

The v1.0.0 release established this PyPI trusted-publisher configuration, and
v1.1.0 uses the same publisher identity:

1. Open PyPI and sign in.
2. Open **Account settings**.
3. Open **Publishing**.
4. Add a pending trusted publisher with:

| Field | Value |
| --- | --- |
| PyPI project name | `silva-networks` |
| Owner | `jseluis` |
| Repository name | `silva-networks` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Then configure the matching GitHub environment:

1. Open the GitHub repository.
2. Open **Settings**.
3. Open **Environments**.
4. Create an environment named `pypi`.
5. Add a required reviewer for the environment.

The environment gate keeps package upload as an explicit release action. The
release workflow builds the source distribution and wheel, runs `twine check`,
publishes the package to PyPI after deployment approval, and creates the GitHub
Release only after the package upload succeeds.

After the workflow succeeds:

```bash
python -m pip install silva-networks
python -c "import silva_networks; print(silva_networks.__version__)"
```

## Zenodo

Zenodo archiving should be connected before each public GitHub Release. The
repository includes `.zenodo.json`, which gives Zenodo software metadata, the
MIT license, keywords, repository relation, and the arXiv article relation.

To enable the archive:

1. Open Zenodo and sign in with GitHub.
2. Open the GitHub integration page.
3. Enable archiving for `jseluis/silva-networks`.
4. Push the validated version tag, currently `v1.1.0`; the release workflow
   creates the corresponding GitHub Release.

Use the concept DOI for the living software citation:

```text
All-versions DOI: 10.5281/zenodo.21770098
Latest record: https://doi.org/10.5281/zenodo.21770098
```

The immutable v1.0.0 archive remains:

```text
Version DOI: 10.5281/zenodo.21770099
Concept DOI: 10.5281/zenodo.21770098
Record: https://zenodo.org/records/21770099
```

Publishing the `v1.1.0` GitHub release creates a new version record under the
same concept DOI. Record its minted version DOI in the release readiness page
after the integration finishes.

Official setup references:

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Publishing with a trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Creating a PyPI project with a trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
- [Zenodo `.zenodo.json`](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)

## Website Mirror

If the site is mirrored to `jsluis.com`, keep GitHub Pages enabled as the
package-native documentation target. That keeps all repository links,
notebook paths, GitHub Actions checks, and release documentation stable.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which checks must pass before publication? | [Release Readiness](release-readiness.md) |
| How should the release be cited? | [Citation-Aware Reporting](examples/citation-aware-reporting.md) |
| Where are article and software identifiers recorded? | [Paper and References](paper/references.md) |

<!-- silva-extension-path:start -->
--8<-- "includes/extension/project.md"
<!-- silva-extension-path:end -->
