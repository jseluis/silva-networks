# Contributing

Thank you for improving SILVA Networks.

## Development Setup

```bash
python -m pip install -e ".[dev,docs,examples]"
pytest
mkdocs build --strict
```

## Coding Standards

- Keep package code typed and readable.
- Prefer small, well-named modules over large monolithic files.
- Add tests for solver behavior, gradients, shapes, and docs/notebook validity.
- Keep examples CPU-first and fast enough for local smoke tests.
- Do not vendor third-party papers or upstream repositories into the public
  repository.

## Pull Request Checklist

- Tests pass with `pytest`.
- Documentation builds with `mkdocs build --strict`.
- New public API entries are documented in `docs/api/reference.md`.
- New examples have a short docs page and a test or smoke command.

