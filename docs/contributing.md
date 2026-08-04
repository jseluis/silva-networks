# Contributing

Development commands:

```bash
python -m pip install -e ".[dev,docs,examples]"
pytest
ruff check src tests examples scripts
mkdocs build --strict
```

Contribution guidelines:

- keep the public API typed and documented;
- add tests for new solvers, layers, examples, and shape behavior;
- keep examples CPU-first;
- cite third-party papers and repositories through canonical links;
- update the API reference when public symbols change.

## Guided Page Endings

Every Markdown documentation page ends with:

```markdown
## Where to Go Next

| Question | Page |
| --- | --- |
| What is the reader likely to ask next? | [Relevant page](relative-link.md) |
```

Use three or four distinct destinations. Phrase the first column as actual
reader questions, and select links that connect explanation, execution, and API
reference where possible. Rendered notebooks use the same table in a final
tagged Markdown cell. Synchronize those cells across notebook copies with:

```bash
python scripts/notebook_navigation.py
```

The documentation audit validates the heading, table shape, question wording,
destination uniqueness, cross-section reach, local link targets, notebook cell
placement, and notebook source synchronization.

## Where to Go Next

| Question | Page |
| --- | --- |
| Which documentation changes have already been recorded? | [Documentation Log](documentation-log.md) |
| Which checks must a contribution pass? | [Release Readiness](release-readiness.md) |
| How can contributors run the complete local workflow? | [Run Everything](run-everything.md) |
