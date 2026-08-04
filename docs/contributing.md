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

Every Markdown page in `docs/get-started`, `docs/learn`, `docs/examples`,
`docs/api`, and `docs/experiments` ends with:

```markdown
## Where to Go Next

| Question | Page |
| --- | --- |
| What is the reader likely to ask next? | [Relevant page](relative-link.md) |
```

Use three or four distinct destinations. Phrase the first column as actual
reader questions, and select links that connect explanation, execution, and API
reference where possible. The documentation audit validates the heading, table
shape, question wording, destination uniqueness, and local link targets.
