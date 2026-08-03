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
