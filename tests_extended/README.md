# Extended Smoke Tests

These tests are optional extended checks for longer tutorial and experiment
verification. They are intentionally outside `pyproject.toml` `testpaths`, so
the public command

```bash
python -m pytest
```

stays fast for package users.

Run them explicitly with:

```bash
python -m pytest tests_extended
```
