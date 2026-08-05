"""Add the shared extension and reproduction path to documentation pages."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
START = "<!-- silva-extension-path:start -->"
END = "<!-- silva-extension-path:end -->"


def _snippet_for(path: Path) -> str | None:
    relative = path.relative_to(DOCS)
    if relative.parts[0] in {"includes", "paper"} or relative.as_posix() == "index.md":
        return None
    if relative.parts[0] == "learn":
        if relative.name == "extending-silva.md":
            return None
        return 'includes/extension/learn.md'
    if relative.parts[0] == "api":
        if relative.name == "extensibility.md":
            return None
        return 'includes/extension/api.md'
    if relative.parts[0] == "examples":
        return 'includes/extension/examples.md'
    if relative.parts[0] == "experiments":
        return 'includes/extension/experiments.md'
    if relative.parts[0] == "get-started":
        return 'includes/extension/get-started.md'
    if relative.parts[0] == "cheatsheets":
        return 'includes/extension/cheatsheets.md'
    if len(relative.parts) == 1:
        return 'includes/extension/project.md'
    return None


def _replace_block(text: str, snippet: str) -> str:
    block = f'{START}\n--8<-- "{snippet}"\n{END}'
    if START in text:
        prefix, remainder = text.split(START, 1)
        if END not in remainder:
            raise ValueError("extension path marker is incomplete")
        _, suffix = remainder.split(END, 1)
        return prefix.rstrip() + "\n\n" + block + suffix
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> int:
    changed = 0
    for path in sorted(DOCS.rglob("*.md")):
        snippet = _snippet_for(path)
        if snippet is None:
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_block(original, snippet)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"updated {changed} documentation pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
