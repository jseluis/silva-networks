#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEFAULT_OUTPUT_DIR="${TMPDIR:-/tmp}/silva-networks-smoke"
OUTPUT_DIR="${SILVA_SMOKE_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

WITH_DOCS=0
WITH_NOTEBOOKS=0
WITH_BUILD=0
WITH_VISION=0
WITH_OPTIMIZATION=0

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_test.sh [options]

Default checks are CPU-first and avoid large downloads:
  - import silva_networks
  - run quick examples
  - run solver and graph public configs through the CLI
  - run focused pytest checks

Options:
  --with-docs          also run mkdocs build --strict
  --with-notebooks     also run the selected notebook smoke set
  --with-build         also build wheel/sdist and run twine check
  --with-vision        also run the CIFAR10 vector smoke; may download CIFAR10
  --with-optimization  also run optimization tests; requires the optimization extra
  --all-local          docs + notebooks + build + optimization, excluding vision downloads
  -h, --help           show this message

Environment:
  PYTHON=/path/to/python
  SILVA_SMOKE_OUTPUT_DIR=/path/to/output_dir
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-docs)
      WITH_DOCS=1
      ;;
    --with-notebooks)
      WITH_NOTEBOOKS=1
      ;;
    --with-build)
      WITH_BUILD=1
      ;;
    --with-vision)
      WITH_VISION=1
      ;;
    --with-optimization)
      WITH_OPTIMIZATION=1
      ;;
    --all-local)
      WITH_DOCS=1
      WITH_NOTEBOOKS=1
      WITH_BUILD=1
      WITH_OPTIMIZATION=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  echo
  echo "+ $*"
  "$@"
}

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

if command -v silva-experiment >/dev/null 2>&1; then
  SILVA_EXPERIMENT=(silva-experiment)
else
  SILVA_EXPERIMENT=("$PYTHON" -m silva_networks.public_experiments)
fi

run "$PYTHON" - <<'PY'
import silva_networks as sn

print("silva_networks", sn.__version__)
print("families", ", ".join(sn.available_silva_families()[:6]) + ", ...")
PY

run "$PYTHON" examples/scalar_deq.py
run "$PYTHON" examples/stacked_architecture.py

run "${SILVA_EXPERIMENT[@]}" \
  --config solver_sweep \
  --device cpu \
  --output-dir "$OUTPUT_DIR"

run "${SILVA_EXPERIMENT[@]}" \
  --config graph_silva_smoke \
  --device cpu \
  --set steps=1 \
  --set solver.max_iter=2 \
  --output-dir "$OUTPUT_DIR"

run "$PYTHON" -m pytest \
  tests/test_solvers.py \
  tests/test_layers.py \
  tests/test_datasets.py \
  tests/test_public_experiments.py \
  -q -ra

if [[ "$WITH_OPTIMIZATION" == "1" ]]; then
  run "$PYTHON" -m pytest tests/test_optimization.py -q -ra
fi

if [[ "$WITH_VISION" == "1" ]]; then
  run "${SILVA_EXPERIMENT[@]}" \
    --config cifar10_vector_smoke \
    --device cpu \
    --set max_samples=8 \
    --set steps=1 \
    --set max_iter=1 \
    --output-dir "$OUTPUT_DIR"
fi

if [[ "$WITH_NOTEBOOKS" == "1" ]]; then
  run "$PYTHON" scripts/run_notebook_smoke.py --timeout 180
fi

if [[ "$WITH_DOCS" == "1" ]]; then
  run "$PYTHON" -m mkdocs build --strict
fi

if [[ "$WITH_BUILD" == "1" ]]; then
  run "$PYTHON" -m build
  run "$PYTHON" -m twine check dist/*
fi

echo
echo "SILVA smoke test completed."
