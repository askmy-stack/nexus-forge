#!/usr/bin/env bash
# Build and publish nexus-forge to PyPI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing build tools"
uv pip install build twine

echo "==> Building wheel and sdist"
rm -rf dist/
uv run python -m build

echo "==> Checking distribution"
uv run twine check dist/*

if [[ "${1:-}" == "--upload" ]]; then
  if [[ -z "${PYPI_API_TOKEN:-}" ]]; then
    echo "Error: set PYPI_API_TOKEN before uploading." >&2
    echo "  export PYPI_API_TOKEN=pypi-..." >&2
    exit 1
  fi
  echo "==> Uploading to PyPI"
  uv run twine upload dist/* -u __token__ -p "$PYPI_API_TOKEN"
else
  echo ""
  echo "Dry run complete. To upload:"
  echo "  export PYPI_API_TOKEN=pypi-..."
  echo "  $0 --upload"
  echo ""
  echo "Or test on TestPyPI first:"
  echo "  uv run twine upload --repository testpypi dist/* -u __token__ -p \"\$TEST_PYPI_API_TOKEN\""
fi
