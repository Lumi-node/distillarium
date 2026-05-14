#!/usr/bin/env bash
# release.sh — local release helper for distillarium
#
# Usage:
#   ./release.sh check      # build + twine check (no upload)
#   ./release.sh test       # publish to TestPyPI
#   ./release.sh publish    # publish to real PyPI
#
# Requires DISTILLARIUM_PYPI_KEY in ../.env (project-scoped token).
# For TestPyPI use TESTPYPI_API_KEY in ../.env (separate token).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

# Load env from project root .env
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

cmd="${1:-check}"

clean_build() {
  echo "▸ Cleaning and rebuilding..."
  rm -rf dist build src/*.egg-info
  python -m build
  echo "▸ Checking distributions..."
  twine check dist/*
}

case "$cmd" in
  check)
    clean_build
    echo
    echo "✓ Build OK. Run './release.sh test' to publish to TestPyPI,"
    echo "  or './release.sh publish' for real PyPI."
    ;;

  test)
    if [ -z "${TESTPYPI_API_KEY:-}" ]; then
      echo "✗ TESTPYPI_API_KEY not set in $PROJECT_ROOT/.env" >&2
      exit 1
    fi
    clean_build
    echo "▸ Uploading to TestPyPI..."
    TWINE_USERNAME=__token__ TWINE_PASSWORD="$TESTPYPI_API_KEY" \
      twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    echo "✓ Published to TestPyPI."
    echo "  Test install:"
    echo "    pip install -i https://test.pypi.org/simple/ distillarium"
    ;;

  publish)
    if [ -z "${DISTILLARIUM_PYPI_KEY:-}" ]; then
      echo "✗ DISTILLARIUM_PYPI_KEY not set in $PROJECT_ROOT/.env" >&2
      exit 1
    fi
    clean_build
    # Read current version from pyproject.toml
    version=$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/version = "([^"]+)"/\1/')
    echo "▸ About to publish distillarium $version to REAL PyPI."
    echo "  This is irreversible — the version cannot be republished."
    read -p "  Continue? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
      echo "  Aborted."
      exit 0
    fi
    TWINE_USERNAME=__token__ TWINE_PASSWORD="$DISTILLARIUM_PYPI_KEY" \
      twine upload dist/*
    echo "✓ Published to PyPI: https://pypi.org/project/distillarium/$version/"
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: $0 {check|test|publish}" >&2
    exit 1
    ;;
esac
