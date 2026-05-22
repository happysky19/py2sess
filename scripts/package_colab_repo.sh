#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/py2sess_colab.zip}"
NAME="$(basename "$ROOT")"
TMP="${OUT}.tmp.$$"

cleanup() {
  rm -f "$TMP"
}
trap cleanup EXIT

cd "$(dirname "$ROOT")"
zip -qr "$TMP" "$NAME" \
  -x "$NAME/.git/*" \
  -x "$NAME/build/*" \
  -x "$NAME/dist/*" \
  -x "$NAME/.mypy_cache/*" \
  -x "$NAME/.pytest_cache/*" \
  -x "$NAME/.ruff_cache/*" \
  -x "$NAME/.gitignore" \
  -x "$NAME/src/*.egg-info/*" \
  -x "$NAME/src/py2sess/_native*.so" \
  -x "$NAME/*.zip" \
  -x "$NAME/*__pycache__*" \
  -x "$NAME/outputs/*" \
  -x "$NAME/docs/py2sess_rt_benchmark_paper.pdf" \
  -x "$NAME/scripts/oco3_paper_support/*"

mv "$TMP" "$OUT"
echo "$OUT"
