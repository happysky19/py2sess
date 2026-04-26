#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_full_benchmark_threads.sh UV_BUNDLE TIR_BUNDLE

Environment:
  PYTHON=python3          Python executable
  BACKEND=both           numpy, torch, or both
  THREADS="1 2 4"        Thread counts to run
  TORCH_DEVICE=cpu       Torch device
  TORCH_DTYPE=float64    Torch dtype
  LIMIT=                 Optional spectral-row limit
  CHUNK_SIZE=            Optional chunk-size override
  OUTPUT_LEVELS=0        Set to 1 to benchmark profile output
  USE_DUMPED_DERIVED_OPTICS=0
                        Set to 1 to bypass Python optical preprocessing
  USE_DUMPED_THERMAL_SOURCE=0
                        Set to 1 to bypass TIR temperature-source generation
  REQUIRE_PYTHON_GENERATED_INPUTS=0
                        Set to 1 to fail on dumped/direct RT input fallback
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BUNDLE="$1"
TIR_BUNDLE="$2"

PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-both}"
THREADS="${THREADS:-1 2 4}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
OUTPUT_LEVELS="${OUTPUT_LEVELS:-0}"

[[ -f "$UV_BUNDLE" ]] || die "UV bundle not found: $UV_BUNDLE"
[[ -f "$TIR_BUNDLE" ]] || die "TIR bundle not found: $TIR_BUNDLE"

case "$BACKEND" in
  numpy | torch | both) ;;
  *) die "BACKEND must be numpy, torch, or both" ;;
esac

common_args=(
  --backend "$BACKEND"
  --torch-device "$TORCH_DEVICE"
  --torch-dtype "$TORCH_DTYPE"
)

if [[ -n "${LIMIT:-}" ]]; then
  common_args+=(--limit "$LIMIT")
fi

if [[ -n "${CHUNK_SIZE:-}" ]]; then
  common_args+=(--chunk-size "$CHUNK_SIZE")
fi

if [[ "$OUTPUT_LEVELS" == "1" ]]; then
  common_args+=(--output-levels)
fi

if [[ "${USE_DUMPED_DERIVED_OPTICS:-0}" == "1" ]]; then
  common_args+=(--use-dumped-derived-optics)
fi

if [[ "${REQUIRE_PYTHON_GENERATED_INPUTS:-0}" == "1" ]]; then
  common_args+=(--require-python-generated-inputs)
fi

run_case() {
  local name="$1"
  local script="$2"
  local bundle="$3"
  local threads="$4"
  local case_args=("${common_args[@]}")

  if [[ "$name" == "TIR" && "${USE_DUMPED_THERMAL_SOURCE:-0}" == "1" ]]; then
    case_args+=(--use-dumped-thermal-source)
  fi

  echo
  echo "== $name | threads=$threads | backend=$BACKEND =="

  OMP_NUM_THREADS="$threads" \
  OPENBLAS_NUM_THREADS="$threads" \
  MKL_NUM_THREADS="$threads" \
  VECLIB_MAXIMUM_THREADS="$threads" \
  NUMEXPR_NUM_THREADS="$threads" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$ROOT_DIR/examples/$script" "$bundle" \
      "${case_args[@]}" \
      --torch-threads "$threads"
}

echo "py2sess full-spectrum benchmark sweep"
echo "  UV bundle:  $UV_BUNDLE"
echo "  TIR bundle: $TIR_BUNDLE"

for threads in $THREADS; do
  run_case "UV" "benchmark_uv_full_spectrum.py" "$UV_BUNDLE" "$threads"
  run_case "TIR" "benchmark_tir_full_spectrum.py" "$TIR_BUNDLE" "$threads"
done
