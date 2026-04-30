#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  UV_PROFILE=profile.txt UV_SCENE=uv.yaml \
  TIR_PROFILE=profile.txt TIR_SCENE=tir.yaml \
  scripts/run_full_benchmark_threads.sh

Environment:
  PYTHON=python3          Python executable
  BACKEND=both           numpy, torch, or both
  THREADS="1 2 4"        Thread counts to run
  TORCH_DEVICE=cpu       Torch device
  TORCH_DTYPE=float64    Torch dtype
  LIMIT=                 Optional spectral-row limit
  CHUNK_SIZE=            Optional chunk-size override
  OUTPUT_LEVELS=0        Set to 1 to benchmark profile output
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-both}"
THREADS="${THREADS:-1 2 4}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
OUTPUT_LEVELS="${OUTPUT_LEVELS:-0}"

[[ -n "${UV_PROFILE:-}" && -n "${UV_SCENE:-}" ]] || die "UV_PROFILE and UV_SCENE are required"
[[ -n "${TIR_PROFILE:-}" && -n "${TIR_SCENE:-}" ]] || die "TIR_PROFILE and TIR_SCENE are required"
[[ -f "$UV_PROFILE" ]] || die "UV profile not found: $UV_PROFILE"
[[ -f "$UV_SCENE" ]] || die "UV scene not found: $UV_SCENE"
[[ -f "$TIR_PROFILE" ]] || die "TIR profile not found: $TIR_PROFILE"
[[ -f "$TIR_SCENE" ]] || die "TIR scene not found: $TIR_SCENE"

case "$BACKEND" in
  numpy | torch | both) ;;
  *) die "BACKEND must be numpy, torch, or both" ;;
esac

common_args=(
  --backend "$BACKEND"
  --torch-device "$TORCH_DEVICE"
  --torch-dtype "$TORCH_DTYPE"
  --require-python-generated-inputs
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

run_case() {
  local name="$1"
  local profile="$2"
  local scene="$3"
  local threads="$4"

  echo
  echo "== $name | threads=$threads | backend=$BACKEND =="

  OMP_NUM_THREADS="$threads" \
  OPENBLAS_NUM_THREADS="$threads" \
  MKL_NUM_THREADS="$threads" \
  VECLIB_MAXIMUM_THREADS="$threads" \
  NUMEXPR_NUM_THREADS="$threads" \
  NUMBA_NUM_THREADS="$threads" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$ROOT_DIR/examples/benchmark_scene_full_spectrum.py" \
      --profile "$profile" \
      --scene "$scene" \
      "${common_args[@]}" \
      --torch-threads "$threads"
}

echo "py2sess full-spectrum benchmark sweep"
echo "  UV profile:  $UV_PROFILE"
echo "  UV scene:    $UV_SCENE"
echo "  TIR profile: $TIR_PROFILE"
echo "  TIR scene:   $TIR_SCENE"

for threads in $THREADS; do
  run_case "UV" "$UV_PROFILE" "$UV_SCENE" "$threads"
  run_case "TIR" "$TIR_PROFILE" "$TIR_SCENE" "$threads"
done
