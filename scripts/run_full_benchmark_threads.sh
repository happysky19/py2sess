#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_full_benchmark_threads.sh UV_INPUT TIR_INPUT
  UV_PROFILE=profile.txt UV_SCENE=uv.yaml TIR_PROFILE=profile.txt TIR_SCENE=tir.yaml scripts/run_full_benchmark_threads.sh

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
  REQUIRE_PYTHON_GENERATED_INPUTS=1
                        Set to 0 to allow legacy .npz and dumped/direct fallback
  UV_PROFILE=           UV atmospheric profile text file for scene-input mode
  UV_SCENE=             UV scene YAML file for scene-input mode
  TIR_PROFILE=          TIR atmospheric profile text file for scene-input mode
  TIR_SCENE=            TIR scene YAML file for scene-input mode
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ $# -ne 0 && $# -ne 2 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_MODE=0
UV_INPUT=""
TIR_INPUT=""
if [[ $# -eq 2 ]]; then
  UV_INPUT="$1"
  TIR_INPUT="$2"
else
  SCENE_MODE=1
fi

PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-both}"
THREADS="${THREADS:-1 2 4}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
OUTPUT_LEVELS="${OUTPUT_LEVELS:-0}"
REQUIRE_PYTHON_GENERATED_INPUTS="${REQUIRE_PYTHON_GENERATED_INPUTS:-1}"

if [[ "$SCENE_MODE" == "1" ]]; then
  [[ -n "${UV_PROFILE:-}" && -n "${UV_SCENE:-}" ]] || die "UV_PROFILE and UV_SCENE are required in scene-input mode"
  [[ -n "${TIR_PROFILE:-}" && -n "${TIR_SCENE:-}" ]] || die "TIR_PROFILE and TIR_SCENE are required in scene-input mode"
  [[ -f "$UV_PROFILE" ]] || die "UV profile not found: $UV_PROFILE"
  [[ -f "$UV_SCENE" ]] || die "UV scene not found: $UV_SCENE"
  [[ -f "$TIR_PROFILE" ]] || die "TIR profile not found: $TIR_PROFILE"
  [[ -f "$TIR_SCENE" ]] || die "TIR scene not found: $TIR_SCENE"
else
  [[ -f "$UV_INPUT" || -d "$UV_INPUT" ]] || die "UV input not found: $UV_INPUT"
  [[ -f "$TIR_INPUT" || -d "$TIR_INPUT" ]] || die "TIR input not found: $TIR_INPUT"
fi

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

if [[ "$REQUIRE_PYTHON_GENERATED_INPUTS" == "1" ]]; then
  common_args+=(--require-python-generated-inputs)
fi

run_case() {
  local name="$1"
  local script="$2"
  local input="$3"
  local profile="$4"
  local scene="$5"
  local threads="$6"
  local case_args=("${common_args[@]}")
  local command=("$PYTHON" "$ROOT_DIR/examples/$script")

  if [[ "$name" == "TIR" && "${USE_DUMPED_THERMAL_SOURCE:-0}" == "1" ]]; then
    case_args+=(--use-dumped-thermal-source)
  fi

  if [[ -n "$input" ]]; then
    command+=("$input")
  else
    command+=(--profile "$profile" --scene "$scene")
  fi

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
    "${command[@]}" \
      "${case_args[@]}" \
      --torch-threads "$threads"
}

echo "py2sess full-spectrum benchmark sweep"
if [[ "$SCENE_MODE" == "1" ]]; then
  echo "  UV profile:  $UV_PROFILE"
  echo "  UV scene:    $UV_SCENE"
  echo "  TIR profile: $TIR_PROFILE"
  echo "  TIR scene:   $TIR_SCENE"
else
  echo "  UV input:  $UV_INPUT"
  echo "  TIR input: $TIR_INPUT"
fi

for threads in $THREADS; do
  if [[ "$SCENE_MODE" == "1" ]]; then
    run_case "UV" "benchmark_uv_full_spectrum.py" "" "$UV_PROFILE" "$UV_SCENE" "$threads"
    run_case "TIR" "benchmark_tir_full_spectrum.py" "" "$TIR_PROFILE" "$TIR_SCENE" "$threads"
  else
    run_case "UV" "benchmark_uv_full_spectrum.py" "$UV_INPUT" "" "" "$threads"
    run_case "TIR" "benchmark_tir_full_spectrum.py" "$TIR_INPUT" "" "" "$threads"
  fi
done
