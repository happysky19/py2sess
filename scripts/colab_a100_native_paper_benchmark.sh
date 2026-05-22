#!/usr/bin/env bash
set -euo pipefail

# Colab A100 paper benchmark runner for feature/add-native-backend.
#
# Expected input layout:
#   ${INPUT_ROOT}/benchmark_bundles/tir_scene_python.yaml
#   ${INPUT_ROOT}/benchmark_bundles/uv_scene_python.yaml
#   ${INPUT_ROOT}/profiles/Profiles_1_2006726_0000.dat
#   ${INPUT_ROOT}/profiles/Profiles_1_2006726_1500.dat
#
# Typical Colab use:
#   from google.colab import drive
#   drive.mount('/content/gdrive')
#   !bash /content/py2sess_v2/scripts/colab_a100_native_paper_benchmark.sh

REPO_DIR="${REPO_DIR:-/content/py2sess_v2}"
INPUT_ROOT="${INPUT_ROOT:-/content/gdrive/MyDrive/input_bundle}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/gdrive/MyDrive/full_spectrum_benchmark_colab_native_a100_w2_r5}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
MAX_JOBS="${MAX_JOBS:-2}"

cd "${REPO_DIR}"

echo "repo=${REPO_DIR}"
echo "input_root=${INPUT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
git rev-parse --abbrev-ref HEAD || true
git rev-parse --short HEAD || true
git status --short || true
nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
PY

test -f "${INPUT_ROOT}/benchmark_bundles/tir_scene_python.yaml"
test -f "${INPUT_ROOT}/benchmark_bundles/uv_scene_python.yaml"
test -f "${INPUT_ROOT}/profiles/Profiles_1_2006726_0000.dat"
test -f "${INPUT_ROOT}/profiles/Profiles_1_2006726_1500.dat"

"${PYTHON_BIN}" scripts/colab_build_native_cuda.py \
  --install-build-tools \
  --clean-first \
  --max-jobs "${MAX_JOBS}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/colab_native_cuda_smoke.py \
  --dtype float64 \
  --layers 114 \
  --parity-rows 256 \
  --tir-rows 200000 \
  --uv-rows 280000 \
  --warmups 2 \
  --repeats 5

PYTHONPATH=src "${PYTHON_BIN}" scripts/benchmark_full_spectrum_rt.py \
  --input-root "${INPUT_ROOT}" \
  --systems python \
  --cases tir,uv \
  --backend-set cuda \
  --torch-dtypes float64 \
  --warmups 2 \
  --repeats 5 \
  --output-dir "${OUTPUT_DIR}"

echo "summary_csv=${OUTPUT_DIR}/summary_full_spectrum.csv"
echo "raw_csv=${OUTPUT_DIR}/raw_full_spectrum_timings.csv"
echo "manifest_csv=${OUTPUT_DIR}/manifest_full_spectrum.csv"
cat "${OUTPUT_DIR}/summary_full_spectrum.csv"
