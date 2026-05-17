# Paper RT Benchmarks

This workflow creates timing tables and figures for py2sess paper runtime
comparisons. It times RT solves after synthetic or scene inputs have already
been prepared.

The default run is the forward-only synthetic scaling benchmark: Solar and TIR
RT inputs are created directly from analytic, inhomogeneous optical profiles,
with no cross-section files or opacity generation in the timed path.

## Local Run

Smoke test:

```bash
PYTHONPATH=src python3 scripts/benchmark_paper_rt.py \
  --preset smoke \
  --groups synthetic-forward \
  --backend-set cpu \
  --torch-dtypes float64 \
  --repeats 2
```

Forward-only scaling run:

```bash
PYTHONPATH=src python3 scripts/benchmark_paper_rt.py \
  --preset paper \
  --groups synthetic-forward \
  --backend-set all \
  --torch-dtypes float64 \
  --warmups 1 \
  --repeats 5 \
  --output-dir outputs/forward_scaling_benchmark
```

Synthetic forward plus Jacobian CUDA run:

```bash
PYTHONPATH=src python3 scripts/benchmark_paper_rt.py \
  --preset paper \
  --groups synthetic-forward synthetic-jacobian \
  --backend-set cuda \
  --torch-dtypes float64 \
  --warmups 1 \
  --repeats 5 \
  --output-dir outputs/synthetic_rt_cuda_benchmark

PYTHONPATH=src python3 scripts/plot_paper_rt_benchmarks.py \
  --summary outputs/synthetic_rt_cuda_benchmark/summary.csv \
  --output-dir outputs/synthetic_rt_cuda_benchmark \
  --formats png,eps
```

The runner writes:

- `raw_timings.csv`: every measured repeat.
- `summary.csv`: best, mean, median, standard deviation, speedup, checksums,
  gradient norms, and validation differences.
- `manifest.csv`: environment metadata and skipped backend/device entries, such
  as CUDA being unavailable.

By default the runner prints progress lines to stderr when each subcase starts
and finishes. These messages are emitted outside the timed RT region and are not
included in `seconds`, `forward_seconds`, or `backward_seconds`. Use
`--no-progress` to silence them.

Plot the generated summary:

```bash
PYTHONPATH=src python3 scripts/plot_paper_rt_benchmarks.py \
  --summary outputs/forward_scaling_benchmark/summary.csv \
  --output-dir outputs/forward_scaling_benchmark \
  --formats png,eps
```

Generated outputs belong under `outputs/` and are not tracked by default.

## Paper Synthetic Forward Summary

The paper-facing synthetic forward summary combines local Apple M2 Pro NumPy and
torch rows with Colab Tesla T4 and A100 CUDA rows. The combined summary CSV is
[`docs/assets/synthetic_forward_scaling_summary.csv`](assets/synthetic_forward_scaling_summary.csv),
and the combined raw repeat table is
[`docs/assets/synthetic_forward_scaling_raw_timings.csv`](assets/synthetic_forward_scaling_raw_timings.csv).

Regenerate the paper synthetic plots from the normalized combined CSV:

```bash
PYTHONPATH=src python3 scripts/plot_paper_rt_benchmarks.py \
  --summary docs/assets/paper_rt_all_timing_summary.csv \
  --output-dir docs/assets \
  --formats png,eps
```

![Synthetic forward scaling](assets/paper_rt_synthetic_forward_publication.png)

The synthetic Jacobian figure is generated from
[`docs/assets/synthetic_jacobian_scaling_summary.csv`](assets/synthetic_jacobian_scaling_summary.csv):

![Synthetic Jacobian scaling](assets/paper_rt_synthetic_jacobian_publication.png)

The Jacobian overhead figure uses the same normalized combined CSV so each
Jacobian row is matched with the corresponding forward-only runtime on the same
backend and hardware:

![Synthetic Jacobian overhead](assets/paper_rt_synthetic_jacobian_overhead_publication.png)

## Combined Timing Summary

The paper-facing all-in-one timing table is
[`docs/assets/paper_rt_all_timing_summary.csv`](assets/paper_rt_all_timing_summary.csv).
It normalizes the full-spectrum benchmark, synthetic forward scaling, and
synthetic Jacobian scaling summaries into one schema with `backend_group` values
for `numpy`, `torch_cpu`, and `torch_cuda`. The full-spectrum Fortran baseline
rows are retained with `backend_group=fortran`. Synthetic Jacobian rows are
torch-only because the benchmark uses torch autograd; NumPy Jacobian timing is
therefore marked in the notes as not applicable rather than filled with a
different finite-difference definition.

Regenerate the combined table:

```bash
PYTHONPATH=src python3 scripts/build_paper_rt_timing_summary.py
```

## Experiments

`fortran-forward` runs the checked-in UV and TIR profile/scene cases and compares
py2sess spectra with the existing Fortran spectrum references.

`synthetic-forward` runs Solar and TIR direct RT inputs without gas
cross-section or opacity files. The current synthetic profiles are deterministic
and retrieval-like rather than constant slabs:

- Solar: smooth gas-like absorption, Rayleigh-like scattering, a low-altitude
  aerosol layer, an elevated cloud-like scattering layer, HG first-order scatter
  inputs, surface albedo `0.05`, and `fbeam=1.0`.
- TIR: smooth gas-like absorption, a weak low-altitude scattering layer,
  emissivity `0.98`, surface albedo `0.02`, and a deterministic
  US-standard-like temperature profile for the thermal source arrays.

The default paper sweep uses RT layer counts `5,10,20,50,100,114,200` and
wavelength counts `300,1000,3000,10000,30000,100000,300000`. Output tables also
include `levels = layers + 1`. Without `--full-grid`, the run uses two
orthogonal sweeps: wavelengths at 114 layers and vertical grid size at 50000
wavelengths.

`synthetic-jacobian` runs torch autograd on already-resident tensors. The main
runtime scaling figure uses `tau` gradients only: spectral grid size, vertical
grid size, and active `tau`-layer count. The timed region starts after tensors
are on the selected device and covers only RT forward plus
`radiance_total.sum().backward()`.

The summary table records `gradient_target`, `active_tau_layers`,
`n_grad_vars`, `forward_mean_s`, `backward_mean_s`, gradient checksums,
gradient norms, and CUDA peak memory where available. The default gradient-size
tau-layer sweep uses active layer counts `1,2,5,10,20,50,114` at 50000
wavelengths and 114 layers. For the overhead figure, the run also times
`omega/ssa` and `g` gradients for one active layer and all layers, plus
wavelength-local surface albedo and thermal surface emissivity, all at the
representative 50000-wavelength, 114-layer setting. For Solar `omega` and `g`,
the differentiable HG first-order scatter source is rebuilt inside the timed
forward graph so those VJPs include the local FO source response. The Jacobian
spectral sweep has a separate bounded default wavelength grid `300,1000,3000,10000`; use
`--jacobian-wavelength-counts` to request larger Jacobian spectral points. Use
`--jacobian-targets` to run only selected targets, for example
`--jacobian-targets omega` when adding only the missing omega overhead rows.

Use `--torch-compile` to time PyTorch rows through `torch.compile`; eager mode is
the default. Compiled rows are labeled with `torch.compile` in `summary.csv` and
the compile mode is recorded in `manifest.csv`. The current pseudo-spherical
Solar `include_fo=True` path still uses NumPy cached geometry precomputation, so
compiled Solar rows are skipped rather than silently mixed with eager timings.

Use `--full-grid` for the full layer-by-wavelength Cartesian grid. Without it,
the default forward-only run writes two orthogonal scaling curves. The Jacobian
run adds a third gradient-variable sweep.

## Synthetic VJP Sensitivity Design

The synthetic VJP case is now patterned after the derivative-validation style in
Ding and Yang (2023): compact finite-difference checks validate selected local
derivatives, while timing focuses on retrieval-relevant variables rather than a
dense finite-difference Jacobian. py2sess remains scalar 2S-ESS, so the closest
phase-shape analogue is the HG asymmetry factor `g`, not the full polarized
phase matrix.

## CUDA / Colab

Colab usually starts with a CUDA-enabled PyTorch wheel. Preserve that wheel by
installing py2sess without the `torch` extra:

No input bundle upload is needed for this synthetic scaling run.

```python
PAPER_REF = "<final-paper-tag-or-commit>"

!git clone https://github.com/happysky19/py2sess.git
%cd py2sess
!git checkout {PAPER_REF}
%pip install -e ".[plot]"
```

Check CUDA:

```python
import torch

print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")
```

Run CUDA-capable benchmarks. The Jacobian output includes `tau`, `omega/ssa`,
and surface-albedo gradient targets; the runtime scaling figure filters to
`tau`, while the overhead figure compares all three targets:

```python
!PYTHONPATH=src python scripts/benchmark_paper_rt.py \
  --preset paper \
  --groups synthetic-forward synthetic-jacobian \
  --backend-set cuda \
  --torch-dtypes float64 \
  --warmups 1 \
  --repeats 5 \
  --output-dir outputs/synthetic_rt_cuda_colab

!PYTHONPATH=src python scripts/plot_paper_rt_benchmarks.py \
  --summary outputs/synthetic_rt_cuda_colab/summary.csv \
  --output-dir outputs/synthetic_rt_cuda_colab \
  --formats png,eps
```

CUDA timing synchronizes before and after each measured repeat and records peak
allocated CUDA memory when available. If CUDA is unavailable, the run continues
and records skipped CUDA entries in `manifest.csv`.

## Fortran Jacobian Follow-Up

This repository does not contain the Fortran source. For Fortran Jacobian timing,
use an external 2S-ESS checkout and export one representative UV and TIR direct
RT-input case from this workflow if needed. The Fortran driver should bypass
optical-property generation, read the same RT arrays, and time forward-only and
Jacobian routines with the same warmup/repeat policy. If the Fortran path exposes
finite-difference Jacobians only, label those timings as finite-difference
Jacobian timing rather than comparing them directly with torch autograd timing.
