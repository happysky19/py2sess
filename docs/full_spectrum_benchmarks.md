# Full-Spectrum Benchmarks

Preferred runtime path:

```text
profile text + scene YAML -> Python preprocessing -> py2sess RT inputs -> RT solve
```

Fortran dumps are validation data in this path, not final RT inputs. When a
full raw spectroscopy/aerosol source is not available yet, a scene can point to
a local Fortran CreateProps provider directory containing component optical
depths and phase-source inputs. Legacy array directories and `.npz` bundles
remain available for parity/debug.

## Commands

Scene inputs:

```bash
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py \
  --profile profile.txt --scene uv_scene.yaml --require-python-generated-inputs

PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py \
  --profile profile.txt --scene tir_scene.yaml --require-python-generated-inputs
```

Runtime array directories:

```bash
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py uv_runtime_dir --require-python-generated-inputs
PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py tir_runtime_dir --require-python-generated-inputs
```

Thread sweep:

```bash
scripts/run_full_benchmark_threads.sh uv_runtime_dir tir_runtime_dir

UV_PROFILE=profile.txt UV_SCENE=uv_scene.yaml \
TIR_PROFILE=profile.txt TIR_SCENE=tir_scene.yaml \
scripts/run_full_benchmark_threads.sh
```

Useful environment variables: `BACKEND=numpy|torch|both`, `THREADS="1 2 4"`,
`LIMIT=1000`, `CHUNK_SIZE=...`, `OUTPUT_LEVELS=1`, and
`REQUIRE_PYTHON_GENERATED_INPUTS=0` for legacy fallback/debug.

## Strict Mode

`--require-python-generated-inputs` rejects runtime use of dumped or direct RT
intermediates:

- direct layer inputs: `tau`, `omega`, `tau_arr`, `omega_arr`
- dumped phase inputs: `asymm`, `scaling`, `fo_exact_scatter`, `asymm_arr`,
  `d2s_scaling`
- dumped TIR source inputs: `thermal_bb_input`, `surfbb`
- legacy `.npz` stores for strict array-directory runs

With `--profile --scene`, dumped-input flags are invalid.

## Timing

Benchmarks print setup and solver timing separately:

- `load (s)`: input-store or profile/scene load
- `layer optical properties`: generation of `tau`, `ssa`, and scattering fractions
- `geometry preprocessing`: geometry-only helper generation
- `optical preprocessing`: `g`, delta-M factor, and solar FO scatter
- `thermal source`: TIR Planck/source generation
- `rt (s)`: solver time only
- `fo (s)` / `2s (s)`: low-level component timings when available

Use `rt (s)` for solver speed claims. `wall (s)` excludes loading and printed
preprocessing, but includes backend-local overhead such as tensor conversion.

## Scene Inputs

Minimal UV scene:

```yaml
mode: solar
gases: [O3]
spectral: {wavelengths_nm: [500.0, 600.0]}
geometry: {angles: [30.0, 20.0, 0.0]}
surface: {albedo: 0.1}
opacity:
  gas_cross_sections: {path: o3_cross_sections.npy}
  aerosol: {moments: {path: aerosol_moments.npy}}
```

UV scene using a local Fortran CreateProps provider:

```yaml
mode: solar
opacity:
  provider: {kind: fortran_createprops, path: uv_createprops_provider}
```

Minimal TIR scene:

```yaml
mode: thermal
gases: [O3]
spectral: {wavenumber_band_cm_inv: [[899.5, 900.5], [900.5, 901.5]]}
geometry: {view_angle: 20.0}
surface: {albedo: 0.02}
opacity:
  gas_cross_sections: {path: o3_cross_sections.npy}
  aerosol: {moments: {path: aerosol_moments.npy}}
```

TIR scene using a local Fortran CreateProps provider:

```yaml
mode: thermal
opacity:
  provider: {kind: fortran_createprops, path: tir_createprops_provider}
```

Array paths are resolved relative to the scene file. The profile loader accepts
GEOCAPE-style files and simple named-column tables, ordered internally from top
to bottom.

Provider directories are created locally, for example:

```bash
PYTHONPATH=src python3 scripts/create_fortran_createprops_provider.py uv \
  /path/to/Dump_9_26_1500.dat_11_114 uv_createprops_provider
```

This provider is an interim Fortran-derived opacity source. It avoids runtime
use of final RT intermediates, but it is not a replacement for raw spectroscopy
readers, aerosol microphysics, or future pyharp adapters. TIR providers must
carry the actual wavenumber coordinate from CreateProps; do not infer it from
the row-wise wavelength array.
