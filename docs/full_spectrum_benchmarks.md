# Full-Spectrum Benchmarks

Preferred path:

```text
profile text + scene YAML -> Python preprocessing -> py2sess RT inputs -> RT solve
```

Fortran dumps are validation data, not normal runtime inputs. Full runs should
use saved gas cross-section tables; direct HITRAN line-by-line scenes are for
limited checks and offline table generation.

## Run

```bash
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py \
  --profile profile_uv.txt --scene uv_scene.yaml --require-python-generated-inputs

PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py \
  --profile profile_tir.txt --scene tir_scene.yaml --require-python-generated-inputs
```

Thread sweep:

```bash
UV_PROFILE=profile_uv.txt UV_SCENE=uv_scene.yaml \
TIR_PROFILE=profile_tir.txt TIR_SCENE=tir_scene.yaml \
scripts/run_full_benchmark_threads.sh
```

Useful environment variables: `BACKEND=numpy|torch|both`, `THREADS="1 2 4"`,
`LIMIT=1000`, `CHUNK_SIZE=...`, `OUTPUT_LEVELS=1`.

## Gas Tables

Fast scenes use NetCDF gas cross-section tables:

```yaml
opacity:
  gas_cross_sections:
    table3d: {path: gas_xsec.nc}
```

Accepted layouts are `cross_section(gas, spectral, pressure, temperature)` for
reusable lookup tables and `cross_section(gas, spectral, level)` for exact
profile caches. Both also need `pressure_hpa`, `temperature_k`, and one spectral
axis: `wavenumber_cm_inv`, `wavelength_nm`, or `wavelength_microns`.

Build an exact local table for one benchmark profile:

```bash
PYTHONPATH=src python3 scripts/create_hitran_opacity_table.py gas_xsec.nc \
  --profile profile.txt --scene scene.yaml
```

Build a reusable lookup table:

```bash
PYTHONPATH=src python3 scripts/create_hitran_opacity_table.py gas_xsec.nc \
  --hitran-dir /path/to/HITRAN --gas H2O --gas CO2 \
  --pressure-hpa 100 300 700 1000 --temperature-k 220 260 300 \
  --wavenumber-start 500 --wavenumber-step 1 --wavenumber-count 1000
```

## Strict Mode

`--require-python-generated-inputs` rejects dumped/direct RT intermediates such
as `tau`, `omega`, `asymm`, `scaling`, `fo_exact_scatter`, `thermal_bb_input`,
and `surfbb`. With `--profile --scene`, dumped-input flags are invalid.

## Timing

Benchmarks print setup and solver timing separately. Use `rt (s)` for solver
speed claims; treat `load (s)`, `layer optical properties`, `geometry
preprocessing`, `optical preprocessing`, and `thermal source` as setup cost.

## Legacy Parity

Current UV/TIR parity checks may use local CreateProps provider directories:

```yaml
mode: solar  # or thermal
opacity:
  provider: {kind: fortran_createprops, path: uv_createprops_provider}
```

Create provider directories locally:

```bash
PYTHONPATH=src python3 scripts/create_fortran_createprops_provider.py uv \
  /path/to/Dump_9_26_1500.dat_11_114 uv_createprops_provider
```

Use `REQUIRE_PYTHON_GENERATED_INPUTS=0` only for legacy `.npz` or fallback runs.
