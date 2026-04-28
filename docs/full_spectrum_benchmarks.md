# Full-Spectrum Benchmarks

Preferred runtime path:

```text
profile text + scene YAML -> Python preprocessing -> py2sess RT inputs -> RT solve
```

Fortran dumps are validation data, not normal RT inputs. Current full UV/TIR
parity runs can use a local CreateProps provider while raw opacity providers
mature.

Direct HITRAN line-by-line scenes are useful for limited-row validation and
offline table generation. They are not yet the recommended full-spectrum timing
path.

Fast gas-opacity scenes should use saved pressure-temperature cross-section
tables:

```yaml
opacity:
  gas_cross_sections:
    table3d: {path: gas_xsec.nc}
```

The NetCDF table must provide `cross_section(gas, spectral, pressure,
temperature)` plus `pressure_hpa`, `temperature_k`, and either
`wavenumber_cm_inv`, `wavelength_nm`, or `wavelength_microns`. For exact
benchmark caches, `cross_section(gas, spectral, level)` with matching profile
`pressure_hpa(level)` and `temperature_k(level)` is also accepted.

To build an exact local HITRAN table for a profile and scene:

```bash
PYTHONPATH=src python3 scripts/create_hitran_opacity_table.py gas_xsec.nc \
  --profile profile.txt --scene scene.yaml
```

To build a reusable pressure-temperature lookup table:

```bash
PYTHONPATH=src python3 scripts/create_hitran_opacity_table.py gas_xsec.nc \
  --hitran-dir /path/to/HITRAN --gas H2O --gas CO2 \
  --pressure-hpa 100 300 700 1000 --temperature-k 220 260 300 \
  --wavenumber-start 500 --wavenumber-step 1 --wavenumber-count 1000
```

## Commands

Scene inputs:

```bash
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py \
  --profile profile.txt --scene uv_scene.yaml --require-python-generated-inputs

PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py \
  --profile profile.txt --scene tir_scene.yaml --require-python-generated-inputs
```

Thread sweep:

```bash
UV_PROFILE=profile_uv.txt UV_SCENE=uv_scene.yaml \
TIR_PROFILE=profile_tir.txt TIR_SCENE=tir_scene.yaml \
scripts/run_full_benchmark_threads.sh
```

For the packaged UV parity run, use the station matching the `D01` reference
output. Some legacy dump filenames contain a different station number.

Useful environment variables: `BACKEND=numpy|torch|both`, `THREADS="1 2 4"`,
`LIMIT=1000`, `CHUNK_SIZE=...`, `OUTPUT_LEVELS=1`, and
`REQUIRE_PYTHON_GENERATED_INPUTS=0` for legacy parity runs.

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

Benchmarks print setup and solver timing separately.

- `load (s)`: profile/scene or input-store loading
- `layer optical properties`: `tau`, `ssa`, and scattering fractions
- `geometry preprocessing`: geometry-only helper generation
- `optical preprocessing`: `g`, delta-M factor, and solar FO scatter
- `thermal source`: TIR Planck/source generation
- `rt (s)`: solver time only

Use `rt (s)` for solver speed claims. Treat `load (s)` and preprocessing as
opacity/setup cost.

## Scene Inputs

Scene YAML should contain `mode`, `spectral`, `geometry`, `surface`, and
`opacity` sections. Solar geometry uses `angles: [sza, vza, raz]`; thermal
geometry uses `view_angle`. Array paths are resolved relative to the scene file.

CreateProps provider scene for current full parity runs:

```yaml
mode: solar  # or thermal
opacity:
  provider: {kind: fortran_createprops, path: uv_createprops_provider}
```

Provider directories are created locally, for example:

```bash
PYTHONPATH=src python3 scripts/create_fortran_createprops_provider.py uv \
  /path/to/Dump_9_26_1500.dat_11_114 uv_createprops_provider
```

The provider stores Fortran `taug` as `gas_absorption_tau`. The separate
`absorption_tau` field is the RT-layer non-scattering term reconstructed from
`taudp` and `omega`; it can include aerosol absorption.
