# Full-Spectrum Benchmarks

Preferred runtime path:

```text
profile text + scene YAML -> Python preprocessing -> py2sess RT inputs -> RT solve
```

Fortran dumps are validation data, not normal RT inputs. Current full UV/TIR
parity runs can use a local CreateProps provider while raw opacity providers
mature.

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
UV_PROFILE=profile.txt UV_SCENE=uv_scene.yaml \
TIR_PROFILE=profile.txt TIR_SCENE=tir_scene.yaml \
scripts/run_full_benchmark_threads.sh
```

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

Use `rt (s)` for solver speed claims.

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
