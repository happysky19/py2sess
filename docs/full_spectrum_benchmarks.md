# Full-Spectrum Benchmarks

Preferred runtime path:

```text
profile text + scene YAML -> Python preprocessing -> py2sess RT inputs -> RT solve
```

Full runs use saved gas cross-section NetCDF tables and aerosol table specs.
Direct HITRAN line processing is only for offline table generation or small
checks.

## Run

```bash
PYTHONPATH=src python3 examples/benchmark_scene_full_spectrum.py \
  --profile profile_uv.txt --scene uv_scene.yaml --require-python-generated-inputs

PYTHONPATH=src python3 examples/benchmark_scene_full_spectrum.py \
  --profile profile_tir.txt --scene tir_scene.yaml --require-python-generated-inputs
```

The command reads `mode: solar` or `mode: thermal` from the scene YAML, so UV
and TIR use the same API.

Thread sweep:

```bash
UV_PROFILE=profile_uv.txt UV_SCENE=uv_scene.yaml \
TIR_PROFILE=profile_tir.txt TIR_SCENE=tir_scene.yaml \
scripts/run_full_benchmark_threads.sh
```

Useful environment variables: `BACKEND=numpy|torch|both`, `THREADS="1 2 4"`,
`LIMIT=1000`, `CHUNK_SIZE=...`, `OUTPUT_LEVELS=1`.

Add `--component-timing` to the benchmark command when you need diagnostic
NumPy FO/2S split timing. The default benchmark reports the public
`scene.forward()` RT time.

## Inputs

Strict scene mode rejects precomputed RT arrays. Gas absorption should come from
NetCDF tables:

```yaml
opacity:
  gas_cross_sections:
    table3d: {path: gas_xsec.nc}
  aerosol:
    loadings:
      kind: geocape_files
      paths: [...]
    ssprops:
      path: SSprops
```

Public benchmark aerosol inputs can keep profile-dependent loading in the
profile CSV and reusable aerosol optics in one NetCDF:

```csv
pressure_hpa,temperature_k,height_km,O3,dust_loading,smoke_loading
1000.0,290.0,0.0,3.0e-8,1.0,0.0
```

```yaml
opacity:
  aerosol:
    properties: aerosol_properties.nc
    loading_columns:
      dust_loading: dust
      smoke_loading: smoke
```

`aerosol_loading` is dimensionless in these benchmark files. The NetCDF
`bulk_extinction` and `bulk_scattering` variables use units
`optical_depth_per_unit_loading`, so py2sess computes
`aerosol_tau[wave, layer] = sum_type loading[layer, type] * bulk[wave, type]`.
If loading is supplied on profile levels, py2sess averages adjacent levels to
layers.

GEOCAPE UV scenes can also point to raw source tables:

```yaml
surface:
  albedo:
    geocape_emissivity: {path: Surface_Data/Emissivity_1.asc}
solar:
  flux_factor:
    geocape_solar_spectrum: {path: newkur.dat, scale: 1.0e4}
```

Create an exact local table for one profile:

```bash
PYTHONPATH=src python3 scripts/create_hitran_opacity_table.py gas_xsec.nc \
  --profile profile.txt --scene scene.yaml
```

## Current Convergence

Local 1-thread clean scene-input runs against the packaged reference outputs:

| Case | Wavelengths | Max absolute difference | Max relative difference |
|---|---:|---:|---:|
| UV | 280000 | `8.737648e-09` | `4.707861e-03 %` |
| TIR | 200000 | `5.017868e-07` | `4.993298e-04 %` |

The UV comparison includes the corrected Python Rayleigh CO2 unit handling.
The original Fortran benchmark has a small Rayleigh optical-depth bug there, so
the remaining UV difference is expected and minor.

Reference output files must carry the spectral grid used for comparison. UV
references should include `wavelength_nm`; TIR references should include
`wavelength_nm` and may also include `wavenumber_cm_inv` or
`wavenumber_band_cm_inv`. The scene loader checks the reference spectral grid
before comparing radiances.

## Timing

Use `rt (s)` for solver speed claims. `load (s)`, `layer optical properties`,
`optical preprocessing`, `thermal source`, and any geometry setup are
preprocessing costs and are printed separately when present.
