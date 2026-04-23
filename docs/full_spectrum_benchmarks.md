# Full-Spectrum Benchmarks

The example scripts consume local or external NumPy bundles that follow the
schema documented here. The large full-spectrum bundles are intended to remain
outside the GitHub repository and can be kept in a local `benchmark_bundles/`
directory or any other path you prefer.

Example commands from the repository root:

```bash
PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py /path/to/tir_full_bundle.npz
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py /path/to/uv_full_bundle.npz
```

The benchmark table reports:

- `load (s)`: one-time bundle read and slice time, printed in the header
- `wall (s)`: total backend wall time after bundle load
- `rt (s)`: solver runtime only
- `setup (s)`: `wall (s) - rt (s)`
- `fo (s)`, `2s (s)`: component timings when available
- `#wavelength/s`: spectral throughput based on `rt (s)`
- `max abs diff`, `max rel (%)`: optional total-radiance accuracy relative to
  bundled saved Fortran outputs when `ref_total` is present

Each script prints low-level optimized rows (`numpy`, `torch-*`) and public API
rows (`numpy-forward`, `torch-*-forward`). The low-level rows keep separate
`fo (s)` and `2s (s)` timings. The public rows show the end-to-end
`TwoStreamEss.forward()` endpoint path.

By design, `wall (s)` excludes opening the `.npz` bundle itself. It does
include backend-local setup within the benchmark path, such as geometry
precompute, PyTorch warmup, tensor conversion, and checksum reduction.

Use `--output-levels` only when timing profile output. Profile timing is not
the endpoint performance target because it allocates and returns
`radiance_profile_*` arrays.

Generated benchmark reports should stay local, for example under `outputs/`,
`local_outputs/`, or `paper_outputs/`, which are ignored by git.

## TIR bundle schema

Required arrays:

- `wavelengths`
- `heights`
- `user_angle`
- `tau_arr`
- `omega_arr`
- `asymm_arr`
- `d2s_scaling`
- `thermal_bb_input`
- `surfbb`
- `albedo`

Expected shapes:

- `tau_arr`, `omega_arr`, `asymm_arr`, `d2s_scaling`: `(n_wavelengths, n_layers)`
- `thermal_bb_input`: `(n_wavelengths, n_layers + 1)`
- `surfbb`, `albedo`, `wavelengths`: `(n_wavelengths,)`
- `heights`: `(n_layers + 1,)`
- `user_angle`: scalar or length-1 array

Optional saved-output arrays:

- `ref_total`: `(n_wavelengths,)`
- `ref_fo`: `(n_wavelengths,)`
- `ref_2s`: `(n_wavelengths,)`

## UV bundle schema

Required arrays:

- `wavelengths`
- `user_obsgeom`
- `heights`
- `tau`
- `omega`
- `asymm`
- `scaling`
- `albedo`
- `flux_factor`
- `fo_exact_scatter`
- `chapman`
- `x0`
- `user_stream`
- `user_secant`
- `azmfac`
- `px11`
- `pxsq`
- `px0x`
- `ulp`

Optional arrays:

- `stream_value`
- `ref_total`
- `ref_fo`
- `ref_2s`

Expected shapes:

- `tau`, `omega`, `asymm`, `scaling`: `(n_wavelengths, n_layers)`
- `fo_exact_scatter`: `(n_wavelengths, n_layers)`
- `albedo`, `flux_factor`, `wavelengths`: `(n_wavelengths,)`
- `user_obsgeom`: `(1, 3)` for the current bundled observation-geometry case
- `heights`: `(n_layers + 1,)`
- `chapman`: `(n_layers, n_layers)`
- `pxsq`, `px0x`: `(n_wavelengths, n_layers)` or broadcast-compatible arrays
  matching the batch-solver inputs
- scalar controls may be stored as scalars or length-1 arrays
- reference radiance arrays, when present, use shape `(n_wavelengths,)`

## Scope

- The TIR benchmark covers the standalone batched FO plus 2S path.
- The UV benchmark covers the standalone FO plus 2S path, using a bundle that
  already includes the FO exact-scatter input required by the standalone
  solver.
