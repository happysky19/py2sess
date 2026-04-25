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

To sweep UV and TIR with 1, 2, and 4 threads:

```bash
scripts/run_full_benchmark_threads.sh /path/to/uv_full_bundle.npz /path/to/tir_full_bundle.npz
```

The benchmark table reports:

- `load (s)`: one-time bundle read and slice time, printed in the header
- `geometry preprocessing`: one-time UV generation of Chapman and auxiliary
  angular factors from `heights`, `user_obsgeom`, and `stream_value`
- `optical preprocessing`: one-time generation of `g`, delta-M truncation
  factor, and solar FO scatter terms when physical optical inputs are present
- `wall (s)`: total backend wall time after bundle load
- `rt (s)`: solver runtime only
- `setup (s)`: `wall (s) - rt (s)`
- `fo (s)`, `2s (s)`: component timings when available
- `#wavelength/s`: spectral throughput based on `rt (s)`
- `chunk`: low-level chunk size or public `.forward()` internal endpoint chunk size
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

By default, the scripts generate derived optical inputs from physical
Rayleigh/aerosol fractions when the bundle contains them. Pass
`--use-dumped-derived-optics` to use stored Fortran-derived `g`, delta-M
factor, and solar FO scatter terms instead. Older bundles without physical
optical fields automatically fall back to the stored derived arrays.

The benchmark loaders read only the arrays needed for the selected mode. With
physical optical inputs available, the scripts do not load stored
`asymm`/`scaling`/`fo_exact_scatter` or `asymm_arr`/`d2s_scaling` arrays.

To enrich a local bundle with physical optical fields from the original
Fortran text dump:

```bash
python3 scripts/enrich_full_benchmark_optics.py uv /path/to/UV_Dump.dat /path/to/uv_full_bundle.npz /path/to/uv_full_bundle_with_optics.npz
python3 scripts/enrich_full_benchmark_optics.py tir /path/to/TIR_Dump.dat /path/to/tir_full_bundle.npz /path/to/tir_full_bundle_with_optics.npz
```

The enrichment step writes a new local bundle and leaves the original bundle
unchanged.

Generated benchmark reports should stay local, for example under `outputs/`,
`local_outputs/`, or `paper_outputs/`, which are ignored by git.

## TIR bundle schema

Required arrays:

- `wavelengths`
- `heights`
- `user_angle`
- `tau_arr`
- `omega_arr`
- `thermal_bb_input`
- `surfbb`
- `albedo`

Optional surface input:

- `emissivity`; when omitted, the benchmark uses `emissivity = 1 - albedo`

Optical phase inputs, preferred:

- `depol`
- `rayleigh_fraction`
- `aerosol_fraction`
- `aerosol_moments`
- `aerosol_interp_fraction`

Legacy derived optical inputs:

- `asymm_arr`
- `d2s_scaling`

Expected shapes:

- `tau_arr`, `omega_arr`: `(n_wavelengths, n_layers)`
- `depol`, `aerosol_interp_fraction`: `(n_wavelengths,)`
- `rayleigh_fraction`: `(n_wavelengths, n_layers)`
- `aerosol_fraction`: `(n_wavelengths, n_layers, n_aerosol)`
- `aerosol_moments`: `(2, n_moments + 1, n_aerosol)`
- `asymm_arr`, `d2s_scaling`: `(n_wavelengths, n_layers)` when using legacy
  derived optical inputs
- `thermal_bb_input`: `(n_wavelengths, n_layers + 1)`
- `surfbb`, `albedo`, `emissivity`, `wavelengths`: `(n_wavelengths,)`
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
- `albedo`
- `flux_factor`

UV geometry-derived arrays are no longer required. The benchmark generates
`chapman`, `x0`, `user_stream`, `user_secant`, `azmfac`, `px11`, `pxsq`,
`px0x`, and `ulp` from the required geometry inputs.

Optical phase inputs, preferred:

- `depol`
- `rayleigh_fraction`
- `aerosol_fraction`
- `aerosol_moments`
- `aerosol_interp_fraction`

Legacy derived optical inputs:

- `asymm`
- `scaling`
- `fo_exact_scatter`

Optional arrays:

- `stream_value`
- `ref_total`
- `ref_fo`
- `ref_2s`

Expected shapes:

- `tau`, `omega`: `(n_wavelengths, n_layers)`
- `depol`, `aerosol_interp_fraction`: `(n_wavelengths,)`
- `rayleigh_fraction`: `(n_wavelengths, n_layers)`
- `aerosol_fraction`: `(n_wavelengths, n_layers, n_aerosol)`
- `aerosol_moments`: `(2, n_moments + 1, n_aerosol)`
- `asymm`, `scaling`, `fo_exact_scatter`: `(n_wavelengths, n_layers)` when
  using legacy derived optical inputs
- `albedo`, `flux_factor`, `wavelengths`: `(n_wavelengths,)`
- `user_obsgeom`: `(1, 3)` for the current bundled observation-geometry case
- `heights`: `(n_layers + 1,)`
- scalar controls may be stored as scalars or length-1 arrays
- reference radiance arrays, when present, use shape `(n_wavelengths,)`

## Scope

- The TIR benchmark covers the standalone batched FO plus 2S path.
- The UV benchmark covers the standalone FO plus 2S path. If physical optical
  phase inputs are present, Python generates the solar `fo_scatter_term`;
  otherwise the script uses the stored `fo_exact_scatter` field.
