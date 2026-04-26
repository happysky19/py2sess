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

The sweep script also accepts environment overrides such as
`BACKEND=numpy`, `THREADS="1 4"`, `LIMIT=1000`,
`USE_DUMPED_DERIVED_OPTICS=1`, and `USE_DUMPED_THERMAL_SOURCE=1`.

The benchmark table reports:

- `load (s)`: one-time bundle read and slice time, printed in the header
- `layer optical properties`: one-time generation of `tau`, `ssa`,
  Rayleigh fraction, and aerosol fractions when component optical depths are
  present
- `geometry preprocessing`: one-time generation of geometry-derived inputs
  from the height grid and viewing geometry
- `optical preprocessing`: one-time generation of `g`, delta-M truncation
  factor, and solar FO scatter terms when physical optical inputs are present
- `thermal source`: one-time generation of thermal Planck/source inputs from
  temperatures when temperature fields are present
- `preprocessing total`: one-time derived-input setup before the backend timing
  table
- `wall (s)`: total backend wall time after bundle load
- `rt (s)`: solver runtime only
- `fo (s)`, `2s (s)`: component timings when available
- `#wavelength/s`: spectral throughput based on `rt (s)`
- `chunk`: low-level chunk size or public `.forward()` internal endpoint chunk size
- `max abs diff`, `max rel (%)`: optional total-radiance accuracy relative to
  bundled saved Fortran outputs when `ref_total` is present

Each script prints low-level optimized rows (`numpy`, `torch-*`) and public API
rows (`numpy-forward`, `torch-*-forward`). The low-level rows keep separate
`fo (s)` and `2s (s)` timings. The public rows show the end-to-end
`TwoStreamEss.forward()` endpoint path.

By design, `wall (s)` excludes opening the `.npz` bundle and the printed
preprocessing steps. It still includes backend-local overhead within the
benchmark row, such as PyTorch warmup, tensor conversion, and checksum
reduction.

Use `--output-levels` only when timing profile output. Profile timing is not
the endpoint performance target because it allocates and returns
`radiance_profile_*` arrays.

By default, the scripts generate layer optical properties from component
optical depths when the bundle contains them, then generate derived phase inputs
from Rayleigh/aerosol fractions. Pass `--use-dumped-derived-optics` to use
stored Fortran-derived `g`, delta-M factor, and solar FO scatter terms instead.
Older bundles without physical optical fields automatically fall back to the
stored derived arrays.

The benchmark loaders read only the arrays needed for the selected mode. With
component optical-depth inputs available, the scripts do not load direct
`tau`/`omega` arrays. With physical phase inputs available, the scripts do not
load stored `asymm`/`scaling`/`fo_exact_scatter` or
`asymm_arr`/`d2s_scaling` arrays.

To enrich a local bundle with physical optical fields from the original
Fortran text dump:

```bash
python3 scripts/enrich_full_benchmark_optics.py uv /path/to/UV_Dump.dat /path/to/uv_full_bundle.npz /path/to/uv_full_bundle_with_optics.npz
python3 scripts/enrich_full_benchmark_optics.py tir /path/to/TIR_Dump.dat /path/to/tir_full_bundle.npz /path/to/tir_full_bundle_with_optics.npz
```

The enrichment step writes a new local bundle and leaves the original bundle
unchanged. It also replaces any spectral row-index placeholder in `wavelengths`
with the physical wavelength grid from the dump.

Generated benchmark reports should stay local, for example under `outputs/`,
`local_outputs/`, or `paper_outputs/`, which are ignored by git.

## TIR bundle schema

Required arrays:

- `wavelengths`
- `heights`
- `user_angle`
- `albedo`

Provide either direct layer optical inputs or component optical-depth inputs.

Layer optical inputs, direct:

- `tau_arr`
- `omega_arr`

Layer optical inputs, generated:

- `absorption_tau`; `gas_absorption_tau` is also accepted when the absorption
  term is specifically gas-only
- `rayleigh_scattering_tau`
- optional `aerosol_scattering_tau` for scattering-only aerosol components
- optional `aerosol_extinction_tau` plus exactly one of
  `aerosol_scattering_tau` or `aerosol_single_scattering_albedo` for absorbing
  aerosol components

Provide either direct thermal source inputs or generated thermal source inputs.

Thermal source inputs, direct:

- `thermal_bb_input`
- `surfbb`

Thermal source inputs, generated:

- `level_temperature_k`
- `surface_temperature_k`
- exactly one of `wavenumber_cm_inv` or `wavelength_microns`

Optional surface input:

- `emissivity`; when omitted, the benchmark uses `emissivity = 1 - albedo`

Optical phase inputs, preferred:

- `depol`
- `rayleigh_fraction` and `aerosol_fraction`, unless generated from component
  optical depths
- `aerosol_moments`

Optional optical phase input:

- `aerosol_interp_fraction`; when omitted, the TIR benchmark derives it from
  a physical spectral coordinate: `wavelengths`, `wavelength_microns`, or
  `wavenumber_cm_inv`. Bundles that store row indices in `wavelengths` must
  provide either `aerosol_interp_fraction` or one of the physical coordinate
  arrays.

Legacy derived optical inputs:

- `asymm_arr`
- `d2s_scaling`

Expected shapes:

- `tau_arr`, `omega_arr`: `(n_wavelengths, n_layers)`
- `absorption_tau` or `gas_absorption_tau`, and `rayleigh_scattering_tau`:
  `(n_wavelengths, n_layers)`
- `aerosol_extinction_tau`, `aerosol_scattering_tau`:
  `(n_wavelengths, n_layers, n_aerosol)`
- `aerosol_single_scattering_albedo`:
  `(n_aerosol,)` or `(n_wavelengths, n_layers, n_aerosol)`
- `depol`, `aerosol_interp_fraction`: `(n_wavelengths,)`
- `rayleigh_fraction`: `(n_wavelengths, n_layers)`
- `aerosol_fraction`: `(n_wavelengths, n_layers, n_aerosol)`
- `aerosol_moments`: `(2, n_moments + 1, n_aerosol)`
- `asymm_arr`, `d2s_scaling`: `(n_wavelengths, n_layers)` when using legacy
  derived optical inputs
- `thermal_bb_input`: `(n_wavelengths, n_layers + 1)`
- `level_temperature_k`: `(n_layers + 1,)`
- `surface_temperature_k`: scalar or `(n_wavelengths,)`
- `wavenumber_cm_inv`, `wavelength_microns`: `(n_wavelengths,)`
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
- `albedo`
- `flux_factor`

Provide either direct layer optical inputs or component optical-depth inputs.

Layer optical inputs, direct:

- `tau`
- `omega`

Layer optical inputs, generated:

- `absorption_tau`; `gas_absorption_tau` is also accepted when the absorption
  term is specifically gas-only
- `rayleigh_scattering_tau`
- optional `aerosol_scattering_tau` for scattering-only aerosol components
- optional `aerosol_extinction_tau` plus exactly one of
  `aerosol_scattering_tau` or `aerosol_single_scattering_albedo` for absorbing
  aerosol components

UV geometry-derived arrays are no longer required. The benchmark generates
`chapman`, `x0`, `user_stream`, `user_secant`, `azmfac`, `px11`, `pxsq`,
`px0x`, and `ulp` from the required geometry inputs.

Optical phase inputs, preferred:

- `depol`
- `rayleigh_fraction` and `aerosol_fraction`, unless generated from component
  optical depths
- `aerosol_moments`

Optional optical phase input:

- `aerosol_interp_fraction`; when omitted, the UV benchmark derives it from
  physical `wavelengths` using the Fortran reverse endpoint interpolation
  convention. Bundles that store row indices in `wavelengths` must provide
  `aerosol_interp_fraction`.

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
- `absorption_tau` or `gas_absorption_tau`, and `rayleigh_scattering_tau`:
  `(n_wavelengths, n_layers)`
- `aerosol_extinction_tau`, `aerosol_scattering_tau`:
  `(n_wavelengths, n_layers, n_aerosol)`
- `aerosol_single_scattering_albedo`:
  `(n_aerosol,)` or `(n_wavelengths, n_layers, n_aerosol)`
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
