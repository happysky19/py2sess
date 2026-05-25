# Level Flux Benchmarks

py2sess level flux arrays use a final TOA-to-BOA level axis of length
`nlyr + 1`.

- `flux_up`: upward flux.
- `flux_down`: downward flux, including the direct solar beam when present.
- `flux_net`: `flux_up - flux_down`.
- `flux_mean`: mean-intensity or actinic-style proxy.

The formula decision and literature review are recorded in
`docs/level_flux_formula_review.md`.

Run the lightweight checks with:

```bash
PYTHONPATH=src python scripts/benchmark_level_flux_references.py
```

The script reports public py2sess fluxes by default. Cases with FO use
`include_fo=True` and compare the returned `flux_*` fields directly against the
reference totals. Solar atmospheric FO radiance corrections are not integrated
into public DISORT-style fluxes; fluxes remain moment-equation outputs, with
only validated separable direct-source replacements applied. Thermal FO flux
output uses exact hemispheric source replacements for direct thermal source
transmission. If `pydisort` is installed, the script also runs DISORT references
through `pydisort.gather_flx()`.

The DISOTEST flux comparison script reports the public moment-flux definition:

```bash
PYTHONPATH=src python scripts/benchmark_disotest_flux.py

PYTHONPATH=src python scripts/benchmark_disotest_flux.py --suite disotest

PYTHONPATH=src python scripts/benchmark_disotest_flux.py --suite pydisort-grid

PYTHONPATH=src python scripts/benchmark_disotest_flux.py \
  --stream 0.5

PYTHONPATH=src python scripts/benchmark_disotest_flux.py \
  --compare-vijay-section6

PYTHONPATH=src python scripts/benchmark_disotest_flux.py \
  --suite disort-test --paper-table --fo-flux-n-mu 8
```

The default `--stream` value is the public solver default, `1/sqrt(3)`. Passing
`--stream 0.5` reproduces the 2S-ESS paper's Section 6 convention for validating
two-stream calculations against DISORT/LIDORT. The script labels these choices as
`stream_mode=public-default` and `stream_mode=section6`; other stream values are
reported as sensitivity runs.

The default suite registers all 48 official DISOTEST cases plus 26 deterministic
`pydisort` grid cases. Of the official DISOTEST cases, 26 are directly runnable
through current public py2sess inputs; the rest are retained in the report with
explicit unsupported reasons. The runnable official set includes Haze-L and
Cloud C.1 tabulated phase-function cases by reducing the DISORT moments to the
two-stream inputs `g = P1/(2 * 1 + 1)` and
`delta_m_truncation_factor = P2/(2 * 2 + 1)`. The grid
cases cover pure absorption, Lambertian surface reflection, isotropic
scattering, Rayleigh scattering, and Henyey-Greenstein scattering. The grid
cases use `pydisort` as both the `benchmark` and `pydisort` columns.

The surface-first diagnostic suite is available as `--suite disotest-surface`.
It registers DISOTEST `6d`, `7d`, and `7e` with their DISORT benchmark fluxes.
`6d` is runnable through py2sess using the cDISORT Hapke surface kernel and an
exact direct-surface hemispheric flux replacement. The `7d`/`7e` rows remain
diagnostic because they combine non-Lambertian surfaces with mixed solar,
top-boundary, and thermal source terms that are not yet represented together in
the public py2sess API.

Top isotropic illumination (`fisot`) is now available for scalar NumPy solar
runs and can be run with `--suite disotest-top-isotropic`. DISOTEST `1c`, `1f`,
and `8a`-`8c` are registered as runnable top-boundary cases. They remain a
separate diagnostic group because this is a diffuse hemispheric boundary, not a
direct beam, and it is not equivalent to choosing a different `fbeam`/`mu0`.
py2sess treats public `fisot` as DISORT-style isotropic boundary radiance and
projects it onto the two-stream boundary with incident flux `pi * fisot`.
DISOTEST `9a`/`9b` are still held back because their official output grid
includes non-boundary optical depths, while py2sess level fluxes currently use
layer boundaries.

For DISOTEST scattering cases, the `benchmark`/`pydisort` columns are the
exact values shown in parentheses in the paper tables. The non-parenthesized
paper values are 2S-ESS outputs; matching those checks implementation parity,
while matching the parenthesized values would require eliminating the physical
two-stream closure error.

Use `--compare-vijay-section6` to show the official cases with transcribed
2S-ESS Section 6 rows in one table: `benchmark`, `pydisort`,
`vijay_section6`, and py2sess at `stream=1/sqrt(3)` for multiple
`fo_flux_n_mu` values. The `vijay_section6` fixture is transcribed from the
Zenodo `DISORT_Comparisons/2S-ESS/2SESSTEST.out.*` outputs associated with
Natraj et al. (2023).

Use `--paper-table` for manuscript-ready rows. That mode prints only `DISORT`,
`py2sess`, and percent error columns, uses `fo_flux_n_mu=8` by default, and
accepts `--py2sess-backend numpy|torch|native` for later backend parity checks.

Current analytic checks include pure absorption, Lambertian direct-surface
reflection, and thermal surface emission. Isotropic and pure Rayleigh
single-scatter formulas are kept as component references for future formula
work, not as a public flux correction.

For diagnostics, add `--components` to also report correction-term
diagnostics:

```bash
PYTHONPATH=src python scripts/benchmark_level_flux_references.py --components
```

KINETICS comparisons are diagnostic, not authoritative. The current command
compares a captured FLXOUT-style table against the built-in absorbing-solar
check, so the captured table must have the same level grid:

```bash
PYTHONPATH=src python scripts/benchmark_level_flux_references.py \
  --kinetics-flxout path/to/captured_flxout.txt
```

The parser accepts columns for `direct_flux`, `diffuse_plus_flux`,
`diffuse_minus_flux`, `net_flux`, `diffuse_radiation_field`,
`total_radiation_field`, and `diffuse_factor`. Common FLXOUT names such as
`YDF`, `YFLP`, `YFLM`, `YTOT`, `DIFFLXR`, `TRADF`, and `DIFFAC` are accepted as
aliases.
