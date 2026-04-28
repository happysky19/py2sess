# Public API Argument Reference

`py2sess` exposes one canonical public API at `TwoStreamEss` and keeps
Fortran-derived names at the internal solver boundary. The public names follow
pydisort where the physics has a direct match: `nlyr`, `tau`, `ssa`, `fbeam`,
and `albedo`. It intentionally keeps `g` instead of DISORT `pmom` because
2S-ESS takes one layer asymmetry factor rather than a full phase-moment array.

`2S` is the two-stream multiple-scattering/emission component. It solves the
layer boundary-value problem with two quadrature directions and is the fast
default path. `FO` is the first-order correction: direct-beam/single-scatter
solar radiance, or first-order thermal source transmission. When `include_fo`
is enabled, `result.radiance_total` is the validated `2S + FO` total.

Angles are in degrees. Solar observation geometry is `[sza, vza, raz]`, where
`sza` is solar zenith, `vza` is viewing zenith, and `raz` is relative azimuth.
Thermal `angles` are viewing zenith angles only. Internally these angles are
converted to cosines, secants, Chapman factors, and phase-angle terms. Relative
azimuth is used through trigonometric factors, so `raz=360` is equivalent to
`raz=0`. Heights `z` are in km and ordered top to bottom. Layer optical
thickness is positive downward. Optical thickness, `ssa`, `g`, albedo,
emissivity, Planck radiance, and radiance use the caller's consistent unit
convention; the solver does not convert radiance units.

`forward()` accepts either a single atmosphere with shape `(nlyr,)` or leading
batch dimensions such as `(nwave, nlyr)` and `(ncol, nwave, nlyr)`. Batch
dimensions are preserved in `result.radiance`; multiple solar or thermal
geometries append one final geometry axis. The default batched path returns TOA
`radiance_2s`, optional `radiance_fo`, and `radiance_total`, but not BOA fluxes.
Set `TwoStreamEssOptions(output_levels=True)` when you need
upwelling radiance profiles; profile arrays use the final axis for levels,
ordered from TOA to BOA.

See [`model_assumptions.md`](model_assumptions.md) for the compact checklist of
defaults and hard-coded RT conventions.

| Public name | Meaning | Shape | Default | Fortran/internal name |
|---|---|---:|---|---|
| `nlyr` | Number of atmospheric layers | scalar | required | `NLAYERS` |
| `mode` | Source mode: `solar`, `solar_lattice`, or `thermal` | scalar | `solar` | source-mode branch in the master drivers |
| `output_levels` | Return level-by-level radiance profiles | scalar | `False` | `DO_LEVEL_OUTPUT` |
| `upwelling` | Compute upwelling outputs | scalar | `True` | `DO_UPWELLING` |
| `downwelling` | Compute downwelling outputs | scalar | `False` | `DO_DNWELLING` |
| `plane_parallel` | Use plane-parallel geometry instead of spherical geometry | scalar | `False` | `DO_PLANE_PARALLEL` |
| `delta_scaling` | Apply delta-M scaling in the 2S core | scalar | `True` | `DO_D2S_SCALING` |
| `brdf_surface` | Enable explicit BRDF coefficients | scalar | `False` | `DO_BRDF_SURFACE` |
| `tau` | Layer optical thickness | `(..., nlyr)` | required | `DELTAU_INPUT` |
| `ssa` | Single-scattering albedo | `(..., nlyr)` | required | `OMEGA_INPUT` |
| `g` | Asymmetry factor | `(..., nlyr)` | required | `ASYMM_INPUT` |
| `z` | Level height grid, top to bottom, km | `(nlyr+1,)` | required for solar and spherical FO paths | `HEIGHT_GRID` |
| `angles` | Solar `[sza, vza, raz]`, degrees | `(3,)` or `(ngeom, 3)` | required for `mode="solar"` | `USER_OBSGEOMS` |
| `angles` | Thermal viewing zenith angle(s), degrees | scalar or `(ngeom,)` | required for `mode="thermal"` | `USER_ANGLES`, converted to `USER_STREAMS` |
| `view_angles` | Advanced lattice/thermal view-angle override, degrees | `(nview,)` | `None` | `USER_ANGLES` / `USER_VZANGLES` |
| `beam_szas` | Advanced solar-lattice solar zenith angles, degrees | `(nbeam,)` | required for `mode="solar_lattice"` | `BEAM_SZAS` |
| `relazms` | Advanced solar-lattice relative azimuth angles, degrees | `(nazm,)` | required for `mode="solar_lattice"` | `USER_RELAZMS` |
| `stream` | Two-stream quadrature cosine | scalar | `1/sqrt(3)` | `STREAM_VALUE` |
| `fbeam` | Direct solar beam/source normalization | scalar or `(...)` | `1.0` | `FLUX_FACTOR` / `FLUXFAC` |
| `albedo` | Lambertian surface albedo | scalar or `(...)` | `0.0` | `LAMBERTIAN_ALBEDO` |
| `delta_m_truncation_factor` | Delta-M truncation factor `f`; use explicit values for mixed phase functions or fixture parity | `(..., nlyr)` | `None` -> `g**2` | `D2S_SCALING`; FO optical path uses `FO_TRUNCFAC` |
| `planck` | Thermal Planck/source value at level boundaries | `(..., nlyr+1)` | required for `mode="thermal"` | `THERMAL_BB_INPUT` |
| `surface_planck` | Surface Planck radiance | scalar or `(...)` | `0.0` | `SURFBB` / `FO_SURFBB` |
| `emissivity` | Surface emissivity | scalar or `(...)` | `0.0` | `EMISSIVITY` / `FO_USER_EMISSIVITY` |
| `geometry` | FO geometry method: `pseudo_spherical` or `regular_pseudo_spherical` | scalar | `pseudo_spherical` | EPS/RPS FO geometry selector |
| `earth_radius` | Planet radius for spherical paths, km | scalar | `6371.0` | `EARTH_RADIUS` |
| `include_fo` | Attach first-order outputs to a main 2S run | scalar | `False` | FO master-call branch |
| `fo_scatter_term` | Solar FO phase/source term: phase function times single-scattering and delta-M source scaling | `(..., nlyr)` or `(..., nlyr, ngeom)` | HG term from `ssa`, `g`, `f` | `FO_EXACTSCAT` |
| `n_moments` / `fo_n_moments` | Advanced solar FO phase control; `0` is isotropic and positive values use the closed-form HG fallback unless explicit phase data are supplied | scalar | `5000` | `NMOMENTS_INPUT`-style FO moment controls |
| `nfine` / `fo_nfine` | Number of fine sub-layers used by EPS FO spherical-path integration | scalar | `3` | `NFINEDIVS` |

## Defaults

- `stream=None` becomes `1 / sqrt(3)`. Fortran parity examples pass their
  benchmark stream explicitly, such as `stream=0.5` for the packaged TIR case.
- `fbeam` defaults to `1.0`.
- `delta_m_truncation_factor=None` derives the HG-like fallback `f = g**2`,
  clipped to `0 <= f < 1`.
- Pass `delta_m_truncation_factor=np.zeros(nlyr)` only when no truncation is
  intended.
- Set `delta_scaling=False` to disable the 2S optical-property transform. Solar
  FO still uses `delta_m_truncation_factor` in its single-scatter source term
  unless you pass `fo_scatter_term` explicitly.
- `geometry="pseudo_spherical"` maps to the EPS FO geometry path.
- Solar and thermal source handling are mode-exclusive. A single `TwoStreamEss`
  call does not combine direct solar and Planck thermal sources in this pass.

## Result Names

- `result.radiance` returns the preferred total radiance for the selected path.
- `result.radiance_2s` returns the two-stream TOA radiance.
- `result.radiance_fo` returns the attached first-order component when
  available.
- `result.radiance_total` returns the best available total radiance. For solar
  and thermal `include_fo=True`, this is `2S + FO`; otherwise it falls back to
  the primary 2S output available for the requested method.
- Profile results are available as `result.radiance_profile_2s`,
  `result.radiance_profile_fo`, and `result.radiance_profile_total` when
  `output_levels=True`, for both scalar and batched calls.
