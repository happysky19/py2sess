# Public API Argument Reference

`py2sess` exposes one canonical public API at `TwoStreamEss` and keeps the
Fortran-derived names at the internal solver boundary. The API follows pydisort
where the underlying 2S-ESS physics has a direct match: `nlyr`, `tau`, `ssa`,
`fbeam`, and `albedo`. It intentionally keeps `g` instead of DISORT `pmom`
because 2S-ESS takes one layer asymmetry factor rather than a full phase-moment
array. The table below maps the public names to their physical meaning and
closest original Fortran/internal variables.

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

| Public name | Meaning | Shape | Default | Old Python name | Fortran/internal name |
|---|---|---:|---|---|---|
| `nlyr` | Number of atmospheric layers | scalar | required | `n_layers` | `NLAYERS` |
| `mode` | Source mode: `solar`, `solar_lattice`, or `thermal` | scalar | `solar` | `source_mode` | source-mode branch in the master drivers |
| `output_levels` | Return level-by-level radiance profiles | scalar | `False` | `do_level_output` | `DO_LEVEL_OUTPUT` |
| `upwelling` | Compute upwelling outputs | scalar | `True` | `do_upwelling` | `DO_UPWELLING` |
| `downwelling` | Compute downwelling outputs | scalar | `False` | `do_dnwelling` | `DO_DNWELLING` |
| `plane_parallel` | Use plane-parallel geometry instead of spherical geometry | scalar | `False` | `do_plane_parallel` | `DO_PLANE_PARALLEL` |
| `delta_scaling` | Apply delta-M scaling in the 2S core | scalar | `True` | `do_delta_scaling` | `DO_D2S_SCALING` |
| `brdf_surface` | Enable explicit BRDF coefficients | scalar | `False` | `do_brdf_surface` | `DO_BRDF_SURFACE` |
| `tau` | Layer optical thickness | `(..., nlyr)` | required | `tau_arr` | `DELTAU_INPUT` |
| `ssa` | Single-scattering albedo | `(..., nlyr)` | required | `omega_arr` | `OMEGA_INPUT` |
| `g` | Asymmetry factor | `(..., nlyr)` | required | `asymm_arr` | `ASYMM_INPUT` |
| `z` | Level height grid, top to bottom | `(nlyr+1,)` | required for solar and spherical FO paths | `height_grid` | `HEIGHT_GRID` |
| `angles` | Solar observation geometry `[sza, vza, raz]`, degrees | `(3,)` or `(ngeom, 3)` | required for `mode="solar"` | `user_obsgeoms` | `USER_OBSGEOMS` |
| `angles` | Thermal viewing zenith angle(s), degrees | scalar or `(ngeom,)` | required for `mode="thermal"` | `user_angles` | `USER_ANGLES`, converted internally to `USER_STREAMS` |
| `view_angles` | Advanced lattice/thermal view-angle override | `(nview,)` | `None` | `user_angles` | `USER_ANGLES` / `USER_VZANGLES` |
| `beam_szas` | Advanced solar-lattice solar zenith angles | `(nbeam,)` | required for `mode="solar_lattice"` | `beam_szas` | `BEAM_SZAS` |
| `relazms` | Advanced solar-lattice relative azimuth angles | `(nazm,)` | required for `mode="solar_lattice"` | `user_relazms` | `USER_RELAZMS` |
| `stream` | Two-stream quadrature cosine | scalar | `1/sqrt(3)` | `stream_value` | `STREAM_VALUE` |
| `fbeam` | Direct solar beam/source normalization. It scales the incident solar beam/source term and is normally `1.0` unless radiance normalization requires a different beam flux. | scalar or `(...)` | `1.0` | `flux_factor` | `FLUX_FACTOR` / `FLUXFAC` |
| `albedo` | Lambertian surface albedo | scalar or `(...)` | `0.0` | `albedo` | `LAMBERTIAN_ALBEDO` |
| `delta_m_truncation_factor` | Delta-M truncation factor `f`; use explicit values for mixed phase functions or fixture parity | `(..., nlyr)` | `None` -> `g**2` | `d2s_scaling` | `D2S_SCALING`; FO optical path uses `FO_TRUNCFAC` |
| `planck` | Thermal Planck/source value at level boundaries | `(..., nlyr+1)` | required for `mode="thermal"` | `thermal_bb_input` | `THERMAL_BB_INPUT` |
| `surface_planck` | Surface Planck radiance | scalar or `(...)` | `0.0` | `surfbb` | `SURFBB` / `FO_SURFBB` |
| `emissivity` | Surface emissivity | scalar or `(...)` | `0.0` | `emissivity` | `EMISSIVITY` / `FO_USER_EMISSIVITY` |
| `geometry` | FO geometry method: `pseudo_spherical` or `regular_pseudo_spherical` | scalar | `pseudo_spherical` | `fo_geometry_mode` | `FO_SSGeometry_Master_Obs_EPS`, `FO_SSGeometry_Master_Obs_RPS`, `FO_DTGeometry_Master_EPS`, `FO_DTGeometry_Master_PP_RPS` |
| `earth_radius` | Planet radius for spherical paths, km | scalar | `6371.0` | `earth_radius` | `EARTH_RADIUS` |
| `include_fo` | Attach first-order outputs to a main 2S run | scalar | `False` | `include_fo` | FO master-call branch |
| `fo_scatter_term` | Optional precomputed solar FO scatter term: phase function times single-scattering and delta-M source scaling. If omitted, solar FO builds the Henyey-Greenstein term from `ssa`, `g`, and `delta_m_truncation_factor` | `(..., nlyr)` or `(..., nlyr, ngeom)` | `None` | `exact_scatter` | `FO_EXACTSCAT` |
| `n_moments` / `fo_n_moments` | Advanced solar FO phase control. For the default HG fallback, any positive value uses the closed-form HG phase function and `0` is isotropic; explicit phase-moment generation is a separate preprocessing path. | scalar | `5000` | `n_moments` / `fo_n_moments` | `NMOMENTS_INPUT`-style FO moment controls |
| `nfine` / `fo_nfine` | Number of fine sub-layers used by EPS FO spherical-path integration. Larger values refine curved-path integration but increase FO preprocessing/work. | scalar | `3` | `nfine` / `fo_nfine` | `NFINEDIVS` |

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
- Batched profile results are available as `result.radiance_profile_2s`,
  `result.radiance_profile_fo`, and `result.radiance_profile_total` when
  `output_levels=True`.
