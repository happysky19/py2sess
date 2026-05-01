# Public API Argument Reference

`py2sess` exposes one public solver API: `TwoStreamEss.forward(...)`. The last
column gives the closest Fortran or internal name for code comparison.

`2S` is the two-stream multiple-scattering/emission solve. `FO` is the
first-order solar single-scatter/direct-beam or thermal source-transmission
correction. With `include_fo=True`, `result.radiance_total` is `2S + FO`.

Conventions: angles are degrees; solar geometry is `[sza, vza, raz]`; thermal
geometry is viewing zenith angle only; heights `z` are km from top to bottom;
layer optical thickness is positive downward. Radiance and Planck units are
caller-defined and are not converted.

`forward()` accepts either a single atmosphere with shape `(nlyr,)` or leading
batch dimensions such as `(nwave, nlyr)` and `(ncol, nwave, nlyr)`. Batch
dimensions are preserved in `result.radiance`; multiple solar or thermal
geometries append one final geometry axis. The default batched path returns TOA
`radiance_2s`, optional `radiance_fo`, and `radiance_total`, but not BOA fluxes.
Set `TwoStreamEssOptions(output_levels=True)` only when level radiance profiles
are needed; profile arrays use the final axis for TOA-to-BOA levels.

| Public name | Meaning | Shape | Default | Fortran/internal name |
|---|---|---:|---|---|
| `nlyr` | Number of atmospheric layers | scalar | required | `NLAYERS` |
| `mode` | Source mode: `solar`, `solar_lattice`, or `thermal` | scalar | `solar` | source-mode branch |
| `output_levels` | Return level-by-level radiance profiles | scalar | `False` | `DO_LEVEL_OUTPUT` |
| `upwelling` | Compute upwelling outputs | scalar | `True` | `DO_UPWELLING` |
| `downwelling` | Compute downwelling outputs | scalar | `False` | `DO_DNWELLING` |
| `plane_parallel` | Use plane-parallel geometry instead of spherical geometry | scalar | `False` | `DO_PLANE_PARALLEL` |
| `delta_scaling` | Apply delta-M scaling in the 2S core | scalar | `True` | `DO_D2S_SCALING` |
| `brdf_surface` | Enable explicit BRDF coefficients | scalar | `False` | `DO_BRDF_SURFACE` |
| `bvp_solver` | 2S boundary-value solver: `auto`, `scipy`, `banded`, or `pentadiag` | scalar | `auto` | BVP solver selector |
| `tau` | Layer optical thickness | `(..., nlyr)` | required | `DELTAU_INPUT` |
| `ssa` | Single-scattering albedo | `(..., nlyr)` | required | `OMEGA_INPUT` |
| `g` | Asymmetry factor | `(..., nlyr)` | required | `ASYMM_INPUT` |
| `z` | Level height grid, top to bottom, km | `(nlyr+1,)` | required for solar and spherical FO paths | `HEIGHT_GRID` |
| `angles` | Solar `[sza, vza, raz]`, degrees, or thermal viewing zenith angle(s), degrees | `(3,)`, `(ngeom, 3)`, scalar, or `(ngeom,)` | required | `USER_OBSGEOMS` / `USER_ANGLES` |
| `view_angles` | Advanced lattice/thermal view-angle override, degrees | `(nview,)` | `None` | `USER_ANGLES` / `USER_VZANGLES` |
| `beam_szas` | Advanced solar-lattice solar zenith angles, degrees | `(nbeam,)` | required for `mode="solar_lattice"` | `BEAM_SZAS` |
| `relazms` | Advanced solar-lattice relative azimuth angles, degrees | `(nazm,)` | required for `mode="solar_lattice"` | `USER_RELAZMS` |
| `stream` | Two-stream quadrature cosine | scalar | `1/sqrt(3)` | `STREAM_VALUE` |
| `fbeam` | Direct solar beam/source normalization | scalar or `(...)` | `1.0` | `FLUX_FACTOR` / `FLUXFAC` |
| `albedo` | Lambertian surface albedo | scalar or `(...)` | `0.0` | `LAMBERTIAN_ALBEDO` |
| `delta_m_truncation_factor` | Delta-M truncation factor `f` | `(..., nlyr)` | `None` -> `g**2` | `D2S_SCALING` / `TRUNCFAC` |
| `planck` | Thermal Planck/source value at level boundaries | `(..., nlyr+1)` | required for `mode="thermal"` | `THERMAL_BB_INPUT` |
| `surface_planck` | Surface Planck radiance | scalar or `(...)` | `0.0` | `SURFBB` / `FO_SURFBB` |
| `emissivity` | Surface emissivity | scalar or `(...)` | `0.0` | `EMISSIVITY` / `FO_USER_EMISSIVITY` |
| `geometry` | FO geometry method: `pseudo_spherical` or `regular_pseudo_spherical` | scalar | `pseudo_spherical` | EPS/RPS selector |
| `earth_radius` | Planet radius for spherical paths, km | scalar | `6371.0` | `EARTH_RADIUS` |
| `include_fo` | Attach first-order outputs to a main 2S run | scalar | `False` | FO master-call branch |
| `fo_scatter_term` | Solar FO phase/source term: phase function times single-scattering and delta-M source scaling | `(..., nlyr)` or `(..., nlyr, ngeom)` | HG term from `ssa`, `g`, `f` | `FO_EXACTSCAT` |
| `n_moments` / `fo_n_moments` | Advanced solar FO phase control; positive values use the closed-form HG fallback unless explicit phase data are supplied | scalar | `5000` | `NMOMENTS_INPUT` |
| `nfine` / `fo_nfine` | Number of fine sub-layers used by EPS FO spherical-path integration | scalar | `3` | `NFINEDIVS` |

## Notes

- `delta_m_truncation_factor=None` uses the HG fallback `f = g**2`, clipped to
  `0 <= f < 1`. Pass explicit values for mixed Rayleigh/aerosol phase inputs.
- `delta_scaling=False` disables the 2S optical-property transform. Solar FO
  still uses the truncation factor in its source term unless `fo_scatter_term`
  is passed explicitly.
- Solar and thermal source handling are mode-exclusive. A single `TwoStreamEss`
  call does not combine direct solar and Planck thermal sources.
- `bvp_solver="auto"` uses the optimized batch dispatch for batched thermal
  runs and the banded scalar path for scalar thermal runs. Use explicit values
  only for debugging or parity checks.
- Direct HITRAN line-by-line opacity is for limited validation/offline table
  generation. Full-spectrum runtime should use saved gas cross-section tables.
- The high-level scene entry point is `load_scene(profile=..., config=...)`;
  it builds the same public `forward()` inputs listed above.

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
