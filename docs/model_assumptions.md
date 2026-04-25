# Model Assumptions And Defaults

This checklist records public defaults that affect solver setup. Reference
fixtures may still carry Fortran-dumped intermediate arrays, but normal API
calls should either derive these values in Python or require the caller to pass
them explicitly.

| Item | Current behavior | Status |
|---|---|---|
| `stream` | Defaults to `1/sqrt(3)`; must satisfy `0 < stream <= 1`. Fortran parity cases pass benchmark-specific values explicitly. | Public RT convention |
| `fbeam` | Defaults to `1.0`; scales the incident solar beam/source term. Batch calls accept scalar or leading batch shape. | Public source normalization |
| `earth_radius` | Defaults to `6371.0 km`; custom positive finite radii are honored. | Public geometry input |
| `thermal_tcutoff` | Defaults to `1e-8`; must be positive and finite. | Advanced thermal source cutoff |
| `nfine` / `fo_nfine` | Defaults to `3`; must be positive when FO is run. It controls fine sub-layer integration in EPS FO geometry. | FO geometry quadrature control |
| `n_moments` / `fo_n_moments` | Defaults to `5000`; `0` means isotropic phase, negative values are rejected. The default HG fallback uses the closed-form phase function for any positive value; explicit phase moments remain a separate preprocessing path. | FO phase-function control |
| `delta_m_truncation_factor` | Defaults to the differentiable HG fallback `g**2`; `py2sess.optical.phase` can derive the mixed Rayleigh/aerosol Fortran value from phase-moment inputs. | Python-generated unless explicit |
| `fo_scatter_term` | Solar FO builds a differentiable HG term from `ssa`, `g`, geometry, and `delta_m_truncation_factor` when omitted; `py2sess.optical.phase` can derive the mixed Rayleigh/aerosol Fortran term from phase-moment inputs. | Python-generated unless explicit |
| BRDF and surface leaving | Disabled unless `brdf_surface` or `surface_leaving` options and matching coefficient dictionaries are provided. | Explicit surface supplements |
| Thermal surface terms | `emissivity` and `albedo` are independent inputs; examples often use `emissivity = 1 - albedo` for opaque surfaces. | Caller-controlled convention |

Known limitation: the helper now covers the RT-adjacent phase mixing step, but
not the full raw CreateProps/GEOCAPE optical-property pipeline. Gas optical
depths, aerosol fractions, Rayleigh fractions, and endpoint aerosol moments
still need to come from an upstream optical-property provider.
