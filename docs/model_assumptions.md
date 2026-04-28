# Model Assumptions And Defaults

This checklist records public defaults that affect solver setup.

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
| Scene optical inputs | `py2sess.optical.scene_io` loads profile text plus YAML scenes. The scene helpers generate layer optical properties, Rayleigh/aerosol fractions, phase inputs, and thermal sources when the required physical inputs are present. A local CreateProps provider remains available for current full-spectrum parity. | Python-generated foundation plus interim provider |
| Scene gas and height overrides | Scene YAML can supply fixed/background gas VMRs such as `gas_vmr: {O2: 0.2095}` and explicit `atmosphere.heights_km` when the profile file does not carry the benchmark vertical grid. | Caller-provided physical scene inputs |
| Layer optical properties | `py2sess.optical.properties` can combine component optical depths into `tau`, `ssa`, Rayleigh scattering fraction, and aerosol scattering fractions. Prefer `absorption_tau` for total non-scattering absorption; `gas_absorption_tau` is accepted when that absorption is specifically gas-only. | Python-generated when component optical depths exist |
| BRDF and surface leaving | Disabled unless `brdf_surface` or `surface_leaving` options and matching coefficient dictionaries are provided. | Explicit surface supplements |
| Thermal source terms | Thermal RT requires `planck` and `surface_planck`. Benchmark bundles may provide legacy `thermal_bb_input`/`surfbb` directly, or provide temperatures plus a wavelength, wavenumber, or wavenumber-band coordinate so Python can generate the public source inputs. | Python-generated when physical inputs exist |
| Thermal surface terms | `emissivity` and `albedo` are independent inputs. Benchmark bundles use explicit `emissivity` when available and fall back to `1 - albedo` only for older bundles. | Caller-controlled convention |

Known limitation: direct HITRAN line-by-line opacity is for limited
validation/offline table generation. Full-spectrum runtime should use saved
pressure-temperature opacity tables. O4, surface table readers, and pyharp-style
adapters remain separate work.
