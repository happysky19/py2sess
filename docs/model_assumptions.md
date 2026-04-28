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
| Scene optical inputs | `py2sess.optical.scene_io` loads profile text plus YAML scene files. `py2sess.optical.scene` covers GEOCAPE-style hydrostatic profile setup, air columns, Rayleigh Bodhaine cross section/depolarization, simple gas optical-depth integration, aerosol table interpolation, and `build_scene_opacity_components()` for the Python equivalents of CreateProps `taug`, `taudp`, `omega`, `fr`, and `fa`. `py2sess.optical.geocape` reads GEOCAPE gas cross-section tables, aerosol loading files, and SSprops bulk/moment tables. `py2sess.optical.hitran` covers the Fortran no-convolution HITRAN path. `py2sess.optical.createprops` can read a local Fortran CreateProps provider directory for current benchmark parity. | Python-generated foundation plus interim provider |
| Scene gas and height overrides | Scene YAML can supply fixed/background gas VMRs such as `gas_vmr: {O2: 0.2095}` and explicit `atmosphere.heights_km` when the profile file does not carry the benchmark vertical grid. | Caller-provided physical scene inputs |
| Layer optical properties | `py2sess.optical.properties` can combine component optical depths into `tau`, `ssa`, Rayleigh scattering fraction, and aerosol scattering fractions. Prefer `absorption_tau` for total non-scattering absorption; `gas_absorption_tau` is accepted when that absorption is specifically gas-only. | Python-generated when component optical depths exist |
| BRDF and surface leaving | Disabled unless `brdf_surface` or `surface_leaving` options and matching coefficient dictionaries are provided. | Explicit surface supplements |
| Thermal source terms | Thermal RT requires `planck` and `surface_planck`. Benchmark bundles may provide legacy `thermal_bb_input`/`surfbb` directly, or provide temperatures plus a wavelength, wavenumber, or wavenumber-band coordinate so Python can generate the public source inputs. | Python-generated when physical inputs exist |
| Thermal surface terms | `emissivity` and `albedo` are independent inputs. Benchmark bundles use explicit `emissivity` when available and fall back to `1 - albedo` only for older bundles. | Caller-controlled convention |

Known limitation: the Fortran CreateProps provider remains an interim full
benchmark source until the Python HITRAN path is validated at full spectrum
size. O4 handling, surface table readers, and pyharp-style adapters remain
separate work.
