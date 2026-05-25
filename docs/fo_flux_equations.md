# First-Order Level Flux Equations

This note records the equations used when py2sess reports level fluxes that
include first-order (FO) contributions. The base `output_fluxes=True` path
returns two-stream level fluxes, with the direct solar beam included in
`flux_down` and `flux_mean` for solar cases. The attached FO solver returns
radiance along requested observation geometries; a single observation-geometry
radiance is not a flux.

## Conventions

The py2sess canonical level axis is top of atmosphere to bottom of atmosphere,
with final axis length `nlyr + 1`.

Use a local level coordinate with optical depth `tau` increasing downward.
For angular integrations, let `mu` be the absolute cosine from the local
vertical, `0 <= mu <= 1`.

For any diffuse radiance field,

```text
F_up(tau)   = integral_up     mu I(tau, Omega) dOmega
F_down(tau) = integral_down   mu I(tau, Omega) dOmega
F_net(tau)  = F_up(tau) - F_down(tau)
J(tau)      = (1 / 4pi) integral_4pi I(tau, Omega) dOmega
```

py2sess stores `J` in `flux_mean`. For a direct solar beam with source
normalization `F0`, solar cosine `mu0`, and direct transmittance `T_sun(tau)`,

```text
F_down,direct(tau) = F0 mu0 T_sun(tau)
J_direct(tau)      = F0 T_sun(tau) / (4pi)
```

These are the direct-beam terms already used by the current solar two-stream
flux output.

## Why A Derivation Is Required

The current solar FO output is

```text
I_FO(tau_l; mu_view, phi_view)
```

for one or more requested observation directions. The desired flux contribution
is instead

```text
F_up,FO(tau_l)   = integral_up   mu I_FO(tau_l, Omega) dOmega
F_down,FO(tau_l) = integral_down mu I_FO(tau_l, Omega) dOmega
J_FO(tau_l)      = (1 / 4pi) integral_4pi I_FO(tau_l, Omega) dOmega
```

Therefore we cannot multiply the existing FO radiance by `2pi mu` or reuse the
user view angle. We need either a closed-form hemispheric integral or an
explicit angular quadrature over FO directions.

## Solar FO Source

For a plane-parallel layer and a local direction `Omega`, the first-order
single-scatter radiance has the form

```text
I_ss(tau, Omega)
  = F0 / (4pi) integral_path
      omega_eff(t) P(cosTheta(Omega_sun, Omega), t)
      T_sun(t) T_view(t -> tau, Omega) ds_tau
```

where

```text
omega_eff = omega / (1 - f omega)
```

matches the current delta-M source-side scaling used by the FO solar path,
`P` is normalized so its full-sphere integral is `4pi`, `T_sun` is direct-beam
transmittance to the scattering point, and `T_view` is transmittance from the
scattering point to the level along `Omega`.

For level fluxes, this radiance must be integrated over the hemisphere:

```text
F_up,ss(tau_l)
  = integral_{Omega up} mu I_ss(tau_l, Omega) dOmega

F_down,ss(tau_l)
  = integral_{Omega down} mu I_ss(tau_l, Omega) dOmega

J_ss(tau_l)
  = (1 / 4pi) integral_{4pi} I_ss(tau_l, Omega) dOmega
```

For isotropic scattering in plane-parallel geometry, `P = 1`, so the azimuth
integral is exact and the layer-source part reduces to exponential-integral
kernels:

```text
F_up,ss(tau) =
  F0 / 2 integral_tau^tau_s
    omega_eff(t) T_sun(t) E2(t - tau) dt

F_down,ss(tau) =
  F0 / 2 integral_0^tau
    omega_eff(t) T_sun(t) E2(tau - t) dt

J_ss(tau) =
  F0 / (8pi) [
    integral_tau^tau_s omega_eff(t) T_sun(t) E1(t - tau) dt
    + integral_0^tau omega_eff(t) T_sun(t) E1(tau - t) dt
  ]
```

where

```text
E_n(x) = integral_0^1 mu^(n - 2) exp(-x / mu) dmu
```

These identities are useful analytic tests, not necessarily the fastest
implementation for general phase functions.

For the pure Rayleigh family used by `py2sess.optical.phase`,

```text
P(cosTheta) = 1 + delta [ 3/4 (1 + cosTheta^2) - 1 ]
```

where `delta = 1` is the usual unpolarized Rayleigh phase function and
`delta = 0` reduces to isotropic scattering. After azimuth averaging over a
hemisphere for a solar beam cosine `mu0`,

```text
Pbar(mu) = a0 + a2 mu^2
a0 = 1 - delta + delta * 3/4 * (3/2 - mu0^2 / 2)
a2 = delta * 3/4 * (3 mu0^2 / 2 - 1/2)
```

so the exact single-scatter flux kernels become

```text
F_up,ss(tau) =
  F0 / 2 integral_tau^tau_s
    omega_eff(t) T_sun(t) [a0 E2(t - tau) + a2 E4(t - tau)] dt

F_down,ss(tau) =
  F0 / 2 integral_0^tau
    omega_eff(t) T_sun(t) [a0 E2(tau - t) + a2 E4(tau - t)] dt

J_ss(tau) =
  F0 / (8pi) [
    integral_tau^tau_s omega_eff(t) T_sun(t)
      [a0 E1(t - tau) + a2 E3(t - tau)] dt
    + integral_0^tau omega_eff(t) T_sun(t)
      [a0 E1(tau - t) + a2 E3(tau - t)] dt
  ]
```

These Rayleigh expressions are used as analytic references for explicit
Rayleigh `fo_scatter_term` cases. Generic explicit `fo_scatter_term` values
remain observation-geometry terms; they do not define a full angular phase
function for flux integration unless they match an identifiable phase family
such as isotropic or pure Rayleigh.

For anisotropic phase functions, the implementation should use angular
quadrature over `(mu, phi)` unless a closed-form hemispheric phase integral is
derived and validated. This keeps the first implementation simple and avoids
mixing observation-geometry shortcuts into a flux calculation.

## Solar Atmospheric Two-Stream Counterpart

The two-stream moment solution already includes a low-order quadrature
representation of atmospheric scattering. It is tempting to replace the
two-stream single-scatter piece with an exact FO hemispheric integral, but that
is not a stable public total-flux operation: in thick or conservative scattering
cases the two-stream single- and multiple-scatter errors are coupled through the
moment closure and can cancel. The formulas below are useful diagnostics for
component tests, not part of the default public `flux_*` total.

This matches the DISORT literature. The DISORT report describes the
Nakajima-Tanaka/TMS idea as subtracting an approximate single-scattering
intensity and adding back a more exact single-scattering intensity, but it also
states that those corrections are not applied to flux, flux divergence, or mean
intensity because delta-M fluxes are already designed to be accurate and
conserve flux. PythonicDISORT documents the same behavior: `only_flux=True`
solves only the zeroth Fourier mode, while Nakajima-Tanaka is an intensity
correction, not a flux correction.

References:

- DISORT Report v1.1, sections 3.6.1-3.6.3:
  https://www.libradtran.org/lib/exe/fetch.php?media=disortreport1.1.pdf
- PythonicDISORT documentation, sections 3.2 and 3.7.2:
  https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html

Let `Ibar_up,2S_ss(tau; mu1)` and `Ibar_down,2S_ss(tau; mu1)` be the azimuthal
mean of the FO single-scatter radiance at `mu1`, with surface reflection turned
off. A direct-beam single-scatter diagnostic counterpart is

```text
F_up,2S_ss(tau)   = 2pi mu1 Ibar_up,2S_ss(tau; mu1)
F_down,2S_ss(tau) = 2pi mu1 Ibar_down,2S_ss(tau; mu1)
J_2S_ss(tau)      = 0.5 [Ibar_up,2S_ss(tau; mu1) + Ibar_down,2S_ss(tau; mu1)]
```

For isotropic scattering this reduces to

```text
Ibar_up,2S_ss(tau; mu1) =
  F0 / (4pi) integral_tau^tau_s
    omega_eff(t) T_sun(t) exp(-(t - tau) / mu1) dt / mu1

Ibar_down,2S_ss(tau; mu1) =
  F0 / (4pi) integral_0^tau
    omega_eff(t) T_sun(t) exp(-(tau - t) / mu1) dt / mu1
```

This diagnostic is not a valid total-flux correction by itself. A true
first-order flux term would need every one-atmospheric-scatter path generated
from the zeroth-order field, including direct-beam surface reflection that later
scatters, downwelling single-scatter radiation reflected by the surface, and the
delta-M choice of optical-depth grid and source scaling. The rejected shortcut

```text
F_2S + F_direct-beam-ss,exact - d/d epsilon F_2S(epsilon omega)|epsilon=0
```

only replaces the direct-beam atmospheric single-scatter piece. It is therefore
not exposed as a public `flux_*` option.

## Solar Direct Surface Reflection

The current solar FO `direct_beam` radiance term is direct-beam reflection from
the lower boundary into the requested view direction. For a Lambertian surface,
the BOA radiance is

```text
I_surf,db(BOA, Omega_up) = F0 mu0 A T_sun(BOA) / pi
```

and the upward flux at BOA is

```text
F_up,surf_db(BOA) = F0 mu0 A T_sun(BOA)
```

At an interior level `tau`, with vertical optical separation `Delta tau` to the
surface in plane-parallel geometry,

```text
F_up,surf_db(tau) =
  2pi I_surf,db(BOA) E3(Delta tau)

J_surf_db(tau) =
  0.5 I_surf,db(BOA) E2(Delta tau)
```

These equations change for BRDF surfaces; a BRDF implementation must integrate
the reflected direct-beam radiance over the upward hemisphere.

## Thermal FO Source

For thermal emission in plane-parallel geometry, the formal solution for
upward radiance at level `tau` is

```text
I_up(tau, mu)
  = I_surf(mu) exp(-(tau_s - tau) / mu)
    + integral_tau^tau_s S(t) exp(-(t - tau) / mu) dt / mu
```

The corresponding flux and mean-intensity contributions are

```text
F_up,th(tau) =
  2pi I_surf E3(tau_s - tau)
  + 2pi integral_tau^tau_s S(t) E2(t - tau) dt

J_up,th(tau) =
  0.5 I_surf E2(tau_s - tau)
  + 0.5 integral_tau^tau_s S(t) E1(t - tau) dt
```

With no incident thermal radiation at TOA, the downwelling terms are

```text
F_down,th(tau) =
  2pi integral_0^tau S(t) E2(tau - t) dt

J_down,th(tau) =
  0.5 integral_0^tau S(t) E1(tau - t) dt
```

The current FO thermal solver uses a line-of-sight source integral for requested
view angles. Reporting FO thermal fluxes requires evaluating these hemispheric
integrals, not reweighting the requested view-angle radiance.

## DISORT-Style Public Flux Definition

DISORT-style fluxes are moment-equation outputs, not automatically the angular
integral of an intensity field after intensity corrections have been applied.
This distinction matters for py2sess because `radiance_total = radiance_2s +
radiance_fo` is an intensity-path quantity, while `flux_up`, `flux_down`,
`flux_net`, and `flux_mean` are reported as level moments.

For scalar NumPy solar runs with `include_fo=True` and `output_fluxes=True`,
the public default is therefore:

```text
F_public = F_2S,moment + Delta F_direct_source
```

where `Delta F_direct_source` is limited to source terms whose hemispheric flux
can be replaced without changing the atmospheric scattering moment closure. For
the current solar implementation this is Lambertian direct-beam surface
reflection:

```text
Delta F_direct_surface =
  F_direct_surface,exact - F_direct_surface,2S-quadrature
```

Atmospheric FO single-scatter radiance is not integrated back into the public
flux by default. DISORT and PythonicDISORT make the same algorithmic separation:
Nakajima-Tanaka/TMS corrections are intensity corrections, while fluxes remain
moment quantities. Consequently, integrating a corrected intensity field can
differ from the flux function.

Thermal FO fluxes use the same source-replacement pattern for direct thermal
source transmission:

```text
F_public = F_2S + (F_FO,exact - F_FO,2S-quadrature)
```

with the rule applied to `flux_up`, `flux_down`, and `flux_mean`, followed by

```text
F_net,total  = F_up,total - F_down,total
```

For the current implementation, this replacement is applied to direct thermal
atmospheric and surface source transmission. No atmospheric solar scattering
flux correction is exposed.

## FO Flux Angular Quadrature

The configurable angular quadrature is used only by FO flux source-replacement
integrals. It is not used by the main 2S moment solve and does not change the
view-angle FO radiance path.

For the polar cosine, py2sess uses Gauss-Legendre quadrature on the positive
hemisphere:

```text
mu_i     = (x_i + 1) / 2
w_mu_i  = w_i / 2
```

where `x_i, w_i` are the standard Gauss-Legendre nodes and weights on
`[-1, 1]`. The user-facing option is `fo_flux_n_mu`, with default `8`.

For azimuth, py2sess uses a uniform midpoint rule:

```text
phi_j   = 2 pi (j + 1/2) / N_phi
w_phi_j = 2 pi / N_phi
```

with `N_phi = fo_flux_n_phi`. Current public source replacements are
axisymmetric or analytic in azimuth, so the default is `fo_flux_n_phi=None`:
py2sess applies the analytic `2 pi` azimuth factor and only performs numerical
quadrature in `mu`. The option is still exposed so future non-axisymmetric FO
flux terms can use the same public control.

The previously considered formula

```text
F_total = integral_angle(I_2S + I_FO) + C_moment
```

is not used as the public DISORT-style flux definition. It mixes an
intensity-corrected radiance path with a moment-equation flux path and can be
substantially worse in conservative, thick, or strongly anisotropic scattering
cases.

## Implementation Checklist

1. Keep current `flux_up`, `flux_down`, `flux_net`, and `flux_mean` fields.
2. For scalar NumPy solar `include_fo=True`, keep public fluxes on the
   DISORT-style moment path. Do not integrate atmospheric FO radiance
   corrections into public flux by default.
3. For any solar FO source term exposed in public fluxes, use a hemispheric
   angular integral, not the user observation geometry alone. Keep atmospheric
   single-scatter flux formulas diagnostic until a complete validated correction
   is available.
4. Implement thermal FO flux from hemispheric source integrals or an equivalent
   angular quadrature.
5. Validate against:
   - zero-source and zero-optical-depth limits;
   - pure absorbing solar Beer-Lambert direct flux;
   - isotropic single-scatter analytic formulas above;
   - isothermal black thermal slab consistency;
   - pydisort `gather_flx()` for small deterministic cases.
6. Treat KINETICS actinic/total radiation field comparisons as diagnostic only.
