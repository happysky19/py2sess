# Level Flux Formula Review

This note records the current conclusion from checking py2sess level-flux
formulas against DISORT-style references.

## Question

A common starting point is

```text
radiance_total = radiance_2s + radiance_fo
flux_total     = integral_angle(radiance_total)
```

This is mathematically true for a single self-consistent angular radiance field.
It is not automatically the right public flux definition for py2sess, because
the reported fluxes are two-stream moment outputs while `radiance_fo` is an
intensity-path correction.

## Literature Check

DISORT separates moment fluxes from intensity corrections:

- DISORT's user outputs define `RFLDIR` as direct-beam flux, `RFLDN` as diffuse
  down-flux, `FLUP` as diffuse up-flux, and `UAVG` as mean intensity. The
  program has a separate `FLUXES` routine for fluxes and an `INTCOR` routine
  for Nakajima-Tanaka intensity correction.
- The DISORT report describes the Nakajima-Tanaka/TMS correction under
  "Correction of the Intensity Field", not as a flux replacement formula:
  https://www.libradtran.org/lib/exe/fetch.php?media=disortreport1.1.pdf
- PythonicDISORT documents the same distinction: `only_flux=True` solves only
  the zeroth Fourier mode, and Nakajima-Tanaka correction is applied to
  intensity, not to flux. It explicitly notes that integrating corrected
  intensity can differ from the flux functions:
  https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html
- Section 6 of Natraj et al. (2023), "The 2 stream-exact single scattering
  (2S-ESS) radiative transfer model", uses DISORT test problems as flux
  references. Its flux definitions separate diffuse up/down flux from the
  direct downwelling beam and define mean intensity with an added direct-beam
  term. The paper also states that `mu_bar = 0.5` should be used when validating
  two-stream calculations against DISORT/LIDORT, because those models use
  Gaussian quadrature over each polar hemisphere.

## Formula Decision

The public DISORT-style py2sess flux definition should stay on the moment path:

```text
F_public = F_2S,moment + Delta F_direct_source
```

For solar, the currently validated `Delta F_direct_source` is Lambertian
direct-beam surface reflection:

```text
Delta F_direct_surface =
  F_direct_surface,exact - F_direct_surface,2S-quadrature
```

Atmospheric FO single-scatter radiance is not integrated into the public
`flux_*` fields by default.

For top isotropic illumination, py2sess treats public `fisot` as the physical
DISORT-style boundary radiance. The two-stream boundary condition projects that
isotropic radiance onto the active stream by preserving incident hemispheric
flux:

```text
pi * fisot = 2 * pi * stream * I_2S
I_2S       = fisot / (2 * stream)
```

This keeps `fisot` independent of the chosen two-stream cosine.

The rejected atmospheric shortcut was

```text
F_2S + F_direct-beam-ss,exact - d/d epsilon F_2S(epsilon omega)|epsilon=0
```

It is not a true first-order total-flux correction. It only replaces the
direct-beam atmospheric single-scatter piece, while a full first-order term
would also include surface-reflected zero-order light that scatters, downward
single-scattered light reflected by the surface, and the delta-M choice of
optical-depth grid and source scaling. It is therefore not exposed in the
public API. The older `single_scatter` and `radiance_integral` correction modes
were also removed because they mixed an intensity-correction path with the
moment-flux path.

## Benchmark Evidence

The pydisort adapter is not the dominant issue: pydisort matches the hard-coded
DISOTEST flux references for the directly comparable cases to roundoff.

The DISOTEST comparison has two distinct reference targets:

- The values in parentheses in Natraj et al. Section 6 are exact/published
  DISORT/LIDORT-style references. Comparing py2sess against these measures the
  physical two-stream approximation error.
- The non-parenthesized values are the paper's 2S-ESS outputs. With
  `stream=0.5`, py2sess reproduces these values for the comparable flux rows;
  for example DISOTEST 1a gives `flux_up(TOA) = 0.0823709`, matching the
  Section 6 2S-ESS value `pi * 0.02622`.

The benchmark script therefore labels `stream=0.5` as `section6` mode and
`stream=1/sqrt(3)` as `public-default` mode. The labels are reporting metadata;
they do not change the solver. Thick, strongly scattering rows such as DISOTEST
7b should be read as two-stream closure stress tests, not as FO angular
quadrature failures.

For direct paper comparison, `scripts/benchmark_disotest_flux.py
--compare-vijay-section6` reports the Zenodo 2S-ESS `2SESSTEST.out.*` values as
`vijay_section6`, alongside the DISORT/LIDORT benchmark and public-default
py2sess outputs.

The rejected direct-beam single-scatter shortcut was tested and failed as a
public correction. On the DISOTEST subset:

```text
case                         default worst %   rejected shortcut worst %
DISOTEST 1a isotropic beam          3.03              -0.82
DISOTEST 1e isotropic beam        -15.62             154.01
DISOTEST 2b Rayleigh beam          -1.94              -9.42
DISOTEST 3a HG beam                17.22              50.30
DISOTEST 3b HG beam               -10.86              23.65
```

The direct-source replacement cases remain accurate:

```text
DISOTEST 6a clear beam             ~0%
DISOTEST 6b absorbing beam         ~0.0002%
DISOTEST 6c absorbing surface      ~0.0017%
```

## Conclusion

The issue was not the pydisort/DISOTEST mapping. The invalid formula was the
documentation claim that public solar flux could be viewed as
`integral_angle(radiance_2s + radiance_fo) + C_moment`.

That formula mixes an intensity-correction path with a moment-flux path and can
worsen thick, conservative, or strongly anisotropic scattering cases. A
direct-beam single-scatter replacement has the same limitation and is not a
complete first-order total-flux formula. The public solver should report
DISORT-style moment fluxes, with only validated source-replacement corrections
enabled by default.
