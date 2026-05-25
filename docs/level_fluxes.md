# Level Fluxes

py2sess can report DISORT-style level flux arrays when
`output_levels=True` and `output_fluxes=True`. The final axis is the atmospheric
level axis ordered from top of atmosphere to bottom of atmosphere, with length
`nlyr + 1`.

## Output Fields

- `flux_up`: upward hemispheric flux.
- `flux_down`: downward hemispheric flux, including the direct solar beam for
  solar cases.
- `flux_net`: `flux_up - flux_down`.
- `flux_mean`: mean-intensity or actinic-style flux proxy.

For a diffuse radiance field,

```text
F_up(tau)   = integral_up     mu I(tau, Omega) dOmega
F_down(tau) = integral_down   mu I(tau, Omega) dOmega
F_net(tau)  = F_up(tau) - F_down(tau)
J(tau)      = (1 / 4pi) integral_4pi I(tau, Omega) dOmega
```

For a direct solar beam with top-of-atmosphere source normalization `F0`,
solar cosine `mu0`, and direct transmittance `T_sun(tau)`,

```text
F_down,direct(tau) = F0 mu0 T_sun(tau)
J_direct(tau)      = F0 T_sun(tau) / (4pi)
```

## Public Flux Definition

The public `flux_*` fields stay on the two-stream moment path. This is the same
separation used by DISORT-style solvers: fluxes are moment outputs, while
single-scattering or TMS-style corrections are intensity corrections.

The default public solar flux is therefore not defined as an angular integral
of `radiance_2s + radiance_fo`. That identity is exact only for one
self-consistent angular radiance field. py2sess radiance output combines a
two-stream moment reconstruction with first-order intensity-path corrections,
so directly integrating that mixed radiance can give worse fluxes in thick,
conservative, or strongly anisotropic scattering cases.

Validated separable source replacements are included where they preserve the
moment-flux convention. Solar direct-beam surface reflection and thermal
source-transmission flux corrections are handled this way. Atmospheric
first-order scattering radiance is not integrated into the public solar
`flux_*` fields by default.

Top isotropic illumination uses a physical DISORT-style boundary radiance
`fisot`. The two-stream boundary condition preserves incident hemispheric flux:

```text
pi * fisot = 2 * pi * stream * I_2S
I_2S       = fisot / (2 * stream)
```

This keeps `fisot` independent of the selected two-stream cosine.

## Angular Quadrature Controls

`fo_flux_n_mu` controls the positive-hemisphere Gaussian quadrature used by
first-order flux source replacements. The default is `8`. Nodes and weights are
the Gauss-Legendre rule on `[0, 1]`; up and down hemispheres reuse those
positive `mu` nodes with the appropriate propagation direction.

`fo_flux_n_phi` is reserved for source replacements that require explicit
azimuth quadrature. Current public source replacements are axisymmetric or
analytic in azimuth, so the default `None` avoids unnecessary azimuth work.
When explicit azimuth is used, py2sess uses midpoint nodes

```text
phi_j = 2*pi*(j + 1/2) / fo_flux_n_phi
```

with uniform weight `2*pi / fo_flux_n_phi`.

## Analytic Check

A clear absorbing solar column has the Beer-Lambert level-flux solution

```text
flux_down(level) = F0 * mu0 * exp(-tau_cumulative(level) / mu0)
flux_up(level)   = 0
flux_net(level)  = -flux_down(level)
```

Run the checked-in example with:

```bash
PYTHONPATH=src python3 examples/level_flux_beer_lambert.py
```

## Benchmark Commands

Lightweight analytic and optional pydisort checks:

```bash
PYTHONPATH=src python3 scripts/benchmark_level_flux_references.py
```

DISOTEST-style comparisons:

```bash
PYTHONPATH=src python3 scripts/benchmark_disotest_flux.py
PYTHONPATH=src python3 scripts/benchmark_disotest_flux.py --suite disort-test
PYTHONPATH=src python3 scripts/benchmark_disotest_flux.py --suite pydisort-grid
```

The default two-stream cosine is `1/sqrt(3)`. Passing `--stream 0.5` reproduces
the Gaussian-hemisphere convention used in published 2S-ESS DISORT comparison
tests.
The stream value changes the two-stream closure, not the physical definition of
the reported flux fields.
