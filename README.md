# py2sess

`py2sess` is a Python implementation of the optimized 2S-ESS radiative-transfer
model. It supports solar and thermal forward calculations with NumPy and
optional torch backends. It does not call the original Fortran code.

## Install

```bash
python3 -m pip install .
python3 -m pip install -e ".[torch,dev]"
```

For source-tree runs without installation, set `PYTHONPATH=src`.

## Build

`py2sess` uses CMake through `scikit-build-core`. The current release is pure
Python, so CMake prepares the build tree and leaves room for future native
kernels without changing the package layout.

```bash
cmake -S . -B build
cmake --build build
python3 -m build
```

Releases are tagged from merged PRs by GitHub Actions. Add a `release:major`,
`release:minor`, or `release:patch` label to the PR before merging; the default
is `release:patch`.

PyPI publication uses Trusted Publishing. Configure PyPI to trust
`happysky19/py2sess`, workflow `.github/workflows/release.yml`, and the `pypi`
GitHub environment, then run the `Publish to PyPI` GitHub Actions workflow for
the tag.

## Quick Start

Solar:

```python
import numpy as np
from py2sess import TwoStreamEss, TwoStreamEssOptions

solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
result = solver.forward(
    tau=np.full(3, 0.02),
    ssa=np.full(3, 0.2),
    g=np.full(3, 0.1),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 20.0, 0.0],  # sza, vza, relative azimuth
    albedo=0.3,
)
print(result.radiance)
```

Thermal:

```python
solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
result = solver.forward(
    tau=np.full(3, 0.1),
    ssa=np.zeros(3),
    g=np.zeros(3),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=20.0,
    planck=np.array([1.0, 1.1, 1.2, 1.3]),
    surface_planck=1.4,
    emissivity=1.0,
)
```

Batched wavelengths use leading dimensions:

```python
solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
tau = np.full((100, 3), 0.02)
result = solver.forward(
    tau=tau,
    ssa=np.zeros_like(tau),
    g=np.zeros_like(tau),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=20.0,
    planck=np.ones((100, 4)),
    surface_planck=np.ones(100),
    emissivity=np.ones(100),
)
print(result.radiance.shape)  # (100,)
```

Level fluxes use the final axis for TOA-to-BOA levels. This clear absorbing
solar case has an analytic Beer-Lambert flux solution:

```python
import numpy as np
from py2sess import TwoStreamEss, TwoStreamEssOptions

sza = 30.0
mu0 = np.cos(np.deg2rad(sza))
fbeam = 1.0
tau = np.array([0.1, 0.2])
z = np.array([2.0, 1.0, 0.0])

solver = TwoStreamEss(
    TwoStreamEssOptions(
        nlyr=tau.size,
        mode="solar",
        plane_parallel=True,
        delta_scaling=False,
        downwelling=True,
        output_levels=True,
        output_fluxes=True,
        fo_flux_n_mu=8,
    )
)
result = solver.forward(
    tau=tau,
    ssa=np.zeros_like(tau),  # pure absorption
    g=np.zeros_like(tau),
    z=z,
    angles=[sza, 0.0, 0.0],
    fbeam=fbeam,
    albedo=0.0,  # black surface: no upward reflected flux
    delta_m_truncation_factor=np.zeros_like(tau),
    include_fo=True,
)

level_tau = np.concatenate(([0.0], np.cumsum(tau)))
analytic_down = fbeam * mu0 * np.exp(-level_tau / mu0)

np.testing.assert_allclose(result.flux_down[0], analytic_down, atol=1.0e-9)
np.testing.assert_allclose(result.flux_up[0], 0.0, atol=1.0e-8)
np.testing.assert_allclose(result.flux_net, result.flux_up - result.flux_down)

print(result.flux_down[0])
```

Torch CPU float64:

```python
solver = TwoStreamEss(
    TwoStreamEssOptions(nlyr=3, mode="solar", backend="torch", torch_dtype="float64")
)
```

## API Notes

Core inputs are `tau`, `ssa`, `g`, `z`, `angles`, and the surface/source terms
needed by the selected mode. Solar angles are `[sza, vza, raz]` in degrees;
thermal angles are viewing zenith angles. Heights are in km, ordered top to
bottom.

See [`docs/api_arguments.md`](docs/api_arguments.md) for the full argument
table and conventions.
Level-flux conventions are summarized in
[`docs/level_fluxes.md`](docs/level_fluxes.md).

## Examples

```bash
python3 examples/level_flux_beer_lambert.py
python3 examples/build_thermal_source_from_temperature.py
python3 examples/retrieve_synthetic_spectra.py --case uv --noise-level 0
```

Scene/profile runs:

```python
from py2sess.scene import load_scene

scene = load_scene(profile="profile.txt", config="scene.yaml")
result = scene.forward(backend="numpy", include_fo=True)
```

Full-spectrum benchmark details are in
[`docs/full_spectrum_benchmarks.md`](docs/full_spectrum_benchmarks.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m ruff check .
python3 -m ruff format --check .
```

Full-spectrum benchmarks use profile text plus scene YAML inputs and Python
optical preprocessing. Keep large local cross-section tables, benchmark bundles,
and generated outputs out of git.

## References

- Fortran repository: [vnatraj1/2S-ESS](https://github.com/vnatraj1/2S-ESS)
- Paper: [Natraj et al., 2022, JQSRT](https://www.sciencedirect.com/science/article/pii/S002240732200351X)
