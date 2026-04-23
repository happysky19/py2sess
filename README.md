# py2sess

`py2sess` is a standalone Python implementation of the optimized 2S-ESS forward radiative-transfer model.

It provides:

- solar observation-geometry forward calculations
- thermal observation-geometry forward calculations
- NumPy and torch execution backends
- packaged benchmark fixtures for regression testing
- NumPy and differentiable torch helpers for thermal source construction
- surface-leaving input helpers

This repository is intentionally independent of the original Fortran codebase. It does not build or invoke Fortran code, and it does not retain the mixed-repository porting history.

Primary upstream references:

- Fortran repository: [vnatraj1/2S-ESS](https://github.com/vnatraj1/2S-ESS)
- Paper: [Natraj et al., 2022, Journal of Quantitative Spectroscopy and Radiative Transfer](https://www.sciencedirect.com/science/article/pii/S002240732200351X)

## Installation

```bash
python3 -m pip install -e .
```

Optional extras:

```bash
python3 -m pip install -e ".[torch]"
python3 -m pip install -e ".[speed]"
python3 -m pip install -e ".[plot]"
python3 -m pip install -e ".[dev]"
```

Editable install is the recommended development mode. After installation, the
package imports as `py2sess` without setting `PYTHONPATH`.

## Quick Start

```python
import numpy as np
from py2sess import TwoStreamEss, TwoStreamEssOptions

solver = TwoStreamEss(
    TwoStreamEssOptions(
        nlyr=3,
        mode="solar",
        output_levels=True,
    )
)

result = solver.forward_fo(
    tau=np.zeros(3),
    ssa=np.zeros(3),
    g=np.zeros(3),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 0.0, 0.0],
    albedo=0.3,
)

print(result.radiance)
```

For the normal 2S forward model, use `forward()`:

```python
result = solver.forward(
    tau=np.full(3, 0.02),
    ssa=np.full(3, 0.2),
    g=np.full(3, 0.1),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 20.0, 0.0],
    albedo=0.3,
)

print(result.radiance)
```

## Public API Names

The public API uses one canonical set of short radiative-transfer names while
keeping the Fortran-derived names inside the solver core. It follows pydisort
where the physics matches directly: `nlyr`, `tau`, `ssa`, `fbeam`, and
`albedo`. It keeps `g` instead of DISORT `pmom` because 2S-ESS consumes a
single layer asymmetry factor, not a full phase-moment array. Advanced controls
such as `stream`, `fbeam`, `delta_m_scaling`, and `geometry` have defaults for
routine calls.

For full-spectrum or multi-column work, pass leading batch dimensions on
`tau`, `ssa`, `g`, `delta_m_scaling`, `planck`, `albedo`, `surface_planck`,
`emissivity`, and `fbeam`. For example, `(nwave, nlyr)` returns
`result.radiance.shape == (nwave,)` for one geometry, while multiple requested
geometries append a final geometry axis. Batched `forward()` uses the optimized
endpoint kernels by default. Set `output_levels=True` to request upwelling
radiance profiles; the final profile axis is ordered from TOA to BOA.

| New API name | Meaning | Shape | Default | Old Python name | Fortran/internal name |
|---|---|---:|---|---|---|
| `nlyr` | Number of atmospheric layers | scalar | required | `n_layers` | `NLAYERS` |
| `mode` | Source mode: `solar`, `solar_lattice`, or `thermal` | scalar | `solar` | `source_mode` | source-mode branch |
| `tau` | Layer optical thickness | `(..., nlyr)` | required | `tau_arr` | `DELTAU_INPUT` |
| `ssa` | Single-scattering albedo | `(..., nlyr)` | required | `omega_arr` | `OMEGA_INPUT` |
| `g` | Asymmetry factor | `(..., nlyr)` | required | `asymm_arr` | `ASYMM_INPUT` |
| `z` | Level height grid, top to bottom | `(nlyr+1,)` | mode-dependent | `height_grid` | `HEIGHT_GRID` |
| `angles` | Solar `[sza, vza, raz]` or thermal view zenith angle(s), degrees | solar `(3,)` or `(ngeom, 3)`; thermal scalar or `(ngeom,)` | mode-dependent | `user_obsgeoms` / `user_angles` | `USER_OBSGEOMS` / `USER_ANGLES` |
| `stream` | Two-stream quadrature cosine | scalar | solar `1/sqrt(3)`, thermal `0.5` | `stream_value` | `STREAM_VALUE` |
| `fbeam` | Direct solar beam/source normalization | scalar or `(...)` | `1.0` | `flux_factor` | `FLUX_FACTOR` |
| `albedo` | Lambertian surface albedo | scalar or `(...)` | `0.0` | `albedo` | `LAMBERTIAN_ALBEDO` |
| `delta_m_scaling` | Delta-M truncation/scaling factor | `(..., nlyr)` | zeros | `d2s_scaling` | `D2S_SCALING` / `FO_TRUNCFAC` |
| `planck` | Thermal level Planck/source input | `(..., nlyr+1)` | thermal required | `thermal_bb_input` | `THERMAL_BB_INPUT` |
| `surface_planck` | Surface Planck radiance | scalar or `(...)` | `0.0` | `surfbb` | `SURFBB` |
| `emissivity` | Surface emissivity | scalar or `(...)` | `0.0` | `emissivity` | `EMISSIVITY` |
| `geometry` | FO solar geometry method | scalar | `pseudo_spherical` | `fo_geometry_mode` | `FO_SSGeometry_Master_Obs_EPS` / `FO_SSGeometry_Master_Obs_RPS` |

See [`docs/api_arguments.md`](docs/api_arguments.md) for the fuller argument
table and notes on advanced thermal and lattice inputs.

## Common Calls

Solar:

```python
solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
result = solver.forward(
    tau=np.full(3, 0.02),
    ssa=np.full(3, 0.2),
    g=np.full(3, 0.1),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 20.0, 0.0],
    albedo=0.3,
)
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

Batched wavelengths:

```python
solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
tau = np.full((100, 3), 0.02)
result = solver.forward(
    tau=tau,
    ssa=np.zeros_like(tau),
    g=np.zeros_like(tau),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 20.0, 0.0],
)
print(result.radiance.shape)  # (100,)
```

Torch CPU float64:

```python
solver = TwoStreamEss(
    TwoStreamEssOptions(nlyr=3, mode="solar", backend="torch", torch_dtype="float64")
)
```

Level profiles:

```python
solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True))
result = solver.forward(
    tau=np.full(3, 0.1),
    ssa=np.zeros(3),
    g=np.zeros(3),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=20.0,
    planck=np.ones(4),
    surface_planck=1.0,
    emissivity=1.0,
)
print(result.radiance_profile_2s.shape)  # (..., nlyr + 1), TOA to BOA
```

Unsupported in the current public API: combined solar+thermal source terms in
one call, gradients with respect to geometry (`angles` or `z`), and batched
thermal `forward_fo()` as an FO-only public path. MPS/float32 is useful for
development speed studies but CPU float64 is the validation path.

Run the bundled examples from the repository root after the editable install:

```bash
python3 examples/run_tir_reference_case.py
python3 examples/run_uv_reference_case.py
python3 examples/run_analytic_cases.py
python3 examples/build_thermal_source_from_temperature.py
python3 examples/retrieve_synthetic_spectra.py
```

The retrieval example defaults to a zero-noise/no-prior sanity check. For a
noisy, weakly regularized demonstration, run:

```bash
python3 examples/retrieve_synthetic_spectra.py \
  --prior-mode weak \
  --thermal-noise 0.003 \
  --solar-noise 0.002 \
  --uv-noise 0.002 \
  --plot-dir outputs/retrieval_plots
```

The retrieval example uses the reusable `py2sess.retrieval` helpers for a
Rodgers-style optimal-estimation residual with torch Jacobians and SciPy
least-squares. It prints Jacobian, Gauss-Newton Hessian,
posterior-covariance, averaging-kernel, and DFS diagnostics. In
zero-noise/no-prior mode, the well-posed thermal, solar, and UV benchmark
retrievals recover the generating truth.

The optional `--plot-dir` argument saves one spectrum PNG per retrieval, each
with pre-noise clean, post-noise observed, and fitted spectra on a log radiance
scale, plus a signed post-noise-minus-fitted residual panel. The post-noise
observed spectrum is the measurement used by the retrieval.

Full-spectrum benchmark examples are included for local or external `.npz`
bundles that follow the documented schema:

```bash
python3 examples/benchmark_tir_full_spectrum.py /path/to/tir_full_bundle.npz
python3 examples/benchmark_uv_full_spectrum.py /path/to/uv_full_bundle.npz
```

If the bundle includes saved Fortran radiance arrays such as `ref_total`, the
benchmark table also reports total-radiance accuracy alongside timing.

If you want to run directly from a source tree without installing, use:

```bash
PYTHONPATH=src python3 examples/run_tir_reference_case.py
PYTHONPATH=src python3 examples/run_uv_reference_case.py
PYTHONPATH=src python3 examples/run_analytic_cases.py
PYTHONPATH=src python3 examples/build_thermal_source_from_temperature.py
PYTHONPATH=src python3 examples/retrieve_synthetic_spectra.py
PYTHONPATH=src python3 examples/benchmark_tir_full_spectrum.py /path/to/tir_full_bundle.npz
PYTHONPATH=src python3 examples/benchmark_uv_full_spectrum.py /path/to/uv_full_bundle.npz
```

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

## Validation

The package includes three validation layers:

- analytic checks for closed-form solver identities
- saved UV and TIR benchmark fixtures containing Python-ready inputs and Fortran reference outputs
- NumPy versus PyTorch backend parity checks

The benchmark fixtures are static subsets of the larger author-supplied cases.
They are small enough for routine regression testing while still exercising the
same 114-layer solver structure used by the benchmark datasets.

Within the standalone test suite, the saved-file checks focus on the stable
quantities for each packaged case:

- UV fixture: saved 2S TOA radiance
- TIR fixture: saved total TOA radiance and FO TOA radiance

Details for the packaged reference fixtures are in [`docs/benchmark_cases.md`](docs/benchmark_cases.md).

## Acknowledgments

`py2sess` is a standalone Python implementation derived from the 2S-ESS
methodology. Its validation workflow includes comparisons against
author-provided benchmark outputs and saved-file reference cases.

## Repository Layout

- `src/py2sess`: core package
- `src/py2sess/data/benchmark`: packaged UV and TIR benchmark fixtures
- `tests`: standalone regression and analytic checks
- `examples`: runnable demos
- `docs/api_arguments.md`: public API argument mapping and Fortran-name crosswalk
- `docs/benchmark_cases.md`: packaged benchmark notes
- `docs/full_spectrum_benchmarks.md`: full-spectrum benchmark bundle notes
- `docs/retrieval_examples.md`: synthetic autograd retrieval examples
- `docs/pypi_release.md`: PyPI release checklist
- Generated benchmark reports, plots, and paper tables should stay in ignored
  local directories such as `outputs/`, `local_outputs/`, or `paper_outputs/`.

## Development Checks

Install the development extras, then run:

```bash
python3 -m pre_commit run --all-files
python3 -m ruff check .
python3 -m unittest discover -s tests -v
```

The pre-commit setup follows the lightweight repository hygiene pattern used in
`pyharp`, with an added `ruff` lint-and-format layer for the Python code.

## Notes

- CPU float64 is the parity-oriented reference path.
- PyTorch support is optional.
- The packaged benchmark cases are validation fixtures, not full-spectrum performance datasets.
- The large full-spectrum benchmark bundles are intended to remain local or be
  hosted separately rather than committed to the GitHub repository.
