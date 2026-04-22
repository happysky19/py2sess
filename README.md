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
        n_layers=3,
        source_mode="solar_obs",
        do_level_output=True,
    )
)

result = solver.forward_fo(
    tau_arr=np.zeros(3),
    omega_arr=np.zeros(3),
    asymm_arr=np.zeros(3),
    height_grid=np.array([3.0, 2.0, 1.0, 0.0]),
    user_obsgeoms=np.array([[30.0, 0.0, 0.0]]),
    stream_value=1.0 / np.sqrt(3.0),
    flux_factor=1.0,
    albedo=0.3,
    d2s_scaling=np.zeros(3),
    fo_geometry_mode="eps",
)

print(result.intensity_total)
```

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

The retrieval example uses a Rodgers-style optimal-estimation residual with
torch Jacobians and SciPy least-squares. It prints Jacobian, Gauss-Newton
Hessian, posterior-covariance, averaging-kernel, and DFS diagnostics. In
zero-noise/no-prior mode, the thermal, solar, and UV benchmark retrievals
recover the generating truth.

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
- `docs/benchmark_cases.md`: packaged benchmark notes
- `docs/full_spectrum_benchmarks.md`: full-spectrum benchmark bundle notes
- `docs/retrieval_examples.md`: synthetic autograd retrieval examples
- `docs/pypi_release.md`: PyPI release checklist

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
