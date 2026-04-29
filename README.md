# py2sess

`py2sess` is a Python implementation of the optimized 2S-ESS radiative-transfer
model. It supports solar and thermal forward calculations with NumPy and
optional torch backends. It does not call the original Fortran code.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[torch,dev]"
```

For source-tree runs without installation, set `PYTHONPATH=src`.

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

## Examples

```bash
python3 examples/run_uv_reference_case.py
python3 examples/run_tir_reference_case.py
python3 examples/build_thermal_source_from_temperature.py
python3 examples/retrieve_synthetic_spectra.py --case uv --noise-level 0
```

Full-spectrum benchmark details are in
[`docs/full_spectrum_benchmarks.md`](docs/full_spectrum_benchmarks.md).
Retrieval notes are in [`docs/retrieval.md`](docs/retrieval.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m ruff check .
python3 -m ruff format --check .
```

Keep large benchmark bundles and generated outputs out of git.

## References

- Fortran repository: [vnatraj1/2S-ESS](https://github.com/vnatraj1/2S-ESS)
- Paper: [Natraj et al., 2022, JQSRT](https://www.sciencedirect.com/science/article/pii/S002240732200351X)
