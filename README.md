# py2sess

`py2sess` is a standalone Python implementation of the optimized 2S-ESS
two-stream radiative-transfer model. It exposes a small public API for solar and
thermal forward calculations, with NumPy and optional torch backends.

The package does not build or call the original Fortran code. Fortran-style
names are kept inside the solver core; user code should use the public names
shown below.

## Install

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[torch,dev]"
```

For source-tree runs without installation, set `PYTHONPATH=src`.

## Quick Start

Solar observation geometry:

```python
import numpy as np
from py2sess import TwoStreamEss, TwoStreamEssOptions

solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))

result = solver.forward(
    tau=np.full(3, 0.02),
    ssa=np.full(3, 0.2),
    g=np.full(3, 0.1),
    z=np.array([3.0, 2.0, 1.0, 0.0]),
    angles=[30.0, 20.0, 0.0],  # sza, vza, relative azimuth in degrees
    albedo=0.3,
)

print(result.radiance)
```

Thermal observation geometry:

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

## Public Names

Core inputs:

| Name | Meaning |
|---|---|
| `nlyr` | number of atmospheric layers |
| `mode` | `solar`, `solar_lattice`, or `thermal` |
| `tau` | layer optical thickness |
| `ssa` | single-scattering albedo |
| `g` | layer asymmetry factor |
| `z` | level height grid, top to bottom |
| `angles` | solar `[sza, vza, raz]` or thermal view zenith angle |
| `albedo` | Lambertian surface albedo |
| `planck` | thermal level Planck/source input |
| `surface_planck` | surface Planck radiance |
| `emissivity` | surface emissivity |

Optional controls include `stream`, `fbeam`, `delta_m_truncation_factor`, `geometry`,
`include_fo`, `fo_scatter_term`, and `output_levels`. When
`delta_m_truncation_factor` is omitted, py2sess uses the Henyey-Greenstein
fallback `g**2`. Pass an explicit factor for mixed phase functions or when
reusing full-spectrum optical-property fixtures.

For mixed Rayleigh/aerosol phase inputs, `py2sess.optical.phase` can generate
`g`, `delta_m_truncation_factor`, and solar `fo_scatter_term` from
Rayleigh/aerosol fractions and endpoint aerosol moments.

Result names:

- `result.radiance`: preferred total radiance
- `result.radiance_2s`: two-stream component
- `result.radiance_fo`: first-order component when available
- `result.radiance_total`: explicit total radiance
- `result.radiance_profile_*`: profile outputs when `output_levels=True`

For the complete argument table and Fortran-name crosswalk, see
[`docs/api_arguments.md`](docs/api_arguments.md).

## Examples

```bash
python3 examples/run_uv_reference_case.py
python3 examples/run_tir_reference_case.py
python3 examples/build_thermal_source_from_temperature.py
python3 examples/retrieve_synthetic_spectra.py --case uv --noise-level 0
```

Full-spectrum benchmark scripts accept local `.npz` bundles:

```bash
python3 examples/benchmark_uv_full_spectrum.py /path/to/uv_full_bundle.npz
python3 examples/benchmark_tir_full_spectrum.py /path/to/tir_full_bundle.npz
```

Bundle details are in
[`docs/full_spectrum_benchmarks.md`](docs/full_spectrum_benchmarks.md).
Rodgers-style synthetic retrieval notes are in [`docs/retrieval.md`](docs/retrieval.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m ruff check .
python3 -m ruff format --check .
```

The packaged UV and TIR fixtures are small regression cases derived from the
original 2S-ESS benchmark outputs. Large full-spectrum bundles and generated
reports should stay outside git, for example under ignored local directories
such as `benchmark_bundles/`, `outputs/`, `local_outputs/`, or
`paper_outputs/`.

## References

- Fortran repository: [vnatraj1/2S-ESS](https://github.com/vnatraj1/2S-ESS)
- Paper: [Natraj et al., 2022, JQSRT](https://www.sciencedirect.com/science/article/pii/S002240732200351X)
