# Benchmark Cases

The package includes two packaged benchmark fixtures:

- `tir_benchmark_fixture.npz`
- `uv_benchmark_fixture.npz`

These fixtures are static Python-ready subsets of larger 114-layer author benchmark cases. They are intended for:

- regression testing
- example scripts
- quick solver validation after refactoring

They are not intended to replace the full-spectrum benchmark datasets used during development.

## TIR fixture

- source mode: thermal observation geometry
- vertical structure: 114 layers
- spectral coverage: 32 sampled wavelengths from the larger author case
- packaged fields:
  - optical properties
  - thermal blackbody inputs
  - surface terms
  - saved Fortran reference outputs for FO and total TOA intensity

For this packaged case, the total TOA radiance is the stable saved-file comparison target. The internal 2S/FO split can depend on the wrapper convention used around the thermal delta-M handling, so the standalone regression suite checks:

- saved-file parity for FO
- saved-file parity for total
- NumPy versus torch parity for the internal component split

## UV fixture

- source mode: solar observation geometry
- vertical structure: 114 layers
- spectral coverage: 16 sampled wavelengths from the larger author case
- packaged fields:
  - optical properties
  - precomputed 2S geometry factors
  - saved Fortran reference outputs for 2S, FO, and total TOA intensity

## Why these fixtures exist

The original mixed development repository used external dumps, cached arrays, and direct Fortran comparisons. That is useful during porting, but it is too heavy for a clean standalone Python package.

The packaged fixtures keep the new repository self-contained:

- no Fortran build is required
- no external dump parsing is required
- the same small cases are available to tests and examples

For the packaged UV fixture, the standalone regression suite uses the saved 2S radiance as the direct benchmark target. FO behavior is covered separately by the analytic tests bundled with the package.

## Runtime dimensions

Both fixtures preserve the original 114-layer structure. The packaged spectral subsets are intentionally small so that routine regression runs stay fast.
