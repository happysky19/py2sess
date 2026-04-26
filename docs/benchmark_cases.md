# Benchmark Cases

The package includes two packaged benchmark fixtures:

- `tir_benchmark_fixture.npz`
- `uv_benchmark_fixture.npz`

These fixtures are static Python-ready subsets of larger 114-layer 2S-ESS
benchmark cases. They are intended for:

- regression testing
- example scripts
- quick solver validation after refactoring

They are not intended to replace the full-spectrum benchmark datasets used during
development.

## TIR fixture

- source mode: thermal observation geometry
- vertical structure: 114 layers
- spectral coverage: 32 sampled wavelengths from the larger benchmark case
- packaged fields:
  - optical properties
  - thermal blackbody inputs
  - surface terms
  - saved Fortran reference outputs for 2S, FO, and total TOA intensity

For this packaged case, the two-stream quadrature stream is `0.5`, matching the
full-spectrum TIR benchmark bundle from which the sampled rows were taken. The
standalone regression suite checks:

- saved-file parity for 2S
- saved-file parity for FO
- saved-file parity for total
- NumPy versus torch parity for the internal component split

## UV fixture

- source mode: solar observation geometry
- vertical structure: 114 layers
- spectral coverage: 16 sampled wavelengths from the larger benchmark case
- packaged fields:
  - optical properties
  - geometry inputs plus saved 2S geometry factors for parity checks
  - Rayleigh/aerosol phase-mixing inputs plus saved FO exact-scatter terms
  - saved Fortran reference outputs for 2S, FO, and total TOA intensity

## Why these fixtures exist

The original mixed development repository used external dumps, cached arrays, and direct Fortran comparisons. That is useful during porting, but it is too heavy for a clean standalone Python package.

The packaged fixtures keep the new repository self-contained:

- no Fortran build is required
- no external dump parsing is required
- the same small cases are available to tests and examples

For the packaged UV fixture, the standalone regression suite checks saved-file
parity for 2S, FO, and total radiance. The reference example builds the mixed
Rayleigh/aerosol `g`, delta-M truncation factor, and solar FO scatter term in
Python from the fixture phase inputs, then compares against the saved Fortran
outputs. The saved geometry and scatter arrays remain in the fixture as compact
truth data for helper-parity tests.

## Runtime dimensions

Both fixtures preserve the original 114-layer structure. The packaged spectral subsets are intentionally small so that routine regression runs stay fast.
