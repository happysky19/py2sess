# solar_jacobian_profile1

Solar 2S Jacobian validation case from the production UVVSWIR Fortran exact driver.

The pass/fail target is the production-driver finite-difference derivative of TOA
2S radiance with respect to Lambertian albedo. The Fortran solar LPS surface
weighting-function output is kept in the reference file as diagnostic data because
it does not match the production-driver finite difference for this case.

Run from the repository root:

```bash
PYTHONPATH=src python3 examples/compare_fortran_jacobian.py \
  --case solar \
  --scene benchmarks/solar_jacobian_profile1/scene.yaml
```
