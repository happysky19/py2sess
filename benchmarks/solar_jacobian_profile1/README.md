# solar_jacobian_profile1

Solar FO+2S Jacobian validation case from the production UVVSWIR Fortran exact
driver.

The pass/fail target is the production-driver finite-difference derivative of TOA
2S, FO, and total radiance with respect to Lambertian albedo.
`jacobian_reference` in `scene.yaml` points to the compact Fortran validation
data.

```bash
PYTHONPATH=src python3 examples/compare_fortran_jacobian.py \
  --case solar \
  --scene benchmarks/solar_jacobian_profile1/scene.yaml \
  --plot outputs/gradient_validation/solar_jacobian_profile1.png
```
