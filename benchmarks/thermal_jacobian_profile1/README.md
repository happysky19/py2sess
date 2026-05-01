# thermal_jacobian_profile1

Thermal FO+2S Jacobian validation case.

```bash
PYTHONPATH=src python3 examples/compare_fortran_jacobian.py \
  --profile benchmarks/thermal_jacobian_profile1/profile.csv \
  --scene benchmarks/thermal_jacobian_profile1/scene.yaml \
  --plot outputs/gradient_validation/thermal_jacobian_profile1.png
```

The scene runs 1000 thermal wavelengths from profile and YAML inputs.
`jacobian_reference` in `scene.yaml` points to sparse validation data: saved
Fortran TOA radiance, surface-emissivity Jacobian, and normalized
surface-temperature Jacobian at 100 wavelengths.
