# thermal_jacobian_profile1

Compact thermal Fortran-Jacobian validation case.

```bash
PYTHONPATH=src python3 examples/compare_fortran_jacobian.py \
  --profile benchmarks/thermal_jacobian_profile1/profile.csv \
  --scene benchmarks/thermal_jacobian_profile1/scene.yaml \
  --reference benchmarks/thermal_jacobian_profile1/fortran_jacobian_reference.npz
```

The scene runs 1000 thermal wavelengths from profile and YAML inputs. The
Fortran reference file is sparse validation data: 100 saved wavelengths with
radiance and selected Jacobians. Total radiance, surface-emissivity Jacobian,
and normalized surface-temperature Jacobian are pass/fail checks. Profile and
component Jacobian columns are diagnostic because their state/convention is not
identical to py2sess public outputs. The reference file is not used to build
runtime optical inputs.

The fixture builder intentionally requires total, 2S, and FO columns. Fortran
outputs that contain only FO columns, or zero surface-temperature Jacobians from
an incomplete local driver, should not be used as references.
