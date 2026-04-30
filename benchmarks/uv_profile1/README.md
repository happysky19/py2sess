# uv_profile1

Solar public py2sess benchmark scene. Run from the repository root:

```bash
PYTHONPATH=src python3 examples/benchmark_scene_full_spectrum.py \
  --profile benchmarks/uv_profile1/profile.csv \
  --scene benchmarks/uv_profile1/scene.yaml \
  --backend numpy \
  --require-python-generated-inputs
```

`profile.csv` contains pressure, temperature, height, gas VMR columns, and dimensionless aerosol loading columns. `aerosol_properties.nc` stores reusable aerosol optical properties with units on the NetCDF variables. `reference_outputs.npz` includes `wavelength_nm`, `ref_2s`, `ref_fo`, and `ref_total`.
