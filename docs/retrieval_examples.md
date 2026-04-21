# Synthetic Retrieval Examples

`examples/retrieve_synthetic_spectra.py` demonstrates how to use `py2sess` in
small differentiable retrieval workflows.

The script creates synthetic spectra with `py2sess` and retrieves small state
vectors with `torch.optim.Adam` followed by `torch.optim.LBFGS`.

## Workflows

The script currently includes three workflows:

- Thermal FO + 2S tau/surface-temperature sanity retrieval:
  retrieves optical-depth scale and surface temperature while holding the
  atmospheric temperature profile fixed. With zero noise and no priors, this
  should recover the synthetic truth to numerical precision.
- Thermal FO + 2S full-state retrieval:
  retrieves a smooth temperature profile, surface temperature, and scalar
  optical-depth scale from synthetic TIR spectra at two viewing angles. This
  is intentionally harder because temperature, optical depth, and surface
  emission can compensate for one another.
- Solar 2S retrieval:
  retrieves surface albedo and scalar optical-depth scale from a synthetic
  solar-observation spectrum.

The solar example uses the differentiable 2S torch kernel only. The package
currently includes a batched NumPy FO solar helper for benchmark parity, but not
a torch-native FO solar helper.

## Run

From the repository root:

```bash
python3 examples/retrieve_synthetic_spectra.py
```

From a source checkout without installation:

```bash
PYTHONPATH=src python3 examples/retrieve_synthetic_spectra.py
```

The default run is a zero-noise/no-prior sanity test.

To run a noisy, weakly regularized demonstration:

```bash
PYTHONPATH=src python3 examples/retrieve_synthetic_spectra.py \
  --prior-mode weak \
  --thermal-noise 0.003 \
  --solar-noise 0.002
```

## Interpretation

These are workflow examples, not full retrieval products. The zero-noise
sanity cases are meant to check differentiability and optimization plumbing.
The full thermal state-vector example is intentionally more ill-conditioned:
optical-depth scale, atmospheric temperature, and surface temperature can
produce very similar radiances. Real retrievals should use physically motivated
state-vector parameterizations, priors, constraints, multiple spectral regions
or geometries when available, and an explicit noise model.

The weak prior mode is opt-in because a prior changes the objective. In a
zero-noise synthetic truth-recovery test, a nonzero prior can deliberately move
the optimum away from the exact generating state.

The thermal workflow builds `thermal_bb_input` and `surfbb` with
`thermal_source_from_temperature_profile_torch`, so gradients propagate from
the radiance loss back to the retrieved level and surface temperatures.
