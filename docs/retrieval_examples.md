# Synthetic Retrieval Examples

`examples/retrieve_synthetic_spectra.py` demonstrates how to use `py2sess` in
small differentiable retrieval workflows.

The script creates spectra with `py2sess` and retrieves small state vectors
with a Rodgers-style optimal-estimation residual:

```text
r = [S_e^-1/2 (F(x) - y), S_a^-1/2 (x - x_a)]
```

The nonlinear solve uses `scipy.optimize.least_squares`. The forward-model
Jacobian is computed from the torch model, so the example also reports Jacobian
singular values, Gauss-Newton Hessian conditioning, posterior covariance,
averaging-kernel diagnostics, and degrees of freedom for signal.

## Workflows

The script currently includes four workflows:

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
- UV benchmark 2S retrieval:
  retrieves scalar albedo and optical-depth multipliers from the packaged UV
  benchmark geometry.

The solar and UV retrieval examples use the differentiable 2S torch kernel
only. The package currently includes a batched NumPy FO solar helper for
benchmark parity, but not a torch-native FO solar helper.

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
  --solar-noise 0.002 \
  --uv-noise 0.002 \
  --plot-dir outputs/retrieval_plots
```

The optional `--plot-dir` argument saves one PNG per retrieval. Each figure has
two spectrum subplots. The first plots the pre-noise clean synthetic spectrum,
the post-noise observed spectrum used by the retrieval, and the fitted spectrum
at the retrieved state on a log radiance scale. The second plots the signed
post-noise-minus-fitted spectral residual. For thermal retrievals, both
viewing-angle spectra are plotted.

Plotting requires matplotlib, available through the package's `plot` extra:

```bash
python3 -m pip install -e ".[plot]"
```

## Interpretation

These are workflow examples, not full retrieval products. The zero-noise
sanity cases are meant to check differentiability, Jacobian construction, and
least-squares retrieval plumbing.
The full thermal state-vector example is intentionally more ill-conditioned:
optical-depth scale, atmospheric temperature, and surface temperature can
produce very similar radiances. Real retrievals should use physically motivated
state-vector parameterizations, priors, constraints, multiple spectral regions
or geometries when available, and an explicit noise model.

The weak prior mode is opt-in because a prior changes the objective by adding
an explicit background state and covariance. In a zero-noise synthetic
truth-recovery test, a nonzero prior can deliberately move the optimum away
from the exact generating state.

The thermal workflow builds `thermal_bb_input` and `surfbb` with
`thermal_source_from_temperature_profile_torch`, so gradients propagate from
the radiance loss back to the retrieved level and surface temperatures.

The reported optimal-estimation matrices are:

```text
K_w = S_e^-1/2 dF/dx
H_GN = K_w^T K_w + S_a^-1
S_hat = H_GN^-1
A = S_hat K_w^T K_w
DFS = trace(A)
```

When `prior-mode=off`, `S_a^-1` is zero and the retrieval reduces to
Jacobian-based nonlinear least squares.
