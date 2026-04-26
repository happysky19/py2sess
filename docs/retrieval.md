# Rodgers-Style Retrieval

`py2sess.retrieval` provides a small optimal-estimation layer for synthetic
inverse tests. It uses torch autograd to evaluate the Jacobian of a user-provided
forward model, then applies the Rodgers-style Gauss-Newton update with explicit
measurement and prior covariance matrices.

The cost function is

```text
J(x) = (y - F(x))^T Se^-1 (y - F(x)) + (x - xa)^T Sa^-1 (x - xa)
```

The diagnostics are the posterior covariance, averaging kernel, degrees of
freedom for signal, singular values of `Se^-1/2 K Sa^1/2`, and Hessian
condition number.

`evaluate_jacobian()` currently builds a dense torch autograd Jacobian. It is
intended for small synthetic retrievals and information-content checks; larger
production retrievals will need a more specialized Jacobian path.

Current intended cases:

- UV/solar synthetic retrieval of optical-depth scale and albedo.
- Thermal sanity retrieval of optical-depth scale and surface temperature.
- Full thermal profile retrieval only as an information-content diagnostic.

Known gaps:

- Raw optical-property generation is not complete. Python helpers can combine
  component optical depths and phase-mixing inputs, but gas spectroscopy,
  aerosol microphysics, and endpoint aerosol moments still need an upstream
  optical-property provider.
- Thermal scattering gradients are correct but can be slow because the torch
  path uses a dense BVP fallback when `ssa` or `g` require gradients.
- No instrument model is included yet: no spectral response convolution,
  correlated noise covariance, calibration offset, or real-data adapter.
