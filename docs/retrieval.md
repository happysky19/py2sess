# Rodgers-Style Retrieval

`py2sess.retrieval` provides a small Rodgers-style optimal-estimation layer for
synthetic inverse tests. It uses torch autograd for Jacobians and explicit
measurement/prior covariance matrices.

The cost function is

```text
J(x) = (y - F(x))^T Se^-1 (y - F(x)) + (x - xa)^T Sa^-1 (x - xa)
```

Diagnostics include posterior covariance, averaging kernel, degrees of freedom
for signal, singular values of `Se^-1/2 K Sa^1/2`, and Hessian condition number.
`evaluate_jacobian()` builds a dense torch autograd Jacobian, so it is intended
for small synthetic retrievals and information-content checks.

Intended examples are UV optical-depth scale plus albedo, thermal optical-depth
scale plus surface temperature, and low-dimensional thermal profile diagnostics.

Known gaps:

- Thermal scattering gradients are correct but can be slow because the torch
  path uses a dense BVP fallback when `ssa` or `g` require gradients.
- No instrument model is included yet.
