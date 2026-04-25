"""Torch optical-property transformations shared by solver paths."""

from __future__ import annotations

from ..core.backend import _load_torch

torch = _load_torch()

_TRUNCATION_FACTOR_MAX = 1.0 - 1.0e-12


def default_delta_m_truncation_factor_torch(asymm):
    """Returns the differentiable HG-like default delta-M truncation factor."""
    return torch.clamp(asymm * asymm, min=0.0, max=_TRUNCATION_FACTOR_MAX)


def validate_delta_m_truncation_factor_torch(truncation_factor, omega) -> None:
    """Validates public delta-M truncation factors before solver use."""
    if not bool(torch.all(torch.isfinite(truncation_factor)).item()):
        raise ValueError("delta_m_truncation_factor must be finite")
    if not bool(torch.all((truncation_factor >= 0.0) & (truncation_factor < 1.0)).item()):
        raise ValueError("delta_m_truncation_factor must satisfy 0 <= f < 1")
    if not bool(torch.all(1.0 - omega * truncation_factor > 0.0).item()):
        raise ValueError("delta_m_truncation_factor gives non-positive 1 - ssa * f")


def delta_m_scale_optical_properties_torch(
    tau,
    omega,
    asymm,
    scaling,
):
    """Applies 2S delta-M scaling to torch optical-property tensors.

    Parameters
    ----------
    tau
        Layer optical thickness tensor.
    omega
        Layer single-scattering albedo tensor.
    asymm
        Layer asymmetry-parameter tensor.
    scaling
        Delta-M truncation-factor tensor.

    Returns
    -------
    tuple
        Delta-scaled optical thickness, single-scattering albedo, and
        asymmetry-parameter tensors.
    """
    omfac = 1.0 - omega * scaling
    m1fac = 1.0 - scaling
    delta_tau = omfac * tau
    omega_total = torch.clamp((m1fac * omega) / omfac, min=1.0e-9, max=0.999999999)
    asymm_total = torch.clamp((asymm - scaling) / m1fac, min=-0.999999999, max=0.999999999)
    asymm_total = torch.where(
        (asymm_total >= 0.0) & (asymm_total < 1.0e-9),
        torch.full_like(asymm_total, 1.0e-9),
        asymm_total,
    )
    asymm_total = torch.where(
        (asymm_total < 0.0) & (asymm_total > -1.0e-9),
        torch.full_like(asymm_total, -1.0e-9),
        asymm_total,
    )
    return delta_tau, omega_total, asymm_total
