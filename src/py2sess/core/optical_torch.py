"""Torch optical-property transformations shared by solver paths."""

from __future__ import annotations

from .backend import _load_torch

torch = _load_torch()


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
