"""Optical-property transformations shared by solver paths."""

from __future__ import annotations

import numpy as np


def delta_m_scale_optical_properties(
    tau: np.ndarray,
    omega: np.ndarray,
    asymm: np.ndarray,
    scaling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies 2S delta-M scaling to optical properties.

    Parameters
    ----------
    tau
        Layer optical thickness.
    omega
        Layer single-scattering albedo.
    asymm
        Layer asymmetry parameter.
    scaling
        Delta-M truncation factor for each layer.

    Returns
    -------
    tuple of ndarray
        Delta-scaled optical thickness, single-scattering albedo, and
        asymmetry parameter. Shapes follow NumPy broadcasting rules.
    """
    omfac = 1.0 - omega * scaling
    m1fac = 1.0 - scaling
    delta_tau = omfac * tau
    omega_total = np.clip((m1fac * omega) / omfac, 1.0e-9, 0.999999999)
    asymm_total = np.clip((asymm - scaling) / m1fac, -0.999999999, 0.999999999)
    asymm_total = np.where(
        (asymm_total >= 0.0) & (asymm_total < 1.0e-9),
        1.0e-9,
        asymm_total,
    )
    asymm_total = np.where(
        (asymm_total < 0.0) & (asymm_total > -1.0e-9),
        -1.0e-9,
        asymm_total,
    )
    return delta_tau, omega_total, asymm_total
