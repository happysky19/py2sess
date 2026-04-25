"""Optical-property transformations shared by solver paths."""

from __future__ import annotations

import numpy as np

_TRUNCATION_FACTOR_MAX = 1.0 - 1.0e-12


def default_delta_m_truncation_factor(asymm: np.ndarray) -> np.ndarray:
    """Returns the HG-like default delta-M truncation factor."""
    asymm_arr = np.asarray(asymm, dtype=float)
    return np.clip(asymm_arr * asymm_arr, 0.0, _TRUNCATION_FACTOR_MAX)


def validate_delta_m_truncation_factor(
    truncation_factor: np.ndarray,
    omega: np.ndarray,
) -> None:
    """Validates public delta-M truncation factors before solver use."""
    factor = np.asarray(truncation_factor, dtype=float)
    ssa = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(factor)):
        raise ValueError("delta_m_truncation_factor must be finite")
    if np.any((factor < 0.0) | (factor >= 1.0)):
        raise ValueError("delta_m_truncation_factor must satisfy 0 <= f < 1")
    if np.any(1.0 - ssa * factor <= 0.0):
        raise ValueError("delta_m_truncation_factor gives non-positive 1 - ssa * f")


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
