"""Optical-property transformations shared by solver paths."""

from __future__ import annotations

import numpy as np

_TRUNCATION_FACTOR_MAX = 1.0 - 1.0e-12
_NUMBA_DELTA_M_MIN_SIZE = 262_144
_DELTA_M_KERNEL = None
_DELTA_M_IMPORT_FAILED = False


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


def _get_delta_m_kernel():
    global _DELTA_M_KERNEL, _DELTA_M_IMPORT_FAILED
    if _DELTA_M_KERNEL is not None:
        return _DELTA_M_KERNEL
    if _DELTA_M_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional acceleration dependency
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(tau, omega, asymm, scaling):
            nrows, nlayers = tau.shape
            delta_tau = np.empty((nrows, nlayers), np.float64)
            omega_total = np.empty((nrows, nlayers), np.float64)
            asymm_total = np.empty((nrows, nlayers), np.float64)
            for row in prange(nrows):
                for layer in range(nlayers):
                    omfac = 1.0 - omega[row, layer] * scaling[row, layer]
                    m1fac = 1.0 - scaling[row, layer]
                    delta_tau[row, layer] = tau[row, layer] * omfac

                    omega_value = m1fac * omega[row, layer] / omfac
                    if omega_value < 1.0e-9:
                        omega_value = 1.0e-9
                    elif omega_value > 0.999999999:
                        omega_value = 0.999999999

                    asymm_value = (asymm[row, layer] - scaling[row, layer]) / m1fac
                    if asymm_value < -0.999999999:
                        asymm_value = -0.999999999
                    elif asymm_value > 0.999999999:
                        asymm_value = 0.999999999
                    if asymm_value >= 0.0 and asymm_value < 1.0e-9:
                        asymm_value = 1.0e-9
                    elif asymm_value < 0.0 and asymm_value > -1.0e-9:
                        asymm_value = -1.0e-9

                    omega_total[row, layer] = omega_value
                    asymm_total[row, layer] = asymm_value
            return delta_tau, omega_total, asymm_total

        _DELTA_M_KERNEL = _kernel
        return _DELTA_M_KERNEL
    except Exception:  # pragma: no cover - optional acceleration dependency
        _DELTA_M_IMPORT_FAILED = True
        return None


def _delta_m_scale_optical_properties_fast(
    tau: np.ndarray,
    omega: np.ndarray,
    asymm: np.ndarray,
    scaling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    tau_arr = np.asarray(tau)
    omega_arr = np.asarray(omega)
    asymm_arr = np.asarray(asymm)
    scaling_arr = np.asarray(scaling)
    if (
        tau_arr.ndim != 2
        or omega_arr.shape != tau_arr.shape
        or asymm_arr.shape != tau_arr.shape
        or scaling_arr.shape != tau_arr.shape
        or tau_arr.size < _NUMBA_DELTA_M_MIN_SIZE
    ):
        return None
    kernel = _get_delta_m_kernel()
    if kernel is None:
        return None
    return kernel(
        np.ascontiguousarray(tau_arr, dtype=np.float64),
        np.ascontiguousarray(omega_arr, dtype=np.float64),
        np.ascontiguousarray(asymm_arr, dtype=np.float64),
        np.ascontiguousarray(scaling_arr, dtype=np.float64),
    )


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
    accelerated = _delta_m_scale_optical_properties_fast(tau, omega, asymm, scaling)
    if accelerated is not None:
        return accelerated

    omfac = 1.0 - omega * scaling
    m1fac = 1.0 - scaling
    delta_tau = omfac * tau
    omega_total = np.asarray((m1fac * omega) / omfac, dtype=float)
    np.clip(omega_total, 1.0e-9, 0.999999999, out=omega_total)
    asymm_total = np.asarray((asymm - scaling) / m1fac, dtype=float)
    np.clip(asymm_total, -0.999999999, 0.999999999, out=asymm_total)
    np.putmask(asymm_total, (asymm_total >= 0.0) & (asymm_total < 1.0e-9), 1.0e-9)
    np.putmask(asymm_total, (asymm_total < 0.0) & (asymm_total > -1.0e-9), -1.0e-9)
    return delta_tau, omega_total, asymm_total
