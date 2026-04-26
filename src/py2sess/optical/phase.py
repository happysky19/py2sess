"""Phase-function preprocessing used by the benchmark optical inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .delta_m import validate_delta_m_truncation_factor

_FRACTION_TOL = 1.0e-10
_FRACTION_SUM_TOL = 1.0e-5


@dataclass(frozen=True)
class TwoStreamPhaseInputs:
    """Two-stream phase inputs derived from Rayleigh/aerosol fractions."""

    g: np.ndarray
    delta_m_truncation_factor: np.ndarray


def ssa_from_optical_depth(total_tau, scattering_tau) -> np.ndarray:
    """Returns single-scattering albedo from total and scattering optical depth."""
    total = np.asarray(total_tau, dtype=float)
    scattering = np.asarray(scattering_tau, dtype=float)
    total_b, scattering_b = np.broadcast_arrays(total, scattering)
    return np.divide(
        scattering_b,
        total_b,
        out=np.zeros_like(total_b, dtype=float),
        where=total_b > 0.0,
    )


def aerosol_interp_fraction(wavelengths, *, reverse: bool = False) -> np.ndarray:
    """Returns the Fortran endpoint interpolation fraction for aerosol moments."""
    grid = np.asarray(wavelengths, dtype=float)
    if grid.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if grid.size == 0:
        raise ValueError("wavelengths must not be empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("wavelengths must be finite")
    span = grid[-1] - grid[0]
    if span == 0.0:
        return np.zeros_like(grid, dtype=float)
    values = grid[::-1] if reverse else grid
    return (values - grid[0]) / span


def _broadcast_leading(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to spectral shape {shape}") from exc


def _aerosol_moments_array(aerosol_moments) -> np.ndarray:
    moments = np.asarray(aerosol_moments, dtype=float)
    if moments.ndim != 3 or moments.shape[0] != 2 or moments.shape[1] < 3:
        raise ValueError("aerosol_moments must have shape (2, nmom + 1, naerosol)")
    if not np.all(np.isfinite(moments)):
        raise ValueError("aerosol_moments must be finite")
    return moments


def _validate_phase_fractions(
    *,
    ssa: np.ndarray,
    rayleigh_fraction: np.ndarray,
    aerosol_fraction: np.ndarray,
) -> None:
    if np.any((ssa < -_FRACTION_TOL) | (ssa > 1.0 + _FRACTION_TOL)):
        raise ValueError("ssa must satisfy 0 <= ssa <= 1")
    if np.any(rayleigh_fraction < -_FRACTION_TOL) or np.any(aerosol_fraction < -_FRACTION_TOL):
        raise ValueError("rayleigh_fraction and aerosol_fraction must be nonnegative")
    fraction_sum = rayleigh_fraction + np.sum(aerosol_fraction, axis=-1)
    if np.any(fraction_sum > 1.0 + _FRACTION_SUM_TOL):
        raise ValueError("rayleigh_fraction and aerosol_fraction must not sum above 1")


def _aerosol_phase_endpoints(moments: np.ndarray, cos_scatter: np.ndarray) -> np.ndarray:
    endpoint_phase = np.zeros((2, moments.shape[2], cos_scatter.size), dtype=float)
    p_lm2 = np.ones_like(cos_scatter, dtype=float)
    endpoint_phase += moments[:, 0, :, None] * p_lm2

    p_lm1 = cos_scatter
    endpoint_phase += moments[:, 1, :, None] * p_lm1
    for ell in range(2, moments.shape[1]):
        p_l = ((2 * ell - 1) * cos_scatter * p_lm1 - (ell - 1) * p_lm2) / ell
        endpoint_phase += moments[:, ell, :, None] * p_l
        p_lm2, p_lm1 = p_lm1, p_l
    return endpoint_phase


def _aerosol_moment(
    *,
    aerosol_fraction: np.ndarray,
    aerosol_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    moment_index: int,
) -> np.ndarray:
    moment = aerosol_moments[0, moment_index] + aerosol_interp_fraction[..., None] * (
        aerosol_moments[1, moment_index] - aerosol_moments[0, moment_index]
    )
    return np.einsum("...la,...a->...l", aerosol_fraction, moment, optimize=True)


def build_two_stream_phase_inputs(
    *,
    ssa,
    depol,
    rayleigh_fraction,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
) -> TwoStreamPhaseInputs:
    """Builds Fortran-style two-stream ``g`` and delta-M truncation factor."""
    ssa_arr = np.asarray(ssa, dtype=float)
    ray_arr = np.asarray(rayleigh_fraction, dtype=float)
    ssa_b, ray_b = np.broadcast_arrays(ssa_arr, ray_arr)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if not np.all(np.isfinite(ssa_b)) or not np.all(np.isfinite(ray_b)):
        raise ValueError("ssa and rayleigh_fraction must be finite")
    lead_shape = ssa_b.shape[:-1]
    depol_b = _broadcast_leading("depol", depol, lead_shape)
    fac_b = _broadcast_leading("aerosol_interp_fraction", aerosol_interp_fraction, lead_shape)

    moments = _aerosol_moments_array(aerosol_moments)
    aer_frac = np.asarray(aerosol_fraction, dtype=float)
    if not np.all(np.isfinite(aer_frac)):
        raise ValueError("aerosol_fraction must be finite")
    if moments.shape[2] != aer_frac.shape[-1]:
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = np.broadcast_to(aer_frac, ssa_b.shape + (moments.shape[2],))
    _validate_phase_fractions(
        ssa=ssa_b,
        rayleigh_fraction=ray_b,
        aerosol_fraction=aer_frac_b,
    )

    ray2mom = (1.0 - depol_b) / (2.0 + depol_b)
    moment1 = _aerosol_moment(
        aerosol_fraction=aer_frac_b,
        aerosol_moments=moments,
        aerosol_interp_fraction=fac_b,
        moment_index=1,
    )
    moment2 = ray_b * ray2mom[..., None] + _aerosol_moment(
        aerosol_fraction=aer_frac_b,
        aerosol_moments=moments,
        aerosol_interp_fraction=fac_b,
        moment_index=2,
    )
    g = moment1 / 3.0
    truncation_factor = moment2 / 5.0
    validate_delta_m_truncation_factor(truncation_factor, ssa_b)
    return TwoStreamPhaseInputs(g=g, delta_m_truncation_factor=truncation_factor)


def _normalize_angles(angles) -> np.ndarray:
    arr = np.asarray(angles, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("angles must be finite")
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError("angles must have shape (3,) or (ngeom, 3)")


def _solar_obs_scattering_cosines(angles) -> np.ndarray:
    geoms = _normalize_angles(angles)
    sza = np.deg2rad(geoms[:, 0])
    vza = np.deg2rad(geoms[:, 1])
    raz = np.deg2rad(geoms[:, 2])
    mu1 = np.cos(vza)
    cosscat = -(np.cos(vza) * np.cos(sza)) + np.sin(vza) * np.sin(sza) * np.cos(raz)
    overhead = np.isclose(geoms[:, 0], 0.0)
    if np.any(overhead):
        cosscat = cosscat.copy()
        cosscat[overhead] = np.where(np.isclose(mu1[overhead], 0.0), 0.0, -mu1[overhead])
    return cosscat


def build_solar_fo_scatter_term(
    *,
    ssa,
    depol,
    rayleigh_fraction,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
    angles,
    delta_m_truncation_factor,
) -> np.ndarray:
    """Builds the Fortran-style solar FO exact-scatter source term."""
    ssa_arr = np.asarray(ssa, dtype=float)
    ray_arr = np.asarray(rayleigh_fraction, dtype=float)
    factor_arr = np.asarray(delta_m_truncation_factor, dtype=float)
    ssa_b, ray_b, factor_b = np.broadcast_arrays(ssa_arr, ray_arr, factor_arr)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if not np.all(np.isfinite(ssa_b)) or not np.all(np.isfinite(ray_b)):
        raise ValueError("ssa and rayleigh_fraction must be finite")
    validate_delta_m_truncation_factor(factor_b, ssa_b)

    lead_shape = ssa_b.shape[:-1]
    depol_b = _broadcast_leading("depol", depol, lead_shape)
    fac_b = _broadcast_leading("aerosol_interp_fraction", aerosol_interp_fraction, lead_shape)
    cos_scatter = _solar_obs_scattering_cosines(angles)

    delta = 2.0 * (1.0 - depol_b) / (2.0 + depol_b)
    raypf = delta[..., None] * 0.75 * (1.0 + cos_scatter * cos_scatter) + 1.0 - delta[..., None]
    phase_total = ray_b[..., :, None] * raypf[..., None, :]

    moments = _aerosol_moments_array(aerosol_moments)
    aer_frac = np.asarray(aerosol_fraction, dtype=float)
    if not np.all(np.isfinite(aer_frac)):
        raise ValueError("aerosol_fraction must be finite")
    if moments.shape[2] != aer_frac.shape[-1]:
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = np.broadcast_to(aer_frac, ssa_b.shape + (moments.shape[2],))
    _validate_phase_fractions(
        ssa=ssa_b,
        rayleigh_fraction=ray_b,
        aerosol_fraction=aer_frac_b,
    )
    aerosol_phase_endpoints = _aerosol_phase_endpoints(moments, cos_scatter)
    aerosol_phase = aerosol_phase_endpoints[0] + fac_b[..., None, None] * (
        aerosol_phase_endpoints[1] - aerosol_phase_endpoints[0]
    )
    phase_total = phase_total + np.einsum(
        "...la,...ag->...lg",
        aer_frac_b,
        aerosol_phase,
        optimize=True,
    )

    scatter = phase_total * ssa_b[..., None] / (1.0 - factor_b[..., None] * ssa_b[..., None])
    if cos_scatter.size == 1:
        return scatter[..., 0]
    return scatter
