"""Phase-function preprocessing used by the benchmark optical inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .delta_m import validate_delta_m_truncation_factor

_FRACTION_TOL = 1.0e-10
_FRACTION_SUM_TOL = 1.0e-5
_NUMBA_PHASE_MIN_ROWS = 4096
_SOLAR_SCATTER_KERNEL = None
_PHASE_NUMBA_IMPORT_FAILED = False


@dataclass(frozen=True)
class TwoStreamPhaseInputs:
    """Two-stream phase inputs derived from Rayleigh/aerosol fractions."""

    g: np.ndarray
    delta_m_truncation_factor: np.ndarray


@dataclass(frozen=True)
class SolarPhaseInputs:
    """Solar phase inputs derived from Rayleigh/aerosol optical-depth components."""

    g: np.ndarray
    delta_m_truncation_factor: np.ndarray
    fo_scatter_term: np.ndarray


def ssa_from_optical_depth(total_tau, scattering_tau) -> np.ndarray:
    """Returns single-scattering albedo from total and scattering optical depth."""
    total = np.asarray(total_tau, dtype=float)
    scattering = np.asarray(scattering_tau, dtype=float)
    total_b, scattering_b = np.broadcast_arrays(total, scattering)
    positive = total_b > 0.0
    out = np.empty_like(total_b, dtype=float)
    np.divide(scattering_b, total_b, out=out, where=positive)
    if not np.all(positive):
        out[~positive] = 0.0
    return out


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
    expected = np.where(ssa > _FRACTION_TOL, 1.0, 0.0)
    if np.any(np.abs(fraction_sum - expected) > _FRACTION_SUM_TOL):
        raise ValueError(
            "rayleigh_fraction and aerosol_fraction must sum to 1 where ssa > 0 "
            "and 0 where ssa == 0"
        )


def _aerosol_phase_endpoints(moments: np.ndarray, cos_scatter: np.ndarray) -> np.ndarray:
    basis = np.polynomial.legendre.legvander(cos_scatter, moments.shape[1] - 1)
    return np.matmul(np.moveaxis(moments, 1, 2), basis.T)


def _aerosol_moment_weights(
    aerosol_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    moment_indices: tuple[int, ...],
) -> np.ndarray:
    lower = aerosol_moments[0, moment_indices, :].T
    span = (aerosol_moments[1, moment_indices, :] - aerosol_moments[0, moment_indices, :]).T
    return lower + aerosol_interp_fraction[..., None, None] * span


def _get_solar_scatter_kernel():
    global _SOLAR_SCATTER_KERNEL, _PHASE_NUMBA_IMPORT_FAILED
    if _SOLAR_SCATTER_KERNEL is not None:
        return _SOLAR_SCATTER_KERNEL
    if _PHASE_NUMBA_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional acceleration dependency
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(
            ssa,
            rayleigh_tau,
            aerosol_tau,
            depol,
            moments,
            fac,
            inv_scattering,
            aerosol_phase_endpoints,
            cos_scatter,
        ):
            nrows, nlayers = rayleigh_tau.shape
            naerosol = aerosol_tau.shape[2]
            g = np.empty((nrows, nlayers), np.float64)
            truncation = np.empty((nrows, nlayers), np.float64)
            scatter = np.empty((nrows, nlayers), np.float64)
            cos2 = cos_scatter * cos_scatter
            for row in prange(nrows):
                ray2mom = (1.0 - depol[row]) / (2.0 + depol[row])
                delta = 2.0 * ray2mom
                raypf = delta * 0.75 * (1.0 + cos2) + 1.0 - delta
                interp = fac[row]
                for layer in range(nlayers):
                    aerosol_moment1 = 0.0
                    aerosol_moment2 = 0.0
                    aerosol_phase = 0.0
                    for aer in range(naerosol):
                        scat = aerosol_tau[row, layer, aer]
                        m1 = moments[0, 1, aer] + interp * (moments[1, 1, aer] - moments[0, 1, aer])
                        m2 = moments[0, 2, aer] + interp * (moments[1, 2, aer] - moments[0, 2, aer])
                        phase = aerosol_phase_endpoints[0, aer, 0] + interp * (
                            aerosol_phase_endpoints[1, aer, 0] - aerosol_phase_endpoints[0, aer, 0]
                        )
                        aerosol_moment1 += scat * m1
                        aerosol_moment2 += scat * m2
                        aerosol_phase += scat * phase
                    inv = inv_scattering[row, layer]
                    layer_ssa = ssa[row, layer]
                    factor = (rayleigh_tau[row, layer] * ray2mom + aerosol_moment2) * inv * 0.2
                    g[row, layer] = aerosol_moment1 * inv / 3.0
                    truncation[row, layer] = factor
                    scatter[row, layer] = (
                        (rayleigh_tau[row, layer] * raypf + aerosol_phase)
                        * inv
                        * layer_ssa
                        / (1.0 - factor * layer_ssa)
                    )
            return g, truncation, scatter

        _SOLAR_SCATTER_KERNEL = _kernel
        return _SOLAR_SCATTER_KERNEL
    except Exception:  # pragma: no cover - optional acceleration dependency
        _PHASE_NUMBA_IMPORT_FAILED = True
        return None


def _solar_from_scattering_fast(
    *,
    ssa: np.ndarray,
    rayleigh_scattering_tau: np.ndarray,
    aerosol_scattering_tau: np.ndarray,
    depol: np.ndarray,
    aerosol_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    inv_scattering_tau: np.ndarray,
    aerosol_phase_endpoints: np.ndarray,
    cos_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if (
        cos_scatter.shape != (1,)
        or ssa.ndim != 2
        or rayleigh_scattering_tau.shape != ssa.shape
        or aerosol_scattering_tau.ndim != 3
        or aerosol_scattering_tau.shape[:2] != ssa.shape
        or depol.shape != (ssa.shape[0],)
        or aerosol_interp_fraction.shape != (ssa.shape[0],)
        or inv_scattering_tau.shape != ssa.shape
        or ssa.shape[0] < _NUMBA_PHASE_MIN_ROWS
    ):
        return None
    kernel = _get_solar_scatter_kernel()
    if kernel is None:
        return None
    return kernel(
        np.ascontiguousarray(ssa, dtype=np.float64),
        np.ascontiguousarray(rayleigh_scattering_tau, dtype=np.float64),
        np.ascontiguousarray(aerosol_scattering_tau, dtype=np.float64),
        np.ascontiguousarray(depol, dtype=np.float64),
        np.ascontiguousarray(aerosol_moments, dtype=np.float64),
        np.ascontiguousarray(aerosol_interp_fraction, dtype=np.float64),
        np.ascontiguousarray(inv_scattering_tau, dtype=np.float64),
        np.ascontiguousarray(aerosol_phase_endpoints, dtype=np.float64),
        float(cos_scatter[0]),
    )


def _aerosol_scattering_array(
    aerosol_scattering_tau,
    layer_shape: tuple[int, ...],
    naerosol: int,
) -> np.ndarray:
    arr = np.asarray(aerosol_scattering_tau, dtype=float)
    if arr.ndim == 0:
        raise ValueError("aerosol_scattering_tau must include an aerosol axis")
    if arr.shape[-1] != naerosol:
        raise ValueError("aerosol_scattering_tau and aerosol_moments disagree on naerosol")
    return np.broadcast_to(arr, layer_shape + (naerosol,))


def _component_scattering_inputs(
    *,
    ssa,
    depol,
    rayleigh_scattering_tau,
    aerosol_scattering_tau,
    aerosol_moments,
    aerosol_interp_fraction,
    scattering_tau,
    validate_inputs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ssa_arr = np.asarray(ssa, dtype=float)
    ray_arr = np.asarray(rayleigh_scattering_tau, dtype=float)
    ssa_b, ray_b = np.broadcast_arrays(ssa_arr, ray_arr)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if validate_inputs and (not np.all(np.isfinite(ssa_b)) or not np.all(np.isfinite(ray_b))):
        raise ValueError("ssa and rayleigh_scattering_tau must be finite")

    lead_shape = ssa_b.shape[:-1]
    depol_b = _broadcast_leading("depol", depol, lead_shape)
    fac_b = _broadcast_leading("aerosol_interp_fraction", aerosol_interp_fraction, lead_shape)
    moments = _aerosol_moments_array(aerosol_moments)
    aerosol_scat = _aerosol_scattering_array(aerosol_scattering_tau, ssa_b.shape, moments.shape[2])
    if validate_inputs and (
        np.any(ray_b < -_FRACTION_TOL) or np.any(aerosol_scat < -_FRACTION_TOL)
    ):
        raise ValueError("rayleigh_scattering_tau and aerosol_scattering_tau must be nonnegative")

    if scattering_tau is None:
        scattering_tau_arr = ray_b + np.sum(aerosol_scat, axis=-1)
    else:
        scattering_tau_arr = np.broadcast_to(np.asarray(scattering_tau, dtype=float), ssa_b.shape)
        if validate_inputs and (
            not np.all(np.isfinite(scattering_tau_arr))
            or np.any(scattering_tau_arr < -_FRACTION_TOL)
        ):
            raise ValueError("scattering_tau must be finite and nonnegative")
    inv_scattering = np.zeros_like(scattering_tau_arr, dtype=float)
    np.divide(1.0, scattering_tau_arr, out=inv_scattering, where=scattering_tau_arr > 0.0)
    return (
        ssa_b,
        ray_b,
        depol_b,
        fac_b,
        moments,
        aerosol_scat,
        inv_scattering,
    )


def _two_stream_moments_from_scattering(
    *,
    rayleigh_scattering_tau: np.ndarray,
    aerosol_scattering_tau: np.ndarray,
    depol: np.ndarray,
    aerosol_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    inv_scattering_tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ray2mom = (1.0 - depol) / (2.0 + depol)
    aerosol_moments12 = np.matmul(
        aerosol_scattering_tau,
        _aerosol_moment_weights(aerosol_moments, aerosol_interp_fraction, (1, 2)),
    )
    aerosol_moment1 = aerosol_moments12[..., 0] * inv_scattering_tau
    aerosol_moment2 = aerosol_moments12[..., 1] * inv_scattering_tau
    ray_moment2 = rayleigh_scattering_tau * ray2mom[..., None] * inv_scattering_tau
    return aerosol_moment1 / 3.0, (ray_moment2 + aerosol_moment2) * 0.2


def build_two_stream_phase_inputs(
    *,
    ssa,
    depol,
    rayleigh_fraction,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
    validate_inputs: bool = True,
) -> TwoStreamPhaseInputs:
    """Builds Fortran-style two-stream ``g`` and delta-M truncation factor."""
    ssa_arr = np.asarray(ssa, dtype=float)
    ray_arr = np.asarray(rayleigh_fraction, dtype=float)
    ssa_b, ray_b = np.broadcast_arrays(ssa_arr, ray_arr)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if validate_inputs and (not np.all(np.isfinite(ssa_b)) or not np.all(np.isfinite(ray_b))):
        raise ValueError("ssa and rayleigh_fraction must be finite")
    lead_shape = ssa_b.shape[:-1]
    depol_b = _broadcast_leading("depol", depol, lead_shape)
    fac_b = _broadcast_leading("aerosol_interp_fraction", aerosol_interp_fraction, lead_shape)

    moments = _aerosol_moments_array(aerosol_moments)
    aer_frac = np.asarray(aerosol_fraction, dtype=float)
    if validate_inputs and not np.all(np.isfinite(aer_frac)):
        raise ValueError("aerosol_fraction must be finite")
    if moments.shape[2] != aer_frac.shape[-1]:
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = np.broadcast_to(aer_frac, ssa_b.shape + (moments.shape[2],))
    if validate_inputs:
        _validate_phase_fractions(
            ssa=ssa_b,
            rayleigh_fraction=ray_b,
            aerosol_fraction=aer_frac_b,
        )

    ray2mom = (1.0 - depol_b) / (2.0 + depol_b)
    aerosol_moments12 = np.matmul(
        aer_frac_b,
        _aerosol_moment_weights(moments, fac_b, (1, 2)),
    )
    moment1 = aerosol_moments12[..., 0]
    moment2 = ray_b * ray2mom[..., None] + aerosol_moments12[..., 1]
    g = moment1 / 3.0
    truncation_factor = moment2 / 5.0
    if validate_inputs:
        validate_delta_m_truncation_factor(truncation_factor, ssa_b)
    return TwoStreamPhaseInputs(g=g, delta_m_truncation_factor=truncation_factor)


def build_two_stream_phase_inputs_from_scattering_tau(
    *,
    ssa,
    depol,
    rayleigh_scattering_tau,
    aerosol_scattering_tau,
    aerosol_moments,
    aerosol_interp_fraction,
    scattering_tau=None,
    validate_inputs: bool = True,
) -> TwoStreamPhaseInputs:
    """Builds two-stream phase inputs without materializing aerosol fractions."""
    ssa_b, ray_b, depol_b, fac_b, moments, aerosol_scat, inv_scattering = (
        _component_scattering_inputs(
            ssa=ssa,
            depol=depol,
            rayleigh_scattering_tau=rayleigh_scattering_tau,
            aerosol_scattering_tau=aerosol_scattering_tau,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=aerosol_interp_fraction,
            scattering_tau=scattering_tau,
            validate_inputs=validate_inputs,
        )
    )
    g, truncation_factor = _two_stream_moments_from_scattering(
        rayleigh_scattering_tau=ray_b,
        aerosol_scattering_tau=aerosol_scat,
        depol=depol_b,
        aerosol_moments=moments,
        aerosol_interp_fraction=fac_b,
        inv_scattering_tau=inv_scattering,
    )
    if validate_inputs:
        validate_delta_m_truncation_factor(truncation_factor, ssa_b)
    return TwoStreamPhaseInputs(g=g, delta_m_truncation_factor=truncation_factor)


def build_solar_phase_inputs_from_scattering_tau(
    *,
    ssa,
    depol,
    rayleigh_scattering_tau,
    aerosol_scattering_tau,
    aerosol_moments,
    aerosol_interp_fraction,
    angles,
    scattering_tau=None,
    validate_inputs: bool = True,
) -> SolarPhaseInputs:
    """Builds solar 2S phase inputs and FO scatter terms from component scattering."""
    ssa_b, ray_b, depol_b, fac_b, moments, aerosol_scat, inv_scattering = (
        _component_scattering_inputs(
            ssa=ssa,
            depol=depol,
            rayleigh_scattering_tau=rayleigh_scattering_tau,
            aerosol_scattering_tau=aerosol_scattering_tau,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=aerosol_interp_fraction,
            scattering_tau=scattering_tau,
            validate_inputs=validate_inputs,
        )
    )
    cos_scatter = _solar_obs_scattering_cosines(angles)
    delta = 2.0 * (1.0 - depol_b) / (2.0 + depol_b)
    aerosol_phase_endpoints = _aerosol_phase_endpoints(moments, cos_scatter)
    if not validate_inputs:
        accelerated = _solar_from_scattering_fast(
            ssa=ssa_b,
            rayleigh_scattering_tau=ray_b,
            aerosol_scattering_tau=aerosol_scat,
            depol=depol_b,
            aerosol_moments=moments,
            aerosol_interp_fraction=fac_b,
            inv_scattering_tau=inv_scattering,
            aerosol_phase_endpoints=aerosol_phase_endpoints,
            cos_scatter=cos_scatter,
        )
        if accelerated is not None:
            return SolarPhaseInputs(
                g=accelerated[0],
                delta_m_truncation_factor=accelerated[1],
                fo_scatter_term=accelerated[2],
            )
    if cos_scatter.size == 1:
        ray2mom = (1.0 - depol_b) / (2.0 + depol_b)
        raypf = delta * 0.75 * (1.0 + float(cos_scatter[0]) ** 2) + 1.0 - delta
        aerosol_phase = aerosol_phase_endpoints[0, :, 0] + fac_b[..., None] * (
            aerosol_phase_endpoints[1, :, 0] - aerosol_phase_endpoints[0, :, 0]
        )
        aerosol_mixed = np.matmul(
            aerosol_scat,
            np.concatenate(
                (
                    _aerosol_moment_weights(moments, fac_b, (1, 2)),
                    aerosol_phase[..., None],
                ),
                axis=-1,
            ),
        )
        g = aerosol_mixed[..., 0] * inv_scattering / 3.0
        ray_moment2 = ray_b * ray2mom[..., None] * inv_scattering
        truncation_factor = (ray_moment2 + aerosol_mixed[..., 1] * inv_scattering) * 0.2
        if validate_inputs:
            validate_delta_m_truncation_factor(truncation_factor, ssa_b)
        phase_numerator = ray_b * raypf[..., None] + aerosol_mixed[..., 2]
        phase_numerator *= inv_scattering
        phase_numerator *= ssa_b
        phase_numerator /= 1.0 - truncation_factor * ssa_b
        scatter = phase_numerator
    else:
        g, truncation_factor = _two_stream_moments_from_scattering(
            rayleigh_scattering_tau=ray_b,
            aerosol_scattering_tau=aerosol_scat,
            depol=depol_b,
            aerosol_moments=moments,
            aerosol_interp_fraction=fac_b,
            inv_scattering_tau=inv_scattering,
        )
        if validate_inputs:
            validate_delta_m_truncation_factor(truncation_factor, ssa_b)
        raypf = delta[..., None] * 0.75 * (1.0 + cos_scatter * cos_scatter) + 1.0 - delta[..., None]
        phase_numerator = ray_b[..., :, None] * raypf[..., None, :]
        aerosol_phase = aerosol_phase_endpoints[0] + fac_b[..., None, None] * (
            aerosol_phase_endpoints[1] - aerosol_phase_endpoints[0]
        )
        phase_numerator = phase_numerator + np.matmul(aerosol_scat, aerosol_phase)
        phase_total = phase_numerator * inv_scattering[..., None]
        scatter = (
            phase_total * ssa_b[..., None] / (1.0 - truncation_factor[..., None] * ssa_b[..., None])
        )
    return SolarPhaseInputs(
        g=g,
        delta_m_truncation_factor=truncation_factor,
        fo_scatter_term=scatter,
    )


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
    validate_inputs: bool = True,
) -> np.ndarray:
    """Builds the Fortran-style solar FO exact-scatter source term."""
    ssa_arr = np.asarray(ssa, dtype=float)
    ray_arr = np.asarray(rayleigh_fraction, dtype=float)
    factor_arr = np.asarray(delta_m_truncation_factor, dtype=float)
    ssa_b, ray_b, factor_b = np.broadcast_arrays(ssa_arr, ray_arr, factor_arr)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if validate_inputs and (not np.all(np.isfinite(ssa_b)) or not np.all(np.isfinite(ray_b))):
        raise ValueError("ssa and rayleigh_fraction must be finite")
    if validate_inputs:
        validate_delta_m_truncation_factor(factor_b, ssa_b)

    lead_shape = ssa_b.shape[:-1]
    depol_b = _broadcast_leading("depol", depol, lead_shape)
    fac_b = _broadcast_leading("aerosol_interp_fraction", aerosol_interp_fraction, lead_shape)
    cos_scatter = _solar_obs_scattering_cosines(angles)

    delta = 2.0 * (1.0 - depol_b) / (2.0 + depol_b)

    moments = _aerosol_moments_array(aerosol_moments)
    aer_frac = np.asarray(aerosol_fraction, dtype=float)
    if validate_inputs and not np.all(np.isfinite(aer_frac)):
        raise ValueError("aerosol_fraction must be finite")
    if moments.shape[2] != aer_frac.shape[-1]:
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = np.broadcast_to(aer_frac, ssa_b.shape + (moments.shape[2],))
    if validate_inputs:
        _validate_phase_fractions(
            ssa=ssa_b,
            rayleigh_fraction=ray_b,
            aerosol_fraction=aer_frac_b,
        )
    aerosol_phase_endpoints = _aerosol_phase_endpoints(moments, cos_scatter)
    if cos_scatter.size == 1:
        raypf = delta * 0.75 * (1.0 + float(cos_scatter[0]) ** 2) + 1.0 - delta
        phase_total = ray_b * raypf[..., None]
        aerosol_phase = aerosol_phase_endpoints[0, :, 0] + fac_b[..., None] * (
            aerosol_phase_endpoints[1, :, 0] - aerosol_phase_endpoints[0, :, 0]
        )
        phase_total += np.matmul(aer_frac_b, aerosol_phase[..., None])[..., 0]
        phase_total *= ssa_b
        phase_total /= 1.0 - factor_b * ssa_b
        return phase_total

    raypf = delta[..., None] * 0.75 * (1.0 + cos_scatter * cos_scatter) + 1.0 - delta[..., None]
    phase_total = ray_b[..., :, None] * raypf[..., None, :]
    aerosol_phase = aerosol_phase_endpoints[0] + fac_b[..., None, None] * (
        aerosol_phase_endpoints[1] - aerosol_phase_endpoints[0]
    )
    phase_total = phase_total + np.matmul(aer_frac_b, aerosol_phase)
    return phase_total * ssa_b[..., None] / (1.0 - factor_b[..., None] * ssa_b[..., None])
