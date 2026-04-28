"""Batched NumPy helpers for solar observation-geometry FO calculations."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .fo_solar_obs import _fo_eps_geometry

_NUMBA_FO_MIN_BATCH = 4096
_NUMBA_FO_IMPORT_FAILED = False
_NUMBA_FO_KERNEL = None


@dataclass(frozen=True)
class FoSolarObsBatchPrecompute:
    """Reusable geometry terms for the batched FO EPS NumPy path."""

    inv_layer_thickness: np.ndarray
    do_nadir: bool
    mu0: float
    ntrav_nl: int
    sunpathsnl: np.ndarray
    cota: np.ndarray
    cotfine: np.ndarray
    csqfine: np.ndarray
    wfine: np.ndarray
    nfinedivs: np.ndarray
    rayconv: float
    xfine: np.ndarray
    sunpathsfine: np.ndarray
    ntraversefine: np.ndarray
    fine_path_matrix: np.ndarray | None = None
    fine_column_index: np.ndarray | None = None


@dataclass(frozen=True)
class FoSolarObsBatchResult:
    """Batched solar FO endpoint and optional profile components."""

    total: np.ndarray
    single_scatter: np.ndarray
    direct_beam: np.ndarray
    total_profile: np.ndarray | None = None
    single_scatter_profile: np.ndarray | None = None
    direct_beam_profile: np.ndarray | None = None


def _exp_cutoff_owned(values: np.ndarray) -> np.ndarray:
    """Applies the Fortran 88-optical-depth cutoff in place."""
    if values.size == 0 or float(np.max(values)) < 88.0:
        np.exp(-values, out=values)
        return values
    too_deep = values >= 88.0
    np.exp(-values, out=values)
    np.putmask(values, too_deep, 0.0)
    return values


def _exp_cutoff(values: np.ndarray) -> np.ndarray:
    """Applies the Fortran 88-optical-depth cutoff to a fresh array."""
    if values.size == 0 or float(np.max(values)) < 88.0:
        return np.exp(-values)
    result = np.exp(-values)
    result[values >= 88.0] = 0.0
    return result


def _ensure_c_contiguous(values: np.ndarray, *, dtype) -> np.ndarray:
    """Returns a C-contiguous array with the requested dtype."""
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def _numba_fo_enabled(batch_size: int) -> bool:
    """Returns whether the optional Numba UV FO kernel should be enabled."""
    flag = os.environ.get("PY2SESS_NUMBA_FO", "auto").lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if flag in {"1", "true", "on", "yes"}:
        return _get_numba_fo_kernel() is not None
    if batch_size < _NUMBA_FO_MIN_BATCH:
        return False
    return _get_numba_fo_kernel() is not None


def _get_numba_fo_kernel():
    """Builds the optional Numba non-nadir FO recurrence kernel on demand."""
    global _NUMBA_FO_IMPORT_FAILED, _NUMBA_FO_KERNEL
    if _NUMBA_FO_KERNEL is not None:
        return _NUMBA_FO_KERNEL
    if _NUMBA_FO_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional dependency
        from numba import njit, prange
    except Exception:  # pragma: no cover - optional dependency
        _NUMBA_FO_IMPORT_FAILED = True
        return None

    @njit(parallel=True, cache=True)
    def _solve_fo_solar_eps_nonnadir_kernel(
        extinction,
        phase_terms,
        solar_flux,
        albedo,
        attenuation_nl,
        fine_attenuation,
        cota,
        cotfine,
        csqfine,
        wfine,
        nfinedivs,
        column_index,
        rayconv,
        mu0,
    ):
        batch_size, nlayers = extinction.shape
        out = np.empty(batch_size, np.float64)
        for row in prange(batch_size):
            cot_1 = cota[nlayers]
            cumsource_up = 0.0
            cumsource_db = 4.0 * mu0 * albedo[row] * attenuation_nl[row]

            for n_fortran in range(nlayers, 0, -1):
                layer = n_fortran - 1
                cot_2 = cota[layer]
                ke = rayconv * extinction[row, layer]
                lostrans = math.exp(-ke * (cot_2 - cot_1))
                layer_sum = 0.0
                nfine_layer = nfinedivs[layer]
                for j in range(nfine_layer):
                    column = column_index[j, layer]
                    tran = math.exp(-ke * (cot_2 - cotfine[j, layer]))
                    layer_sum += (
                        phase_terms[row, layer]
                        * fine_attenuation[row, column]
                        * csqfine[j, layer]
                        * tran
                        * wfine[j, layer]
                    )
                source = layer_sum * ke
                cumsource_db = lostrans * cumsource_db
                cumsource_up = lostrans * cumsource_up + source
                cot_1 = cot_2
            out[row] = 0.25 * solar_flux[row] / math.pi * (cumsource_up + cumsource_db)
        return out

    _NUMBA_FO_KERNEL = _solve_fo_solar_eps_nonnadir_kernel
    return _NUMBA_FO_KERNEL


@lru_cache(maxsize=8)
def _fo_solar_obs_batch_precompute_from_key(
    user_obsgeom_key: tuple[float, ...],
    heights_key: tuple[float, ...],
    earth_radius: float,
    nfine: int,
) -> FoSolarObsBatchPrecompute:
    """Builds reusable geometry arrays for the batched FO EPS solver."""
    user_obsgeom = np.asarray(user_obsgeom_key, dtype=float).reshape(1, 3)
    heights = np.asarray(heights_key, dtype=float)
    geometry = _fo_eps_geometry(
        user_obsgeoms=user_obsgeom,
        height_grid=heights,
        earth_radius=earth_radius,
        nfine=nfine,
        vsign=1.0,
    )
    geometry_arrays = {name: np.asarray(value) for name, value in geometry.items()}
    nlayers = heights.size - 1
    v = 0
    do_nadir = bool(np.asarray(geometry_arrays["do_nadir"], dtype=bool)[v])
    inv_layer_thickness = 1.0 / (heights[:-1] - heights[1:])
    ntrav_nl = int(np.asarray(geometry_arrays["ntraversenl"], dtype=int)[v])
    fine_path_matrix = None
    fine_column_index = None
    if not do_nadir:
        sunpathsfine = np.asarray(geometry_arrays["sunpathsfine"], dtype=float)
        ntraversefine = np.asarray(geometry_arrays["ntraversefine"], dtype=int)
        max_fine = int(ntraversefine.shape[0])
        path_columns: list[np.ndarray] = []
        fine_column_index = np.full((max_fine, nlayers), -1, dtype=int)
        for n in range(nlayers, 0, -1):
            nfine_layer = int(geometry_arrays["nfinedivs"][n - 1, v])
            for j in range(nfine_layer):
                ntrav = int(ntraversefine[j, n - 1, v])
                column = np.zeros(nlayers, dtype=float)
                column[:ntrav] = sunpathsfine[:ntrav, j, n - 1, v]
                fine_column_index[j, n - 1] = len(path_columns)
                path_columns.append(column)
        fine_path_matrix = np.stack(path_columns, axis=1)

    return FoSolarObsBatchPrecompute(
        inv_layer_thickness=np.ascontiguousarray(inv_layer_thickness, dtype=float),
        do_nadir=do_nadir,
        mu0=float(np.asarray(geometry_arrays["mu0"], dtype=float)[v]),
        ntrav_nl=ntrav_nl,
        sunpathsnl=np.ascontiguousarray(
            np.asarray(geometry_arrays["sunpathsnl"], dtype=float)[:ntrav_nl, v]
        ),
        cota=np.ascontiguousarray(np.asarray(geometry_arrays["cota"], dtype=float)[:, v]),
        cotfine=np.ascontiguousarray(np.asarray(geometry_arrays["cotfine"], dtype=float)[:, :, v]),
        csqfine=np.ascontiguousarray(np.asarray(geometry_arrays["csqfine"], dtype=float)[:, :, v]),
        wfine=np.ascontiguousarray(np.asarray(geometry_arrays["wfine"], dtype=float)[:, :, v]),
        nfinedivs=np.ascontiguousarray(np.asarray(geometry_arrays["nfinedivs"], dtype=int)[:, v]),
        rayconv=float(np.asarray(geometry_arrays["raycon"], dtype=float)[v]),
        xfine=np.asarray(geometry_arrays["xfine"], dtype=float),
        sunpathsfine=np.asarray(geometry_arrays["sunpathsfine"], dtype=float),
        ntraversefine=np.asarray(geometry_arrays["ntraversefine"], dtype=int),
        fine_path_matrix=(
            None
            if fine_path_matrix is None
            else np.ascontiguousarray(fine_path_matrix, dtype=float)
        ),
        fine_column_index=(
            None
            if fine_column_index is None
            else np.ascontiguousarray(fine_column_index, dtype=np.int64)
        ),
    )


def fo_solar_obs_batch_precompute(
    *,
    user_obsgeom: np.ndarray,
    heights: np.ndarray,
    earth_radius: float = 6371.0,
    nfine: int = 3,
) -> FoSolarObsBatchPrecompute:
    """Returns cached geometry terms for batched FO EPS evaluation."""
    return _fo_solar_obs_batch_precompute_from_key(
        tuple(np.asarray(user_obsgeom, dtype=float).ravel()),
        tuple(np.asarray(heights, dtype=float).ravel()),
        float(earth_radius),
        int(nfine),
    )


def solve_fo_solar_obs_eps_batch_numpy(
    *,
    tau: np.ndarray,
    omega: np.ndarray,
    scaling: np.ndarray,
    albedo: np.ndarray,
    flux_factor: np.ndarray,
    exact_scatter: np.ndarray,
    precomputed: FoSolarObsBatchPrecompute,
    return_profile: bool = False,
    return_components: bool = False,
) -> np.ndarray | FoSolarObsBatchResult:
    """Evaluates FO EPS TOA radiance for a spectral batch.

    Notes
    -----
    This matches the optimized Fortran driver usage for the UV benchmark path.
    FO optical depth receives delta-M scaling before entering the FO
    recurrence, while ``exact_scatter`` already includes the corresponding TMS
    factor.
    """
    tau_scaled = np.asarray(tau, dtype=float) * (
        1.0 - np.asarray(omega, dtype=float) * np.asarray(scaling, dtype=float)
    )
    extinction = tau_scaled * precomputed.inv_layer_thickness[np.newaxis, :]
    phase_terms = np.asarray(exact_scatter, dtype=float)
    total_tau = extinction[:, : precomputed.ntrav_nl] @ precomputed.sunpathsnl
    attenuation_nl = _exp_cutoff_owned(total_tau)

    if (
        _numba_fo_enabled(int(extinction.shape[0]))
        and not return_profile
        and not return_components
        and not precomputed.do_nadir
        and precomputed.fine_path_matrix is not None
        and precomputed.fine_column_index is not None
    ):
        kernel = _get_numba_fo_kernel()
        if kernel is not None:
            fine_attenuation = _exp_cutoff_owned(extinction @ precomputed.fine_path_matrix)
            return kernel(
                _ensure_c_contiguous(extinction, dtype=np.float64),
                _ensure_c_contiguous(phase_terms, dtype=np.float64),
                _ensure_c_contiguous(np.asarray(flux_factor, dtype=float), dtype=np.float64),
                _ensure_c_contiguous(np.asarray(albedo, dtype=float), dtype=np.float64),
                _ensure_c_contiguous(attenuation_nl, dtype=np.float64),
                _ensure_c_contiguous(fine_attenuation, dtype=np.float64),
                precomputed.cota,
                precomputed.cotfine,
                precomputed.csqfine,
                precomputed.wfine,
                precomputed.nfinedivs,
                precomputed.fine_column_index,
                precomputed.rayconv,
                precomputed.mu0,
            )

    batch_size, nlayers = extinction.shape
    cumsource_up = np.zeros(batch_size, dtype=float)
    cumsource_db = 4.0 * precomputed.mu0 * np.asarray(albedo, dtype=float) * attenuation_nl
    profile_up = None
    profile_db = None
    if return_profile:
        profile_up = np.zeros((batch_size, nlayers + 1), dtype=float)
        profile_db = np.empty((batch_size, nlayers + 1), dtype=float)
        profile_db[:, nlayers] = cumsource_db

    if precomputed.do_nadir:
        v = 0
        for n in range(nlayers, 0, -1):
            kn = extinction[:, n - 1]
            layer_sum = np.zeros(batch_size, dtype=float)
            nfine_layer = int(precomputed.nfinedivs[n - 1])
            for j in range(nfine_layer):
                ntrav = int(precomputed.ntraversefine[j, n - 1, v])
                paths = precomputed.sunpathsfine[:ntrav, j, n - 1, v]
                fine_tau = extinction[:, :ntrav] @ paths
                attenuation = _exp_cutoff(fine_tau)
                layer_sum += (
                    phase_terms[:, n - 1]
                    * attenuation
                    * np.exp(-precomputed.xfine[j, n - 1, v] * kn)
                    * precomputed.wfine[j, n - 1]
                )
            lostrans = _exp_cutoff(tau_scaled[:, n - 1])
            source = layer_sum * kn
            cumsource_db = lostrans * cumsource_db
            cumsource_up = lostrans * cumsource_up + source
            if return_profile:
                profile_up[:, n - 1] = cumsource_up
                profile_db[:, n - 1] = cumsource_db
    else:
        if precomputed.fine_path_matrix is None or precomputed.fine_column_index is None:
            raise ValueError("missing non-nadir FO batch geometry terms")
        cot_1 = precomputed.cota[nlayers]
        fine_attenuation = _exp_cutoff_owned(extinction @ precomputed.fine_path_matrix)
        for n in range(nlayers, 0, -1):
            cot_2 = precomputed.cota[n - 1]
            ke = precomputed.rayconv * extinction[:, n - 1]
            lostrans = np.exp(-ke * (cot_2 - cot_1))
            layer_sum = np.zeros(batch_size, dtype=float)
            nfine_layer = int(precomputed.nfinedivs[n - 1])
            for j in range(nfine_layer):
                column = int(precomputed.fine_column_index[j, n - 1])
                tran = np.exp(-ke * (cot_2 - precomputed.cotfine[j, n - 1]))
                layer_sum += (
                    phase_terms[:, n - 1]
                    * fine_attenuation[:, column]
                    * precomputed.csqfine[j, n - 1]
                    * tran
                    * precomputed.wfine[j, n - 1]
                )
            source = layer_sum * ke
            cumsource_db = lostrans * cumsource_db
            cumsource_up = lostrans * cumsource_up + source
            if return_profile:
                profile_up[:, n - 1] = cumsource_up
                profile_db[:, n - 1] = cumsource_db
            cot_1 = cot_2

    scale = 0.25 * np.asarray(flux_factor, dtype=float) / math.pi
    single_scatter = scale * cumsource_up
    direct_beam = scale * cumsource_db
    if return_profile:
        single_scatter_profile = scale[:, np.newaxis] * profile_up
        direct_beam_profile = scale[:, np.newaxis] * profile_db
        total_profile = single_scatter_profile + direct_beam_profile
        if return_components:
            return FoSolarObsBatchResult(
                total=single_scatter + direct_beam,
                single_scatter=single_scatter,
                direct_beam=direct_beam,
                total_profile=total_profile,
                single_scatter_profile=single_scatter_profile,
                direct_beam_profile=direct_beam_profile,
            )
        return total_profile
    if return_components:
        return FoSolarObsBatchResult(
            total=single_scatter + direct_beam,
            single_scatter=single_scatter,
            direct_beam=direct_beam,
        )
    return single_scatter + direct_beam
