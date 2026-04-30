"""Batched boundary-value problem solvers for two-stream calculations."""

from __future__ import annotations

import numpy as np

_NUMBA_BVP_MIN_BATCH = 8192
_PENTADIAGONAL_BVP_KERNEL = None
_PENTADIAGONAL_BVP_IMPORT_FAILED = False


def _normalize_numpy_bvp_engine(engine: str) -> str:
    """Normalizes the public NumPy BVP engine names."""
    normalized = engine.lower()
    allowed = {"auto", "numpy", "numba", "pentadiagonal", "block", "banded"}
    if normalized not in allowed:
        raise ValueError(
            "NumPy BVP engine must be 'auto', 'numpy', 'numba', 'pentadiagonal', 'block', or 'banded'"
        )
    if normalized in {"auto", "numba"}:
        return normalized
    if normalized in {"numpy", "pentadiagonal"}:
        return "pentadiagonal"
    return normalized


def _get_pentadiagonal_bvp_kernel():
    """Lazily builds the optional row-parallel pentadiagonal BVP kernel."""
    global _PENTADIAGONAL_BVP_KERNEL, _PENTADIAGONAL_BVP_IMPORT_FAILED
    if _PENTADIAGONAL_BVP_KERNEL is not None:
        return _PENTADIAGONAL_BVP_KERNEL
    if _PENTADIAGONAL_BVP_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional acceleration dependency
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(
            albedo, bottom_source, surface_factor, stream_value, x1, x2, et, wu0, wu1, wl0, wl1
        ):
            batch, nlay = x1.shape
            ntotal = 2 * nlay
            lcon = np.empty((batch, nlay), np.float64)
            mcon = np.empty((batch, nlay), np.float64)
            for b in prange(batch):
                col = np.empty(ntotal, np.float64)
                elm1 = np.empty(ntotal - 1, np.float64)
                elm2 = np.empty(ntotal - 2, np.float64)
                factor = surface_factor * albedo[b]
                xpnet = x2[b, nlay - 1] - factor * x1[b, nlay - 1] * stream_value
                xmnet = x1[b, nlay - 1] - factor * x2[b, nlay - 1] * stream_value

                col[0] = -wu0[b, 0]
                for n in range(1, nlay):
                    prev = n - 1
                    row_m = 2 * n - 1
                    row_p = row_m + 1
                    col[row_m] = wu0[b, n] - wl0[b, prev]
                    col[row_p] = wu1[b, n] - wl1[b, prev]
                col[ntotal - 1] = (
                    -wl1[b, nlay - 1] + wl0[b, nlay - 1] * stream_value * factor + bottom_source[b]
                )

                elm31 = 1.0 / x1[b, 0]
                elm1_i2 = -(x2[b, 0] * et[b, 0]) * elm31
                elm1[0] = elm1_i2
                elm2_i2 = 0.0
                elm2[0] = elm2_i2
                col_i2 = col[0] * elm31
                col[0] = col_i2

                mat22 = x1[b, 0] * et[b, 0]
                bet = x2[b, 0] + mat22 * elm1_i2
                bet = -1.0 / bet
                elm1_i1 = -x1[b, 1] * bet
                elm1[1] = elm1_i1
                elm2_i1 = (-x2[b, 1] * et[b, 1]) * bet
                elm2[1] = elm2_i1
                col_i1 = (mat22 * col_i2 - col[1]) * bet
                col[1] = col_i1

                for i in range(2, ntotal - 2):
                    if i % 2 == 0:
                        n = i // 2
                        prev = n - 1
                        mat1_i = x2[b, prev] * et[b, prev]
                        mat2_i = x1[b, prev]
                        mat3_i = -x2[b, n]
                        mat4_i = -x1[b, n] * et[b, n]
                        mat5_i = 0.0
                    else:
                        n = (i + 1) // 2
                        prev = n - 1
                        mat1_i = 0.0
                        mat2_i = x1[b, prev] * et[b, prev]
                        mat3_i = x2[b, prev]
                        mat4_i = -x1[b, n]
                        mat5_i = -x2[b, n] * et[b, n]

                    bet = mat2_i + mat1_i * elm1_i2
                    den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
                    den = -1.0 / den
                    elm1_i2 = elm1_i1
                    elm1_i1 = (mat4_i + bet * elm2_i1) * den
                    elm1[i] = elm1_i1
                    elm2_i2 = elm2_i1
                    elm2_i1 = mat5_i * den
                    elm2[i] = elm2_i1
                    col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
                    col_i2 = col_i1
                    col_i1 = col_i
                    col[i] = col_i

                i = ntotal - 2
                n = i // 2
                prev = n - 1
                mat1_i = x2[b, prev] * et[b, prev]
                mat2_i = x1[b, prev]
                mat3_i = -x2[b, n]
                mat4_i = -x1[b, n] * et[b, n]
                bet = mat2_i + mat1_i * elm1_i2
                den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
                den = -1.0 / den
                elm1_i2 = elm1_i1
                elm1_i1 = (mat4_i + bet * elm2_i1) * den
                elm1[i] = elm1_i1
                elm2_i2 = elm2_i1
                col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
                col_i2 = col_i1
                col_i1 = col_i
                col[i] = col_i

                i = ntotal - 1
                bet = xpnet * et[b, nlay - 1]
                den = xmnet + bet * elm1_i1
                den = -1.0 / den
                col_i = (bet * col_i1 - col[i]) * den
                col_i2 = col_i1
                col_i1 = col_i
                col[i] = col_i

                i = ntotal - 2
                col_i = col_i2 + elm1[i] * col_i1
                col[i] = col_i
                col_i2 = col_i1
                col_i1 = col_i
                for i in range(ntotal - 3, -1, -1):
                    col_i = col[i] + elm1[i] * col_i1 + elm2[i] * col_i2
                    col[i] = col_i
                    col_i2 = col_i1
                    col_i1 = col_i

                for n in range(nlay):
                    lcon[b, n] = col[2 * n]
                    mcon[b, n] = col[2 * n + 1]
            return lcon, mcon

        _PENTADIAGONAL_BVP_KERNEL = _kernel
        return _PENTADIAGONAL_BVP_KERNEL
    except Exception:  # pragma: no cover - optional acceleration dependency
        _PENTADIAGONAL_BVP_IMPORT_FAILED = True
        return None


def _solve_pentadiagonal_bvp_batch_numba(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Runs the optional Numba BVP kernel for large batches."""
    if xpos1.shape[0] < _NUMBA_BVP_MIN_BATCH or xpos1.shape[1] < 2:
        return None
    kernel = _get_pentadiagonal_bvp_kernel()
    if kernel is None:
        return None
    return kernel(
        np.ascontiguousarray(albedo, dtype=float),
        np.ascontiguousarray(bottom_source, dtype=float),
        float(surface_factor),
        float(stream_value),
        np.ascontiguousarray(xpos1, dtype=float),
        np.ascontiguousarray(xpos2, dtype=float),
        np.ascontiguousarray(eigentrans, dtype=float),
        np.ascontiguousarray(wupper[0], dtype=float),
        np.ascontiguousarray(wupper[1], dtype=float),
        np.ascontiguousarray(wlower[0], dtype=float),
        np.ascontiguousarray(wlower[1], dtype=float),
    )


def _solve_pentadiagonal_bvp_batch(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Solves a batch of regular two-stream pentadiagonal BVP systems."""
    x1 = np.ascontiguousarray(xpos1.T)
    x2 = np.ascontiguousarray(xpos2.T)
    et = np.ascontiguousarray(eigentrans.T)
    wupper0 = np.ascontiguousarray(wupper[0].T)
    wupper1 = np.ascontiguousarray(wupper[1].T)
    wlower0 = np.ascontiguousarray(wlower[0].T)
    wlower1 = np.ascontiguousarray(wlower[1].T)
    nlay, batch = x1.shape
    ntotal = 2 * nlay
    col = np.zeros((ntotal, batch), dtype=float)

    factor = surface_factor * albedo
    xpnet = x2[-1] - factor * x1[-1] * stream_value
    xmnet = x1[-1] - factor * x2[-1] * stream_value

    col[0] = -wupper0[0]
    for n in range(1, nlay):
        prev = n - 1
        row_m = 2 * n - 1
        row_p = row_m + 1
        col[row_m] = wupper0[n] - wlower0[prev]
        col[row_p] = wupper1[n] - wlower1[prev]
    col[-1] = -wlower1[-1] + wlower0[-1] * stream_value * factor + bottom_source

    elm1 = np.zeros((ntotal - 1, batch), dtype=float)
    elm2 = np.zeros((ntotal - 2, batch), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        elm31 = 1.0 / x1[0]
        elm1_i2 = -(x2[0] * et[0]) * elm31
        elm1[0] = elm1_i2
        elm2_i2 = np.zeros(batch, dtype=float)
        elm2[0] = elm2_i2

        col_i2 = col[0] * elm31
        col[0] = col_i2

        mat22 = x1[0] * et[0]
        bet = x2[0] + mat22 * elm1_i2
        bet = -1.0 / bet
        elm1_i1 = -x1[1] * bet
        elm1[1] = elm1_i1
        elm2_i1 = (-x2[1] * et[1]) * bet
        elm2[1] = elm2_i1
        col_i1 = (mat22 * col_i2 - col[1]) * bet
        col[1] = col_i1

        for i in range(2, ntotal - 2):
            if i % 2 == 0:
                n = i // 2
                prev = n - 1
                mat1_i = x2[prev] * et[prev]
                mat2_i = x1[prev]
                mat3_i = -x2[n]
                mat4_i = -x1[n] * et[n]
                mat5_i = 0.0
            else:
                n = (i + 1) // 2
                prev = n - 1
                mat1_i = 0.0
                mat2_i = x1[prev] * et[prev]
                mat3_i = x2[prev]
                mat4_i = -x1[n]
                mat5_i = -x2[n] * et[n]
            bet = mat2_i + mat1_i * elm1_i2
            den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
            den = -1.0 / den
            elm1_i2 = elm1_i1
            elm1_i1 = (mat4_i + bet * elm2_i1) * den
            elm1[i] = elm1_i1
            elm2_i2 = elm2_i1
            elm2_i1 = mat5_i * den
            elm2[i] = elm2_i1

            col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
            col_i2 = col_i1
            col_i1 = col_i
            col[i] = col_i

        i = ntotal - 2
        n = i // 2
        prev = n - 1
        mat1_i = x2[prev] * et[prev]
        mat2_i = x1[prev]
        mat3_i = -x2[n]
        mat4_i = -x1[n] * et[n]
        bet = mat2_i + mat1_i * elm1_i2
        den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
        den = -1.0 / den
        elm1_i2 = elm1_i1
        elm1_i1 = (mat4_i + bet * elm2_i1) * den
        elm1[i] = elm1_i1
        elm2_i2 = elm2_i1

        col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
        col_i2 = col_i1
        col_i1 = col_i
        col[i] = col_i

        i = ntotal - 1
        bet = xpnet * et[-1]
        den = xmnet + bet * elm1_i1
        den = -1.0 / den
        col_i = (bet * col_i1 - col[i]) * den
        col_i2 = col_i1
        col_i1 = col_i
        col[i] = col_i

        i = ntotal - 2
        col_i = col_i2 + elm1[i] * col_i1
        col[i] = col_i
        col_i2 = col_i1
        col_i1 = col_i
        for i in range(ntotal - 3, -1, -1):
            col_i = col[i] + elm1[i] * col_i1 + elm2[i] * col_i2
            col[i] = col_i
            col_i2 = col_i1
            col_i1 = col_i

    return col[0::2].T, col[1::2].T


def _build_dense_matrix_rhs_batch(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds dense BVP matrices and RHS for fallback solves."""
    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    batch, nlay = xpos1.shape
    ntotal = 2 * nlay
    mat = np.zeros((batch, ntotal, ntotal), dtype=float)
    rhs = np.zeros((batch, ntotal), dtype=float)
    factor = surface_factor * albedo
    xpnet = xpos2[:, -1] - factor * xpos1[:, -1] * stream_value
    xmnet = xpos1[:, -1] - factor * xpos2[:, -1] * stream_value
    mat[:, 0, 0] = xpos1[:, 0]
    mat[:, 0, 1] = xpos2[:, 0] * eigentrans[:, 0]
    rhs[:, 0] = -wupper0[:, 0]
    row = 1
    for n in range(1, nlay):
        prev = n - 1
        mat[:, row, 2 * prev] = xpos1[:, prev] * eigentrans[:, prev]
        mat[:, row, 2 * prev + 1] = xpos2[:, prev]
        mat[:, row, 2 * n] = -xpos1[:, n]
        mat[:, row, 2 * n + 1] = -xpos2[:, n] * eigentrans[:, n]
        rhs[:, row] = wupper0[:, n] - wlower0[:, prev]
        row += 1
        mat[:, row, 2 * prev] = xpos2[:, prev] * eigentrans[:, prev]
        mat[:, row, 2 * prev + 1] = xpos1[:, prev]
        mat[:, row, 2 * n] = -xpos2[:, n]
        mat[:, row, 2 * n + 1] = -xpos1[:, n] * eigentrans[:, n]
        rhs[:, row] = wupper1[:, n] - wlower1[:, prev]
        row += 1
    mat[:, -1, -2] = xpnet * eigentrans[:, -1]
    mat[:, -1, -1] = xmnet
    rhs[:, -1] = -wlower1[:, -1] + wlower0[:, -1] * stream_value * factor + bottom_source
    return mat, rhs


def _solve_dense_bvp_batch(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Solves dense BVP matrices row-by-row for correctness-oriented paths."""
    mat, rhs = _build_dense_matrix_rhs_batch(
        albedo=albedo,
        bottom_source=bottom_source,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )
    sol = np.linalg.solve(mat, rhs[..., np.newaxis])[..., 0]
    return sol[:, 0::2], sol[:, 1::2]


def _repair_nonfinite_rows_with_dense(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
    lcon: np.ndarray,
    mcon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repairs only non-finite BVP rows with the dense direct solver."""
    bad = (~np.isfinite(lcon).all(axis=1)) | (~np.isfinite(mcon).all(axis=1))
    if not np.any(bad):
        return lcon, mcon
    lcon_bad, mcon_bad = _solve_dense_bvp_batch(
        albedo=albedo[bad],
        bottom_source=bottom_source[bad],
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1[bad],
        xpos2=xpos2[bad],
        eigentrans=eigentrans[bad],
        wupper=(wupper[0][bad], wupper[1][bad]),
        wlower=(wlower[0][bad], wlower[1][bad]),
    )
    repaired_lcon = lcon.copy()
    repaired_mcon = mcon.copy()
    repaired_lcon[bad] = lcon_bad
    repaired_mcon[bad] = mcon_bad
    return repaired_lcon, repaired_mcon


def _solve_bvp_batch(
    *,
    albedo: np.ndarray,
    bottom_source: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
    bvp_engine: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    engine = _normalize_numpy_bvp_engine(bvp_engine)
    if engine in {"auto", "numba"}:
        result = _solve_pentadiagonal_bvp_batch_numba(
            albedo=albedo,
            bottom_source=bottom_source,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=wupper,
            wlower=wlower,
        )
        if result is not None:
            lcon, mcon = result
            return _repair_nonfinite_rows_with_dense(
                albedo=albedo,
                bottom_source=bottom_source,
                surface_factor=surface_factor,
                stream_value=stream_value,
                xpos1=xpos1,
                xpos2=xpos2,
                eigentrans=eigentrans,
                wupper=wupper,
                wlower=wlower,
                lcon=lcon,
                mcon=mcon,
            )
    if engine in {"auto", "numba", "pentadiagonal"}:
        lcon, mcon = _solve_pentadiagonal_bvp_batch(
            albedo=albedo,
            bottom_source=bottom_source,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=wupper,
            wlower=wlower,
        )
        return _repair_nonfinite_rows_with_dense(
            albedo=albedo,
            bottom_source=bottom_source,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=wupper,
            wlower=wlower,
            lcon=lcon,
            mcon=mcon,
        )
    return _solve_dense_bvp_batch(
        albedo=albedo,
        bottom_source=bottom_source,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def solve_solar_observation_bvp_batch(
    *,
    albedo: np.ndarray,
    direct_beam: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
    bvp_engine: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Solves batched solar-observation BVP systems."""
    return _solve_bvp_batch(
        albedo=albedo,
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
        bvp_engine=bvp_engine,
    )


def solve_thermal_bvp_batch(
    *,
    albedo: np.ndarray,
    emissivity: np.ndarray,
    surfbb: np.ndarray,
    surface_factor: float,
    stream_value: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigentrans: np.ndarray,
    wupper: tuple[np.ndarray, np.ndarray],
    wlower: tuple[np.ndarray, np.ndarray],
    bvp_engine: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Solves batched thermal BVP systems."""
    return _solve_bvp_batch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
        bvp_engine=bvp_engine,
    )
