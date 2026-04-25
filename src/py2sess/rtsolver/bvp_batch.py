"""Batched boundary-value problem solvers for two-stream calculations."""

from __future__ import annotations

import numpy as np


def _normalize_numpy_bvp_engine(engine: str) -> str:
    """Normalizes the public NumPy BVP engine names."""
    normalized = engine.lower()
    allowed = {"auto", "numpy", "numba", "pentadiagonal", "block", "banded"}
    if normalized not in allowed:
        raise ValueError(
            "NumPy BVP engine must be 'auto', 'numpy', 'numba', 'pentadiagonal', 'block', or 'banded'"
        )
    if normalized in {"auto", "numpy", "numba", "pentadiagonal"}:
        return "pentadiagonal"
    return normalized


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
    engine = _normalize_numpy_bvp_engine(bvp_engine)
    if engine == "pentadiagonal":
        lcon, mcon = _solve_pentadiagonal_bvp_batch(
            albedo=albedo,
            bottom_source=direct_beam,
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
            bottom_source=direct_beam,
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
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
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
    bottom_source = surfbb * emissivity
    engine = _normalize_numpy_bvp_engine(bvp_engine)
    if engine == "pentadiagonal":
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


def solve_solar_observation_dense_bvp_batch(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Solves dense solar-observation BVP matrices."""
    return _solve_dense_bvp_batch(
        albedo=albedo,
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def build_thermal_dense_matrix_rhs_batch(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Builds dense thermal BVP matrices and RHS for fallback solves."""
    return _build_dense_matrix_rhs_batch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def solve_thermal_dense_bvp_batch(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Solves dense thermal BVP matrices for fallback rows."""
    return _solve_dense_bvp_batch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )
