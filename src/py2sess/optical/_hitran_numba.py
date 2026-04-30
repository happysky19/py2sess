"""Optional Numba kernels for HITRAN line-by-line accumulation."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import numpy as np

_C2 = 1.4387752
_T0 = 296.0
_VOIGT_EXTRA = 25.0
_RRTPI = 0.5641895835
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_HUMLIK_Y0 = 1.5
_HUMLIK_Y0PY0 = 3.0
_HUMLIK_Y0Q = 2.25
_HUMLIK_C = (1.0117281, -0.75197147, 0.012557727, 0.010022008, -0.00024206814, 5.0084806e-7)
_HUMLIK_S = (1.393237, 0.23115241, -0.15535147, 0.0062183662, 9.1908299e-5, -6.2752596e-7)
_HUMLIK_T = (0.31424038, 0.94778839, 1.5976826, 2.2795071, 3.0206370, 3.8897249)
_MIN_WORK = 2_000_000


def accumulate_hitran_numba(
    *,
    nspec: int,
    lines: Any,
    q296: np.ndarray,
    q: np.ndarray,
    pressure_atm: np.ndarray,
    temperature_k: np.ndarray,
    nvoigt: int,
    voigt_grid: np.ndarray,
) -> np.ndarray | None:
    kernel = _accumulation_kernel(nspec, pressure_atm.size, lines.size)
    if kernel is None:
        return None
    return kernel(
        nspec,
        voigt_grid,
        nvoigt,
        lines.center_cm_inv,
        lines.strength,
        lines.air_half_width,
        lines.lower_state_energy,
        lines.temperature_exponent,
        lines.pressure_shift,
        lines.molecular_mass_amu,
        q296,
        q,
        pressure_atm,
        temperature_k,
    )


def _accumulation_kernel(nspec: int, nlevel: int, nlines: int):
    flag = os.environ.get("PY2SESS_NUMBA_HITRAN", "auto").lower()
    if flag in {"0", "false", "off", "no"}:
        return None
    if flag in {"1", "true", "on", "yes"}:
        kernel = _kernel()
        if kernel is None:
            raise ImportError("PY2SESS_NUMBA_HITRAN is enabled, but numba is not installed")
        return kernel
    if flag != "auto":
        raise ValueError("PY2SESS_NUMBA_HITRAN must be auto, on, or off")
    if nspec * nlevel * nlines < _MIN_WORK:
        return None
    return _kernel()


@lru_cache(maxsize=1)
def _kernel():
    try:  # pragma: no cover - optional dependency
        from numba import njit, prange
    except Exception:  # pragma: no cover - optional dependency
        return None

    @njit(cache=True)
    def _humlicek(x, y):
        yq = y * y
        yrrtpi = y * _RRTPI
        abx = abs(x)
        xq = abx * abx
        if y >= 70.55:
            return yrrtpi / (xq + yq)

        xlim0 = (15100.0 + y * (40.0 - y * 3.6)) ** 0.5
        xlim1 = 0.0
        if y < 8.425:
            xlim1 = (164.0 - y * (4.3 + y * 1.8)) ** 0.5
        xlim2 = 6.8 - y
        xlim3 = 2.4 * y
        xlim4 = 18.1 * y + 1.65
        if y <= 1.0e-6:
            xlim1 = xlim0
            xlim2 = xlim0

        if abx >= xlim0:
            return yrrtpi / (xq + yq)
        if abx >= xlim1:
            a0 = yq + 0.5
            d = _RRTPI / (a0 * a0 + xq * (yq + yq - 1.0 + xq))
            return d * y * (a0 + xq)
        if abx > xlim2:
            h0 = 0.5625 + yq * (4.5 + yq * (10.5 + yq * (6.0 + yq)))
            h2 = -4.5 + yq * (9.0 + yq * (6.0 + yq * 4.0))
            h4 = 10.5 - yq * (6.0 - yq * 6.0)
            h6 = -6.0 + yq * 4.0
            e0 = 1.875 + yq * (8.25 + yq * (5.5 + yq))
            e2 = 5.25 + yq * (1.0 + yq * 3.0)
            e4 = 0.75 * h6
            d = _RRTPI / (h0 + xq * (h2 + xq * (h4 + xq * (h6 + xq))))
            return d * y * (e0 + xq * (e2 + xq * (e4 + xq)))
        if abx < xlim3:
            z0 = 272.1014 + y * (
                1280.829
                + y
                * (
                    2802.870
                    + y
                    * (
                        3764.966
                        + y
                        * (
                            3447.629
                            + y
                            * (
                                2256.981
                                + y
                                * (1074.409 + y * (369.1989 + y * (88.26741 + y * (13.39880 + y))))
                            )
                        )
                    )
                )
            )
            z2 = 211.678 + y * (
                902.3066
                + y
                * (
                    1758.336
                    + y
                    * (
                        2037.310
                        + y
                        * (1549.675 + y * (793.4273 + y * (266.2987 + y * (53.59518 + y * 5.0))))
                    )
                )
            )
            z4 = 78.86585 + y * (
                308.1852
                + y * (497.3014 + y * (479.2576 + y * (269.2916 + y * (80.39278 + y * 10.0))))
            )
            z6 = 22.03523 + y * (55.02933 + y * (92.75679 + y * (53.59518 + y * 10.0)))
            z8 = 1.496460 + y * (13.39880 + y * 5.0)
            p0 = 153.5168 + y * (
                549.3954
                + y
                * (
                    919.4955
                    + y
                    * (
                        946.8970
                        + y
                        * (
                            662.8097
                            + y
                            * (
                                328.2151
                                + y * (115.3772 + y * (27.93941 + y * (4.264678 + y * 0.3183291)))
                            )
                        )
                    )
                )
            )
            p2 = -34.16955 + y * (
                -1.322256
                + y
                * (
                    124.5975
                    + y
                    * (189.7730 + y * (139.4665 + y * (56.81652 + y * (12.79458 + y * 1.2733163))))
                )
            )
            p4 = 2.584042 + y * (
                10.46332 + y * (24.01655 + y * (29.81482 + y * (12.79568 + y * 1.9099744)))
            )
            p6 = -0.07272979 + y * (0.9377051 + y * (4.266322 + y * 1.273316))
            p8 = 0.0005480304 + y * 0.3183291
            d = 1.7724538 / (z0 + xq * (z2 + xq * (z4 + xq * (z6 + xq * (z8 + xq)))))
            return d * (p0 + xq * (p2 + xq * (p4 + xq * (p6 + xq * p8))))

        ypy0 = y + _HUMLIK_Y0
        ypy0q = ypy0 * ypy0
        acc = 0.0
        near = abx <= xlim4
        for j in range(6):
            dm = x - _HUMLIK_T[j]
            mq = dm * dm
            mf = 1.0 / (mq + ypy0q)
            xm = mf * dm
            ym = mf * ypy0

            dp = x + _HUMLIK_T[j]
            pq = dp * dp
            pf = 1.0 / (pq + ypy0q)
            xp = pf * dp
            yp = pf * ypy0

            if near:
                acc += _HUMLIK_C[j] * (ym + yp) - _HUMLIK_S[j] * (xm - xp)
            else:
                yf = y + _HUMLIK_Y0PY0
                acc += (_HUMLIK_C[j] * (mq * mf - _HUMLIK_Y0 * ym) + _HUMLIK_S[j] * yf * xm) / (
                    mq + _HUMLIK_Y0Q
                ) + (_HUMLIK_C[j] * (pq * pf - _HUMLIK_Y0 * yp) - _HUMLIK_S[j] * yf * xp) / (
                    pq + _HUMLIK_Y0Q
                )
        if not near:
            acc = y * acc + np.exp(-xq)
        return acc

    @njit(parallel=True, cache=True)
    def _accumulate(
        nspec,
        voigt_grid,
        nvoigt,
        center,
        strength,
        air_half_width,
        lower_state_energy,
        temperature_exponent,
        pressure_shift,
        molecular_mass_amu,
        q296,
        q,
        pressure_atm,
        temperature_k,
    ):
        nlines = center.size
        nlevel = pressure_atm.size
        npoints = voigt_grid.size
        spec = np.zeros((nspec, nlevel))
        for level in prange(nlevel):
            temp = temperature_k[level]
            press = pressure_atm[level]
            rt0t = _T0 / temp
            rc2t = _C2 / temp
            rc2t0 = _C2 / _T0
            for i in range(nlines):
                sigma = center[i] + pressure_shift[i] * press
                insert = np.searchsorted(voigt_grid, sigma)
                idx = insert - 1
                if idx >= 0 and insert < npoints:
                    nvlo = max(1, insert - nvoigt)
                    nvhi = min(npoints, insert + nvoigt)
                elif (
                    sigma < voigt_grid[0] - _VOIGT_EXTRA
                    or sigma > voigt_grid[npoints - 1] + _VOIGT_EXTRA
                ):
                    nvlo = 0
                    nvhi = 0
                elif sigma < voigt_grid[0]:
                    nvlo = 1
                    nvhi = nvoigt
                else:
                    nvlo = npoints - nvoigt + 1
                    nvhi = npoints
                if nvhi < nvlo:
                    continue

                vg = 4.30140e-7 * sigma * (temp / molecular_mass_amu[i]) ** 0.5
                voigta = press * air_half_width[i] * rt0t ** temperature_exponent[i] / vg
                ratio1 = np.exp(-lower_state_energy[i] * rc2t) - np.exp(
                    -(sigma + lower_state_energy[i]) * rc2t
                )
                ratio2 = np.exp(-lower_state_energy[i] * rc2t0) - np.exp(
                    -(sigma + lower_state_energy[i]) * rc2t0
                )
                vnorm = ratio1 / ratio2 * q296[i] / q[level, i] * strength[i] / vg * _INV_SQRT_PI
                for pos_index in range(nvlo - 1, nvhi):
                    spec_index = pos_index - nvoigt
                    if 0 <= spec_index < nspec:
                        x = (voigt_grid[pos_index] - sigma) / vg
                        spec[spec_index, level] += vnorm * _humlicek(x, voigta)
        return spec

    return _accumulate
