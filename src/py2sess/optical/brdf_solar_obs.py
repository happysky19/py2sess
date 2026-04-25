"""Solar observation-geometry BRDF kernel coefficient generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


LAMBERTIAN_IDX = 1
ROSSTHIN_IDX = 2
ROSSTHICK_IDX = 3


@dataclass(frozen=True)
class SolarObsBrdfResult:
    """BRDF Fourier coefficients for solar observation geometry."""

    brdf_f_0: np.ndarray
    brdf_f: np.ndarray
    ubrdf_f: np.ndarray


def _gauleg_unit(n_half: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n_half)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def _ross_kernel(
    *, xi: float, sxi: float, xj: float, sxj: float, cphi: float, thick: bool
) -> float:
    pie = math.acos(-1.0)
    ds1 = xi * xj
    ds2 = sxi * sxj
    ds3 = xi + xj
    cksi = ds1 - ds2 * cphi
    cksi = min(1.0, cksi)
    sksi = math.sqrt(max(0.0, 1.0 - cksi * cksi))
    ksi = math.acos(cksi)
    if thick:
        func = ((0.5 * pie - ksi) * cksi + sksi) / ds3
        return func - 0.25 * pie
    func = ((0.5 * pie - ksi) * cksi + sksi) / ds1
    return func - 0.5 * pie


def solar_obs_brdf_from_kernels(
    *,
    kernel_specs: list[dict[str, Any]],
    user_obsgeoms: np.ndarray | None = None,
    stream_value: float = 1.0 / math.sqrt(3.0),
    n_geoms: int,
) -> SolarObsBrdfResult:
    """Generates solar observation-geometry BRDF Fourier coefficients.

    Parameters
    ----------
    kernel_specs
        Sequence of Fortran-style kernel specifications. Each item must
        provide ``which_brdf`` and may provide ``factor`` and
        ``nstreams_brdf``.
    user_obsgeoms
        Observation geometries with columns ``(sza, vza, azimuth)`` in
        degrees.
    stream_value
        Two-stream quadrature cosine used for quadrature-stream terms.
    n_geoms
        Number of observation geometries to generate.

    Returns
    -------
    SolarObsBrdfResult
        Fourier-0 and Fourier-1 BRDF coefficients for beam, quadrature, and
        user-stream reflection terms.
    """
    brdf_f_0 = np.zeros((n_geoms, 2), dtype=float)
    brdf_f = np.zeros(2, dtype=float)
    ubrdf_f = np.zeros((n_geoms, 2), dtype=float)

    if user_obsgeoms is None:
        user_obsgeoms = np.zeros((n_geoms, 3), dtype=float)
    user_obsgeoms = np.asarray(user_obsgeoms, dtype=float)
    if user_obsgeoms.shape != (n_geoms, 3):
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")

    nstreams_brdf = int(kernel_specs[0].get("nstreams_brdf", 4)) if kernel_specs else 4
    if nstreams_brdf % 2 != 0:
        raise ValueError("nstreams_brdf must be even")
    n_half = nstreams_brdf // 2
    x_half, a_half = _gauleg_unit(n_half)
    x_brdf = np.zeros(nstreams_brdf, dtype=float)
    a_brdf = np.zeros(nstreams_brdf, dtype=float)
    for i in range(n_half):
        i1 = i + n_half
        x_brdf[i] = x_half[i]
        x_brdf[i1] = -x_half[i]
        a_brdf[i] = a_half[i]
        a_brdf[i1] = a_half[i]
    phi = math.pi * x_brdf
    cphi = np.cos(phi)

    stream_sine = math.sqrt(max(0.0, 1.0 - stream_value * stream_value))
    sza_cos = np.cos(np.deg2rad(user_obsgeoms[:, 0]))
    sza_sin = np.sqrt(np.clip(1.0 - sza_cos * sza_cos, 0.0, None))
    user_streams = np.cos(np.deg2rad(user_obsgeoms[:, 1]))
    user_sines = np.sqrt(np.clip(1.0 - user_streams * user_streams, 0.0, None))

    for spec in kernel_specs:
        which_brdf = int(spec["which_brdf"])
        factor = float(spec.get("factor", 1.0))
        if which_brdf == LAMBERTIAN_IDX:
            brdf_f_0[:, 0] += factor
            brdf_f[0] += factor
            ubrdf_f[:, 0] += factor
            continue
        if which_brdf not in {ROSSTHIN_IDX, ROSSTHICK_IDX}:
            raise NotImplementedError(
                "solar observational BRDF kernel generation currently supports "
                "Lambertian, RossThin, and RossThick only"
            )
        thick = which_brdf == ROSSTHICK_IDX
        brdfunc = np.zeros(nstreams_brdf, dtype=float)
        brdfunc_0 = np.zeros((nstreams_brdf, n_geoms), dtype=float)
        user_brdfunc = np.zeros((nstreams_brdf, n_geoms), dtype=float)
        for k in range(nstreams_brdf):
            brdfunc[k] = _ross_kernel(
                xi=stream_value,
                sxi=stream_sine,
                xj=stream_value,
                sxj=stream_sine,
                cphi=float(cphi[k]),
                thick=thick,
            )
            for ig in range(n_geoms):
                brdfunc_0[k, ig] = _ross_kernel(
                    xi=stream_value,
                    sxi=stream_sine,
                    xj=float(sza_cos[ig]),
                    sxj=float(sza_sin[ig]),
                    cphi=float(cphi[k]),
                    thick=thick,
                )
                user_brdfunc[k, ig] = _ross_kernel(
                    xi=float(user_streams[ig]),
                    sxi=float(user_sines[ig]),
                    xj=stream_value,
                    sxj=stream_sine,
                    cphi=float(cphi[k]),
                    thick=thick,
                )
        for m in (0, 1):
            delfac = 1.0 if m == 0 else 2.0
            azmfac = a_brdf if m == 0 else a_brdf * np.cos(m * phi)
            helpv = 0.5 * delfac
            brdf_f[m] += factor * helpv * float(np.dot(brdfunc, azmfac))
            for ig in range(n_geoms):
                brdf_f_0[ig, m] += factor * helpv * float(np.dot(brdfunc_0[:, ig], azmfac))
                ubrdf_f[ig, m] += factor * helpv * float(np.dot(user_brdfunc[:, ig], azmfac))

    return SolarObsBrdfResult(
        brdf_f_0=brdf_f_0,
        brdf_f=brdf_f,
        ubrdf_f=ubrdf_f,
    )
