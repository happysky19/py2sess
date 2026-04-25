"""Thermal BRDF kernel coefficient and emissivity generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


LAMBERTIAN_IDX = 1
ROSSTHIN_IDX = 2
ROSSTHICK_IDX = 3


@dataclass(frozen=True)
class ThermalBrdfResult:
    """Thermal BRDF coefficient bundle plus surface emissivity."""

    brdf_f: float
    ubrdf_f: np.ndarray
    emissivity: float


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


def thermal_brdf_from_kernels(
    *,
    kernel_specs: list[dict[str, Any]],
    user_angles: np.ndarray,
    do_surface_emission: bool,
) -> ThermalBrdfResult:
    """Generates thermal BRDF and emissivity coefficients.

    Parameters
    ----------
    kernel_specs
        Sequence of Fortran-style kernel specifications. Each item must
        provide ``which_brdf`` and may provide ``factor``, ``stream_value``,
        and ``nstreams_brdf``.
    user_angles
        Viewing zenith angles in degrees for user-stream reflection terms.
    do_surface_emission
        Whether to compute the non-Lambertian reflected-emission correction
        and return the corresponding emissivity.

    Returns
    -------
    ThermalBrdfResult
        Thermal BRDF quadrature coefficient, user-stream BRDF coefficients,
        and surface emissivity.
    """
    stream_value = (
        float(kernel_specs[0].get("stream_value", 1.0 / math.sqrt(3.0)))
        if kernel_specs
        else 1.0 / math.sqrt(3.0)
    )
    nstreams_brdf = int(kernel_specs[0].get("nstreams_brdf", 4)) if kernel_specs else 4
    if nstreams_brdf % 2 != 0:
        raise ValueError("nstreams_brdf must be even")
    n_half = nstreams_brdf // 2
    x_half, a_half = _gauleg_unit(n_half)
    x_brdf = np.zeros(nstreams_brdf, dtype=float)
    a_brdf = np.zeros(nstreams_brdf, dtype=float)
    cxe_brdf = np.zeros(n_half, dtype=float)
    sxe_brdf = np.zeros(n_half, dtype=float)
    for i in range(n_half):
        i1 = i + n_half
        x_brdf[i] = x_half[i]
        x_brdf[i1] = -x_half[i]
        a_brdf[i] = a_half[i]
        a_brdf[i1] = a_half[i]
        cxe_brdf[i] = x_half[i]
        sxe_brdf[i] = math.sqrt(max(0.0, 1.0 - x_half[i] * x_half[i]))
    phi = math.pi * x_brdf
    cphi = np.cos(phi)
    bax_brdf = phi[:n_half] * a_brdf[:n_half] / math.pi

    stream_sine = math.sqrt(max(0.0, 1.0 - stream_value * stream_value))
    user_streams = np.cos(np.deg2rad(user_angles))
    user_sines = np.sqrt(np.clip(1.0 - user_streams * user_streams, 0.0, None))

    brdf_f = 0.0
    ubrdf_f = np.zeros(user_angles.size, dtype=float)
    emissivity = 1.0

    for spec in kernel_specs:
        which_brdf = int(spec["which_brdf"])
        factor = float(spec.get("factor", 1.0))
        local_brdf_f = 0.0
        local_ubrdf_f = np.zeros(user_angles.size, dtype=float)
        local_emissivity = factor if which_brdf == LAMBERTIAN_IDX and do_surface_emission else 0.0
        if which_brdf == LAMBERTIAN_IDX:
            local_brdf_f = 1.0
            local_ubrdf_f.fill(1.0)
        elif which_brdf in {ROSSTHIN_IDX, ROSSTHICK_IDX}:
            thick = which_brdf == ROSSTHICK_IDX
            brdfunc = np.zeros(nstreams_brdf, dtype=float)
            user_brdfunc = np.zeros((nstreams_brdf, user_angles.size), dtype=float)
            ebrdfunc = np.zeros((n_half, nstreams_brdf), dtype=float)
            for k in range(nstreams_brdf):
                brdfunc[k] = _ross_kernel(
                    xi=stream_value,
                    sxi=stream_sine,
                    xj=stream_value,
                    sxj=stream_sine,
                    cphi=cphi[k],
                    thick=thick,
                )
                for ui in range(user_angles.size):
                    user_brdfunc[k, ui] = _ross_kernel(
                        xi=user_streams[ui],
                        sxi=user_sines[ui],
                        xj=stream_value,
                        sxj=stream_sine,
                        cphi=cphi[k],
                        thick=thick,
                    )
                if do_surface_emission:
                    for ke in range(n_half):
                        ebrdfunc[ke, k] = _ross_kernel(
                            xi=stream_value,
                            sxi=stream_sine,
                            xj=cxe_brdf[ke],
                            sxj=sxe_brdf[ke],
                            cphi=cphi[k],
                            thick=thick,
                        )
            local_brdf_f = 0.5 * float(np.dot(brdfunc, a_brdf))
            for ui in range(user_angles.size):
                local_ubrdf_f[ui] = 0.5 * float(np.dot(user_brdfunc[:, ui], a_brdf))
            if do_surface_emission:
                refl = 0.0
                for k in range(nstreams_brdf):
                    refl += a_brdf[k] * float(np.dot(ebrdfunc[:, k], bax_brdf))
                local_emissivity = refl * factor
        else:
            raise NotImplementedError(
                "thermal BRDF kernel generation currently supports Lambertian, RossThin, and RossThick only"
            )
        brdf_f += factor * local_brdf_f
        ubrdf_f += factor * local_ubrdf_f
        if do_surface_emission:
            emissivity -= local_emissivity

    return ThermalBrdfResult(
        brdf_f=brdf_f,
        ubrdf_f=ubrdf_f,
        emissivity=emissivity,
    )
