"""Solar observation-geometry BRDF kernel coefficient generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


LAMBERTIAN_IDX = 1
ROSSTHIN_IDX = 2
ROSSTHICK_IDX = 3
RPV_IDX = 4
COXMUNK_IDX = 5
DISORT_HAPKE_IDX = 6


@dataclass(frozen=True)
class SolarObsBrdfResult:
    """BRDF Fourier coefficients for solar observation geometry."""

    brdf_f_0: np.ndarray
    brdf_f: np.ndarray
    ubrdf_f: np.ndarray
    direct_brf: np.ndarray


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
    cksi = max(-1.0, min(1.0, cksi))
    sksi = math.sqrt(max(0.0, 1.0 - cksi * cksi))
    ksi = math.acos(cksi)
    if thick:
        func = ((0.5 * pie - ksi) * cksi + sksi) / ds3
        return func - 0.25 * pie
    func = ((0.5 * pie - ksi) * cksi + sksi) / ds1
    return func - 0.5 * pie


def _rpv_kernel(
    *,
    mu_i: float,
    mu_r: float,
    cphi: float,
    hotspot: float,
    asymmetry: float,
    anisotropy: float,
    normalization: float,
) -> float:
    if mu_i <= 0.0 or mu_r <= 0.0:
        return 0.0
    if normalization <= 0.0:
        raise ValueError("RPV normalization must be positive")
    sin_i = math.sqrt(max(0.0, 1.0 - mu_i * mu_i))
    sin_r = math.sqrt(max(0.0, 1.0 - mu_r * mu_r))
    cos_g = max(-1.0, min(1.0, mu_i * mu_r + sin_i * sin_r * cphi))
    minnaert = (mu_i * mu_r) ** (anisotropy - 1.0) / (mu_i + mu_r) ** (1.0 - anisotropy)
    phase = (1.0 - asymmetry * asymmetry) / (
        1.0 + asymmetry * asymmetry + 2.0 * asymmetry * cos_g
    ) ** 1.5
    tan_i = sin_i / mu_i
    tan_r = sin_r / mu_r
    geom = math.sqrt(max(0.0, tan_i * tan_i + tan_r * tan_r - 2.0 * tan_i * tan_r * cphi))
    hotspot_term = 1.0 + (1.0 - hotspot) / (1.0 + geom)
    return minnaert * phase * hotspot_term / normalization


def _coxmunk_kernel(
    *,
    mu_i: float,
    mu_r: float,
    cphi: float,
    wind_speed: float,
    refractive_index: float,
    shadow: bool = False,
) -> float:
    """Scalar Cox-Munk ocean-glint kernel following the RtRetrieval/LIDORT form."""
    if mu_i <= 0.0 or mu_r <= 0.0:
        return 0.0
    if wind_speed < 0.0:
        raise ValueError("Cox-Munk wind_speed must be nonnegative")
    if refractive_index <= 1.0:
        raise ValueError("Cox-Munk refractive_index must be greater than 1")

    sin_i = math.sqrt(max(0.0, 1.0 - mu_i * mu_i))
    sin_r = math.sqrt(max(0.0, 1.0 - mu_r * mu_r))
    ckphi = -max(-1.0, min(1.0, cphi))
    z = max(-1.0, min(1.0, mu_i * mu_r + sin_i * sin_r * ckphi))
    half_scatter_cosine = math.cos(0.5 * math.acos(z))
    if half_scatter_cosine <= 0.0:
        return 0.0

    refractive_index_sq = refractive_index * refractive_index
    h1 = refractive_index_sq * half_scatter_cosine
    h2_sq = refractive_index_sq + half_scatter_cosine * half_scatter_cosine - 1.0
    if h2_sq <= 0.0:
        return 0.0
    h2 = math.sqrt(h2_sq)
    rp = (h1 - h2) / (h1 + h2)
    rl = (half_scatter_cosine - h2) / (half_scatter_cosine + h2)
    fresnel = 0.5 * (rp * rp + rl * rl)

    slope_variance = 0.003 + 0.00512 * wind_speed
    b = min(1.0, (mu_i + mu_r) / (2.0 * half_scatter_cosine))
    if b <= 0.0:
        return 0.0
    tan_alpha = math.tan(0.5 * math.pi - math.asin(b))
    argument = tan_alpha * tan_alpha / slope_variance
    if argument >= 88.0:
        return 0.0
    kernel = fresnel * (math.exp(-argument) / slope_variance) * (0.25 / mu_i / (b**4)) / mu_r
    if not shadow:
        return kernel

    def shadow_term(mu: float) -> float:
        cot = mu / math.sqrt(max(1.0e-30, 1.0 - mu * mu))
        s1 = math.sqrt(slope_variance / math.pi)
        inv_sigma = 1.0 / math.sqrt(slope_variance)
        return 0.5 * (s1 * math.exp(-((cot * inv_sigma) ** 2)) / cot - math.erfc(cot * inv_sigma))

    return kernel / (1.0 + shadow_term(mu_i) + shadow_term(mu_r))


def _disort_hapke_kernel(*, mu_i: float, mu_r: float, cphi: float) -> float:
    if mu_i <= 0.0 or mu_r <= 0.0:
        return 0.0
    sin_i = math.sqrt(max(0.0, 1.0 - mu_i * mu_i))
    sin_r = math.sqrt(max(0.0, 1.0 - mu_r * mu_r))
    ctheta = max(-1.0, min(1.0, mu_r * mu_i + sin_r * sin_i * cphi))
    theta = math.acos(ctheta)
    phase = 1.0 + 0.5 * ctheta
    hotspot_width = 0.06
    opposition = hotspot_width / (hotspot_width + math.tan(0.5 * theta))
    single_scatter_albedo = 0.6
    gamma = math.sqrt(1.0 - single_scatter_albedo)
    h_i = (1.0 + 2.0 * mu_i) / (1.0 + 2.0 * gamma * mu_i)
    h_r = (1.0 + 2.0 * mu_r) / (1.0 + 2.0 * gamma * mu_r)
    return (
        0.25
        * single_scatter_albedo
        * ((1.0 + opposition) * phase + h_i * h_r - 1.0)
        / (mu_r + mu_i)
    )


def coxmunk_giss_stokes_direct_kernel(
    *,
    sza_deg: float,
    vza_deg: float,
    relative_azimuth_deg: float,
    wind_speed: float,
    refractive_index: float,
) -> np.ndarray:
    """Return the GISS Cox-Munk direct-reflection ``I,Q,U`` kernel terms."""
    if wind_speed < 0.0:
        raise ValueError("Cox-Munk wind_speed must be nonnegative")
    if refractive_index <= 1.0:
        raise ValueError("Cox-Munk refractive_index must be greater than 1")
    xi = math.cos(math.radians(float(sza_deg)))
    xj = math.cos(math.radians(float(vza_deg)))
    if xi <= 0.0 or xj <= 0.0:
        return np.zeros(3, dtype=float)
    sxi = math.sqrt(max(0.0, 1.0 - xi * xi))
    sxj = math.sqrt(max(0.0, 1.0 - xj * xj))
    ckphi = math.cos(math.radians(float(relative_azimuth_deg)))
    skphi = math.sin(math.radians(float(relative_azimuth_deg)))
    sigma2 = 0.5 * (0.003 + 0.00512 * float(wind_speed))

    vi1, vi2, vi3 = sxi, 0.0, -xi
    vr1, vr2, vr3 = sxj * ckphi, sxj * skphi, xj
    unit1 = vi1 - vr1
    unit2 = vi2 - vr2
    unit3 = vi3 - vr3
    fact1 = unit1 * unit1 + unit2 * unit2 + unit3 * unit3
    if fact1 <= 0.0:
        return np.zeros(3, dtype=float)
    factor = math.sqrt(1.0 / fact1)

    xi1 = factor * (unit1 * vi1 + unit2 * vi2 + unit3 * vi3)
    cxi2 = 1.0 - (1.0 - xi1 * xi1) / (refractive_index * refractive_index)
    if cxi2 <= 0.0:
        return np.zeros(3, dtype=float)
    cxi2 = math.sqrt(cxi2)
    crper = (xi1 - refractive_index * cxi2) / (xi1 + refractive_index * cxi2)
    crpar = (refractive_index * xi1 - cxi2) / (refractive_index * xi1 + cxi2)

    ti1, ti2, ti3 = -xi, 0.0, -sxi
    tr1, tr2, tr3 = xj * ckphi, xj * skphi, -sxj
    pi1, pi2, pi3 = 0.0, 1.0, 0.0
    pr1, pr2, pr3 = -skphi, ckphi, 0.0

    pikr = pi1 * vr1 + pi2 * vr2 + pi3 * vr3
    prki = pr1 * vi1 + pr2 * vi2 + pr3 * vi3
    tikr = ti1 * vr1 + ti2 * vr2 + ti3 * vr3
    trki = tr1 * vi1 + tr2 * vi2 + tr3 * vi3

    e1 = pikr * prki
    e2 = tikr * trki
    e3 = tikr * prki
    e4 = pikr * trki
    cf11 = e1 * crper + e2 * crpar
    cf12 = -e3 * crper + e4 * crpar
    cf21 = -e4 * crper + e3 * crpar
    cf22 = e2 * crper + e1 * crpar

    vp1 = vi2 * vr3 - vi3 * vr2
    vp2 = vi3 * vr1 - vi1 * vr3
    vp3 = vi1 * vr2 - vi2 * vr1
    dmod = (vp1 * vp1 + vp2 * vp2 + vp3 * vp3) ** 2
    if abs(dmod) < 1.0e-8:
        cf11 = crpar
        cf22 = crper
        dmod = 1.0

    rdz2 = unit3 * unit3
    if rdz2 <= 0.0:
        return np.zeros(3, dtype=float)
    argument = (unit1 * unit1 + unit2 * unit2) / (2.0 * sigma2 * rdz2)
    dex = 0.0 if argument > 88.0 else math.exp(-argument)
    dcoeff = 1.0 / (8.0 * xi * xj * dmod * rdz2 * rdz2 * sigma2) * fact1 * fact1 * dex

    af = 0.5 * dcoeff
    af11 = abs(cf11) ** 2
    af12 = abs(cf12) ** 2
    af21 = abs(cf21) ** 2
    af22 = abs(cf22) ** 2
    return np.array(
        [
            (af11 + af12 + af21 + af22) * af,
            (af11 - af22 + af12 - af21) * af,
            -(cf11 * cf21 + cf12 * cf22) * dcoeff,
        ],
        dtype=float,
    )


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
    direct_brf = np.zeros(n_geoms, dtype=float)

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
            direct_brf += factor
            continue
        if which_brdf not in {
            ROSSTHIN_IDX,
            ROSSTHICK_IDX,
            RPV_IDX,
            COXMUNK_IDX,
            DISORT_HAPKE_IDX,
        }:
            raise NotImplementedError(
                "solar observational BRDF kernel generation currently supports "
                "Lambertian, RossThin, RossThick, RPV, Cox-Munk, and DISORT-Hapke only"
            )
        brdfunc = np.zeros(nstreams_brdf, dtype=float)
        brdfunc_0 = np.zeros((nstreams_brdf, n_geoms), dtype=float)
        user_brdfunc = np.zeros((nstreams_brdf, n_geoms), dtype=float)
        if which_brdf == DISORT_HAPKE_IDX:
            for k in range(nstreams_brdf):
                brdfunc[k] = _disort_hapke_kernel(
                    mu_i=stream_value,
                    mu_r=stream_value,
                    cphi=float(cphi[k]),
                )
                for ig in range(n_geoms):
                    brdfunc_0[k, ig] = _disort_hapke_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=stream_value,
                        cphi=float(cphi[k]),
                    )
                    user_brdfunc[k, ig] = _disort_hapke_kernel(
                        mu_i=stream_value,
                        mu_r=float(user_streams[ig]),
                        cphi=float(cphi[k]),
                    )
            direct_brf += factor * np.array(
                [
                    _disort_hapke_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=float(user_streams[ig]),
                        cphi=math.cos(math.radians(float(user_obsgeoms[ig, 2]))),
                    )
                    for ig in range(n_geoms)
                ],
                dtype=float,
            )
        elif which_brdf == RPV_IDX:
            hotspot = float(spec["hotspot"])
            asymmetry = float(spec["asymmetry"])
            anisotropy = float(spec["anisotropy"])
            normalization = float(spec.get("normalization", 20.0))
            for k in range(nstreams_brdf):
                brdfunc[k] = _rpv_kernel(
                    mu_i=stream_value,
                    mu_r=stream_value,
                    cphi=float(cphi[k]),
                    hotspot=hotspot,
                    asymmetry=asymmetry,
                    anisotropy=anisotropy,
                    normalization=normalization,
                )
                for ig in range(n_geoms):
                    brdfunc_0[k, ig] = _rpv_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=stream_value,
                        cphi=float(cphi[k]),
                        hotspot=hotspot,
                        asymmetry=asymmetry,
                        anisotropy=anisotropy,
                        normalization=normalization,
                    )
                    user_brdfunc[k, ig] = _rpv_kernel(
                        mu_i=stream_value,
                        mu_r=float(user_streams[ig]),
                        cphi=float(cphi[k]),
                        hotspot=hotspot,
                        asymmetry=asymmetry,
                        anisotropy=anisotropy,
                        normalization=normalization,
                    )
            direct_brf += factor * np.array(
                [
                    _rpv_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=float(user_streams[ig]),
                        cphi=math.cos(math.radians(float(user_obsgeoms[ig, 2]))),
                        hotspot=hotspot,
                        asymmetry=asymmetry,
                        anisotropy=anisotropy,
                        normalization=normalization,
                    )
                    for ig in range(n_geoms)
                ],
                dtype=float,
            )
        elif which_brdf == COXMUNK_IDX:
            wind_speed = float(spec["wind_speed"])
            refractive_index = float(spec["refractive_index"])
            shadow = bool(spec.get("shadow", False))
            for k in range(nstreams_brdf):
                brdfunc[k] = _coxmunk_kernel(
                    mu_i=stream_value,
                    mu_r=stream_value,
                    cphi=float(cphi[k]),
                    wind_speed=wind_speed,
                    refractive_index=refractive_index,
                    shadow=shadow,
                )
                for ig in range(n_geoms):
                    brdfunc_0[k, ig] = _coxmunk_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=stream_value,
                        cphi=float(cphi[k]),
                        wind_speed=wind_speed,
                        refractive_index=refractive_index,
                        shadow=shadow,
                    )
                    user_brdfunc[k, ig] = _coxmunk_kernel(
                        mu_i=stream_value,
                        mu_r=float(user_streams[ig]),
                        cphi=float(cphi[k]),
                        wind_speed=wind_speed,
                        refractive_index=refractive_index,
                        shadow=shadow,
                    )
            direct_brf += factor * np.array(
                [
                    _coxmunk_kernel(
                        mu_i=float(sza_cos[ig]),
                        mu_r=float(user_streams[ig]),
                        cphi=math.cos(math.radians(float(user_obsgeoms[ig, 2]))),
                        wind_speed=wind_speed,
                        refractive_index=refractive_index,
                        shadow=shadow,
                    )
                    for ig in range(n_geoms)
                ],
                dtype=float,
            )
        else:
            thick = which_brdf == ROSSTHICK_IDX
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
            direct_brf += factor * np.array(
                [
                    _ross_kernel(
                        xi=float(user_streams[ig]),
                        sxi=float(user_sines[ig]),
                        xj=float(sza_cos[ig]),
                        sxj=float(sza_sin[ig]),
                        cphi=math.cos(math.radians(float(user_obsgeoms[ig, 2]))),
                        thick=thick,
                    )
                    for ig in range(n_geoms)
                ],
                dtype=float,
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
        direct_brf=direct_brf,
    )
