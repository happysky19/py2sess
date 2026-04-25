"""First-order solar observation-geometry forward solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .lattice_result import add_lattice_axes, lattice_shape, reshape_lattice_array
from ..optical.delta_m import (
    default_delta_m_truncation_factor,
    validate_delta_m_truncation_factor,
)
from .preprocess import PreparedInputs


@dataclass(frozen=True)
class FoSolarObsResult:
    """FO solar outputs for one or more observation geometries."""

    intensity_total: np.ndarray
    intensity_ss: np.ndarray
    intensity_db: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    cosscat: np.ndarray
    do_nadir: np.ndarray
    intensity_total_profile: np.ndarray | None = None
    intensity_ss_profile: np.ndarray | None = None
    intensity_db_profile: np.ndarray | None = None
    lattice_counts: tuple[int, int, int] | None = None
    lattice_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def radiance(self) -> np.ndarray:
        """Preferred public total FO radiance output."""
        return self.intensity_total

    @property
    def radiance_total(self) -> np.ndarray:
        """Total FO radiance output."""
        return self.intensity_total

    @property
    def radiance_ss(self) -> np.ndarray:
        """Single-scatter FO radiance component."""
        return self.intensity_ss

    @property
    def radiance_db(self) -> np.ndarray:
        """Direct-beam FO radiance component."""
        return self.intensity_db

    @property
    def radiance_profile(self) -> np.ndarray | None:
        """Total FO radiance profile when level output is available."""
        return self.intensity_total_profile

    def _lattice_shape(self) -> tuple[int, int, int]:
        """Returns the expected lattice shape for reshaping helpers."""
        return lattice_shape(self.lattice_counts)

    def _reshape_lattice_array(self, values: np.ndarray) -> np.ndarray:
        """Reshapes a 1D geometry array back to lattice form."""
        return reshape_lattice_array(values, self.lattice_counts)

    def _reshape_lattice_fields(self) -> dict[str, np.ndarray]:
        """Builds the reshaped lattice dictionary without axes."""
        return {
            "intensity_total": self.intensity_total_lattice(),
            "intensity_ss": self.intensity_ss_lattice(),
            "intensity_db": self.intensity_db_lattice(),
            "mu0": self.mu0_lattice(),
            "mu1": self.mu1_lattice(),
            "cosscat": self.cosscat_lattice(),
            "do_nadir": self.do_nadir_lattice(),
        }

    def reshape_lattice(self) -> dict[str, Any]:
        """Returns all FO outputs reshaped to the original lattice grid."""
        return add_lattice_axes(self._reshape_lattice_fields(), self.lattice_axes)

    def intensity_total_lattice(self) -> np.ndarray:
        """Returns total FO intensity in lattice shape."""
        return self._reshape_lattice_array(self.intensity_total)

    def intensity_ss_lattice(self) -> np.ndarray:
        """Returns single-scatter FO intensity in lattice shape."""
        return self._reshape_lattice_array(self.intensity_ss)

    def intensity_db_lattice(self) -> np.ndarray:
        """Returns direct-beam FO intensity in lattice shape."""
        return self._reshape_lattice_array(self.intensity_db)

    def mu0_lattice(self) -> np.ndarray:
        """Returns solar zenith cosine in lattice shape."""
        return self._reshape_lattice_array(self.mu0)

    def mu1_lattice(self) -> np.ndarray:
        """Returns view zenith cosine in lattice shape."""
        return self._reshape_lattice_array(self.mu1)

    def cosscat_lattice(self) -> np.ndarray:
        """Returns scattering-angle cosine in lattice shape."""
        return self._reshape_lattice_array(self.cosscat)

    def do_nadir_lattice(self) -> np.ndarray:
        """Returns the nadir-view mask in lattice shape."""
        return self._reshape_lattice_array(self.do_nadir)


@lru_cache(maxsize=None)
def _gauss_legendre_unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns cached Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def _gauss_legendre_interval(x1: float, x2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns Gauss-Legendre quadrature nodes and weights on an interval."""
    nodes, weights = _gauss_legendre_unit(n)
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    return xm + xl * nodes, xl * weights


def _find_sunpaths_direct(
    *,
    do_zero_sun_boa: bool,
    radstart: float,
    radii: np.ndarray,
    theta_start: float,
    sin_theta_start: float,
    n: int,
) -> np.ndarray:
    """Computes direct sun paths when the line of sight does not graze tangent."""
    sunpaths = np.zeros(radii.size - 1, dtype=float)
    n1 = n - 1
    if do_zero_sun_boa:
        radiik = radii[n1]
        sunpaths[n - 1] = radiik - radstart
        for k in range(n1, 0, -1):
            radiik1 = radii[k - 1]
            sunpaths[k - 1] = radiik1 - radiik
            radiik = radiik1
        return sunpaths

    radiik = radii[n1]
    sth0 = sin_theta_start
    th0 = theta_start
    sth1 = sth0 * radstart / radiik
    th1 = math.asin(sth1)
    ks1 = th0 - th1
    sunpaths[n - 1] = math.sin(ks1) * radstart / sth1

    sth0 = sth1
    th0 = th1
    for k in range(n1, 0, -1):
        radiik1 = radii[k - 1]
        sth1 = sth0 * radiik / radiik1
        th1 = math.asin(sth1)
        ks1 = th0 - th1
        sunpaths[k - 1] = math.sin(ks1) * radii[k] / sth1
        sth0 = sth1
        th0 = th1
        radiik = radiik1
    return sunpaths


def _find_sunpaths_tangent(
    *,
    radstart: float,
    radii: np.ndarray,
    theta_start: float,
    sin_theta_start: float,
    n: int,
) -> tuple[np.ndarray, int]:
    """Computes sun paths for tangent-geometry traversal."""
    pie = math.acos(-1.0)
    sunpaths = np.zeros(radii.size - 1, dtype=float)
    n1 = n - 1
    tanr = sin_theta_start * radstart
    k = n1
    while k >= n1 and radii[k] > tanr:
        k += 1
    nt = k - 1

    if nt > n:
        th0 = pie - theta_start
        sth0 = sin_theta_start
        radiik1 = radii[n]
        sth1 = sth0 * radstart / radiik1
        th1 = math.asin(sth1)
        ks1 = th0 - th1
        sunpaths[n - 1] = 2.0 * math.sin(ks1) * radstart / sth1
        sth0 = sth1
        th0 = th1
        for k in range(n + 1, nt):
            radiik = radii[k]
            sth1 = sth0 * radiik1 / radiik
            th1 = math.asin(sth1)
            ks1 = th0 - th1
            sunpaths[k - 1] = 2.0 * math.sin(ks1) * radiik / sth0
            sth0 = sth1
            th0 = th1
            radiik1 = radiik
        ks1 = 0.5 * pie - th0
        sunpaths[nt - 1] = 2.0 * math.sin(ks1) * radiik1
    elif nt == n:
        sunpaths[n - 1] = -2.0 * radstart * math.cos(theta_start)

    radiik = radii[n1]
    th0 = pie - theta_start
    sth0 = sin_theta_start
    sth1 = sth0 * radstart / radiik
    th1 = math.asin(sth1)
    ks1 = th0 - th1
    sunpaths[n - 1] = sunpaths[n - 1] + math.sin(ks1) * radstart / sth1
    sth0 = sth1
    th0 = th1
    for k in range(n1, 0, -1):
        radiik1 = radii[k - 1]
        sth1 = sth0 * radiik / radiik1
        th1 = math.asin(sth1)
        ks1 = th0 - th1
        sunpaths[k - 1] = math.sin(ks1) * radiik / sth1
        sth0 = sth1
        th0 = th1
        radiik = radiik1
    return sunpaths, nt


def _find_sun(
    *,
    do_nadir: bool,
    do_overhead_sun: bool,
    radius: float,
    solar_direction: np.ndarray,
    cum_angle: float,
    theta_boa: float,
) -> tuple[float, float, bool]:
    """Computes local solar geometry at a point along the line of sight."""
    if do_nadir:
        return theta_boa, math.sin(theta_boa), True
    if do_overhead_sun:
        return cum_angle, math.sin(cum_angle), True
    px = np.array(
        [-radius * math.sin(cum_angle), 0.0, radius * math.cos(cum_angle)],
        dtype=float,
    )
    b = float(np.dot(px, solar_direction))
    ctheta = -b / radius
    ctheta = max(-1.0, min(1.0, ctheta))
    direct_sun = ctheta >= 0.0
    stheta = math.sqrt(max(0.0, 1.0 - ctheta * ctheta))
    theta = math.acos(ctheta)
    return theta, stheta, direct_sun


def _fo_eps_geometry_uncached(
    *,
    user_obsgeoms: np.ndarray,
    height_grid: np.ndarray,
    earth_radius: float,
    nfine: int,
    vsign: float,
) -> dict[str, np.ndarray]:
    """Builds enhanced pseudo-spherical FO geometry terms."""
    nlayers = height_grid.size - 1
    ngeoms = user_obsgeoms.shape[0]
    radii = earth_radius + height_grid
    alpha = np.zeros((nlayers + 1, ngeoms), dtype=float)
    cota = np.zeros((nlayers + 1, ngeoms), dtype=float)
    raycon = np.zeros(ngeoms, dtype=float)
    do_nadir = np.zeros(ngeoms, dtype=bool)
    nfinedivs = np.full((nlayers, ngeoms), nfine, dtype=int)
    xfine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    wfine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    csqfine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    cotfine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    alphafine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    radiifine = np.zeros((nfine, nlayers, ngeoms), dtype=float)
    mu0 = np.zeros(ngeoms, dtype=float)
    cosscat = np.zeros(ngeoms, dtype=float)
    sunpathsnl = np.zeros((nlayers, ngeoms), dtype=float)
    ntraversenl = np.zeros(ngeoms, dtype=int)
    sunpathsfine = np.zeros((nlayers, nfine, nlayers, ngeoms), dtype=float)
    ntraversefine = np.zeros((nfine, nlayers, ngeoms), dtype=int)

    for v in range(ngeoms):
        vza = float(user_obsgeoms[v, 1])
        sza = float(user_obsgeoms[v, 0])
        azm = float(user_obsgeoms[v, 2])
        if math.isclose(vza, 0.0):
            do_nadir[v] = True
            radiin = radii[nlayers]
            for n in range(nlayers, 0, -1):
                radiin1 = radii[n - 1]
                radiin = radiin1
        else:
            alpha_boa_r = math.radians(vza)
            salpha_boa = 1.0 if math.isclose(vza, 90.0) else math.sin(alpha_boa_r)
            calpha1 = 0.0 if math.isclose(vza, 90.0) else math.cos(alpha_boa_r)
            cota[nlayers, v] = calpha1 / salpha_boa
            alpha[nlayers, v] = alpha_boa_r
            rayconv = salpha_boa * radii[nlayers]
            raycon[v] = rayconv
            radiin1 = radii[nlayers]
            for n in range(nlayers - 1, -1, -1):
                radiin = radii[n]
                sinanv = rayconv / radiin
                alphanv = math.asin(sinanv)
                alpha[n, v] = alphanv
                calpha = math.cos(alphanv)
                cota[n, v] = calpha / sinanv
                radiin1 = radiin
                calpha1 = calpha

        if do_nadir[v]:
            for n in range(nlayers, 0, -1):
                difh = radii[n - 1] - radii[n]
                x_nodes, w_nodes = _gauss_legendre_interval(0.0, float(difh), nfine)
                for j in range(nfine):
                    tfinej = x_nodes[j]
                    radiifine[j, n - 1, v] = radii[n - 1] - tfinej
                    xfine[j, n - 1, v] = tfinej
                    wfine[j, n - 1, v] = w_nodes[j]
        else:
            rayconv = raycon[v]
            alphanv = alpha[nlayers, v]
            for n in range(nlayers, 0, -1):
                alphan1v = alpha[n - 1, v]
                t_nodes, w_nodes = _gauss_legendre_interval(alphan1v, alphanv, nfine)
                for j in range(nfine):
                    tfinej = t_nodes[j]
                    csfine = 1.0 / math.sin(tfinej)
                    rf = rayconv * csfine
                    radiifine[j, n - 1, v] = rf
                    alphafine[j, n - 1, v] = tfinej
                    xfine[j, n - 1, v] = radii[n - 1] - rf
                    wfine[j, n - 1, v] = w_nodes[j]
                    cotfine[j, n - 1, v] = math.cos(tfinej) * csfine
                    csqfine[j, n - 1, v] = csfine * csfine
                alphanv = alphan1v

        do_overhead_sun = math.isclose(sza, 0.0)
        if math.isclose(vza, 90.0):
            calpha_boa = 0.0
            salpha_boa = 1.0
        else:
            salpha_boa = (
                math.sin(alpha[nlayers, v]) if not do_nadir[v] else math.sin(math.radians(vza))
            )
            calpha_boa = (
                math.cos(alpha[nlayers, v]) if not do_nadir[v] else math.cos(math.radians(vza))
            )
        theta_boa_r = math.radians(sza)
        stheta_boa = 1.0 if math.isclose(sza, 90.0) else math.sin(theta_boa_r)
        ctheta_boa = 0.0 if math.isclose(sza, 90.0) else math.cos(theta_boa_r)
        mu0[v] = ctheta_boa
        phi_boa_r = math.radians(azm)
        cphi_boa = math.cos(phi_boa_r)
        sphi_boa = math.sin(phi_boa_r)
        if do_overhead_sun:
            solar_direction = np.zeros(3, dtype=float)
        else:
            solar_direction = np.array(
                [
                    -stheta_boa * cphi_boa * vsign,
                    -stheta_boa * sphi_boa,
                    -ctheta_boa,
                ],
                dtype=float,
            )
        if do_overhead_sun:
            cosscat[v] = -vsign * calpha_boa if not math.isclose(calpha_boa, 0.0) else 0.0
        else:
            term1 = salpha_boa * stheta_boa * cphi_boa
            term2 = calpha_boa * ctheta_boa
            cosscat[v] = -vsign * term2 + term1

        alphanlv = alpha[nlayers, v]
        for n in range(nlayers, 0, -1):
            do_zero_sun_boa = do_overhead_sun and (n == nlayers or do_nadir[v])
            if n == nlayers:
                radstart = radii[n]
                theta_all = theta_boa_r
                stheta = stheta_boa
                direct_sun = True
            else:
                cum_angle = alphanlv - alpha[n, v]
                theta_all, stheta, direct_sun = _find_sun(
                    do_nadir=bool(do_nadir[v]),
                    do_overhead_sun=do_overhead_sun,
                    radius=float(radii[n]),
                    solar_direction=solar_direction,
                    cum_angle=cum_angle,
                    theta_boa=theta_boa_r,
                )
                radstart = radii[n]

            if direct_sun:
                sunpaths, _ = (
                    _find_sunpaths_direct(
                        do_zero_sun_boa=do_zero_sun_boa,
                        radstart=float(radstart),
                        radii=radii,
                        theta_start=float(theta_all),
                        sin_theta_start=float(stheta),
                        n=n,
                    ),
                    n,
                )
                if n == nlayers:
                    sunpathsnl[:, v] = sunpaths
                    ntraversenl[v] = nlayers
            else:
                sunpaths, nt = _find_sunpaths_tangent(
                    radstart=float(radstart),
                    radii=radii,
                    theta_start=float(theta_all),
                    sin_theta_start=float(stheta),
                    n=n,
                )
                if n == nlayers:
                    sunpathsnl[:, v] = sunpaths
                    ntraversenl[v] = nt

            for j in range(nfinedivs[n - 1, v]):
                cum_angle = alphanlv - alphafine[j, n - 1, v]
                thetaf, sthetaf, direct_sunf = _find_sun(
                    do_nadir=bool(do_nadir[v]),
                    do_overhead_sun=do_overhead_sun,
                    radius=float(radiifine[j, n - 1, v]),
                    solar_direction=solar_direction,
                    cum_angle=cum_angle,
                    theta_boa=theta_boa_r,
                )
                if direct_sunf:
                    sunpaths = _find_sunpaths_direct(
                        do_zero_sun_boa=do_zero_sun_boa,
                        radstart=float(radiifine[j, n - 1, v]),
                        radii=radii,
                        theta_start=float(thetaf),
                        sin_theta_start=float(sthetaf),
                        n=n,
                    )
                    ntraversefine[j, n - 1, v] = n
                else:
                    sunpaths, nt = _find_sunpaths_tangent(
                        radstart=float(radiifine[j, n - 1, v]),
                        radii=radii,
                        theta_start=float(thetaf),
                        sin_theta_start=float(sthetaf),
                        n=n,
                    )
                    ntraversefine[j, n - 1, v] = nt
                sunpathsfine[:, j, n - 1, v] = sunpaths

    return {
        "radii": radii,
        "alpha": alpha,
        "cota": cota,
        "raycon": raycon,
        "do_nadir": do_nadir,
        "nfinedivs": nfinedivs,
        "xfine": xfine,
        "wfine": wfine,
        "csqfine": csqfine,
        "cotfine": cotfine,
        "sunpathsnl": sunpathsnl,
        "ntraversenl": ntraversenl,
        "sunpathsfine": sunpathsfine,
        "ntraversefine": ntraversefine,
        "mu0": mu0,
        "cosscat": cosscat,
    }


@lru_cache(maxsize=16)
def _fo_eps_geometry_from_key(
    user_obsgeoms_shape: tuple[int, ...],
    user_obsgeoms_bytes: bytes,
    height_grid_shape: tuple[int, ...],
    height_grid_bytes: bytes,
    earth_radius: float,
    nfine: int,
    vsign: float,
) -> dict[str, np.ndarray]:
    """Builds and caches EPS geometry for repeated invariant geometry calls."""
    user_obsgeoms = np.frombuffer(user_obsgeoms_bytes, dtype=np.float64).reshape(
        user_obsgeoms_shape
    )
    height_grid = np.frombuffer(height_grid_bytes, dtype=np.float64).reshape(height_grid_shape)
    return _fo_eps_geometry_uncached(
        user_obsgeoms=user_obsgeoms,
        height_grid=height_grid,
        earth_radius=earth_radius,
        nfine=nfine,
        vsign=vsign,
    )


def _fo_eps_geometry(
    *,
    user_obsgeoms: np.ndarray,
    height_grid: np.ndarray,
    earth_radius: float,
    nfine: int,
    vsign: float,
) -> dict[str, np.ndarray]:
    """Builds enhanced pseudo-spherical FO geometry terms with cache reuse."""
    user_obsgeoms_arr = np.ascontiguousarray(user_obsgeoms, dtype=np.float64)
    height_grid_arr = np.ascontiguousarray(height_grid, dtype=np.float64)
    return _fo_eps_geometry_from_key(
        tuple(user_obsgeoms_arr.shape),
        user_obsgeoms_arr.tobytes(),
        tuple(height_grid_arr.shape),
        height_grid_arr.tobytes(),
        float(earth_radius),
        int(nfine),
        float(vsign),
    )


def _phase_function_hg(mu: float, asymmetry: float, n_moments: int) -> float:
    """Evaluates the Henyey-Greenstein phase function."""
    if n_moments < 0:
        raise ValueError("n_moments must be non-negative")
    if n_moments == 0:
        return 1.0
    if abs(asymmetry) >= 1.0:
        raise ValueError("g must satisfy -1 < g < 1 for Henyey-Greenstein scattering")
    denominator = 1.0 + asymmetry * asymmetry - 2.0 * asymmetry * mu
    return (1.0 - asymmetry * asymmetry) / (denominator**1.5)


def _normalize_solar_obs_angles(angles) -> np.ndarray:
    arr = np.asarray(angles, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("angles must be finite")
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("angles must have shape (3,) or (ngeom, 3)")
    return arr


def _solar_obs_scattering_cosines(angles) -> np.ndarray:
    geoms = _normalize_solar_obs_angles(angles)
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


def _phase_function_hg_array(mu: np.ndarray, asymmetry: np.ndarray, n_moments: int) -> np.ndarray:
    if n_moments < 0:
        raise ValueError("n_moments must be non-negative")
    mu_arr, asymm_arr = np.broadcast_arrays(np.asarray(mu, dtype=float), asymmetry)
    if n_moments == 0:
        return np.ones_like(mu_arr, dtype=float)
    if np.any(np.abs(asymm_arr) >= 1.0):
        raise ValueError("g must satisfy -1 < g < 1 for Henyey-Greenstein scattering")
    denominator = 1.0 + asymm_arr * asymm_arr - 2.0 * asymm_arr * mu_arr
    return (1.0 - asymm_arr * asymm_arr) / np.power(denominator, 1.5)


def fo_scatter_term_henyey_greenstein(
    *,
    ssa,
    g,
    angles,
    delta_m_truncation_factor=None,
    n_moments: int = 5000,
) -> np.ndarray:
    """Build the solar FO scatter term for Henyey-Greenstein phase functions.

    The returned value matches the Fortran convention:
    ``phase_function(cos_scatter, g) * ssa / (1 - delta_m_truncation_factor * ssa)``.
    A single geometry returns shape ``(..., nlyr)``. Multiple geometries return
    ``(..., nlyr, ngeom)``. Any positive ``n_moments`` uses the closed-form HG
    phase function; ``n_moments=0`` selects isotropic scattering.
    """
    ssa_arr = np.asarray(ssa, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    if ssa_arr.ndim == 0 or g_arr.ndim == 0:
        raise ValueError("ssa and g must have a layer axis")
    if delta_m_truncation_factor is None:
        scaling_arr = default_delta_m_truncation_factor(g_arr)
    else:
        scaling_arr = np.asarray(delta_m_truncation_factor, dtype=float)
        if scaling_arr.ndim == 0 and ssa_arr.ndim > 0:
            scaling_arr = np.full_like(ssa_arr, float(scaling_arr))
    try:
        target_shape = np.broadcast_shapes(ssa_arr.shape, g_arr.shape, scaling_arr.shape)
    except ValueError as exc:
        raise ValueError(
            "ssa, g, and delta_m_truncation_factor must be broadcast-compatible"
        ) from exc
    if len(target_shape) == 0:
        raise ValueError("ssa and g must have a layer axis")
    ssa_b = np.broadcast_to(ssa_arr, target_shape).astype(float, copy=False)
    g_b = np.broadcast_to(g_arr, target_shape).astype(float, copy=False)
    scaling_b = np.broadcast_to(scaling_arr, target_shape).astype(float, copy=False)
    if not (np.all(np.isfinite(ssa_b)) and np.all(np.isfinite(g_b))):
        raise ValueError("ssa and g must be finite")
    validate_delta_m_truncation_factor(scaling_b, ssa_b)

    denominator = 1.0 - scaling_b * ssa_b
    if np.any(np.abs(denominator) <= np.finfo(float).eps):
        raise ValueError("1 - delta_m_truncation_factor * ssa is too close to zero")

    cosscat = _solar_obs_scattering_cosines(angles)
    mu = cosscat.reshape((1,) * (len(target_shape) - 1) + (1, cosscat.size))
    phase = _phase_function_hg_array(mu, g_b[..., np.newaxis], int(n_moments))
    exact = phase * (ssa_b / denominator)[..., np.newaxis]
    if cosscat.size == 1:
        exact = exact[..., 0]
    return np.ascontiguousarray(exact, dtype=float)


def _fo_rps_geometry(
    *,
    user_obsgeoms: np.ndarray,
    height_grid: np.ndarray,
    earth_radius: float,
    vsign: float,
) -> dict[str, np.ndarray]:
    """Builds regular pseudo-spherical FO geometry terms."""
    nlayers = height_grid.size - 1
    ngeoms = user_obsgeoms.shape[0]
    radii = earth_radius + height_grid
    mu0 = np.zeros(ngeoms, dtype=float)
    mu1 = np.zeros(ngeoms, dtype=float)
    cosscat = np.zeros(ngeoms, dtype=float)
    sunpaths = np.zeros((nlayers, nlayers, ngeoms), dtype=float)

    for v in range(ngeoms):
        sza = float(user_obsgeoms[v, 0])
        vza = float(user_obsgeoms[v, 1])
        azm = float(user_obsgeoms[v, 2])
        theta_boa_r = math.radians(sza)
        alpha_boa_r = math.radians(vza)
        do_overhead_sun = math.isclose(sza, 0.0)
        stheta_boa = 1.0 if math.isclose(sza, 90.0) else math.sin(theta_boa_r)
        ctheta_boa = 0.0 if math.isclose(sza, 90.0) else math.cos(theta_boa_r)
        salpha_boa = 1.0 if math.isclose(vza, 90.0) else math.sin(alpha_boa_r)
        calpha_boa = 0.0 if math.isclose(vza, 90.0) else math.cos(alpha_boa_r)
        cphi_boa = math.cos(math.radians(azm))
        mu0[v] = ctheta_boa
        mu1[v] = calpha_boa
        if do_overhead_sun:
            cosscat[v] = -vsign * calpha_boa if not math.isclose(calpha_boa, 0.0) else 0.0
        else:
            term1 = salpha_boa * stheta_boa * cphi_boa
            term2 = calpha_boa * ctheta_boa
            cosscat[v] = -vsign * term2 + term1
        for n in range(1, nlayers + 1):
            sunpaths_local = _find_sunpaths_direct(
                do_zero_sun_boa=do_overhead_sun,
                radstart=float(radii[n]),
                radii=radii,
                theta_start=theta_boa_r,
                sin_theta_start=stheta_boa,
                n=n,
            )
            sunpaths[:n, n - 1, v] = sunpaths_local[:n]

    return {
        "mu0": mu0,
        "mu1": mu1,
        "cosscat": cosscat,
        "sunpaths": sunpaths,
    }


def solve_fo_solar_obs(
    prepared: PreparedInputs,
    *,
    do_plane_parallel: bool,
    geometry_mode: str = "eps",
    n_moments: int = 5000,
    nfine: int = 3,
    exact_scatter: np.ndarray | None = None,
) -> FoSolarObsResult:
    """Solves the optimized FO solar observation-geometry problem."""
    if prepared.source_mode not in {"solar_obs", "solar_lat"}:
        raise NotImplementedError(
            "FO solar observation geometry is implemented for source_mode='solar_obs' "
            "and 'solar_lat' only"
        )
    if geometry_mode not in {"eps", "rps"}:
        raise ValueError("geometry_mode must be 'eps' or 'rps'")
    if prepared.user_obsgeoms is None:
        raise ValueError("user_obsgeoms are required for FO solar observation geometry")
    if prepared.user_obsgeoms.ndim != 2 or prepared.user_obsgeoms.shape[1] != 3:
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")
    if prepared.height_grid is None:
        raise ValueError("height_grid is required for FO solar observation geometry")
    nlayers = prepared.tau_arr.size
    ngeoms = prepared.user_obsgeoms.shape[0]
    extinction = prepared.tau_arr / (prepared.height_grid[:-1] - prepared.height_grid[1:])
    deltaus = prepared.tau_arr.copy()
    mu0 = np.cos(np.deg2rad(prepared.user_obsgeoms[:, 0]))
    mu1 = np.cos(np.deg2rad(prepared.user_obsgeoms[:, 1]))
    cosscat = np.zeros(ngeoms, dtype=float)
    do_nadir = np.isclose(prepared.user_obsgeoms[:, 1], 0.0)
    for v in range(ngeoms):
        sza = math.radians(float(prepared.user_obsgeoms[v, 0]))
        vza = math.radians(float(prepared.user_obsgeoms[v, 1]))
        azm = math.radians(float(prepared.user_obsgeoms[v, 2]))
        if math.isclose(prepared.user_obsgeoms[v, 0], 0.0):
            cosscat[v] = -mu1[v] if not math.isclose(mu1[v], 0.0) else 0.0
        else:
            cosscat[v] = -(math.cos(vza) * math.cos(sza)) + math.sin(vza) * math.sin(
                sza
            ) * math.cos(azm)

    intensity_ss = np.zeros(ngeoms, dtype=float)
    intensity_db = np.zeros(ngeoms, dtype=float)
    intensity_ss_profile = np.zeros((ngeoms, nlayers + 1), dtype=float)
    intensity_db_profile = np.zeros((ngeoms, nlayers + 1), dtype=float)
    exact_scatter_arr = None if exact_scatter is None else np.asarray(exact_scatter, dtype=float)
    if exact_scatter_arr is not None:
        if exact_scatter_arr.ndim == 1:
            exact_scatter_arr = exact_scatter_arr[:, np.newaxis]
        if exact_scatter_arr.shape != (nlayers, ngeoms):
            raise ValueError(
                "exact_scatter must have shape (n_layers,) or (n_layers, n_geometries)"
            )

    if do_plane_parallel:
        for v in range(ngeoms):
            phase_terms = np.zeros(nlayers, dtype=float)
            for n in range(nlayers):
                if exact_scatter_arr is not None:
                    phase_terms[n] = exact_scatter_arr[n, v]
                else:
                    phase = _phase_function_hg(
                        float(cosscat[v]), float(prepared.asymm_arr[n]), n_moments
                    )
                    truncfac = prepared.d2s_scaling[n]
                    omw = prepared.omega_arr[n]
                    tms = omw / (1.0 - truncfac * omw)
                    phase_terms[n] = phase * tms

            flux = 0.25 * prepared.flux_factor / math.pi
            attenuations = np.zeros(nlayers + 1, dtype=float)
            attenuations[0] = 1.0
            cumtau = 0.0
            for n in range(nlayers):
                cumtau += deltaus[n]
                sumd = cumtau / mu0[v]
                attenuations[n + 1] = math.exp(-sumd) if sumd < 88.0 else 0.0

            lostrans_up = np.zeros(nlayers, dtype=float)
            sources_up = np.zeros(nlayers, dtype=float)
            solutions = np.zeros(nlayers, dtype=float)
            factor1 = np.zeros(nlayers, dtype=float)
            factor2 = np.zeros(nlayers, dtype=float)
            nut = 1
            nstart = nlayers
            attn_prev = attenuations[0]
            if math.isclose(mu1[v], 0.0):
                for n in range(nlayers):
                    solutions[n] = phase_terms[n] * attn_prev
                    attn_prev = attenuations[n + 1]
                    if n + 1 >= nut and not math.isclose(attn_prev, 0.0):
                        factor1[n] = attenuations[n + 1] / attn_prev
                        nstart = n + 1
            else:
                for n in range(nlayers):
                    lostau = deltaus[n] / mu1[v]
                    solutions[n] = phase_terms[n] * attn_prev
                    lostrans_up[n] = math.exp(-lostau) if lostau < 88.0 else 0.0
                    attn = attenuations[n + 1]
                    if n + 1 >= nut:
                        if not math.isclose(attn_prev, 0.0):
                            factor1[n] = attn / attn_prev
                            factor2[n] = mu1[v] / mu0[v]
                            nstart = n + 1
                    attn_prev = attn

            for n in range(nlayers - 1, nstart - 1, -1):
                sources_up[n] = 0.0
            for n in range(nstart - 1, nut - 2, -1):
                multiplier = (1.0 - factor1[n] * lostrans_up[n]) / (factor2[n] + 1.0)
                sources_up[n] = solutions[n] * multiplier
            for n in range(nut - 2, -1, -1):
                sources_up[n] = solutions[n]

            cumsource_up = 0.0
            cumsource_db = 4.0 * mu0[v] * prepared.albedo * attenuations[-1]
            intensity_ss_profile[v, nlayers] = flux * cumsource_up
            intensity_db_profile[v, nlayers] = flux * cumsource_db
            for n in range(nlayers - 1, -1, -1):
                cumsource_db = lostrans_up[n] * cumsource_db
                cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
                intensity_ss_profile[v, n] = flux * cumsource_up
                intensity_db_profile[v, n] = flux * cumsource_db
            intensity_ss[v] = flux * cumsource_up
            intensity_db[v] = flux * cumsource_db

        return FoSolarObsResult(
            intensity_total=intensity_ss + intensity_db,
            intensity_ss=intensity_ss,
            intensity_db=intensity_db,
            mu0=mu0,
            mu1=mu1,
            cosscat=cosscat,
            do_nadir=do_nadir,
            intensity_total_profile=intensity_ss_profile + intensity_db_profile,
            intensity_ss_profile=intensity_ss_profile,
            intensity_db_profile=intensity_db_profile,
        )

    if geometry_mode == "rps":
        geometry = _fo_rps_geometry(
            user_obsgeoms=prepared.user_obsgeoms,
            height_grid=prepared.height_grid,
            earth_radius=prepared.earth_radius,
            vsign=1.0,
        )
        mu0 = geometry["mu0"]
        mu1 = geometry["mu1"]
        cosscat = geometry["cosscat"]
        do_nadir = np.isclose(prepared.user_obsgeoms[:, 1], 0.0)
        for v in range(ngeoms):
            phase_terms = np.zeros(nlayers, dtype=float)
            for n in range(nlayers):
                if exact_scatter_arr is not None:
                    phase_terms[n] = exact_scatter_arr[n, v]
                else:
                    phase = _phase_function_hg(
                        float(cosscat[v]), float(prepared.asymm_arr[n]), n_moments
                    )
                    truncfac = prepared.d2s_scaling[n]
                    omw = prepared.omega_arr[n]
                    tms = omw / (1.0 - truncfac * omw)
                    phase_terms[n] = phase * tms

            attenuations = np.zeros(nlayers + 1, dtype=float)
            attenuations[0] = 1.0
            suntau = np.zeros(nlayers, dtype=float)
            for n in range(nlayers):
                sumd = float(np.dot(extinction[: n + 1], geometry["sunpaths"][: n + 1, n, v]))
                suntau[n] = sumd
                attenuations[n + 1] = math.exp(-sumd) if sumd < 88.0 else 0.0

            solutions = np.zeros(nlayers, dtype=float)
            sources_up = np.zeros(nlayers, dtype=float)
            lostrans_up = np.zeros(nlayers, dtype=float)
            factor1 = np.zeros(nlayers, dtype=float)
            factor2 = np.zeros(nlayers, dtype=float)
            nstart = nlayers
            mu1v = mu1[v]
            attn_prev = attenuations[0]
            suntaun1 = 0.0
            if math.isclose(mu1v, 0.0):
                for n in range(nlayers):
                    solutions[n] = phase_terms[n] * attn_prev
                    attn_prev = attenuations[n + 1]
            else:
                for n in range(nlayers):
                    lostau = deltaus[n] / mu1v
                    solutions[n] = phase_terms[n] * attn_prev
                    lostrans_up[n] = math.exp(-lostau) if lostau < 88.0 else 0.0
                    attn = attenuations[n + 1]
                    suntaun = suntau[n]
                    if not math.isclose(attn_prev, 0.0):
                        factor1[n] = attn / attn_prev
                        factor2[n] = (suntaun - suntaun1) / lostau
                        nstart = n + 1
                    attn_prev = attn
                    suntaun1 = suntaun

            for n in range(nlayers - 1, nstart, -1):
                sources_up[n] = 0.0
            for n in range(nstart - 1, -1, -1):
                if math.isclose(mu1v, 0.0):
                    sources_up[n] = solutions[n]
                else:
                    multiplier = (1.0 - factor1[n] * lostrans_up[n]) / (factor2[n] + 1.0)
                    sources_up[n] = solutions[n] * multiplier

            flux = 0.25 * prepared.flux_factor / math.pi
            cumsource_up = 0.0
            cumsource_db = 4.0 * mu0[v] * prepared.albedo * attenuations[-1]
            intensity_ss_profile[v, nlayers] = flux * cumsource_up
            intensity_db_profile[v, nlayers] = flux * cumsource_db
            for n in range(nlayers - 1, -1, -1):
                cumsource_db = lostrans_up[n] * cumsource_db
                cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
                intensity_ss_profile[v, n] = flux * cumsource_up
                intensity_db_profile[v, n] = flux * cumsource_db
            intensity_ss[v] = flux * cumsource_up
            intensity_db[v] = flux * cumsource_db

        return FoSolarObsResult(
            intensity_total=intensity_ss + intensity_db,
            intensity_ss=intensity_ss,
            intensity_db=intensity_db,
            mu0=mu0,
            mu1=mu1,
            cosscat=cosscat,
            do_nadir=do_nadir,
            intensity_total_profile=intensity_ss_profile + intensity_db_profile,
            intensity_ss_profile=intensity_ss_profile,
            intensity_db_profile=intensity_db_profile,
        )

    geometry = _fo_eps_geometry(
        user_obsgeoms=prepared.user_obsgeoms,
        height_grid=prepared.height_grid,
        earth_radius=prepared.earth_radius,
        nfine=nfine,
        vsign=1.0,
    )
    mu0 = geometry["mu0"]
    cosscat = geometry["cosscat"]
    do_nadir = geometry["do_nadir"]
    for v in range(ngeoms):
        phase_terms = np.zeros(nlayers, dtype=float)
        for n in range(nlayers):
            if exact_scatter_arr is not None:
                phase_terms[n] = exact_scatter_arr[n, v]
            else:
                phase = _phase_function_hg(
                    float(cosscat[v]), float(prepared.asymm_arr[n]), n_moments
                )
                truncfac = prepared.d2s_scaling[n]
                omw = prepared.omega_arr[n]
                tms = omw / (1.0 - truncfac * omw)
                phase_terms[n] = phase * tms

        ntrav_nl = int(geometry["ntraversenl"][v])
        sunpathsnl = geometry["sunpathsnl"][:, v]
        total_tau = float(np.dot(extinction[:ntrav_nl], sunpathsnl[:ntrav_nl]))
        attenuations_nl = math.exp(-total_tau) if total_tau < 88.0 else 0.0
        sources_up = np.zeros(nlayers, dtype=float)
        lostrans_up = np.zeros(nlayers, dtype=float)
        if bool(do_nadir[v]):
            lostrans_up[:] = np.exp(-deltaus)
            for n in range(nlayers, 0, -1):
                kn = extinction[n - 1]
                layer_sum = 0.0
                nfine_layer = int(geometry["nfinedivs"][n - 1, v])
                for j in range(nfine_layer):
                    ntrav = int(geometry["ntraversefine"][j, n - 1, v])
                    fine_tau = float(
                        np.dot(extinction[:ntrav], geometry["sunpathsfine"][:ntrav, j, n - 1, v])
                    )
                    attenuation = math.exp(-fine_tau) if fine_tau < 88.0 else 0.0
                    solution = phase_terms[n - 1] * attenuation
                    layer_sum += (
                        solution
                        * math.exp(-geometry["xfine"][j, n - 1, v] * kn)
                        * geometry["wfine"][j, n - 1, v]
                    )
                sources_up[n - 1] = layer_sum * kn
        else:
            cot_1 = geometry["cota"][nlayers, v]
            rayconv = geometry["raycon"][v]
            for n in range(nlayers, 0, -1):
                cot_2 = geometry["cota"][n - 1, v]
                ke = rayconv * extinction[n - 1]
                tran_1 = math.exp(-ke * (cot_2 - cot_1))
                lostrans_up[n - 1] = tran_1
                nfine_layer = int(geometry["nfinedivs"][n - 1, v])
                layer_sum = 0.0
                for j in range(nfine_layer):
                    tran = math.exp(-ke * (cot_2 - geometry["cotfine"][j, n - 1, v]))
                    ntrav = int(geometry["ntraversefine"][j, n - 1, v])
                    fine_tau = float(
                        np.dot(extinction[:ntrav], geometry["sunpathsfine"][:ntrav, j, n - 1, v])
                    )
                    attenuation = math.exp(-fine_tau) if fine_tau < 88.0 else 0.0
                    solution = phase_terms[n - 1] * attenuation
                    layer_sum += (
                        solution
                        * geometry["csqfine"][j, n - 1, v]
                        * tran
                        * geometry["wfine"][j, n - 1, v]
                    )
                sources_up[n - 1] = layer_sum * ke
                cot_1 = cot_2

        flux = 0.25 * prepared.flux_factor / math.pi
        cumsource_up = 0.0
        cumsource_db = 4.0 * mu0[v] * prepared.albedo * attenuations_nl
        intensity_ss_profile[v, nlayers] = flux * cumsource_up
        intensity_db_profile[v, nlayers] = flux * cumsource_db
        for n in range(nlayers - 1, -1, -1):
            cumsource_db = lostrans_up[n] * cumsource_db
            cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
            intensity_ss_profile[v, n] = flux * cumsource_up
            intensity_db_profile[v, n] = flux * cumsource_db
        intensity_ss[v] = flux * cumsource_up
        intensity_db[v] = flux * cumsource_db

    return FoSolarObsResult(
        intensity_total=intensity_ss + intensity_db,
        intensity_ss=intensity_ss,
        intensity_db=intensity_db,
        mu0=mu0,
        mu1=mu1,
        cosscat=cosscat,
        do_nadir=do_nadir,
        intensity_total_profile=intensity_ss_profile + intensity_db_profile,
        intensity_ss_profile=intensity_ss_profile,
        intensity_db_profile=intensity_db_profile,
    )
