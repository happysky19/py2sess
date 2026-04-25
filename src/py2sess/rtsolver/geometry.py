"""Geometry and Chapman-factor helpers translated from optimized 2S-ESS."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


def _chapman_factors_uncached(
    heights: np.ndarray,
    earth_radius: float,
    sza_deg: float,
) -> np.ndarray:
    """Computes pseudo-spherical Chapman factors for a solar beam.

    Parameters
    ----------
    heights
        Monotonically decreasing level heights in kilometers.
    earth_radius
        Planetary radius in kilometers.
    sza_deg
        Solar zenith angle at the top of atmosphere, in degrees.

    Returns
    -------
    numpy.ndarray
        Lower-triangular Chapman-factor matrix with shape
        ``(n_layers, n_layers)``.
    """
    n_layers = heights.shape[0] - 1
    delz = heights[:-1] - heights[1:]
    factors = np.zeros((n_layers, n_layers), dtype=float)

    deg_to_rad = math.atan(1.0) / 45.0
    th_toa = sza_deg * deg_to_rad
    mu_toa = math.cos(th_toa)
    gm_toa = math.sqrt(1.0 - mu_toa * mu_toa)

    h0 = heights[0] + earth_radius
    for n in range(n_layers):
        hn = heights[n + 1] + earth_radius
        sinth1 = gm_toa * hn / h0
        sth1 = math.asin(sinth1)
        re_upper = h0

        for k in range(n + 1):
            delzk = delz[k]
            re_lower = re_upper - delzk
            sinth2 = re_upper * sinth1 / re_lower
            sth2 = math.asin(sinth2)
            phi = sth2 - sth1
            sinphi = math.sin(phi)
            dist = re_upper * sinphi / sinth2
            factors[k, n] = dist / delzk
            re_upper = re_lower
            sinth1 = sinth2
            sth1 = sth2

    return factors


@lru_cache(maxsize=128)
def _chapman_factors_from_key(
    heights_shape: tuple[int, ...],
    heights_bytes: bytes,
    earth_radius: float,
    sza_deg: float,
) -> np.ndarray:
    """Returns cached Chapman factors keyed by immutable numeric inputs."""
    heights = np.frombuffer(heights_bytes, dtype=np.float64).reshape(heights_shape)
    return _chapman_factors_uncached(heights, earth_radius, sza_deg)


def chapman_factors(
    heights: np.ndarray,
    earth_radius: float,
    sza_deg: float,
) -> np.ndarray:
    """Computes pseudo-spherical Chapman factors for a solar beam.

    Repeated spectral calculations often reuse the same geometry for many
    wavelengths, so this wrapper caches the geometry-only calculation without
    changing the numerical algorithm.
    """
    heights64 = np.ascontiguousarray(heights, dtype=np.float64)
    cached = _chapman_factors_from_key(
        heights64.shape,
        heights64.tobytes(),
        float(earth_radius),
        float(sza_deg),
    )
    return cached.copy()


def auxgeom_solar_obs(
    x0: np.ndarray,
    user_streams: np.ndarray,
    stream_value: float,
    do_postprocessing: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Builds auxiliary two-stream angular factors for solar observation mode.

    Parameters
    ----------
    x0
        Cosines of solar zenith angles for each geometry.
    user_streams
        Cosines of user viewing zenith angles.
    stream_value
        Two-stream quadrature cosine.
    do_postprocessing
        Whether user-stream postprocessing terms are active.

    Returns
    -------
    tuple
        ``px11``, ``pxsq``, ``px0x``, and ``ulp`` angular factors used by
        the optimized solver.
    """
    if do_postprocessing:
        ulp = -np.sqrt(0.5 * (1.0 - user_streams * user_streams))
    else:
        ulp = np.zeros_like(user_streams)

    svsq = stream_value * stream_value
    px11 = math.sqrt(0.5 * (1.0 - svsq))
    pxsq = np.array([svsq, px11 * px11], dtype=float)

    pox = np.sqrt(0.5 * (1.0 - x0 * x0))
    px0x = np.column_stack((x0 * stream_value, pox * px11))
    return px11, pxsq, px0x, ulp
