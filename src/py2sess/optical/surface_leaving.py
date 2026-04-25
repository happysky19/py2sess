"""Surface-leaving utilities ported from the 2S-ESS Fortran supplement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_INDWAT_WAVELENGTHS = np.array(
    [
        0.250,
        0.275,
        0.300,
        0.325,
        0.345,
        0.375,
        0.400,
        0.425,
        0.445,
        0.475,
        0.500,
        0.525,
        0.550,
        0.575,
        0.600,
        0.625,
        0.650,
        0.675,
        0.700,
        0.725,
        0.750,
        0.775,
        0.800,
        0.825,
        0.850,
        0.875,
        0.900,
        0.925,
        0.950,
        0.975,
        1.000,
        1.200,
        1.400,
        1.600,
        1.800,
        2.000,
        2.200,
        2.400,
        2.600,
        2.650,
        2.700,
        2.750,
        2.800,
        2.850,
        2.900,
        2.950,
        3.000,
        3.050,
        3.100,
        3.150,
        3.200,
        3.250,
        3.300,
        3.350,
        3.400,
        3.450,
        3.500,
        3.600,
        3.700,
        3.800,
        3.900,
        4.000,
    ],
    dtype=np.float64,
)

_INDWAT_REAL_INDEX = np.array(
    [
        1.362,
        1.354,
        1.349,
        1.346,
        1.343,
        1.341,
        1.339,
        1.338,
        1.337,
        1.336,
        1.335,
        1.334,
        1.333,
        1.333,
        1.332,
        1.332,
        1.331,
        1.331,
        1.331,
        1.330,
        1.330,
        1.330,
        1.329,
        1.329,
        1.329,
        1.328,
        1.328,
        1.328,
        1.327,
        1.327,
        1.327,
        1.324,
        1.321,
        1.317,
        1.312,
        1.306,
        1.296,
        1.279,
        1.242,
        1.219,
        1.188,
        1.157,
        1.142,
        1.149,
        1.201,
        1.292,
        1.371,
        1.426,
        1.467,
        1.483,
        1.478,
        1.467,
        1.450,
        1.432,
        1.420,
        1.410,
        1.400,
        1.385,
        1.374,
        1.364,
        1.357,
        1.351,
    ],
    dtype=np.float64,
)

_INDWAT_IMAG_INDEX = np.array(
    [
        3.35e-08,
        2.35e-08,
        1.60e-08,
        1.08e-08,
        6.50e-09,
        3.50e-09,
        1.86e-09,
        1.30e-09,
        1.02e-09,
        9.35e-10,
        1.00e-09,
        1.32e-09,
        1.96e-09,
        3.60e-09,
        1.09e-08,
        1.39e-08,
        1.64e-08,
        2.23e-08,
        3.35e-08,
        9.15e-08,
        1.56e-07,
        1.48e-07,
        1.25e-07,
        1.82e-07,
        2.93e-07,
        3.91e-07,
        4.86e-07,
        1.06e-06,
        2.93e-06,
        3.48e-06,
        2.89e-06,
        9.89e-06,
        1.38e-04,
        8.55e-05,
        1.15e-04,
        1.10e-03,
        2.89e-04,
        9.56e-04,
        3.17e-03,
        6.70e-03,
        1.90e-02,
        5.90e-02,
        1.15e-01,
        1.85e-01,
        2.68e-01,
        2.98e-01,
        2.72e-01,
        2.40e-01,
        1.92e-01,
        1.35e-01,
        9.24e-02,
        6.10e-02,
        3.68e-02,
        2.61e-02,
        1.95e-02,
        1.32e-02,
        9.40e-03,
        5.15e-03,
        3.60e-03,
        3.40e-03,
        3.80e-03,
        4.60e-03,
    ],
    dtype=np.float64,
)

_MORCASIWAT_KW = np.array(
    [
        0.0271,
        0.0238,
        0.0216,
        0.0188,
        0.0177,
        0.0159,
        0.0151,
        0.01376,
        0.01271,
        0.01208,
        0.01042,
        0.0089,
        0.00812,
        0.00765,
        0.00758,
        0.00768,
        0.00771,
        0.00792,
        0.00885,
        0.0099,
        0.01148,
        0.01182,
        0.01188,
        0.01211,
        0.01251,
        0.0132,
        0.01444,
        0.01526,
        0.0166,
        0.01885,
        0.02188,
        0.02701,
        0.03385,
        0.0409,
        0.04214,
        0.04287,
        0.04454,
        0.0463,
        0.04846,
        0.05212,
        0.05746,
        0.06053,
        0.0628,
        0.06507,
        0.07034,
        0.07801,
        0.09038,
        0.11076,
        0.13584,
        0.16792,
        0.2331,
        0.25838,
        0.26506,
        0.26843,
        0.27612,
        0.28401,
        0.29218,
        0.30176,
        0.31134,
        0.32553,
        0.34052,
        0.3715,
        0.41048,
        0.42947,
        0.43946,
        0.44844,
        0.46543,
        0.48643,
        0.5164,
        0.55939,
        0.62438,
    ],
    dtype=np.float64,
)

_MORCASIWAT_XC = np.array(
    [
        0.1903,
        0.1809,
        0.1731,
        0.1669,
        0.1613,
        0.1563,
        0.1513,
        0.146,
        0.142,
        0.138,
        0.134,
        0.1333,
        0.1347,
        0.1346,
        0.1322,
        0.12961,
        0.12728,
        0.12485,
        0.12065,
        0.1157,
        0.1103,
        0.1068,
        0.1038,
        0.1005,
        0.0971,
        0.0933,
        0.0891,
        0.08612,
        0.08323,
        0.08028,
        0.0774,
        0.0733,
        0.0691,
        0.0675,
        0.06602,
        0.06578,
        0.064,
        0.063,
        0.0623,
        0.0603,
        0.0571,
        0.0561,
        0.0555,
        0.0551,
        0.0545,
        0.0542,
        0.0535,
        0.0525,
        0.0522,
        0.0521,
        0.0522,
        0.0525,
        0.0529,
        0.0538,
        0.0555,
        0.0561,
        0.057,
        0.0585,
        0.0598,
        0.0605,
        0.0621,
        0.0615,
        0.0641,
        0.0675,
        0.0705,
        0.0735,
        0.074,
        0.067,
        0.058,
        0.046,
        0.027,
    ],
    dtype=np.float64,
)

_MORCASIWAT_E = np.array(
    [
        0.6523,
        0.6579,
        0.653,
        0.653,
        0.6534,
        0.6595,
        0.6627,
        0.6651,
        0.661,
        0.642,
        0.638,
        0.628,
        0.628,
        0.631,
        0.6342,
        0.6378,
        0.6366,
        0.6374,
        0.6434,
        0.6449,
        0.6432,
        0.639,
        0.6345,
        0.6384,
        0.6326,
        0.6287,
        0.6326,
        0.6269,
        0.625,
        0.6236,
        0.6246,
        0.6255,
        0.625,
        0.615,
        0.592,
        0.575,
        0.559,
        0.5514,
        0.544,
        0.5332,
        0.5303,
        0.525,
        0.52,
        0.515,
        0.505,
        0.501,
        0.501,
        0.502,
        0.502,
        0.502,
        0.495,
        0.491,
        0.489,
        0.482,
        0.481,
        0.481,
        0.483,
        0.488,
        0.491,
        0.501,
        0.505,
        0.508,
        0.511,
        0.513,
        0.511,
        0.495,
        0.465,
        0.432,
        0.405,
        0.365,
        0.331,
    ],
    dtype=np.float64,
)

_MORCASIWAT_MUDA = np.array(
    [
        [0.770, 0.765, 0.800, 0.841, 0.872, 0.892, 0.911, 0.914],
        [0.769, 0.770, 0.797, 0.824, 0.855, 0.879, 0.908, 0.912],
        [0.766, 0.774, 0.796, 0.808, 0.834, 0.858, 0.902, 0.909],
        [0.767, 0.779, 0.797, 0.797, 0.811, 0.827, 0.890, 0.901],
        [0.767, 0.782, 0.799, 0.791, 0.796, 0.795, 0.871, 0.890],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SurfaceLeavingCoefficients:
    """Surface-leaving coefficients compatible with ``py2sess`` forward calls."""

    slterm_isotropic: np.ndarray
    slterm_f_0: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return coefficients using the mapping shape expected by ``py2sess``."""

        return {
            "slterm_isotropic": self.slterm_isotropic,
            "slterm_f_0": self.slterm_f_0,
        }


def _fortran_nint_positive(value: float) -> int:
    """Return Fortran ``NINT`` behavior for positive values."""

    return int(np.floor(value + 0.5))


def seawater_refractive_index(
    wavelength_microns: float, salinity_ppt: float
) -> tuple[float, float]:
    """Return sea-water real and imaginary refractive indices from ``INDWAT``.

    For wavelengths outside the tabulated interval, the Fortran routine uses
    the nearest table interval for linear extrapolation rather than clamping or
    raising an error. This function keeps that behavior for core parity.
    """

    if not np.isfinite(wavelength_microns):
        raise ValueError("wavelength_microns must be finite")
    if not np.isfinite(salinity_ppt):
        raise ValueError("salinity_ppt must be finite")

    index = int(np.searchsorted(_INDWAT_WAVELENGTHS, wavelength_microns, side="right"))
    index = min(max(index, 1), _INDWAT_WAVELENGTHS.size - 1)
    low = index - 1
    high = index
    span = _INDWAT_WAVELENGTHS[high] - _INDWAT_WAVELENGTHS[low]
    fraction = (wavelength_microns - _INDWAT_WAVELENGTHS[low]) / span
    real_index = _INDWAT_REAL_INDEX[low] + fraction * (
        _INDWAT_REAL_INDEX[high] - _INDWAT_REAL_INDEX[low]
    )
    imag_index = _INDWAT_IMAG_INDEX[low] + fraction * (
        _INDWAT_IMAG_INDEX[high] - _INDWAT_IMAG_INDEX[low]
    )
    real_index += 0.006 * (salinity_ppt / 34.3)
    return float(real_index), float(imag_index)


def morcasiwat_reflectance(wavelength_microns: float, chlorophyll_mg_m3: float) -> float:
    """Return the below-surface water reflectance ratio from ``MORCASIWAT``."""

    if not np.isfinite(wavelength_microns):
        raise ValueError("wavelength_microns must be finite")
    if not np.isfinite(chlorophyll_mg_m3):
        raise ValueError("chlorophyll_mg_m3 must be finite")
    if wavelength_microns < 0.350 or wavelength_microns > 0.700:
        return 0.0

    iwl = _fortran_nint_positive((wavelength_microns - 0.350) / 0.005)
    iwl = min(max(iwl, 0), _MORCASIWAT_KW.size - 1)
    kw = _MORCASIWAT_KW[iwl]
    xc = _MORCASIWAT_XC[iwl]
    exponent = _MORCASIWAT_E[iwl]
    bw = np.exp(
        1.63886
        - 25.9836 * wavelength_microns
        + 26.9625 * wavelength_microns**2
        - 12.0565 * wavelength_microns**3
    )

    if abs(chlorophyll_mg_m3) < 0.001:
        bb = 0.5 * bw
        return float(0.33 * bb / 0.75 / kw)

    if chlorophyll_mg_m3 <= 0.0:
        raise ValueError(
            "chlorophyll_mg_m3 must be positive unless its absolute value is below 0.001"
        )

    bcoef = 0.416 * chlorophyll_mg_m3**0.766
    vu = 0.0
    if 0.02 <= chlorophyll_mg_m3 < 2.0:
        vu = 0.5 * (np.log10(chlorophyll_mg_m3) - 0.3)
    if chlorophyll_mg_m3 < 0.02:
        vu = -1.0
    bbt = (
        0.002
        + 0.01 * (0.5 - 0.25 * np.log10(chlorophyll_mg_m3)) * (0.550 / wavelength_microns) ** vu
    )
    bb = 0.5 * bw + bbt * bcoef
    kd = kw + xc * chlorophyll_mg_m3**exponent
    r1 = 0.33 * bb / 0.75 / kd

    i1 = _fortran_nint_positive((np.log10(chlorophyll_mg_m3) + 2.0) * 2.0) - 1
    i1 = min(max(i1, 0), 4)
    i2 = 7
    if wavelength_microns < 0.645:
        i2 = 6
    if wavelength_microns < 0.580:
        i2 = 5
    if wavelength_microns < 0.530:
        i2 = 4
    if wavelength_microns < 0.500:
        i2 = 3
    if wavelength_microns < 0.460:
        i2 = 2
    if wavelength_microns < 0.430:
        i2 = 1
    if wavelength_microns < 0.406:
        i2 = 0

    mud = _MORCASIWAT_MUDA[i1, i2]
    muu = 0.40
    for _ in range(100):
        u2 = mud * (1.0 - r1) / (1.0 + mud * r1 / muu)
        r2 = 0.33 * bb / u2 / kd
        if abs((r2 - r1) / r2) < 0.0001:
            return float(r2)
        r1 = r2
    raise RuntimeError("MORCASIWAT iteration did not converge")


def surface_leaving_from_water(
    *,
    n_beams: int,
    wavelength_microns: float,
    salinity_ppt: float,
    chlorophyll_mg_m3: float,
    do_isotropic: bool = True,
) -> SurfaceLeavingCoefficients:
    """Generate Fortran-compatible isotropic water-leaving coefficients.

    This ports the non-fluorescence branch of ``TWOSTREAM_SLEAVE_MASTER``. The
    returned object can be passed to ``TwoStreamEss.forward`` via
    ``surface_leaving=coefficients.as_dict()``.
    """

    if n_beams <= 0:
        raise ValueError("n_beams must be positive")
    slterm_isotropic = np.zeros(n_beams, dtype=np.float64)
    slterm_f_0 = np.zeros((n_beams, 2), dtype=np.float64)
    if not do_isotropic:
        return SurfaceLeavingCoefficients(slterm_isotropic=slterm_isotropic, slterm_f_0=slterm_f_0)

    real_index, imag_index = seawater_refractive_index(wavelength_microns, salinity_ppt)
    reflectance = morcasiwat_reflectance(wavelength_microns, chlorophyll_mg_m3)
    index_squared = real_index * real_index + imag_index * imag_index
    sleave = (1.0 / index_squared) * reflectance / (1.0 - 0.485 * reflectance)
    slterm_isotropic.fill(float(sleave))
    return SurfaceLeavingCoefficients(slterm_isotropic=slterm_isotropic, slterm_f_0=slterm_f_0)
