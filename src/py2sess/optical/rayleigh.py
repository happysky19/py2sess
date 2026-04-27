"""Rayleigh optical-property helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayleighOpticalProperties:
    """Rayleigh cross section and depolarization ratio."""

    cross_section: np.ndarray
    depolarization: np.ndarray


def rayleigh_bodhaine(
    wavelengths_nm,
    *,
    co2_ppmv: float = 385.0,
) -> RayleighOpticalProperties:
    """Compute Rayleigh cross sections using the Fortran Bodhaine-style formula.

    Wavelengths are in nm. Cross sections are in cm^2 per molecule, matching
    the GEOCAPE CreateProps convention where multiplying by air column
    molecules/cm^2 gives layer Rayleigh optical depth.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("wavelengths_nm must be finite")
    if np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_nm must be positive")
    if not np.isfinite(co2_ppmv) or co2_ppmv < 0.0:
        raise ValueError("co2_ppmv must be finite and nonnegative")

    mo2 = 20.946
    mn2 = 78.084
    marg = 0.934

    s1_a = 8060.51
    s1_b = 2.48099e6
    s1_c = 132.274
    s1_d = 1.74557e4
    s1_e = 39.32957
    s2_a = 0.54
    s3_a = 1.034
    s3_b = 3.17e-4
    s3_c = 1.096
    s3_d = 1.385e-3
    s3_e = 1.448e-4

    nmol = 2.546899e19
    cons = 24.0 * np.pi**3
    co2 = 1.0e-6 * co2_ppmv

    wav_angstrom = wavelengths * 10.0
    lambda_microns = 1.0e-4 * wav_angstrom
    lambda_cm = 1.0e-8 * wav_angstrom
    inv_lambda_um2 = 1.0 / (lambda_microns * lambda_microns)

    n300m1 = s1_a + s1_b / (s1_c - inv_lambda_um2) + s1_d / (s1_e - inv_lambda_um2)
    n300m1 = n300m1 * 1.0e-8
    nco2m1 = n300m1 * (1.0 + s2_a * (co2 - 0.0003))
    nco2 = nco2m1 + 1.0
    nco2sq = nco2 * nco2

    fn2 = s3_a + s3_b * inv_lambda_um2
    fo2 = s3_c + s3_d * inv_lambda_um2 + s3_e * inv_lambda_um2 * inv_lambda_um2

    farg = 1.0
    fco2 = 1.15
    mair = mn2 + mo2 + marg + co2
    fair = (mn2 * fn2 + mo2 * fo2 + marg * farg + co2 * fco2) / mair
    depol = 6.0 * (fair - 1.0) / (3.0 + 7.0 * fair)

    nsqm1 = nco2sq - 1.0
    nsqp2 = nco2sq + 2.0
    term = nsqm1 / (lambda_cm * lambda_cm) / nmol / nsqp2
    cross_section = cons * term * term * fair

    return RayleighOpticalProperties(
        cross_section=cross_section,
        depolarization=depol,
    )
