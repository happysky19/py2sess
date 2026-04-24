"""Closed-form formulas used by the analytic validation suite."""

from __future__ import annotations

import math

import numpy as np


def lambertian_surface_fo_radiance(
    *,
    fbeam: float,
    albedo: float,
    solar_zenith_degrees: float,
) -> float:
    return fbeam * albedo * math.cos(math.radians(solar_zenith_degrees)) / math.pi


def thermal_surface_only_up_profile(
    tau: np.ndarray,
    *,
    user_angle_degrees: float,
    surface_planck: float,
    emissivity: float,
) -> np.ndarray:
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_above = np.concatenate(([np.sum(tau)], np.cumsum(tau[1:][::-1])[::-1], [0.0]))
    return surface_planck * emissivity * np.exp(-cumulative_tau_above / mu)


def thermal_atmosphere_only_up_profile(
    tau: np.ndarray,
    *,
    user_angle_degrees: float,
    blackbody_value: float,
) -> np.ndarray:
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_above = np.concatenate(([np.sum(tau)], np.cumsum(tau[1:][::-1])[::-1], [0.0]))
    return blackbody_value * (1.0 - np.exp(-cumulative_tau_above / mu))


def thermal_atmosphere_only_down_profile(
    tau: np.ndarray,
    *,
    user_angle_degrees: float,
    blackbody_value: float,
) -> np.ndarray:
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_below = np.concatenate(([0.0], np.cumsum(tau)))
    return blackbody_value * (1.0 - np.exp(-cumulative_tau_below / mu))


def twostream_upward_flux_pair_from_isotropic_intensity(
    *,
    intensity: float,
    stream: float,
) -> np.ndarray:
    return np.array(
        [
            0.5 * intensity,
            2.0 * math.pi * stream * intensity,
        ],
        dtype=float,
    )


def solar_fo_single_scatter_isotropic_one_layer(
    *,
    tau: float,
    omega: float,
    solar_zenith_degrees: float,
    view_zenith_degrees: float,
    fbeam: float,
) -> float:
    mu0 = math.cos(math.radians(solar_zenith_degrees))
    mu = math.cos(math.radians(view_zenith_degrees))
    attenuation_rate = 1.0 / mu0 + 1.0 / mu
    return (
        fbeam
        * omega
        / (4.0 * math.pi)
        * (1.0 / mu)
        * (1.0 - math.exp(-tau * attenuation_rate))
        / attenuation_rate
    )


def thermal_fo_single_layer_uniform_source(
    *,
    tau: float,
    omega: float,
    user_angle_degrees: float,
    blackbody_value: float,
) -> float:
    mu = math.cos(math.radians(user_angle_degrees))
    return blackbody_value * (1.0 - omega) * (1.0 - math.exp(-tau / mu))
