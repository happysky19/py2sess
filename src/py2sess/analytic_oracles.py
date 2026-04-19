"""Closed-form reference formulas used by the analytic validation suite."""

from __future__ import annotations

import math

import numpy as np


def lambertian_surface_fo_radiance(
    *,
    flux_factor: float,
    albedo: float,
    solar_zenith_degrees: float,
) -> float:
    """Returns the closed-form Lambertian surface-only solar FO radiance."""
    return flux_factor * albedo * math.cos(math.radians(solar_zenith_degrees)) / math.pi


def thermal_surface_only_up_profile(
    tau_arr: np.ndarray,
    *,
    user_angle_degrees: float,
    surfbb: float,
    emissivity: float,
) -> np.ndarray:
    """Returns the closed-form thermal surface-only upwelling profile."""
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_above = np.concatenate(
        ([np.sum(tau_arr)], np.cumsum(tau_arr[1:][::-1])[::-1], [0.0])
    )
    return surfbb * emissivity * np.exp(-cumulative_tau_above / mu)


def thermal_atmosphere_only_up_profile(
    tau_arr: np.ndarray,
    *,
    user_angle_degrees: float,
    blackbody_value: float,
) -> np.ndarray:
    """Returns the closed-form thermal atmosphere-only upwelling profile."""
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_above = np.concatenate(
        ([np.sum(tau_arr)], np.cumsum(tau_arr[1:][::-1])[::-1], [0.0])
    )
    return blackbody_value * (1.0 - np.exp(-cumulative_tau_above / mu))


def thermal_atmosphere_only_down_profile(
    tau_arr: np.ndarray,
    *,
    user_angle_degrees: float,
    blackbody_value: float,
) -> np.ndarray:
    """Returns the closed-form thermal atmosphere-only downwelling profile."""
    mu = math.cos(math.radians(user_angle_degrees))
    cumulative_tau_below = np.concatenate(([0.0], np.cumsum(tau_arr)))
    return blackbody_value * (1.0 - np.exp(-cumulative_tau_below / mu))


def twostream_upward_flux_pair_from_isotropic_intensity(
    *,
    intensity: float,
    stream_value: float,
) -> np.ndarray:
    """Returns the public 2S flux pair for an isotropic upwelling hemisphere."""
    return np.array(
        [
            0.5 * intensity,
            2.0 * math.pi * stream_value * intensity,
        ],
        dtype=float,
    )


def solar_fo_single_scatter_isotropic_one_layer(
    *,
    tau: float,
    omega: float,
    solar_zenith_degrees: float,
    view_zenith_degrees: float,
    flux_factor: float,
) -> float:
    """Returns the one-layer isotropic single-scatter solar FO radiance."""
    mu0 = math.cos(math.radians(solar_zenith_degrees))
    mu = math.cos(math.radians(view_zenith_degrees))
    attenuation_rate = 1.0 / mu0 + 1.0 / mu
    return (
        flux_factor
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
    """Returns the one-layer thermal FO radiance for a uniform source layer."""
    mu = math.cos(math.radians(user_angle_degrees))
    return blackbody_value * (1.0 - omega) * (1.0 - math.exp(-tau / mu))
