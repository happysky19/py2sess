"""Hemispheric level-flux helpers for FO source replacements."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.special import expn

from ..optical.brdf_solar_obs import DISORT_HAPKE_IDX, _disort_hapke_kernel
from .fo_thermal import (
    _OPTICAL_THICKNESS_MIN,
    _plane_parallel_down_sources,
    _plane_parallel_up_sources,
    _thermal_coefficients,
)
from .preprocess import PreparedInputs


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mu_quadrature(n_mu: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(_validate_positive_integer("n_mu", n_mu))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _brdf_from_kernel_specs(
    kernel_specs: tuple[dict[str, object], ...],
    *,
    mu_i: float,
    mu_r: float,
    cphi: float,
) -> float:
    value = 0.0
    for spec in kernel_specs:
        which_brdf = int(spec["which_brdf"])
        factor = float(spec.get("factor", 1.0))
        if which_brdf == DISORT_HAPKE_IDX:
            value += factor * _disort_hapke_kernel(mu_i=mu_i, mu_r=mu_r, cphi=cphi)
        else:
            raise NotImplementedError(
                "direct-surface flux with exact BRDF integration is implemented "
                "for DISORT-Hapke kernel_specs only"
            )
    return value


def _brdf_direct_surface_profiles(
    prepared: PreparedInputs,
    *,
    use_twostream_quadrature: bool,
) -> dict[str, np.ndarray]:
    if prepared.brdf is None or prepared.brdf.kernel_specs is None:
        raise NotImplementedError("direct-surface flux with BRDF surfaces requires kernel_specs")

    deltaus = prepared.tau_arr
    tau_levels = np.concatenate(([0.0], np.cumsum(deltaus)))
    distance_to_surface = tau_levels[-1] - tau_levels

    ngeom = prepared.user_obsgeoms.shape[0]
    nlevels = deltaus.size + 1
    flux_up = np.zeros((ngeom, nlevels), dtype=float)
    flux_down = np.zeros_like(flux_up)
    flux_mean = np.zeros_like(flux_up)

    if use_twostream_quadrature:
        mu_nodes = np.array([float(prepared.stream_value)], dtype=float)
        mu_weights = np.array([1.0], dtype=float)
        phi_nodes = np.array([0.0], dtype=float)
        phi_weight = 2.0 * math.pi
    else:
        mu_nodes, mu_weights = _mu_quadrature(64)
        phi_nodes = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        phi_weight = 2.0 * math.pi / phi_nodes.size

    for geom_index in range(ngeom):
        mu0 = math.cos(math.radians(float(prepared.user_obsgeoms[geom_index, 0])))
        incident = prepared.flux_factor * mu0 * math.exp(-tau_levels[-1] / mu0) / math.pi
        for mu, mu_weight in zip(mu_nodes, mu_weights):
            attenuation = np.exp(-distance_to_surface / mu)
            if use_twostream_quadrature:
                brdf_value = float(prepared.brdf.brdf_f_0[geom_index, 0])
                angular_flux_weight = 2.0 * math.pi * mu
                angular_mean_weight = 2.0 * math.pi
            else:
                brdf_sum = 0.0
                for phi in phi_nodes:
                    brdf_sum += _brdf_from_kernel_specs(
                        prepared.brdf.kernel_specs,
                        mu_i=mu0,
                        mu_r=float(mu),
                        cphi=math.cos(float(phi)),
                    )
                brdf_value = brdf_sum * phi_weight
                angular_flux_weight = mu * mu_weight
                angular_mean_weight = mu_weight
            flux_up[geom_index] += incident * brdf_value * angular_flux_weight * attenuation
            flux_mean[geom_index] += (
                incident * brdf_value * angular_mean_weight * attenuation / (4.0 * math.pi)
            )

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def solar_direct_surface_flux_plane_parallel(
    prepared: PreparedInputs,
) -> dict[str, np.ndarray]:
    """Exact hemispheric flux from Lambertian direct-beam surface reflection."""
    if prepared.source_mode not in {"solar_obs", "solar_lat"}:
        raise ValueError("solar direct-surface flux requires solar inputs")
    if prepared.user_obsgeoms is None:
        raise ValueError("solar direct-surface flux requires observation geometries")
    if prepared.brdf is not None:
        return _brdf_direct_surface_profiles(prepared, use_twostream_quadrature=False)
    if prepared.surface_leaving is not None:
        raise NotImplementedError("direct-surface flux with surface leaving is not implemented")

    deltaus = prepared.tau_arr
    tau_levels = np.concatenate(([0.0], np.cumsum(deltaus)))
    distance_to_surface = tau_levels[-1] - tau_levels

    ngeom = prepared.user_obsgeoms.shape[0]
    nlevels = deltaus.size + 1
    flux_up = np.zeros((ngeom, nlevels), dtype=float)
    flux_down = np.zeros_like(flux_up)
    flux_mean = np.zeros_like(flux_up)

    for geom_index in range(ngeom):
        mu0 = math.cos(math.radians(float(prepared.user_obsgeoms[geom_index, 0])))
        surface_flux = (
            prepared.flux_factor * mu0 * float(prepared.albedo) * math.exp(-tau_levels[-1] / mu0)
        )
        flux_up[geom_index] = 2.0 * surface_flux * expn(3, distance_to_surface)
        flux_mean[geom_index] = 0.5 * surface_flux / math.pi * expn(2, distance_to_surface)

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def solar_twostream_direct_surface_flux_plane_parallel(
    prepared: PreparedInputs,
) -> dict[str, np.ndarray]:
    """Two-stream quadrature counterpart of Lambertian direct-beam reflection."""
    if prepared.source_mode not in {"solar_obs", "solar_lat"}:
        raise ValueError("solar direct-surface flux requires solar inputs")
    if prepared.user_obsgeoms is None:
        raise ValueError("solar direct-surface flux requires observation geometries")
    if prepared.brdf is not None:
        return _brdf_direct_surface_profiles(prepared, use_twostream_quadrature=True)
    if prepared.surface_leaving is not None:
        raise NotImplementedError("direct-surface flux with surface leaving is not implemented")

    deltaus = prepared.tau_arr
    tau_levels = np.concatenate(([0.0], np.cumsum(deltaus)))
    distance_to_surface = tau_levels[-1] - tau_levels
    mu1 = float(prepared.stream_value)
    attenuation = np.exp(-distance_to_surface / mu1)

    ngeom = prepared.user_obsgeoms.shape[0]
    nlevels = deltaus.size + 1
    flux_up = np.zeros((ngeom, nlevels), dtype=float)
    flux_down = np.zeros_like(flux_up)
    flux_mean = np.zeros_like(flux_up)

    for geom_index in range(ngeom):
        mu0 = math.cos(math.radians(float(prepared.user_obsgeoms[geom_index, 0])))
        surface_flux = (
            prepared.flux_factor * mu0 * float(prepared.albedo) * math.exp(-tau_levels[-1] / mu0)
        )
        flux_up[geom_index] = 2.0 * mu1 * surface_flux * attenuation
        flux_mean[geom_index] = surface_flux * attenuation / (2.0 * math.pi)

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def _thermal_profiles_for_mu(
    *,
    deltaus: np.ndarray,
    tcom: np.ndarray,
    mu: float,
    surface_source: float,
) -> tuple[np.ndarray, np.ndarray]:
    lostrans_up, sources_up = _plane_parallel_up_sources(
        deltaus=deltaus,
        tcom=tcom,
        mu1v=mu,
        cutoff=88.0,
    )
    lostrans_down, sources_down = _plane_parallel_down_sources(
        deltaus=deltaus,
        tcom=tcom,
        mu1v=mu,
        cutoff=88.0,
    )
    nlevels = deltaus.size + 1
    up = np.zeros(nlevels, dtype=float)
    down = np.zeros(nlevels, dtype=float)

    cumulative = surface_source
    up[-1] = cumulative
    for n in range(deltaus.size - 1, -1, -1):
        cumulative = lostrans_up[n] * cumulative + sources_up[n]
        up[n] = cumulative

    cumulative = 0.0
    down[0] = cumulative
    for n in range(deltaus.size):
        cumulative = sources_down[n] + lostrans_down[n] * cumulative
        down[n + 1] = cumulative
    return up, down


def thermal_fo_flux_plane_parallel(
    prepared: PreparedInputs,
    *,
    do_optical_deltam_scaling: bool,
    do_source_deltam_scaling: bool,
    n_mu: int = 32,
    n_phi: int | None = None,
) -> dict[str, np.ndarray]:
    """Computes plane-parallel FO thermal fluxes by hemispheric quadrature."""
    if prepared.source_mode != "thermal" or prepared.thermal is None:
        raise ValueError("thermal FO flux requires thermal inputs")
    if n_phi is not None:
        _validate_positive_integer("n_phi", n_phi)

    if do_optical_deltam_scaling:
        deltaus = prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling)
    else:
        deltaus = prepared.tau_arr.copy()
    np.putmask(deltaus, deltaus <= 0.0, _OPTICAL_THICKNESS_MIN)
    tcom = _thermal_coefficients(
        prepared,
        deltaus=deltaus,
        do_source_deltam_scaling=do_source_deltam_scaling,
    )
    mu_nodes, mu_weights = _mu_quadrature(n_mu)
    nlevels = deltaus.size + 1
    flux_up = np.zeros((1, nlevels), dtype=float)
    flux_down = np.zeros_like(flux_up)
    flux_mean = np.zeros_like(flux_up)
    surface_source = prepared.thermal.surfbb * prepared.thermal.emissivity

    for mu, weight in zip(mu_nodes, mu_weights):
        up, down = _thermal_profiles_for_mu(
            deltaus=deltaus,
            tcom=tcom,
            mu=float(mu),
            surface_source=float(surface_source),
        )
        flux_up[0] += 2.0 * math.pi * weight * mu * up
        flux_down[0] += 2.0 * math.pi * weight * mu * down
        flux_mean[0] += 0.5 * weight * (up + down)

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def thermal_twostream_source_flux_plane_parallel(
    prepared: PreparedInputs,
    *,
    do_optical_deltam_scaling: bool,
    do_source_deltam_scaling: bool,
) -> dict[str, np.ndarray]:
    """Two-stream quadrature counterpart of direct thermal source transmission."""
    if prepared.source_mode != "thermal" or prepared.thermal is None:
        raise ValueError("thermal source flux requires thermal inputs")

    if do_optical_deltam_scaling:
        deltaus = prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling)
    else:
        deltaus = prepared.tau_arr.copy()
    np.putmask(deltaus, deltaus <= 0.0, _OPTICAL_THICKNESS_MIN)

    tcom = _thermal_coefficients(
        prepared,
        deltaus=deltaus,
        do_source_deltam_scaling=do_source_deltam_scaling,
    )
    up, down = _thermal_profiles_for_mu(
        deltaus=deltaus,
        tcom=tcom,
        mu=float(prepared.stream_value),
        surface_source=float(prepared.thermal.surfbb * prepared.thermal.emissivity),
    )
    mu1 = float(prepared.stream_value)
    flux_up = (2.0 * math.pi * mu1 * up)[np.newaxis, :]
    flux_down = (2.0 * math.pi * mu1 * down)[np.newaxis, :]
    flux_mean = (0.5 * (up + down))[np.newaxis, :]

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }
