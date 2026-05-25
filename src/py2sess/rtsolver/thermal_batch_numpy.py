"""Batched NumPy helpers for thermal observation-geometry cases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .batch_accumulation import (
    accumulate_upwelling_profile_numpy,
    accumulate_upwelling_sources_numpy,
)
from .bvp_batch import solve_thermal_bvp_batch
from ..optical.delta_m import delta_m_scale_optical_properties
from .taylor import vectorized_taylor_series_1

_NUMBA_THERMAL_FO_MIN_BATCH = 8192
_OPTICAL_THICKNESS_MIN = 1.0e-12
_THERMAL_FO_KERNEL = None
_THERMAL_FO_IMPORT_FAILED = False


@dataclass(frozen=True)
class ThermalBatchNumpyResult:
    """Batched thermal endpoint radiances."""

    two_stream_toa: np.ndarray
    fo_total_up_toa: np.ndarray
    two_stream_profile: np.ndarray | None = None
    fo_total_up_profile: np.ndarray | None = None

    @property
    def total_toa(self) -> np.ndarray:
        """Returns 2S plus FO upwelling TOA radiance."""
        return self.two_stream_toa + self.fo_total_up_toa

    @property
    def total_profile(self) -> np.ndarray | None:
        """Returns 2S plus FO upwelling level radiance when available."""
        if self.two_stream_profile is None:
            return None
        if self.fo_total_up_profile is None:
            return self.two_stream_profile
        return self.two_stream_profile + self.fo_total_up_profile


@lru_cache(maxsize=None)
def _legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns cached Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    return np.polynomial.legendre.leggauss(n)


def _gauss_legendre_interval(x1: float, x2: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns Gauss-Legendre quadrature nodes and weights on an interval."""
    nodes, weights = _legendre_nodes_weights(n)
    midpoint = 0.5 * (x2 + x1)
    half_width = 0.5 * (x2 - x1)
    return midpoint + half_width * nodes, half_width * weights


def _exp_cutoff_owned(values: np.ndarray, cutoff: float) -> np.ndarray:
    if values.size == 0 or float(np.max(values)) <= cutoff:
        np.exp(-values, out=values)
        return values
    too_deep = values > cutoff
    np.exp(-values, out=values)
    np.putmask(values, too_deep, 0.0)
    return values


def _hom_solution_thermal(
    *,
    stream_value: float,
    pxsq: float,
    omega: np.ndarray,
    asymm: np.ndarray,
    delta_tau: np.ndarray,
):
    """Builds thermal homogeneous solutions for a wavelength batch."""
    xinv = 1.0 / stream_value
    omega_asymm_3 = 3.0 * omega * asymm
    sab = xinv * (omega - 1.0)
    dab = xinv * (pxsq * omega_asymm_3 - 1.0)
    eigenvalue = np.sqrt(sab * dab)
    helpv = eigenvalue * delta_tau
    eigentrans = _exp_cutoff_owned(helpv, 88.0)
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos1, xpos2, norm_saved


def _thermal_coefficients(delta_tau: np.ndarray, thermal_bb_input: np.ndarray):
    """Builds linear thermal-source coefficients for a wavelength batch."""
    lower = thermal_bb_input[:, :-1]
    upper = thermal_bb_input[:, 1:]
    slope = np.zeros_like(delta_tau)
    np.divide(upper - lower, delta_tau, out=slope, where=delta_tau != 0.0)
    return lower, slope


def _thermal_green_function(
    *,
    omega,
    delta_tau,
    therm0,
    therm1,
    tcutoff,
    eigenvalue,
    eigentrans,
    xpos1,
    xpos2,
    norm_saved,
):
    """Builds thermal Green-function source terms in batch."""
    active = delta_tau > tcutoff
    tterm = (1.0 - omega) * (xpos1 + xpos2) / norm_saved
    k1 = 1.0 / eigenvalue
    tcm2 = k1 * therm1
    tcp2 = tcm2
    tcm1 = k1 * (therm0 - tcm2)
    tcp1 = k1 * (therm0 + tcp2)
    sum_m = tcm1 + tcm2 * delta_tau
    sum_p = tcp1 + tcp2 * delta_tau
    tcm0 = -tcm1
    tcp0 = -sum_p
    t_gmult_dn = tterm * (eigentrans * tcm0 + sum_m)
    t_gmult_up = tterm * (eigentrans * tcp0 + tcp1)
    t_wupper0 = t_gmult_up * xpos2
    t_wupper1 = t_gmult_up * xpos1
    t_wlower0 = t_gmult_dn * xpos1
    t_wlower1 = t_gmult_dn * xpos2
    inactive = ~active
    if np.any(inactive):
        t_wupper0[inactive] = 0.0
        t_wupper1[inactive] = 0.0
        t_wlower0[inactive] = 0.0
        t_wlower1[inactive] = 0.0
        tterm[inactive] = 0.0
    return (
        (tcp0, tcp1, tcp2),
        (tcm0, tcm1, tcm2),
        tterm,
        (t_wupper0, t_wupper1),
        (t_wlower0, t_wlower1),
    )


def _homogeneous_multipliers(
    *, delta_tau, user_secant: float, eigenvalue, eigentrans, t_delt_userm
):
    """Builds thermal homogeneous multipliers for one user stream."""
    zp = user_secant + eigenvalue
    zm = user_secant - eigenvalue
    hmult_2 = user_secant * (1.0 - eigentrans * t_delt_userm) / zp
    with np.errstate(divide="ignore", invalid="ignore"):
        hmult_1 = user_secant * (eigentrans - t_delt_userm) / zm
    near = np.abs(zm) < 1.0e-3
    if np.any(near):
        hmult_1 = hmult_1.copy()
        hmult_1[near] = vectorized_taylor_series_1(
            3,
            zm[near],
            delta_tau[near],
            t_delt_userm[near],
            user_secant,
        )
    return hmult_1, hmult_2


def _thermal_user_solution(*, stream_value: float, user_stream: float, xpos1, xpos2, omega, asymm):
    """Builds user-angle homogeneous solutions for one thermal user angle."""
    hmu_stream = 0.5 * stream_value
    u_help_p0 = (xpos2 + xpos1) * 0.5
    u_help_p1 = (xpos2 - xpos1) * hmu_stream
    omega_mom = 3.0 * omega * asymm
    u_xpos = u_help_p0 * omega + u_help_p1 * omega_mom * user_stream
    u_xneg = u_help_p0 * omega - u_help_p1 * omega_mom * user_stream
    return u_xpos, u_xneg


def _thermal_layer_sources_up(
    *,
    user_stream,
    t_delt_userm,
    delta_tau,
    u_xpos,
    u_xneg,
    hmult_1,
    hmult_2,
    t_c_plus,
    t_c_minus,
    tterm_save,
):
    """Computes batched upwelling thermal layer sources for one user angle."""
    tcp0, tcp1, tcp2 = t_c_plus
    tcm0, tcm1, tcm2 = t_c_minus
    tsgm_uu1 = tcp1 + user_stream * tcp2
    tsgm_ud1 = tcm1 + user_stream * tcm2
    one_minus_t_delt_userm = 1.0 - t_delt_userm
    su = tcp0 * hmult_1 + tsgm_uu1 * one_minus_t_delt_userm - tcp2 * delta_tau * t_delt_userm
    sd = tcm0 * hmult_2 + tsgm_ud1 * one_minus_t_delt_userm - tcm2 * delta_tau * t_delt_userm
    return tterm_save * (u_xpos * sd + u_xneg * su)


def _transparent_thermal_flux_numpy(
    *,
    tau,
    surfbb,
    emissivity,
    stream_value: float,
    return_profile: bool,
    return_fluxes: bool,
    do_upwelling: bool,
    do_dnwelling: bool,
):
    nrows, nlay = tau.shape
    nlev = nlay + 1
    radiance = np.zeros((nrows, nlev) if return_profile else nrows, dtype=float)
    if not return_fluxes:
        return radiance

    surface_source = surfbb * emissivity
    flux_up = np.zeros((nrows, nlev), dtype=float)
    flux_down = np.zeros_like(flux_up)
    flux_mean = np.zeros_like(flux_up)
    if do_upwelling:
        flux_up += 2.0 * np.pi * stream_value * surface_source[:, None]
        flux_mean += 0.5 * surface_source[:, None]
    if do_dnwelling:
        flux_down += 0.0
    return {
        "radiance": radiance,
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def _flux_profile_thermal_batch_numpy(
    *,
    do_upwelling: bool,
    do_dnwelling: bool,
    stream_value: float,
    active_layers,
    lcon,
    mcon,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
):
    """Builds DISORT-style level fluxes from batched thermal 2S quadrature values."""
    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    up_upper = wupper1 + lcon * xpos2 + mcon * xpos1 * eigentrans
    down_upper = wupper0 + lcon * xpos1 + mcon * xpos2 * eigentrans
    up_lower = (
        wlower1[:, -1] + lcon[:, -1] * xpos2[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos1[:, -1]
    )
    down_lower = (
        wlower0[:, -1] + lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
    )
    up_quad = np.concatenate((up_upper, up_lower[:, None]), axis=1)
    down_quad = np.concatenate((down_upper, down_lower[:, None]), axis=1)

    flux_up = np.zeros_like(up_quad)
    flux_down = np.zeros_like(down_quad)
    flux_mean = np.zeros_like(up_quad)
    if do_upwelling:
        flux_up = 2.0 * np.pi * stream_value * up_quad
        flux_mean += 0.5 * up_quad
    if do_dnwelling:
        flux_down = 2.0 * np.pi * stream_value * down_quad
        flux_mean += 0.5 * down_quad

    inactive_layers = ~active_layers
    if np.any(inactive_layers):
        flux_up = flux_up.copy()
        flux_down = flux_down.copy()
        flux_mean = flux_mean.copy()
        for level in range(inactive_layers.shape[1] - 1, -1, -1):
            mask = inactive_layers[:, level]
            flux_up[mask, level] = flux_up[mask, level + 1]
            flux_down[mask, level] = flux_down[mask, level + 1]
            flux_mean[mask, level] = flux_mean[mask, level + 1]
    return flux_up, flux_down, flux_up - flux_down, flux_mean


def _two_stream_thermal_toa(
    *,
    tau,
    omega,
    asymm,
    scaling,
    thermal_bb_input,
    surfbb,
    emissivity,
    albedo,
    stream_value,
    user_stream,
    thermal_tcutoff,
    bvp_engine: str = "auto",
    return_profile: bool = False,
    return_fluxes: bool = False,
    do_upwelling: bool = True,
    do_dnwelling: bool = False,
):
    """Computes batched 2S thermal upwelling TOA radiance."""
    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties(
        tau, omega, asymm, scaling
    )
    transparent_rows = np.all(delta_tau == 0.0, axis=1)
    if np.all(transparent_rows):
        return _transparent_thermal_flux_numpy(
            tau=tau,
            surfbb=surfbb,
            emissivity=emissivity,
            stream_value=stream_value,
            return_profile=return_profile,
            return_fluxes=return_fluxes,
            do_upwelling=do_upwelling,
            do_dnwelling=do_dnwelling,
        )
    if np.any(transparent_rows):
        active_rows = ~transparent_rows
        subset = _two_stream_thermal_toa(
            tau=tau[active_rows],
            omega=omega[active_rows],
            asymm=asymm[active_rows],
            scaling=scaling[active_rows],
            thermal_bb_input=thermal_bb_input[active_rows],
            surfbb=surfbb[active_rows],
            emissivity=emissivity[active_rows],
            albedo=albedo[active_rows],
            stream_value=stream_value,
            user_stream=user_stream,
            thermal_tcutoff=thermal_tcutoff,
            bvp_engine=bvp_engine,
            return_profile=return_profile,
            return_fluxes=return_fluxes,
            do_upwelling=do_upwelling,
            do_dnwelling=do_dnwelling,
        )
        result = _transparent_thermal_flux_numpy(
            tau=tau,
            surfbb=surfbb,
            emissivity=emissivity,
            stream_value=stream_value,
            return_profile=return_profile,
            return_fluxes=return_fluxes,
            do_upwelling=do_upwelling,
            do_dnwelling=do_dnwelling,
        )
        if return_fluxes:
            result["radiance"][active_rows] = subset["radiance"]
            for key in ("flux_up", "flux_down", "flux_net", "flux_mean"):
                result[key][active_rows] = subset[key]
            return result
        result[active_rows] = subset
        return result
    therm0, therm1 = _thermal_coefficients(delta_tau, thermal_bb_input)
    eigenvalue, eigentrans, xpos1, xpos2, norm_saved = _hom_solution_thermal(
        stream_value=stream_value,
        pxsq=stream_value * stream_value,
        omega=omega_total,
        asymm=asymm_total,
        delta_tau=delta_tau,
    )
    user_secant = 1.0 / user_stream
    t_delt_userm = np.exp(-delta_tau * user_secant)
    u_xpos, u_xneg = _thermal_user_solution(
        stream_value=stream_value,
        user_stream=user_stream,
        xpos1=xpos1,
        xpos2=xpos2,
        omega=omega_total,
        asymm=asymm_total,
    )
    hmult_1, hmult_2 = _homogeneous_multipliers(
        delta_tau=delta_tau,
        user_secant=user_secant,
        eigenvalue=eigenvalue,
        eigentrans=eigentrans,
        t_delt_userm=t_delt_userm,
    )
    t_c_plus, t_c_minus, tterm_save, t_wupper, t_wlower = _thermal_green_function(
        omega=omega_total,
        delta_tau=delta_tau,
        therm0=therm0,
        therm1=therm1,
        tcutoff=thermal_tcutoff,
        eigenvalue=eigenvalue,
        eigentrans=eigentrans,
        xpos1=xpos1,
        xpos2=xpos2,
        norm_saved=norm_saved,
    )
    layer_tsup_up = _thermal_layer_sources_up(
        user_stream=user_stream,
        t_delt_userm=t_delt_userm,
        delta_tau=delta_tau,
        u_xpos=u_xpos,
        u_xneg=u_xneg,
        hmult_1=hmult_1,
        hmult_2=hmult_2,
        t_c_plus=t_c_plus,
        t_c_minus=t_c_minus,
        tterm_save=tterm_save,
    )

    surface_factor = 2.0
    lcon, mcon = solve_thermal_bvp_batch(
        albedo=albedo,
        emissivity=emissivity,
        surfbb=surfbb,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=t_wupper,
        wlower=t_wlower,
        bvp_engine=bvp_engine,
    )
    flux_up = None
    flux_down = None
    flux_net = None
    flux_mean = None
    if return_fluxes:
        flux_up, flux_down, flux_net, flux_mean = _flux_profile_thermal_batch_numpy(
            do_upwelling=do_upwelling,
            do_dnwelling=do_dnwelling,
            stream_value=stream_value,
            active_layers=delta_tau > thermal_tcutoff,
            lcon=lcon,
            mcon=mcon,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=t_wupper,
            wlower=t_wlower,
        )

    wlower0, _wlower1 = t_wlower
    idownsurf = (
        wlower0[:, -1] + lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
    ) * stream_value
    surface_source = surface_factor * albedo * idownsurf
    layer_source = lcon * u_xpos * hmult_2 + mcon * u_xneg * hmult_1 + layer_tsup_up
    if return_profile:
        radiance = accumulate_upwelling_profile_numpy(
            layer_source=layer_source,
            layer_trans=t_delt_userm,
            surface_source=surface_source,
        )
    else:
        radiance = accumulate_upwelling_sources_numpy(
            layer_source=layer_source,
            layer_trans=t_delt_userm,
            surface_source=surface_source,
        )
    if return_fluxes:
        return {
            "radiance": radiance,
            "flux_up": flux_up,
            "flux_down": flux_down,
            "flux_net": flux_net,
            "flux_mean": flux_mean,
        }
    return radiance


def precompute_fo_thermal_geometry_numpy(
    *,
    heights,
    user_angle_degrees: float,
    earth_radius: float = 6371.0,
    nfine: int = 3,
):
    """Precomputes FO thermal line-of-sight geometry for repeated batch chunks.

    The solar EPS geometry builder also computes direct-sun paths. Thermal FO
    only needs the line-of-sight quadrature terms, so this keeps the same
    geometry convention without doing the unused solar-path work.
    """
    height_grid = np.asarray(heights, dtype=float)
    nlayers = height_grid.size - 1
    ngeoms = 1
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

    vza = float(user_angle_degrees)
    if math.isclose(vza, 0.0):
        do_nadir[0] = True
        for n in range(nlayers, 0, -1):
            layer_thickness = radii[n - 1] - radii[n]
            x_nodes, w_nodes = _gauss_legendre_interval(0.0, float(layer_thickness), nfine)
            for j in range(nfine):
                xfine[j, n - 1, 0] = x_nodes[j]
                wfine[j, n - 1, 0] = w_nodes[j]
    else:
        alpha_boa_r = math.radians(vza)
        sin_alpha_boa = 1.0 if math.isclose(vza, 90.0) else math.sin(alpha_boa_r)
        cos_alpha_boa = 0.0 if math.isclose(vza, 90.0) else math.cos(alpha_boa_r)
        cota[nlayers, 0] = cos_alpha_boa / sin_alpha_boa
        alpha[nlayers, 0] = alpha_boa_r
        rayconv = sin_alpha_boa * radii[nlayers]
        raycon[0] = rayconv
        for n in range(nlayers - 1, -1, -1):
            sin_alpha = rayconv / radii[n]
            alpha_n = math.asin(sin_alpha)
            alpha[n, 0] = alpha_n
            cota[n, 0] = math.cos(alpha_n) / sin_alpha

        alpha_lower = alpha[nlayers, 0]
        for n in range(nlayers, 0, -1):
            alpha_upper = alpha[n - 1, 0]
            t_nodes, w_nodes = _gauss_legendre_interval(
                float(alpha_upper), float(alpha_lower), nfine
            )
            for j in range(nfine):
                sin_t = math.sin(t_nodes[j])
                cosec_t = 1.0 / sin_t
                fine_radius = rayconv * cosec_t
                xfine[j, n - 1, 0] = radii[n - 1] - fine_radius
                wfine[j, n - 1, 0] = w_nodes[j]
                cotfine[j, n - 1, 0] = math.cos(t_nodes[j]) * cosec_t
                csqfine[j, n - 1, 0] = cosec_t * cosec_t
            alpha_lower = alpha_upper

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
    }


def _get_thermal_fo_kernel():
    global _THERMAL_FO_KERNEL, _THERMAL_FO_IMPORT_FAILED
    if _THERMAL_FO_KERNEL is not None:
        return _THERMAL_FO_KERNEL
    if _THERMAL_FO_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional acceleration dependency
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(
            tau,
            omega,
            scaling,
            thermal_bb_input,
            surfbb,
            emissivity,
            heights,
            do_nadir,
            rayconv,
            cota,
            xfine,
            wfine,
            csqfine,
            cotfine,
            do_optical_deltam_scaling,
            do_source_deltam_scaling,
        ):
            batch, nlay = tau.shape
            nfine = xfine.shape[0]
            out = np.empty(batch, np.float64)
            for row in prange(batch):
                cum_atmos = 0.0
                cum_surface = surfbb[row] * emissivity[row]
                for layer in range(nlay - 1, -1, -1):
                    tau_layer = tau[row, layer]
                    omega_layer = omega[row, layer]
                    scaling_layer = scaling[row, layer]
                    if do_optical_deltam_scaling:
                        deltaus = tau_layer * (1.0 - omega_layer * scaling_layer)
                    else:
                        deltaus = tau_layer
                    extinction = deltaus / (heights[layer] - heights[layer + 1])
                    source_scale = 1.0 - omega_layer
                    if do_source_deltam_scaling:
                        source_scale = source_scale / (1.0 - omega_layer * scaling_layer)
                    lower_bb = thermal_bb_input[row, layer]
                    therm0 = lower_bb * source_scale
                    therm1 = (thermal_bb_input[row, layer + 1] - lower_bb) / deltaus * source_scale

                    source = 0.0
                    if do_nadir:
                        lostrans = math.exp(-deltaus) if deltaus < 88.0 else 0.0
                        for j in range(nfine):
                            xjkn = extinction * xfine[j, layer]
                            solution = therm0 + xjkn * therm1
                            source += solution * extinction * math.exp(-xjkn) * wfine[j, layer]
                    else:
                        ke = rayconv * extinction
                        lostau = ke * (cota[layer] - cota[layer + 1])
                        lostrans = math.exp(-lostau) if lostau < 88.0 else 0.0
                        for j in range(nfine):
                            xjkn = extinction * xfine[j, layer]
                            solution = therm0 + xjkn * therm1
                            weight = ke * csqfine[j, layer] * wfine[j, layer]
                            optical_path = ke * (cota[layer] - cotfine[j, layer])
                            source += solution * weight * math.exp(-optical_path)
                    cum_surface = lostrans * cum_surface
                    cum_atmos = lostrans * cum_atmos + source
                out[row] = cum_atmos + cum_surface
            return out

        _THERMAL_FO_KERNEL = _kernel
        return _THERMAL_FO_KERNEL
    except Exception:  # pragma: no cover - optional acceleration dependency
        _THERMAL_FO_IMPORT_FAILED = True
        return None


def _fo_thermal_toa_numba(
    *,
    tau,
    omega,
    scaling,
    thermal_bb_input,
    surfbb,
    emissivity,
    heights,
    geometry,
    do_optical_deltam_scaling: bool,
    do_source_deltam_scaling: bool,
) -> np.ndarray | None:
    if tau.shape[0] < _NUMBA_THERMAL_FO_MIN_BATCH:
        return None
    kernel = _get_thermal_fo_kernel()
    if kernel is None:
        return None
    return kernel(
        np.ascontiguousarray(tau, dtype=np.float64),
        np.ascontiguousarray(omega, dtype=np.float64),
        np.ascontiguousarray(scaling, dtype=np.float64),
        np.ascontiguousarray(thermal_bb_input, dtype=np.float64),
        np.ascontiguousarray(surfbb, dtype=np.float64),
        np.ascontiguousarray(emissivity, dtype=np.float64),
        np.ascontiguousarray(heights, dtype=np.float64),
        bool(geometry["do_nadir"][0]),
        float(geometry["raycon"][0]),
        np.ascontiguousarray(geometry["cota"][:, 0], dtype=np.float64),
        np.ascontiguousarray(geometry["xfine"][:, :, 0], dtype=np.float64),
        np.ascontiguousarray(geometry["wfine"][:, :, 0], dtype=np.float64),
        np.ascontiguousarray(geometry["csqfine"][:, :, 0], dtype=np.float64),
        np.ascontiguousarray(geometry["cotfine"][:, :, 0], dtype=np.float64),
        bool(do_optical_deltam_scaling),
        bool(do_source_deltam_scaling),
    )


def _fo_thermal_toa(
    *,
    tau,
    omega,
    scaling,
    thermal_bb_input,
    surfbb,
    emissivity,
    heights,
    user_angle_degrees,
    earth_radius,
    nfine,
    geometry=None,
    return_profile: bool = False,
    do_optical_deltam_scaling: bool = True,
    do_source_deltam_scaling: bool = False,
):
    """Computes batched FO thermal upwelling TOA radiance."""
    deltaus = tau * (1.0 - omega * scaling) if do_optical_deltam_scaling else tau
    deltaus = np.array(deltaus, dtype=float, copy=True)
    np.putmask(deltaus, deltaus <= 0.0, _OPTICAL_THICKNESS_MIN)
    lower_bb = thermal_bb_input[:, :-1]
    upper_bb = thermal_bb_input[:, 1:]
    single_scatter_scale = 1.0 - omega
    if do_source_deltam_scaling:
        single_scatter_scale = single_scatter_scale / (1.0 - omega * scaling)
    therm0 = lower_bb * single_scatter_scale
    therm1 = ((upper_bb - lower_bb) / deltaus) * single_scatter_scale
    extinction = deltaus / (heights[:-1] - heights[1:])
    if geometry is None:
        geometry = precompute_fo_thermal_geometry_numpy(
            heights=heights,
            user_angle_degrees=user_angle_degrees,
            earth_radius=earth_radius,
            nfine=nfine,
        )
    if not return_profile:
        accelerated = _fo_thermal_toa_numba(
            tau=tau,
            omega=omega,
            scaling=scaling,
            thermal_bb_input=thermal_bb_input,
            surfbb=surfbb,
            emissivity=emissivity,
            heights=heights,
            geometry=geometry,
            do_optical_deltam_scaling=do_optical_deltam_scaling,
            do_source_deltam_scaling=do_source_deltam_scaling,
        )
        if accelerated is not None:
            return accelerated
    do_nadir = bool(geometry["do_nadir"][0])
    if do_nadir:
        xfine = geometry["xfine"][:, :, 0]
        wfine = geometry["wfine"][:, :, 0]
        lostrans = np.where(deltaus < 88.0, np.exp(-deltaus), 0.0)
        sources_up = np.zeros_like(deltaus)
        for j in range(xfine.shape[0]):
            xjkn = extinction * xfine[j]
            solution = therm0 + xjkn * therm1
            sources_up += solution * extinction * np.exp(-xjkn) * wfine[j]
    else:
        rayconv = float(geometry["raycon"][0])
        cota = geometry["cota"][:, 0]
        cot_upper = cota[:-1]
        cot_lower = cota[1:]
        ke = rayconv * extinction
        lostau = ke * (cot_upper - cot_lower)
        lostrans = np.where(lostau < 88.0, np.exp(-lostau), 0.0)
        xfine = geometry["xfine"][:, :, 0]
        wfine = geometry["wfine"][:, :, 0]
        cotfine = geometry["cotfine"][:, :, 0]
        csqfine = geometry["csqfine"][:, :, 0]
        sources_up = np.zeros_like(deltaus)
        for j in range(xfine.shape[0]):
            xjkn = extinction * xfine[j]
            solution = therm0 + xjkn * therm1
            weight = ke * csqfine[j] * wfine[j]
            optical_path = ke * (cot_upper - cotfine[j])
            sources_up += solution * weight * np.exp(-optical_path)
    cum_atmos = np.zeros(tau.shape[0], dtype=float)
    cum_surface = surfbb * emissivity
    profile = None
    if return_profile:
        profile = np.empty((tau.shape[0], tau.shape[1] + 1), dtype=float)
        profile[:, tau.shape[1]] = cum_atmos + cum_surface
    for n in range(tau.shape[1] - 1, -1, -1):
        cum_surface = lostrans[:, n] * cum_surface
        cum_atmos = lostrans[:, n] * cum_atmos + sources_up[:, n]
        if return_profile:
            profile[:, n] = cum_atmos + cum_surface
    if return_profile:
        return profile
    return cum_atmos + cum_surface


def _mu_quadrature(n_mu: int) -> tuple[np.ndarray, np.ndarray]:
    if int(n_mu) <= 0:
        raise ValueError("n_mu must be a positive integer")
    nodes, weights = _legendre_nodes_weights(int(n_mu))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _thermal_fo_profiles_for_mu_numpy(
    *,
    deltaus: np.ndarray,
    therm0: np.ndarray,
    therm1: np.ndarray,
    mu: float,
    surface_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lostau = deltaus / mu
    lostrans = np.where(lostau < 88.0, np.exp(-lostau), 0.0)
    one_minus_trans = np.where(lostau < 88.0, -np.expm1(-lostau), 1.0)
    ratio = one_minus_trans / lostau
    source_delta = therm1 * deltaus
    sources_up = therm0 * one_minus_trans + source_delta * (ratio - lostrans)
    sources_down = therm0 * one_minus_trans + source_delta * (1.0 - ratio)

    nrows, nlay = deltaus.shape
    up = np.zeros((nrows, nlay + 1), dtype=float)
    down = np.zeros_like(up)
    cumulative = surface_source.copy()
    up[:, nlay] = cumulative
    for n in range(nlay - 1, -1, -1):
        cumulative = lostrans[:, n] * cumulative + sources_up[:, n]
        up[:, n] = cumulative

    cumulative = np.zeros(nrows, dtype=float)
    down[:, 0] = cumulative
    for n in range(nlay):
        cumulative = sources_down[:, n] + lostrans[:, n] * cumulative
        down[:, n + 1] = cumulative
    return up, down


def thermal_fo_flux_correction_numpy(
    *,
    tau,
    omega,
    scaling,
    thermal_bb_input,
    surfbb,
    emissivity,
    stream_value: float,
    do_optical_deltam_scaling: bool,
    do_source_deltam_scaling: bool,
    n_mu: int = 8,
    n_phi: int | None = None,
):
    """Returns exact-minus-2S thermal FO level-flux correction for a batch."""
    if n_phi is not None and int(n_phi) <= 0:
        raise ValueError("n_phi must be a positive integer")
    deltaus = tau * (1.0 - omega * scaling) if do_optical_deltam_scaling else tau
    deltaus = np.array(deltaus, dtype=float, copy=True)
    np.putmask(deltaus, deltaus <= 0.0, _OPTICAL_THICKNESS_MIN)

    lower_bb = thermal_bb_input[:, :-1]
    upper_bb = thermal_bb_input[:, 1:]
    single_scatter_scale = 1.0 - omega
    if do_source_deltam_scaling:
        single_scatter_scale = single_scatter_scale / (1.0 - omega * scaling)
    therm0 = lower_bb * single_scatter_scale
    therm1 = ((upper_bb - lower_bb) / deltaus) * single_scatter_scale
    surface_source = surfbb * emissivity

    nrows, nlay = tau.shape
    nlev = nlay + 1
    exact_up = np.zeros((nrows, nlev), dtype=float)
    exact_down = np.zeros_like(exact_up)
    exact_mean = np.zeros_like(exact_up)
    mu_nodes, mu_weights = _mu_quadrature(n_mu)
    for mu, weight in zip(mu_nodes, mu_weights):
        up, down = _thermal_fo_profiles_for_mu_numpy(
            deltaus=deltaus,
            therm0=therm0,
            therm1=therm1,
            mu=float(mu),
            surface_source=surface_source,
        )
        exact_up += 2.0 * np.pi * float(weight) * float(mu) * up
        exact_down += 2.0 * np.pi * float(weight) * float(mu) * down
        exact_mean += 0.5 * float(weight) * (up + down)

    up_2s, down_2s = _thermal_fo_profiles_for_mu_numpy(
        deltaus=deltaus,
        therm0=therm0,
        therm1=therm1,
        mu=float(stream_value),
        surface_source=surface_source,
    )
    embedded_up = 2.0 * np.pi * stream_value * up_2s
    embedded_down = 2.0 * np.pi * stream_value * down_2s
    embedded_mean = 0.5 * (up_2s + down_2s)

    flux_up = exact_up - embedded_up
    flux_down = exact_down - embedded_down
    flux_mean = exact_mean - embedded_mean
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def solve_thermal_batch_numpy(
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    d2s_scaling,
    thermal_bb_input,
    surfbb,
    albedo,
    heights,
    user_angle_degrees: float,
    stream_value: float = 0.5,
    earth_radius: float = 6371.0,
    thermal_tcutoff: float = 1.0e-8,
    nfine: int = 3,
    emissivity=None,
    bvp_engine: str = "auto",
    fo_geometry=None,
    return_profiles: bool = False,
    do_fo_optical_deltam_scaling: bool = True,
    do_fo_source_deltam_scaling: bool = False,
) -> ThermalBatchNumpyResult:
    """Solves thermal observation-geometry spectra with NumPy arrays.

    Parameters
    ----------
    tau_arr, omega_arr, asymm_arr, d2s_scaling
        Layer optical properties with shape ``(n_spectral, n_layers)``.
    thermal_bb_input
        Atmospheric blackbody source values at layer boundaries with shape
        ``(n_spectral, n_layers + 1)``.
    surfbb, albedo
        Surface blackbody source and Lambertian albedo for each spectral point.
    heights
        Atmospheric level heights, ordered consistently with the scalar solver.
    user_angle_degrees
        Viewing zenith angle for the thermal observation geometry.
    stream_value
        Two-stream quadrature stream value.
    earth_radius, nfine
        Spherical-geometry controls used by the FO thermal path.
    emissivity
        Optional surface emissivity. When omitted, ``1 - albedo`` is used.
    fo_geometry
        Optional cached FO thermal geometry from
        :func:`precompute_fo_thermal_geometry_numpy`.
    do_fo_optical_deltam_scaling, do_fo_source_deltam_scaling
        Delta-M controls for the FO thermal path. The default applies the
        corrected optical-depth scaling and leaves source-side scaling off.

    Returns
    -------
    ThermalBatchNumpyResult
        Batched 2S thermal, FO thermal, and total TOA radiances.
    """
    tau = np.asarray(tau_arr, dtype=float)
    omega = np.asarray(omega_arr, dtype=float)
    asymm = np.asarray(asymm_arr, dtype=float)
    scaling = np.asarray(d2s_scaling, dtype=float)
    bb = np.asarray(thermal_bb_input, dtype=float)
    surfbb_arr = np.asarray(surfbb, dtype=float)
    albedo_arr = np.asarray(albedo, dtype=float)
    emissivity_arr = 1.0 - albedo_arr if emissivity is None else np.asarray(emissivity, dtype=float)
    heights_arr = np.asarray(heights, dtype=float)
    user_stream = float(np.cos(np.deg2rad(user_angle_degrees)))
    two_stream = _two_stream_thermal_toa(
        tau=tau,
        omega=omega,
        asymm=asymm,
        scaling=scaling,
        thermal_bb_input=bb,
        surfbb=surfbb_arr,
        emissivity=emissivity_arr,
        albedo=albedo_arr,
        stream_value=stream_value,
        user_stream=user_stream,
        thermal_tcutoff=thermal_tcutoff,
        bvp_engine=bvp_engine,
        return_profile=return_profiles,
    )
    fo = _fo_thermal_toa(
        tau=tau,
        omega=omega,
        scaling=scaling,
        thermal_bb_input=bb,
        surfbb=surfbb_arr,
        emissivity=emissivity_arr,
        heights=heights_arr,
        user_angle_degrees=user_angle_degrees,
        earth_radius=earth_radius,
        nfine=nfine,
        geometry=fo_geometry,
        return_profile=return_profiles,
        do_optical_deltam_scaling=do_fo_optical_deltam_scaling,
        do_source_deltam_scaling=do_fo_source_deltam_scaling,
    )
    if return_profiles:
        return ThermalBatchNumpyResult(
            two_stream_toa=two_stream[:, 0],
            fo_total_up_toa=fo[:, 0],
            two_stream_profile=two_stream,
            fo_total_up_profile=fo,
        )
    return ThermalBatchNumpyResult(two_stream_toa=two_stream, fo_total_up_toa=fo)
