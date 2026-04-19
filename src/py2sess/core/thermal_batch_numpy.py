"""Batched NumPy helpers for thermal observation-geometry cases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .batch_accumulation import accumulate_upwelling_sources_numpy
from .bvp_batch import solve_thermal_bvp_batch
from .optical import delta_m_scale_optical_properties
from .taylor import vectorized_taylor_series_1


@dataclass(frozen=True)
class ThermalBatchNumpyResult:
    """Batched thermal endpoint radiances."""

    two_stream_toa: np.ndarray
    fo_total_up_toa: np.ndarray

    @property
    def total_toa(self) -> np.ndarray:
        """Returns 2S plus FO upwelling TOA radiance."""
        return self.two_stream_toa + self.fo_total_up_toa


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
    eigentrans = np.where(helpv > 88.0, 0.0, np.exp(-helpv))
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos1, xpos2, norm_saved


def _thermal_coefficients(delta_tau: np.ndarray, thermal_bb_input: np.ndarray):
    """Builds linear thermal-source coefficients for a wavelength batch."""
    lower = thermal_bb_input[:, :-1]
    upper = thermal_bb_input[:, 1:]
    return lower, (upper - lower) / delta_tau


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
    t_wupper0 = np.where(active, t_gmult_up * xpos2, 0.0)
    t_wupper1 = np.where(active, t_gmult_up * xpos1, 0.0)
    t_wlower0 = np.where(active, t_gmult_dn * xpos1, 0.0)
    t_wlower1 = np.where(active, t_gmult_dn * xpos2, 0.0)
    tterm_save = np.where(active, tterm, 0.0)
    return (
        (tcp0, tcp1, tcp2),
        (tcm0, tcm1, tcm2),
        tterm_save,
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
    tcutoff,
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
    tsgm_uu0 = -tsgm_uu1 - tcp2 * delta_tau
    tsgm_ud0 = -tsgm_ud1 - tcm2 * delta_tau
    su = tcp0 * hmult_1 + tsgm_uu0 * t_delt_userm + tsgm_uu1
    sd = tcm0 * hmult_2 + tsgm_ud0 * t_delt_userm + tsgm_ud1
    return tterm_save * (u_xpos * sd + u_xneg * su)


def _accumulate_upwelling_sources(
    *,
    layer_source: np.ndarray,
    layer_trans: np.ndarray,
    surface_source: np.ndarray,
) -> np.ndarray:
    """Evaluates the backward layer recurrence in vectorized batch form."""
    return accumulate_upwelling_sources_numpy(
        layer_source=layer_source,
        layer_trans=layer_trans,
        surface_source=surface_source,
    )


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
):
    """Computes batched 2S thermal upwelling TOA radiance."""
    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties(
        tau, omega, asymm, scaling
    )
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
        tcutoff=thermal_tcutoff,
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

    wlower0, _wlower1 = t_wlower
    idownsurf = (
        wlower0[:, -1] + lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
    ) * stream_value
    surface_source = surface_factor * albedo * idownsurf
    layer_source = lcon * u_xpos * hmult_2 + mcon * u_xneg * hmult_1 + layer_tsup_up
    return _accumulate_upwelling_sources(
        layer_source=layer_source,
        layer_trans=t_delt_userm,
        surface_source=surface_source,
    )


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
):
    """Computes batched FO thermal upwelling TOA radiance."""
    deltaus = tau * (1.0 - omega * scaling)
    lower_bb = thermal_bb_input[:, :-1]
    upper_bb = thermal_bb_input[:, 1:]
    single_scatter_scale = 1.0 - omega
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
    for n in range(tau.shape[1] - 1, -1, -1):
        cum_surface = lostrans[:, n] * cum_surface
        cum_atmos = lostrans[:, n] * cum_atmos + sources_up[:, n]
    return cum_atmos + cum_surface


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
    )
    return ThermalBatchNumpyResult(two_stream_toa=two_stream, fo_total_up_toa=fo)
