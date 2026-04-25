"""Batched NumPy helpers for solar observation-geometry cases."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from .batch_accumulation import (
    accumulate_upwelling_profile_numpy,
    accumulate_upwelling_sources_numpy,
)
from .bvp_batch import solve_solar_observation_bvp_batch
from ..optical.delta_m import delta_m_scale_optical_properties
from .taylor import vectorized_taylor_series_1, vectorized_taylor_series_2


MAX_TAU_PATH = 88.0
MAX_TAU_QPATH = 88.0
TAYLOR_SMALL = 1.0e-3
TAYLOR_ORDER = 3


class _QsPrepBatch(TypedDict):
    """Container for precomputed spherical-transmittance terms."""

    layer_pis_cutoff: np.ndarray
    initial_trans: np.ndarray
    average_secant: np.ndarray
    trans_solar_beam: np.ndarray
    t_delt_mubar: np.ndarray
    itrans_userm: np.ndarray
    t_delt_userm: np.ndarray
    sigma_p: np.ndarray
    emult_up: np.ndarray
    all_active: bool


def _layer_cutoff_mask(nlayers: int, layer_pis_cutoff: np.ndarray) -> np.ndarray:
    """Returns the active-layer mask implied by the PI cutoff for each row."""
    return np.arange(1, nlayers + 1, dtype=int)[np.newaxis, :] <= layer_pis_cutoff[:, np.newaxis]


def _exp_cutoff(values: np.ndarray, cutoff: float) -> np.ndarray:
    """Returns ``exp(-values)`` with the Fortran optical-depth cutoff."""
    if values.size == 0 or float(np.max(values)) <= cutoff:
        return np.exp(-values)
    result = np.exp(-values)
    result[values > cutoff] = 0.0
    return result


def _exp_cutoff_owned(values: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies the cutoff exponential in place."""
    if values.size == 0 or float(np.max(values)) <= cutoff:
        np.exp(-values, out=values)
        return values
    too_deep = values > cutoff
    np.exp(-values, out=values)
    np.putmask(values, too_deep, 0.0)
    return values


def _qsprep_obs_batch(
    delta_tau: np.ndarray, chapman: np.ndarray, user_secant: float
) -> _QsPrepBatch:
    """Builds spherical solar transmittance and multiplier inputs in batch."""
    batch, nlayers = delta_tau.shape
    tauslant_all = delta_tau @ chapman
    delta_tauslant = np.empty_like(tauslant_all)
    delta_tauslant[:, 0] = tauslant_all[:, 0]
    delta_tauslant[:, 1:] = tauslant_all[:, 1:] - tauslant_all[:, :-1]
    too_deep = tauslant_all > MAX_TAU_PATH
    if not np.any(too_deep):
        average_secant = delta_tauslant / delta_tau
        np.exp(-delta_tauslant, out=delta_tauslant)
        t_delt_mubar = delta_tauslant
        np.exp(-tauslant_all, out=tauslant_all)
        trans_solar_beam = tauslant_all[:, -1].copy()
        initial_trans = np.empty_like(tauslant_all)
        initial_trans[:, 0] = 1.0
        initial_trans[:, 1:] = tauslant_all[:, :-1]
        itrans_userm = initial_trans * user_secant
        user_spher = delta_tau * user_secant
        t_delt_userm = _exp_cutoff_owned(user_spher, MAX_TAU_PATH)
        sigma_p = average_secant + user_secant
        emult_up = itrans_userm * (1.0 - t_delt_mubar * t_delt_userm) / sigma_p
        return {
            "layer_pis_cutoff": np.full(batch, nlayers, dtype=int),
            "initial_trans": initial_trans,
            "average_secant": average_secant,
            "trans_solar_beam": trans_solar_beam,
            "t_delt_mubar": t_delt_mubar,
            "itrans_userm": itrans_userm,
            "t_delt_userm": t_delt_userm,
            "sigma_p": sigma_p,
            "emult_up": emult_up,
            "all_active": True,
        }
    has_too_deep = np.any(too_deep, axis=1)
    first_too_deep = np.argmax(too_deep, axis=1) + 1
    cutoff = np.where(has_too_deep, first_too_deep, nlayers).astype(int, copy=False)
    active = _layer_cutoff_mask(nlayers, cutoff)
    zero = np.zeros_like(delta_tau)
    tauslant_previous = np.empty_like(tauslant_all)
    tauslant_previous[:, 0] = 0.0
    tauslant_previous[:, 1:] = tauslant_all[:, :-1]
    initial_trans_raw = np.exp(-tauslant_previous)
    initial_trans = np.where(active, initial_trans_raw, zero)
    average_secant_raw = delta_tauslant / delta_tau
    average_secant = np.where(active, average_secant_raw, zero)
    t_delt_mubar = np.where(
        active & (delta_tauslant <= MAX_TAU_PATH), np.exp(-delta_tauslant), zero
    )
    itrans_userm = initial_trans * user_secant
    trans_solar_beam = np.where(
        tauslant_all[:, -1] > MAX_TAU_PATH, 0.0, np.exp(-tauslant_all[:, -1])
    )
    user_spher = delta_tau * user_secant
    t_delt_userm = _exp_cutoff_owned(user_spher, MAX_TAU_PATH)
    sigma_p_raw = average_secant_raw + user_secant
    sigma_p = np.where(active, sigma_p_raw, zero)
    emult_up = np.where(
        active,
        itrans_userm * (1.0 - t_delt_mubar * t_delt_userm) / sigma_p_raw,
        zero,
    )
    return {
        "layer_pis_cutoff": cutoff,
        "initial_trans": initial_trans,
        "average_secant": average_secant,
        "trans_solar_beam": trans_solar_beam,
        "t_delt_mubar": t_delt_mubar,
        "itrans_userm": itrans_userm,
        "t_delt_userm": t_delt_userm,
        "sigma_p": sigma_p,
        "emult_up": emult_up,
        "all_active": False,
    }


def _hom_solution_solar_batch(
    *,
    fourier: int,
    stream_value: float,
    pxsq: float,
    omega: np.ndarray,
    omega_asymm_3: np.ndarray,
    delta_tau: np.ndarray,
):
    """Builds the solar homogeneous eigensystem for a wavelength batch."""
    xinv = 1.0 / stream_value
    if fourier == 0:
        sab = xinv * (omega - 1.0)
        dab = xinv * (pxsq * omega_asymm_3 - 1.0)
    else:
        sab = xinv * (pxsq * omega_asymm_3 - 1.0)
        dab = -xinv
    eigenvalue = np.sqrt(sab * dab)
    helpv = eigenvalue * delta_tau
    eigentrans = _exp_cutoff_owned(helpv, MAX_TAU_QPATH)
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos1, xpos2, norm_saved


def _hom_user_solution_solar_batch(
    *,
    fourier: int,
    stream_value: float,
    px11: float,
    user_stream: float,
    ulp: float,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    omega: np.ndarray,
    omega_asymm_3: np.ndarray,
):
    """Builds solar user-angle homogeneous solutions for one observation geometry."""
    hmu_stream = 0.5 * stream_value
    if fourier == 0:
        u_help_p1 = (xpos2 - xpos1) * hmu_stream
        common = 0.5 * omega
        scatter = u_help_p1 * omega_asymm_3 * user_stream
        u_xpos = common + scatter
        u_xneg = common - scatter
    else:
        u_xpos = (-0.5 * px11 * ulp) * omega_asymm_3
        u_xneg = u_xpos
    return u_xpos, u_xneg


def _hmult_master_batch(
    *,
    delta_tau: np.ndarray,
    user_secant: float,
    eigenvalue: np.ndarray,
    eigentrans: np.ndarray,
    t_delt_userm: np.ndarray,
):
    """Builds user-angle homogeneous multipliers for a wavelength batch."""
    zp = user_secant + eigenvalue
    zm = user_secant - eigenvalue
    zudel = eigentrans * t_delt_userm
    hmult_2 = user_secant * (1.0 - zudel) / zp
    near = np.abs(zm) < TAYLOR_SMALL
    with np.errstate(divide="ignore", invalid="ignore"):
        hmult_1 = user_secant * (eigentrans - t_delt_userm) / zm
    if np.any(near):
        hmult_1 = hmult_1.copy()
        hmult_1[near] = vectorized_taylor_series_1(
            TAYLOR_ORDER,
            zm[near],
            delta_tau[near],
            t_delt_userm[near],
            user_secant,
        )
    return hmult_1, hmult_2


def _gbeam_solution_batch(
    *,
    fourier: int,
    pi4: float,
    flux_factor: np.ndarray,
    layer_pis_cutoff: np.ndarray,
    px0x: float,
    omega: np.ndarray,
    omega_asymm_3: np.ndarray | None = None,
    asymm: np.ndarray | None = None,
    average_secant: np.ndarray,
    initial_trans: np.ndarray,
    t_delt_mubar: np.ndarray,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    eigenvalue: np.ndarray,
    eigentrans: np.ndarray,
    norm_saved: np.ndarray,
    delta_tau: np.ndarray,
    all_layers_active: bool = False,
):
    """Builds solar Green-function beam terms for a wavelength batch."""
    batch, nlayers = delta_tau.shape
    if omega_asymm_3 is None:
        if asymm is None:
            raise TypeError("provide either omega_asymm_3 or asymm")
        omega_asymm_3 = 3.0 * omega * asymm
    f1 = flux_factor / pi4
    gamma_p_raw = average_secant + eigenvalue
    gamma_m_raw = average_secant - eigenvalue
    if all_layers_active:
        gamma_p = gamma_p_raw
        gamma_m = gamma_m_raw
        zdel = eigentrans
        wdel = t_delt_mubar
        zwdel = zdel * wdel
        with np.errstate(divide="ignore", invalid="ignore"):
            cfunc = (zdel - wdel) / gamma_m_raw
        near = np.abs(gamma_m_raw) < TAYLOR_SMALL
        if np.any(near):
            cfunc = cfunc.copy()
            cfunc[near] = vectorized_taylor_series_1(
                TAYLOR_ORDER,
                gamma_m_raw[near],
                delta_tau[near],
                wdel[near],
                1.0,
            )
        dfunc = (1.0 - zwdel) / gamma_p_raw
        scaled_flux = f1[:, np.newaxis]
        if fourier == 0:
            common = omega * scaled_flux
            scatter = (px0x * omega_asymm_3) * (xpos1 - xpos2)
            scatter *= scaled_flux
            aterm = (common + scatter) / norm_saved
            bterm = (common - scatter) / norm_saved
        else:
            aterm = (px0x * omega_asymm_3) * scaled_flux / norm_saved
            bterm = aterm
        gfunc_dn = cfunc * aterm * initial_trans
        gfunc_up = dfunc * bterm * initial_trans
        wupper0 = gfunc_up * xpos2
        wupper1 = gfunc_up * xpos1
        wlower0 = gfunc_dn * xpos1
        wlower1 = gfunc_dn * xpos2
        return gamma_m, gamma_p, aterm, bterm, (wupper0, wupper1), (wlower0, wlower1)

    active = np.arange(1, nlayers + 1, dtype=int)[np.newaxis, :] <= layer_pis_cutoff[:, np.newaxis]
    zero = np.zeros((batch, nlayers), dtype=float)
    gamma_p = np.where(active, gamma_p_raw, zero)
    gamma_m = np.where(active, gamma_m_raw, zero)

    zdel = eigentrans
    wdel = t_delt_mubar
    zwdel = zdel * wdel
    with np.errstate(divide="ignore", invalid="ignore"):
        cfunc = (zdel - wdel) / gamma_m_raw
    near = active & (np.abs(gamma_m_raw) < TAYLOR_SMALL)
    if np.any(near):
        cfunc = cfunc.copy()
        cfunc[near] = vectorized_taylor_series_1(
            TAYLOR_ORDER,
            gamma_m_raw[near],
            delta_tau[near],
            wdel[near],
            1.0,
        )
    cfunc = np.where(active, cfunc, zero)
    dfunc = np.where(active, (1.0 - zwdel) / gamma_p_raw, zero)

    scaled_flux = f1[:, np.newaxis]
    if fourier == 0:
        common = omega * scaled_flux
        scatter = (px0x * omega_asymm_3) * (xpos1 - xpos2)
        scatter *= scaled_flux
        aterm_raw = (common + scatter) / norm_saved
        bterm_raw = (common - scatter) / norm_saved
    else:
        aterm_raw = (px0x * omega_asymm_3) * scaled_flux / norm_saved
        bterm_raw = aterm_raw
    aterm = np.where(active, aterm_raw, zero)
    bterm = np.where(active, bterm_raw, zero)

    gfunc_dn = cfunc * aterm * initial_trans
    gfunc_up = dfunc * bterm * initial_trans
    wupper0 = gfunc_up * xpos2
    wupper1 = gfunc_up * xpos1
    wlower0 = gfunc_dn * xpos1
    wlower1 = gfunc_dn * xpos2
    return gamma_m, gamma_p, aterm, bterm, (wupper0, wupper1), (wlower0, wlower1)


def _upuser_intensity_batch(
    *,
    layer_pis_cutoff: np.ndarray,
    surface_factor: float,
    albedo: np.ndarray,
    fluxmult: float,
    stream_value: float,
    delta_tau: np.ndarray,
    gamma_p: np.ndarray,
    gamma_m: np.ndarray,
    sigma_p: np.ndarray,
    aterm: np.ndarray,
    bterm: np.ndarray,
    initial_trans: np.ndarray,
    itrans_userm: np.ndarray,
    t_delt_userm: np.ndarray,
    t_delt_mubar: np.ndarray,
    eigentrans: np.ndarray,
    lcon: np.ndarray,
    mcon: np.ndarray,
    wlower1: np.ndarray,
    xpos1: np.ndarray,
    xpos2: np.ndarray,
    u_xpos: np.ndarray,
    u_xneg: np.ndarray,
    hmult_1: np.ndarray,
    hmult_2: np.ndarray,
    emult_up: np.ndarray,
    all_layers_active: bool = False,
    surface_source_zero: bool = False,
    return_profile: bool = False,
):
    """Computes upwelling TOA user intensity for a wavelength batch."""
    nlay = delta_tau.shape[1]
    if surface_source_zero:
        cumsource = np.zeros(lcon.shape[0], dtype=float)
    else:
        par = wlower1[:, -1]
        hom = lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
        idownsurf = (par + hom) * stream_value
        cumsource = surface_factor * albedo * idownsurf
    layersource = np.empty_like(lcon)
    np.multiply(lcon, u_xpos, out=layersource)
    layersource *= hmult_2
    tmp = mcon * u_xneg
    tmp *= hmult_1
    layersource += tmp
    if all_layers_active:
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = initial_trans * hmult_2
            sd -= emult_up
            sd /= gamma_m
            su = t_delt_mubar * hmult_1
            su *= initial_trans
            np.negative(su, out=su)
            su += emult_up
            su /= gamma_p
        near = np.abs(gamma_m) < TAYLOR_SMALL
        if np.any(near):
            taylor = vectorized_taylor_series_2(
                TAYLOR_ORDER,
                TAYLOR_SMALL,
                gamma_m[near],
                sigma_p[near],
                delta_tau[near],
                np.ones(np.count_nonzero(near), dtype=float),
                (t_delt_mubar * t_delt_userm)[near],
                1.0,
            )
            sd[near] = itrans_userm[near] * taylor
        sd *= aterm
        sd *= u_xpos
        layersource += sd
        su *= bterm
        su *= u_xneg
        layersource += su
        if return_profile:
            return fluxmult * accumulate_upwelling_profile_numpy(
                layer_source=layersource,
                layer_trans=t_delt_userm,
                surface_source=cumsource,
            )
        return fluxmult * accumulate_upwelling_sources_numpy(
            layer_source=layersource,
            layer_trans=t_delt_userm,
            surface_source=cumsource,
        )

    active = _layer_cutoff_mask(nlay, layer_pis_cutoff)
    zero = np.zeros_like(gamma_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = initial_trans * hmult_2
        sd -= emult_up
        sd /= gamma_m
        su = t_delt_mubar * hmult_1
        su *= initial_trans
        np.negative(su, out=su)
        su += emult_up
        su /= gamma_p
    near = active & (np.abs(gamma_m) < TAYLOR_SMALL)
    if np.any(near):
        taylor = vectorized_taylor_series_2(
            TAYLOR_ORDER,
            TAYLOR_SMALL,
            gamma_m[near],
            sigma_p[near],
            delta_tau[near],
            np.ones(np.count_nonzero(near), dtype=float),
            (t_delt_mubar * t_delt_userm)[near],
            1.0,
        )
        sd[near] = itrans_userm[near] * taylor
    sd *= aterm
    sd *= u_xpos
    su *= bterm
    su *= u_xneg
    layersource += np.where(active, sd + su, zero)
    if return_profile:
        return fluxmult * accumulate_upwelling_profile_numpy(
            layer_source=layersource,
            layer_trans=t_delt_userm,
            surface_source=cumsource,
        )
    return fluxmult * accumulate_upwelling_sources_numpy(
        layer_source=layersource,
        layer_trans=t_delt_userm,
        surface_source=cumsource,
    )


def solve_solar_obs_batch_numpy(
    *,
    tau: np.ndarray,
    omega: np.ndarray,
    asymm: np.ndarray,
    scaling: np.ndarray,
    albedo: np.ndarray,
    flux_factor: np.ndarray,
    stream_value: float,
    chapman: np.ndarray,
    x0: float,
    user_stream: float,
    user_secant: float,
    azmfac: float,
    px11: float,
    pxsq: np.ndarray,
    px0x: np.ndarray,
    ulp: float,
    bvp_engine: str = "auto",
    return_profile: bool = False,
) -> np.ndarray:
    """Solves the solar-observation 2S problem for a spectral batch.

    Parameters
    ----------
    tau, omega, asymm, scaling
        Layer optical properties with shape ``(n_spectral, n_layers)``.
    albedo, flux_factor
        Surface albedo and incident flux factor for each spectral point.
    stream_value
        Two-stream quadrature stream value.
    chapman
        Geometry Chapman matrix for the selected solar observation geometry.
    x0, user_stream, user_secant, azmfac, px11, pxsq, px0x, ulp
        Precomputed geometry and phase-function factors for the selected
        observation geometry.

    Returns
    -------
    ndarray
        Upwelling TOA 2S radiance for each spectral point.

    Notes
    -----
    The caller supplies already-prepared optical properties and geometry
    factors, so this routine is independent of wavelength region and file
    layout.
    """
    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties(
        tau,
        omega,
        asymm,
        scaling,
    )
    qsprep = _qsprep_obs_batch(delta_tau, chapman, user_secant)
    pi4 = 4.0 * np.pi
    omega_asymm_3 = 3.0 * omega_total * asymm_total
    all_layers_active = bool(qsprep["all_active"])
    layer_pis_cutoff = qsprep["layer_pis_cutoff"]
    average_secant = qsprep["average_secant"]
    initial_trans = qsprep["initial_trans"]
    trans_solar_beam = qsprep["trans_solar_beam"]
    t_delt_mubar = qsprep["t_delt_mubar"]
    itrans_userm = qsprep["itrans_userm"]
    t_delt_userm = qsprep["t_delt_userm"]
    sigma_p = qsprep["sigma_p"]
    emult_up = qsprep["emult_up"]
    total = np.zeros(tau.shape[0], dtype=float)
    total_profile = None
    if return_profile:
        total_profile = np.zeros((tau.shape[0], tau.shape[1] + 1), dtype=float)
    zero_surface = np.zeros_like(albedo)

    for fourier in (0, 1):
        surface_factor = 2.0 if fourier == 0 else 1.0
        delta_factor = 1.0 if fourier == 0 else 2.0
        (
            eigenvalue,
            eigentrans,
            xpos1,
            xpos2,
            norm_saved,
        ) = _hom_solution_solar_batch(
            fourier=fourier,
            stream_value=stream_value,
            pxsq=float(pxsq[fourier]),
            omega=omega_total,
            omega_asymm_3=omega_asymm_3,
            delta_tau=delta_tau,
        )
        u_xpos, u_xneg = _hom_user_solution_solar_batch(
            fourier=fourier,
            stream_value=stream_value,
            px11=px11,
            user_stream=user_stream,
            ulp=ulp,
            xpos1=xpos1,
            xpos2=xpos2,
            omega=omega_total,
            omega_asymm_3=omega_asymm_3,
        )
        hmult_1, hmult_2 = _hmult_master_batch(
            delta_tau=delta_tau,
            user_secant=user_secant,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            t_delt_userm=t_delt_userm,
        )
        gamma_m, gamma_p, aterm, bterm, wupper, wlower = _gbeam_solution_batch(
            fourier=fourier,
            pi4=pi4,
            flux_factor=flux_factor,
            layer_pis_cutoff=layer_pis_cutoff,
            px0x=float(px0x[fourier]),
            omega=omega_total,
            omega_asymm_3=omega_asymm_3,
            average_secant=average_secant,
            initial_trans=initial_trans,
            t_delt_mubar=t_delt_mubar,
            xpos1=xpos1,
            xpos2=xpos2,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            norm_saved=norm_saved,
            delta_tau=delta_tau,
            all_layers_active=all_layers_active,
        )
        if fourier == 0:
            direct_beam = flux_factor * x0 / delta_factor / np.pi * trans_solar_beam * albedo
            bvp_albedo = albedo
        else:
            direct_beam = zero_surface
            bvp_albedo = zero_surface
        lcon, mcon = solve_solar_observation_bvp_batch(
            albedo=bvp_albedo,
            direct_beam=direct_beam,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=wupper,
            wlower=wlower,
            bvp_engine=bvp_engine,
        )
        contribution = _upuser_intensity_batch(
            layer_pis_cutoff=layer_pis_cutoff,
            surface_factor=surface_factor,
            albedo=bvp_albedo,
            fluxmult=delta_factor,
            stream_value=stream_value,
            delta_tau=delta_tau,
            gamma_p=gamma_p,
            gamma_m=gamma_m,
            sigma_p=sigma_p,
            aterm=aterm,
            bterm=bterm,
            initial_trans=initial_trans,
            itrans_userm=itrans_userm,
            t_delt_userm=t_delt_userm,
            t_delt_mubar=t_delt_mubar,
            eigentrans=eigentrans,
            lcon=lcon,
            mcon=mcon,
            wlower1=wlower[0],
            xpos1=xpos1,
            xpos2=xpos2,
            u_xpos=u_xpos,
            u_xneg=u_xneg,
            hmult_1=hmult_1,
            hmult_2=hmult_2,
            emult_up=emult_up,
            all_layers_active=all_layers_active,
            surface_source_zero=(fourier == 1),
            return_profile=return_profile,
        )
        if return_profile:
            total_profile = contribution if fourier == 0 else total_profile + azmfac * contribution
            total = total_profile[:, 0]
        else:
            total = contribution if fourier == 0 else total + azmfac * contribution
    return total_profile if return_profile else total
