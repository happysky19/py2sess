"""NumPy-based two-stream forward solver implementations."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.linalg import solve, solve_banded

from ..optical.delta_m import delta_m_scale_optical_properties
from .preprocess import PreparedInputs
from .solver_common import (
    accumulate_scalar_and_levels,
    accumulate_flux_pair,
    include_surface_term,
    include_thermal_surface_term,
    prepare_thermal_boundary_terms,
    prepare_solar_direct_beam_terms,
    prepare_solar_fourier_postprocessing,
    prepare_solar_geometry_solution,
    prepare_solar_misc,
    prepare_thermal_postprocessing,
    solar_problem_size,
    thermal_problem_size,
)
from .taylor import taylor_series_1, taylor_series_2


MAX_TAU_QPATH = 88.0
MAX_TAU_PATH = 88.0
_OPTICAL_THICKNESS_MIN = 1.0e-12


def _initialize_solution_storage(
    size: int,
    nlay: int,
    *,
    flux_geometry_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Allocates the standard NumPy forward-solver output arrays.

    Parameters
    ----------
    size
        Number of geometries or user angles in the solve.
    nlay
        Number of atmospheric layers.
    flux_geometry_count
        Optional number of flux columns to allocate when it differs from the
        scalar-radiance geometry count.

    Returns
    -------
    dict of ndarray
        Fresh output arrays for scalar radiances, fluxes, and level profiles.
    """
    flux_count = size if flux_geometry_count is None else flux_geometry_count
    return {
        "intensity_toa": np.zeros(size, dtype=float),
        "intensity_boa": np.zeros(size, dtype=float),
        "fluxes_toa": np.zeros((2, flux_count), dtype=float),
        "fluxes_boa": np.zeros((2, flux_count), dtype=float),
        "radlevel_up": np.zeros((size, nlay + 1), dtype=float),
        "radlevel_dn": np.zeros((size, nlay + 1), dtype=float),
    }


def _finalize_solution_storage(
    solved: dict[str, np.ndarray],
    prepared: PreparedInputs,
) -> dict[str, np.ndarray]:
    """Applies source-mode-specific output reshaping.

    Parameters
    ----------
    solved
        Output arrays produced by the solver.
    prepared
        Preprocessed solver inputs.

    Returns
    -------
    dict of ndarray
        Output arrays with any source-mode-specific reshaping applied.
    """
    if prepared.source_mode == "solar_lat" and prepared.lattice_counts is not None:
        nbeams, nusers, nazms = prepared.lattice_counts
        stride = nusers * nazms
        solved["fluxes_toa"] = solved["fluxes_toa"][:, np.arange(nbeams, dtype=int) * stride]
        solved["fluxes_boa"] = solved["fluxes_boa"][:, np.arange(nbeams, dtype=int) * stride]
    return solved


def _check_optical_inputs(delta_tau: np.ndarray, omega: np.ndarray, asymm: np.ndarray) -> None:
    """Validates scaled optical properties before solving.

    Parameters
    ----------
    delta_tau, omega, asymm
        Delta-scaled optical thickness, single-scattering albedo, and
        asymmetry arrays.
    """
    if np.any(delta_tau <= 0.0):
        raise ValueError("All optical thickness values must be positive after scaling")
    if np.any(omega > 0.999999999):
        raise ValueError("Single scattering albedo too close to 1 after scaling")
    if np.any(omega < 1.0e-9):
        raise ValueError("Single scattering albedo too small after scaling")
    if np.any((asymm <= -1.0) | (asymm >= 1.0)):
        raise ValueError("Asymmetry parameter outside (-1, 1) after scaling")


def _floor_zero_optical_thickness(delta_tau: np.ndarray) -> np.ndarray:
    if np.any(delta_tau < 0.0):
        raise ValueError("tau must be nonnegative")
    if not np.any(delta_tau == 0.0):
        return delta_tau
    floored = np.asarray(delta_tau, dtype=float).copy()
    np.putmask(floored, floored == 0.0, _OPTICAL_THICKNESS_MIN)
    return floored


def _apply_delta_scaling(
    prepared: PreparedInputs,
    do_delta_scaling: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies delta-M scaling to the optical-property inputs.

    Parameters
    ----------
    prepared
        Preprocessed solver inputs.
    do_delta_scaling
        Whether delta-M scaling is enabled.

    Returns
    -------
    tuple of ndarray
        Scaled optical thickness, single-scattering albedo, and asymmetry
        arrays.
    """
    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties(
        prepared.tau_arr,
        prepared.omega_arr,
        prepared.asymm_arr,
        prepared.d2s_scaling if do_delta_scaling else np.zeros_like(prepared.tau_arr),
    )

    delta_tau = _floor_zero_optical_thickness(delta_tau)
    _check_optical_inputs(delta_tau, omega_total, asymm_total)
    return delta_tau, omega_total, asymm_total


def _qsprep_obs(
    delta_tau: np.ndarray,
    chapman_factors: np.ndarray,
    user_secants: np.ndarray,
    do_upwelling: bool,
    do_dnwelling: bool,
    do_postprocessing: bool,
    taylor_small: float,
    taylor_order: int,
) -> dict[str, np.ndarray]:
    """Builds spherical solar transmittance and multiplier inputs."""
    nlayers, _, ngeoms = chapman_factors.shape
    do_directbeam = np.ones(ngeoms, dtype=bool)
    layer_pis_cutoff = np.full(ngeoms, nlayers, dtype=int)
    initial_trans = np.zeros((nlayers, ngeoms), dtype=float)
    average_secant = np.zeros((nlayers, ngeoms), dtype=float)
    trans_solar_beam = np.zeros(ngeoms, dtype=float)
    t_delt_mubar = np.zeros((nlayers, ngeoms), dtype=float)
    itrans_userm = np.zeros((nlayers, ngeoms), dtype=float)
    t_delt_userm = np.zeros((nlayers, ngeoms), dtype=float)
    sigma_p = np.zeros((nlayers, ngeoms), dtype=float)
    sigma_m = np.zeros((nlayers, ngeoms), dtype=float)
    emult_up = np.zeros((nlayers, ngeoms), dtype=float)
    emult_dn = np.zeros((nlayers, ngeoms), dtype=float)

    for ib in range(ngeoms):
        usib = user_secants[ib] if do_postprocessing else 0.0
        s_t_0 = 1.0
        s_t_1 = 0.0
        cutoff = nlayers
        tauslant = 0.0

        for n in range(nlayers):
            deltau = delta_tau[n]
            tauslantn1 = tauslant
            tauslant = np.dot(delta_tau[: n + 1], chapman_factors[: n + 1, n, ib])

            if (n + 1) <= cutoff:
                if tauslant > MAX_TAU_PATH:
                    cutoff = n + 1
                else:
                    s_t_1 = math.exp(-tauslant)
                sb = (tauslant - tauslantn1) / deltau
                initial_n = s_t_0
                s_t_0 = s_t_1
                spher = deltau * sb
                wdel = 0.0 if spher > MAX_TAU_PATH else math.exp(-spher)
            else:
                sb = 0.0
                initial_n = 0.0
                wdel = 0.0

            average_secant[n, ib] = sb
            initial_trans[n, ib] = initial_n
            t_delt_mubar[n, ib] = wdel

            if do_postprocessing:
                itudel = initial_n * usib
                itrans_userm[n, ib] = itudel
                spher = deltau * usib
                udel = 0.0 if spher > MAX_TAU_PATH else math.exp(-spher)
                t_delt_userm[n, ib] = udel

                if (n + 1) <= cutoff:
                    if do_upwelling:
                        sigma_pn = sb + usib
                        sigma_p[n, ib] = sigma_pn
                        wudel = wdel * udel
                        su = (1.0 - wudel) / sigma_pn
                        emult_up[n, ib] = itudel * su
                    if do_dnwelling:
                        sigma_mn = sb - usib
                        sigma_m[n, ib] = sigma_mn
                        diff = abs(usib - sb)
                        if diff < taylor_small:
                            sd = taylor_series_1(taylor_order, sigma_mn, deltau, wdel, 1.0)
                        else:
                            sd = (udel - wdel) / sigma_mn
                        emult_dn[n, ib] = itudel * sd

        layer_pis_cutoff[ib] = cutoff
        if tauslant > MAX_TAU_PATH:
            trans_solar_beam[ib] = 0.0
            do_directbeam[ib] = False
        else:
            trans_solar_beam[ib] = math.exp(-tauslant)

    return {
        "do_directbeam": do_directbeam,
        "layer_pis_cutoff": layer_pis_cutoff,
        "initial_trans": initial_trans,
        "average_secant": average_secant,
        "trans_solar_beam": trans_solar_beam,
        "t_delt_mubar": t_delt_mubar,
        "itrans_userm": itrans_userm,
        "t_delt_userm": t_delt_userm,
        "sigma_p": sigma_p,
        "sigma_m": sigma_m,
        "emult_up": emult_up,
        "emult_dn": emult_dn,
    }


def _qsprep_obs_pp(
    delta_tau: np.ndarray,
    average_secant_pp: np.ndarray,
    user_secants: np.ndarray,
    do_upwelling: bool,
    do_dnwelling: bool,
    do_postprocessing: bool,
    taylor_small: float,
    taylor_order: int,
) -> dict[str, np.ndarray]:
    """Builds plane-parallel solar transmittance and multiplier inputs."""
    nlayers = delta_tau.size
    ngeoms = average_secant_pp.size
    do_directbeam = np.ones(ngeoms, dtype=bool)
    layer_pis_cutoff = np.full(ngeoms, nlayers, dtype=int)
    initial_trans = np.zeros((nlayers, ngeoms), dtype=float)
    average_secant = np.zeros((nlayers, ngeoms), dtype=float)
    trans_solar_beam = np.zeros(ngeoms, dtype=float)
    t_delt_mubar = np.zeros((nlayers, ngeoms), dtype=float)
    itrans_userm = np.zeros((nlayers, ngeoms), dtype=float)
    t_delt_userm = np.zeros((nlayers, ngeoms), dtype=float)
    sigma_p = np.zeros((nlayers, ngeoms), dtype=float)
    sigma_m = np.zeros((nlayers, ngeoms), dtype=float)
    emult_up = np.zeros((nlayers, ngeoms), dtype=float)
    emult_dn = np.zeros((nlayers, ngeoms), dtype=float)

    for ib in range(ngeoms):
        usib = user_secants[ib] if do_postprocessing else 0.0
        sb = average_secant_pp[ib]
        cutoff = nlayers
        taugrid = 0.0

        for n in range(nlayers):
            deltau = delta_tau[n]
            taugridn1 = taugrid
            taugrid += deltau
            tauslant = taugrid * sb
            if (n + 1) <= cutoff:
                if tauslant > MAX_TAU_PATH:
                    cutoff = n + 1
                initial_n = math.exp(-taugridn1 * sb)
                spher = deltau * sb
                wdel = 0.0 if spher > MAX_TAU_PATH else math.exp(-spher)
            else:
                initial_n = 0.0
                wdel = 0.0
            initial_trans[n, ib] = initial_n
            average_secant[n, ib] = sb
            t_delt_mubar[n, ib] = wdel

            if do_postprocessing:
                itudel = initial_n * usib
                itrans_userm[n, ib] = itudel
                spher = deltau * usib
                udel = 0.0 if spher > MAX_TAU_PATH else math.exp(-spher)
                t_delt_userm[n, ib] = udel

                if (n + 1) <= cutoff:
                    if do_upwelling:
                        sigma_pn = sb + usib
                        sigma_p[n, ib] = sigma_pn
                        wudel = wdel * udel
                        su = (1.0 - wudel) / sigma_pn
                        emult_up[n, ib] = itudel * su
                    if do_dnwelling:
                        sigma_mn = sb - usib
                        sigma_m[n, ib] = sigma_mn
                        diff = abs(usib - sb)
                        if diff < taylor_small:
                            sd = taylor_series_1(taylor_order, sigma_mn, deltau, wdel, 1.0)
                        else:
                            sd = (udel - wdel) / sigma_mn
                        emult_dn[n, ib] = itudel * sd

        layer_pis_cutoff[ib] = cutoff
        if taugrid * sb > MAX_TAU_PATH:
            trans_solar_beam[ib] = 0.0
            do_directbeam[ib] = False
        else:
            trans_solar_beam[ib] = math.exp(-(taugrid * sb))

    return {
        "do_directbeam": do_directbeam,
        "layer_pis_cutoff": layer_pis_cutoff,
        "initial_trans": initial_trans,
        "average_secant": average_secant,
        "trans_solar_beam": trans_solar_beam,
        "t_delt_mubar": t_delt_mubar,
        "itrans_userm": itrans_userm,
        "t_delt_userm": t_delt_userm,
        "sigma_p": sigma_p,
        "sigma_m": sigma_m,
        "emult_up": emult_up,
        "emult_dn": emult_dn,
    }


def _hom_solution_solar(
    fourier: int,
    stream_value: float,
    pxsq: float,
    omega: np.ndarray,
    asymm: np.ndarray,
    delta_tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds the homogeneous solar eigensystem for one Fourier mode."""
    nlayers = delta_tau.size
    eigenvalue = np.zeros(nlayers, dtype=float)
    eigentrans = np.zeros(nlayers, dtype=float)
    xpos = np.zeros((2, nlayers), dtype=float)
    norm_saved = np.zeros(nlayers, dtype=float)
    xinv = 1.0 / stream_value

    for n in range(nlayers):
        omegan = omega[n]
        omega_asymm_3 = 3.0 * omegan * asymm[n]
        if fourier == 0:
            ep = omegan + pxsq * omega_asymm_3
            em = omegan - pxsq * omega_asymm_3
        else:
            ep = omega_asymm_3 * pxsq
            em = omega_asymm_3 * pxsq
        sab = xinv * (((ep + em) * 0.5) - 1.0)
        dab = xinv * (((ep - em) * 0.5) - 1.0)
        eigenvaluen = math.sqrt(sab * dab)
        eigenvalue[n] = eigenvaluen
        helpv = eigenvaluen * delta_tau[n]
        eigentrans[n] = 0.0 if helpv > MAX_TAU_QPATH else math.exp(-helpv)
        difvec = -sab / eigenvaluen
        xpos1 = 0.5 * (1.0 + difvec)
        xpos2 = 0.5 * (1.0 - difvec)
        xpos[0, n] = xpos1
        xpos[1, n] = xpos2
        norm_saved[n] = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos, norm_saved


def _hom_user_solution_solar(
    fourier: int,
    stream_value: float,
    px11: float,
    user_streams: np.ndarray,
    ulp: np.ndarray,
    xpos: np.ndarray,
    omega: np.ndarray,
    asymm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds solar user-angle homogeneous solutions for one Fourier mode."""
    nlayers = xpos.shape[1]
    ngeoms = user_streams.shape[0]
    u_xpos = np.zeros((ngeoms, nlayers), dtype=float)
    u_xneg = np.zeros((ngeoms, nlayers), dtype=float)
    hmu_stream = 0.5 * stream_value

    for n in range(nlayers):
        xpos1 = xpos[0, n]
        xpos2 = xpos[1, n]
        if fourier == 0:
            u_help_p0 = (xpos2 + xpos1) * 0.5
            u_help_p1 = (xpos2 - xpos1) * hmu_stream
            u_help_m0 = u_help_p0
            u_help_m1 = -u_help_p1
        else:
            u_help_p1 = -(xpos2 + xpos1) * px11 * 0.5
            u_help_m1 = u_help_p1
        omegan = omega[n]
        omega_mom = 3.0 * omegan * asymm[n]
        for um in range(ngeoms):
            if fourier == 0:
                mu = user_streams[um]
                sum_pos = u_help_p0 * omegan + u_help_p1 * omega_mom * mu
                sum_neg = u_help_m0 * omegan + u_help_m1 * omega_mom * mu
            else:
                ulp_um = ulp[um]
                sum_pos = u_help_p1 * omega_mom * ulp_um
                sum_neg = u_help_m1 * omega_mom * ulp_um
            u_xpos[um, n] = sum_pos
            u_xneg[um, n] = sum_neg
    return u_xpos, u_xneg


def _hmult_master(
    taylor_order: int,
    taylor_small: float,
    delta_tau: np.ndarray,
    user_secants: np.ndarray,
    eigenvalue: np.ndarray,
    eigentrans: np.ndarray,
    t_delt_userm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds user-angle homogeneous multipliers for solar or thermal mode."""
    nlayers = delta_tau.size
    ngeoms = user_secants.size
    hmult_1 = np.zeros((ngeoms, nlayers), dtype=float)
    hmult_2 = np.zeros((ngeoms, nlayers), dtype=float)
    for n in range(nlayers):
        eigvn = eigenvalue[n]
        eigtn = eigentrans[n]
        dn = delta_tau[n]
        for um in range(ngeoms):
            udel = t_delt_userm[n, um]
            sm = user_secants[um]
            zp = sm + eigvn
            zm = sm - eigvn
            zudel = eigtn * udel
            hmult_2[um, n] = sm * (1.0 - zudel) / zp
            if abs(zm) < taylor_small:
                hmult_1[um, n] = taylor_series_1(taylor_order, zm, dn, udel, sm)
            else:
                hmult_1[um, n] = sm * (eigtn - udel) / zm
    return hmult_1, hmult_2


def _hom_solution_thermal(
    stream_value: float,
    pxsq: float,
    omega: np.ndarray,
    asymm: np.ndarray,
    delta_tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds the homogeneous thermal eigensystem."""
    nlayers = delta_tau.size
    eigenvalue = np.zeros(nlayers, dtype=float)
    eigentrans = np.zeros(nlayers, dtype=float)
    xpos = np.zeros((2, nlayers), dtype=float)
    norm_saved = np.zeros(nlayers, dtype=float)
    xinv = 1.0 / stream_value
    for n in range(nlayers):
        omegan = omega[n]
        omega_asymm_3 = 3.0 * omegan * asymm[n]
        ep = omegan + pxsq * omega_asymm_3
        em = omegan - pxsq * omega_asymm_3
        sab = xinv * (((ep + em) * 0.5) - 1.0)
        dab = xinv * (((ep - em) * 0.5) - 1.0)
        eig = math.sqrt(sab * dab)
        eigenvalue[n] = eig
        helpv = eig * delta_tau[n]
        eigentrans[n] = 0.0 if helpv > MAX_TAU_QPATH else math.exp(-helpv)
        difvec = -sab / eig
        xpos1 = 0.5 * (1.0 + difvec)
        xpos2 = 0.5 * (1.0 - difvec)
        xpos[0, n] = xpos1
        xpos[1, n] = xpos2
        norm_saved[n] = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos, norm_saved


def _hom_user_solution_thermal(
    stream_value: float,
    user_streams: np.ndarray,
    xpos: np.ndarray,
    omega: np.ndarray,
    asymm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds thermal user-angle homogeneous solutions."""
    nlayers = xpos.shape[1]
    n_users = user_streams.size
    u_xpos = np.zeros((n_users, nlayers), dtype=float)
    u_xneg = np.zeros((n_users, nlayers), dtype=float)
    hmu_stream = 0.5 * stream_value
    for n in range(nlayers):
        xpos1 = xpos[0, n]
        xpos2 = xpos[1, n]
        u_help_p0 = (xpos2 + xpos1) * 0.5
        u_help_p1 = (xpos2 - xpos1) * hmu_stream
        u_help_m0 = u_help_p0
        u_help_m1 = -u_help_p1
        omegan = omega[n]
        omega_mom = 3.0 * omegan * asymm[n]
        for um in range(n_users):
            mu = user_streams[um]
            sum_pos = u_help_p0 * omegan + u_help_p1 * omega_mom * mu
            sum_neg = u_help_m0 * omegan + u_help_m1 * omega_mom * mu
            u_xpos[um, n] = sum_pos
            u_xneg[um, n] = sum_neg
    return u_xpos, u_xneg


def _thermal_setup(delta_tau: np.ndarray, thermal_bb_input: np.ndarray) -> np.ndarray:
    """Builds layerwise thermal source coefficients from blackbody inputs."""
    thermcoeffs = np.zeros((2, delta_tau.size), dtype=float)
    tcn1 = thermal_bb_input[0]
    for n in range(delta_tau.size):
        tcn = thermal_bb_input[n + 1]
        thermcoeffs[0, n] = tcn1
        thermcoeffs[1, n] = (tcn - tcn1) / delta_tau[n]
        tcn1 = tcn
    return thermcoeffs


def _thermal_gf_solution(
    omega: np.ndarray,
    delta_tau: np.ndarray,
    thermcoeffs: np.ndarray,
    tcutoff: float,
    eigenvalue: np.ndarray,
    eigentrans: np.ndarray,
    xpos: np.ndarray,
    norm_saved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds Green-function thermal source terms and boundary vectors."""
    nlayers = delta_tau.size
    t_c_minus = np.zeros((3, nlayers), dtype=float)
    t_c_plus = np.zeros((3, nlayers), dtype=float)
    tterm_save = np.zeros(nlayers, dtype=float)
    t_wupper = np.zeros((2, nlayers), dtype=float)
    t_wlower = np.zeros((2, nlayers), dtype=float)
    for n in range(nlayers):
        if delta_tau[n] <= tcutoff:
            continue
        xpos1 = xpos[0, n]
        xpos2 = xpos[1, n]
        tc1 = thermcoeffs[0, n]
        tc2 = thermcoeffs[1, n]
        omega1 = 1.0 - omega[n]
        helpv = xpos1 + xpos2
        tterm = omega1 * helpv / norm_saved[n]
        tterm_save[n] = tterm
        k1 = 1.0 / eigenvalue[n]
        tk = eigentrans[n]
        tcm2 = k1 * tc2
        tcp2 = tcm2
        t_c_minus[2, n] = tcm2
        t_c_plus[2, n] = tcp2
        tcm1 = k1 * (tc1 - tcm2)
        tcp1 = k1 * (tc1 + tcp2)
        t_c_minus[1, n] = tcm1
        t_c_plus[1, n] = tcp1
        sum_m = tcm1 + tcm2 * delta_tau[n]
        sum_p = tcp1 + tcp2 * delta_tau[n]
        tcm0 = -tcm1
        tcp0 = -sum_p
        t_c_minus[0, n] = tcm0
        t_c_plus[0, n] = tcp0
        t_gmult_dn = tterm * (tk * tcm0 + sum_m)
        t_gmult_up = tterm * (tk * tcp0 + tcp1)
        t_wupper[0, n] = t_gmult_up * xpos2
        t_wupper[1, n] = t_gmult_up * xpos1
        t_wlower[0, n] = t_gmult_dn * xpos1
        t_wlower[1, n] = t_gmult_dn * xpos2
    return t_c_plus, t_c_minus, tterm_save, t_wupper, t_wlower


def _thermal_terms(
    do_upwelling: bool,
    do_dnwelling: bool,
    user_streams: np.ndarray,
    tcutoff: float,
    t_delt_userm: np.ndarray,
    delta_tau: np.ndarray,
    u_xpos: np.ndarray,
    u_xneg: np.ndarray,
    hmult_1: np.ndarray,
    hmult_2: np.ndarray,
    t_c_plus: np.ndarray,
    t_c_minus: np.ndarray,
    tterm_save: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_users, nlayers = u_xpos.shape
    layer_tsup_up = np.zeros((n_users, nlayers), dtype=float)
    layer_tsup_dn = np.zeros((n_users, nlayers), dtype=float)
    if do_upwelling:
        for n in range(nlayers):
            if delta_tau[n] <= tcutoff:
                continue
            tcp1 = t_c_plus[1, n]
            tcm1 = t_c_minus[1, n]
            tcp0 = t_c_plus[0, n]
            tcm0 = t_c_minus[0, n]
            tsgm_uu2 = t_c_plus[2, n]
            tsgm_ud2 = t_c_minus[2, n]
            for um in range(n_users):
                mu = user_streams[um]
                tdel = t_delt_userm[n, um]
                tsgm_uu1 = tcp1 + mu * tsgm_uu2
                tsgm_ud1 = tcm1 + mu * tsgm_ud2
                tsgm_uu0 = -tsgm_uu1 - tsgm_uu2 * delta_tau[n]
                tsgm_ud0 = -tsgm_ud1 - tsgm_ud2 * delta_tau[n]
                su = tcp0 * hmult_1[um, n] + tsgm_uu0 * tdel + tsgm_uu1
                sd = tcm0 * hmult_2[um, n] + tsgm_ud0 * tdel + tsgm_ud1
                layer_tsup_up[um, n] = tterm_save[n] * (u_xpos[um, n] * sd + u_xneg[um, n] * su)
    if do_dnwelling:
        for n in range(nlayers):
            if delta_tau[n] <= tcutoff:
                continue
            tcp1 = t_c_plus[1, n]
            tcm1 = t_c_minus[1, n]
            tcp0 = t_c_plus[0, n]
            tcm0 = t_c_minus[0, n]
            tsgm_du2 = t_c_plus[2, n]
            tsgm_dd2 = t_c_minus[2, n]
            for um in range(n_users):
                mu = user_streams[um]
                tdel = t_delt_userm[n, um]
                tsgm_du1 = tcp1 - mu * tsgm_du2
                tsgm_dd1 = tcm1 - mu * tsgm_dd2
                tsgm_du0 = -tsgm_du1
                tsgm_dd0 = -tsgm_dd1
                sum_p = tsgm_du1 + tsgm_du2 * delta_tau[n]
                sum_m = tsgm_dd1 + tsgm_dd2 * delta_tau[n]
                su = tcp0 * hmult_2[um, n] + tsgm_du0 * tdel + sum_p
                sd = tcm0 * hmult_1[um, n] + tsgm_dd0 * tdel + sum_m
                layer_tsup_dn[um, n] = tterm_save[n] * (u_xneg[um, n] * sd + u_xpos[um, n] * su)
    return layer_tsup_up, layer_tsup_dn


def _gbeam_solution(
    do_plane_parallel: bool,
    do_postprocessing: bool,
    taylor_order: int,
    taylor_small: float,
    delta_tau: np.ndarray,
    fourier: int,
    pi4: float,
    flux_factor: float,
    layer_pis_cutoffb: int,
    px0xb: float,
    omega: np.ndarray,
    asymm: np.ndarray,
    average_secant_ppb: float,
    average_secantb: np.ndarray,
    initial_transb: np.ndarray,
    t_delt_mubarb: np.ndarray,
    xpos: np.ndarray,
    eigenvalue: np.ndarray,
    eigentrans: np.ndarray,
    norm_saved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nlayers = delta_tau.size
    aterm_save = np.zeros(nlayers, dtype=float)
    bterm_save = np.zeros(nlayers, dtype=float)
    gamma_m = np.zeros(nlayers, dtype=float)
    gamma_p = np.zeros(nlayers, dtype=float)
    wupper = np.zeros((2, nlayers), dtype=float)
    wlower = np.zeros((2, nlayers), dtype=float)
    f1 = flux_factor / pi4

    for n in range(layer_pis_cutoffb):
        secbar = average_secantb[n] if not do_plane_parallel else average_secant_ppb
        const = initial_transb[n]
        wdel = t_delt_mubarb[n]
        eig = eigenvalue[n]

        gamma_pn = secbar + eig
        gamma_mn = secbar - eig
        if do_postprocessing:
            gamma_p[n] = gamma_pn
            gamma_m[n] = gamma_mn
        zdel = eigentrans[n]
        zwdel = zdel * wdel
        if abs(gamma_mn) < taylor_small:
            cfunc = taylor_series_1(taylor_order, gamma_mn, delta_tau[n], wdel, 1.0)
        else:
            cfunc = (zdel - wdel) / gamma_mn
        dfunc = (1.0 - zwdel) / gamma_pn

        omegan = omega[n]
        omega_asymm = omegan * asymm[n] * 3.0
        if fourier == 0:
            tp = omegan + px0xb * omega_asymm
            tm = omegan - px0xb * omega_asymm
        else:
            tp = px0xb * omega_asymm
            tm = px0xb * omega_asymm
        dpin = tp * f1
        dmin = tm * f1

        xpos1 = xpos[0, n]
        xpos2 = xpos[1, n]
        sum_la = dpin * xpos1 + dmin * xpos2
        sum_lb = dmin * xpos1 + dpin * xpos2
        aterm = sum_la / norm_saved[n]
        bterm = sum_lb / norm_saved[n]
        if do_postprocessing:
            aterm_save[n] = aterm
            bterm_save[n] = bterm
        gfunc_dn = cfunc * aterm * const
        gfunc_up = dfunc * bterm * const
        wupper[0, n] = gfunc_up * xpos2
        wupper[1, n] = gfunc_up * xpos1
        wlower[0, n] = gfunc_dn * xpos1
        wlower[1, n] = gfunc_dn * xpos2

    return gamma_m, gamma_p, aterm_save, bterm_save, wupper, wlower


def _build_bvp_matrix(
    do_include_surface: bool,
    do_brdf_surface: bool,
    nlay: int,
    albedo: float,
    brdf_fm: float,
    surface_factorm: float,
    xpos: np.ndarray,
    eigentrans: np.ndarray,
    stream_value: float,
) -> np.ndarray:
    ntotal = 2 * nlay
    mat = np.zeros((ntotal, ntotal), dtype=float)
    h_homp = xpos[0, -1] * stream_value
    h_homm = xpos[1, -1] * stream_value
    factor = 0.0
    if do_include_surface:
        factor = surface_factorm * (brdf_fm if do_brdf_surface else albedo)
    r2_homp = factor * h_homp
    r2_homm = factor * h_homm
    xpnet = xpos[1, -1] - r2_homp if do_include_surface else xpos[1, -1]
    xmnet = xpos[0, -1] - r2_homm if do_include_surface else xpos[0, -1]

    row = 0
    mat[row, 0] = xpos[0, 0]
    mat[row, 1] = xpos[1, 0] * eigentrans[0]
    row += 1

    for n in range(1, nlay):
        n1 = n - 1
        mat[row, 2 * n1] = xpos[0, n1] * eigentrans[n1]
        mat[row, 2 * n1 + 1] = xpos[1, n1]
        mat[row, 2 * n] = -xpos[0, n]
        mat[row, 2 * n + 1] = -xpos[1, n] * eigentrans[n]
        row += 1

        mat[row, 2 * n1] = xpos[1, n1] * eigentrans[n1]
        mat[row, 2 * n1 + 1] = xpos[0, n1]
        mat[row, 2 * n] = -xpos[1, n]
        mat[row, 2 * n + 1] = -xpos[0, n] * eigentrans[n]
        row += 1

    mat[row, -2] = xpnet * eigentrans[-1]
    mat[row, -1] = xmnet
    return mat


def _solve_bvp(
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_include_surface_emission: bool,
    nlay: int,
    albedo: float,
    brdf_fm: float,
    emissivity: float,
    surfbb: float,
    surface_factorm: float,
    xpos: np.ndarray,
    eigentrans: np.ndarray,
    stream_value: float,
    direct_beam: float,
    wupper: np.ndarray,
    wlower: np.ndarray,
    bvp_solver: str = "scipy",
) -> tuple[np.ndarray, np.ndarray]:
    ntotal = 2 * nlay
    rhs = np.zeros(ntotal, dtype=float)
    rhs[0] = -wupper[0, 0]
    row = 1
    for n in range(1, nlay):
        rhs[row] = wupper[0, n] - wlower[0, n - 1]
        row += 1
        rhs[row] = wupper[1, n] - wlower[1, n - 1]
        row += 1
    h_partic = wlower[0, -1] * stream_value
    if do_include_surface:
        factor = surface_factorm * (brdf_fm if do_brdf_surface else albedo)
        r2_partic = h_partic * factor
    else:
        r2_partic = 0.0
    surface_term = -wlower[1, -1]
    if do_include_surface:
        surface_term = surface_term + r2_partic + direct_beam
    if do_include_surface_emission:
        surface_term = surface_term + surfbb * emissivity
    rhs[-1] = surface_term

    if bvp_solver == "pentadiag":
        return _solve_bvp_pentadiag(
            do_include_surface=do_include_surface,
            do_brdf_surface=do_brdf_surface,
            nlay=nlay,
            albedo=albedo,
            brdf_fm=brdf_fm,
            surface_factorm=surface_factorm,
            xpos=xpos,
            eigentrans=eigentrans,
            stream_value=stream_value,
            rhs=rhs,
        )
    if bvp_solver not in {"scipy", "banded"}:
        raise ValueError("bvp_solver must be 'scipy', 'banded', or 'pentadiag'")

    if bvp_solver == "banded":
        banded = _build_bvp_banded_matrix(
            do_include_surface=do_include_surface,
            do_brdf_surface=do_brdf_surface,
            nlay=nlay,
            albedo=albedo,
            brdf_fm=brdf_fm,
            surface_factorm=surface_factorm,
            xpos=xpos,
            eigentrans=eigentrans,
            stream_value=stream_value,
        )
        sol = solve_banded((2, 2), banded, rhs, check_finite=False)
        return sol[0::2].copy(), sol[1::2].copy()
    mat = _build_bvp_matrix(
        do_include_surface=do_include_surface,
        do_brdf_surface=do_brdf_surface,
        nlay=nlay,
        albedo=albedo,
        brdf_fm=brdf_fm,
        surface_factorm=surface_factorm,
        xpos=xpos,
        eigentrans=eigentrans,
        stream_value=stream_value,
    )
    sol = solve(mat, rhs, assume_a="gen", check_finite=True)
    lcon = sol[0::2].copy()
    mcon = sol[1::2].copy()
    return lcon, mcon


def _build_bvp_banded_matrix(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    nlay: int,
    albedo: float,
    brdf_fm: float,
    surface_factorm: float,
    xpos: np.ndarray,
    eigentrans: np.ndarray,
    stream_value: float,
) -> np.ndarray:
    """Builds the five-diagonal BVP matrix in SciPy ``solve_banded`` layout."""

    ntotal = 2 * nlay
    banded = np.zeros((5, ntotal), dtype=float)

    h_homp = xpos[0, -1] * stream_value
    h_homm = xpos[1, -1] * stream_value
    factor = 0.0
    if do_include_surface:
        factor = surface_factorm * (brdf_fm if do_brdf_surface else albedo)
    r2_homp = factor * h_homp
    r2_homm = factor * h_homm
    xpnet = xpos[1, -1] - r2_homp if do_include_surface else xpos[1, -1]
    xmnet = xpos[0, -1] - r2_homm if do_include_surface else xpos[0, -1]

    banded[2, 0] = xpos[0, 0]
    if ntotal > 1:
        banded[1, 1] = xpos[1, 0] * eigentrans[0]

    if nlay > 1:
        prev, row_m, row_p = _bvp_banded_layer_indices(nlay)
        banded[3, row_m - 1] = xpos[0, prev] * eigentrans[prev]
        banded[2, row_m] = xpos[1, prev]
        banded[1, row_m + 1] = -xpos[0, prev + 1]
        banded[0, row_m + 2] = -xpos[1, prev + 1] * eigentrans[prev + 1]
        banded[4, row_p - 2] = xpos[1, prev] * eigentrans[prev]
        banded[3, row_p - 1] = xpos[0, prev]
        banded[2, row_p] = -xpos[1, prev + 1]
        banded[1, row_p + 1] = -xpos[0, prev + 1] * eigentrans[prev + 1]

    banded[3, ntotal - 2] = xpnet * eigentrans[-1]
    banded[2, ntotal - 1] = xmnet
    return banded


@lru_cache(maxsize=16)
def _bvp_banded_layer_indices(nlay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns cached row indices for the regular two-stream BVP stencil."""

    n = np.arange(1, nlay)
    prev = n - 1
    row_m = 2 * n - 1
    row_p = row_m + 1
    return prev, row_m, row_p


def _solve_bvp_pentadiag(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    nlay: int,
    albedo: float,
    brdf_fm: float,
    surface_factorm: float,
    xpos: np.ndarray,
    eigentrans: np.ndarray,
    stream_value: float,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the BVP with the regular Fortran pentadiagonal elimination path."""

    ntotal = 2 * nlay
    if nlay == 1:
        mat = _build_bvp_matrix(
            do_include_surface=do_include_surface,
            do_brdf_surface=do_brdf_surface,
            nlay=nlay,
            albedo=albedo,
            brdf_fm=brdf_fm,
            surface_factorm=surface_factorm,
            xpos=xpos,
            eigentrans=eigentrans,
            stream_value=stream_value,
        )
        sol = np.linalg.solve(mat, rhs)
        return sol[0::2].copy(), sol[1::2].copy()

    smallnum = 1.0e-20
    mat1 = np.zeros(ntotal, dtype=float)
    mat2 = np.zeros(ntotal, dtype=float)
    mat3 = np.zeros(ntotal, dtype=float)
    mat4 = np.zeros(ntotal, dtype=float)
    mat5 = np.zeros(ntotal, dtype=float)

    h_homp = xpos[0, -1] * stream_value
    h_homm = xpos[1, -1] * stream_value
    factor = 0.0
    if do_include_surface:
        factor = surface_factorm * (brdf_fm if do_brdf_surface else albedo)
    r2_homp = factor * h_homp
    r2_homm = factor * h_homm
    xpnet = xpos[1, -1] - r2_homp if do_include_surface else xpos[1, -1]
    xmnet = xpos[0, -1] - r2_homm if do_include_surface else xpos[0, -1]

    mat3[0] = xpos[0, 0]
    mat4[0] = xpos[1, 0] * eigentrans[0]

    for n in range(1, nlay):
        prev = n - 1
        row_m = 2 * n - 1
        row_p = row_m + 1
        if n > 1:
            mat1[row_m] = 0.0
        mat2[row_m] = xpos[0, prev] * eigentrans[prev]
        mat3[row_m] = xpos[1, prev]
        mat4[row_m] = -xpos[0, n]
        mat5[row_m] = -xpos[1, n] * eigentrans[n]
        mat1[row_p] = xpos[1, prev] * eigentrans[prev]
        mat2[row_p] = xpos[0, prev]
        mat3[row_p] = -xpos[1, n]
        mat4[row_p] = -xpos[0, n] * eigentrans[n]
        mat5[row_p] = 0.0

    mat1[-1] = 0.0
    mat2[-1] = xpnet * eigentrans[-1]
    mat3[-1] = xmnet

    elm1 = np.zeros(ntotal - 1, dtype=float)
    elm2 = np.zeros(ntotal - 2, dtype=float)
    elm3 = np.zeros(ntotal, dtype=float)
    elm4 = np.zeros(ntotal, dtype=float)

    elm31 = 1.0 / mat3[0]
    elm3[0] = elm31
    elm1_i2 = -mat4[0] * elm31
    elm1[0] = elm1_i2
    elm2_i2 = -mat5[0] * elm31
    elm2[0] = elm2_i2

    mat22 = mat2[1]
    bet = mat3[1] + mat22 * elm1_i2
    if abs(bet) < smallnum:
        raise np.linalg.LinAlgError("Singularity in Pentadiagonal Matrix, Row #  2")
    bet = -1.0 / bet
    elm1_i1 = (mat4[1] + mat22 * elm2_i2) * bet
    elm1[1] = elm1_i1
    elm2_i1 = mat5[1] * bet
    elm2[1] = elm2_i1
    elm3[1] = bet

    for i in range(2, ntotal - 2):
        mat1_i = mat1[i]
        bet = mat2[i] + mat1_i * elm1_i2
        den = mat3[i] + mat1_i * elm2_i2 + bet * elm1_i1
        if abs(den) < smallnum:
            raise np.linalg.LinAlgError(f"Singularity in Pentadiagonal Matrix, Row #{i + 1:3d}")
        den = -1.0 / den
        elm1_i2 = elm1_i1
        elm1_i1 = (mat4[i] + bet * elm2_i1) * den
        elm1[i] = elm1_i1
        elm2_i2 = elm2_i1
        elm2_i1 = mat5[i] * den
        elm2[i] = elm2_i1
        elm3[i] = bet
        elm4[i] = den

    i = ntotal - 2
    mat1_i = mat1[i]
    bet = mat2[i] + mat1_i * elm1_i2
    den = mat3[i] + mat1_i * elm2_i2 + bet * elm1_i1
    if abs(den) < smallnum:
        raise np.linalg.LinAlgError(f"Singularity in Pentadiagonal Matrix, Row #{i + 1:3d}")
    den = -1.0 / den
    elm1_i2 = elm1_i1
    elm1_i1 = (mat4[i] + bet * elm2_i1) * den
    elm1[i] = elm1_i1
    elm2_i2 = elm2_i1
    elm3[i] = bet
    elm4[i] = den

    i = ntotal - 1
    mat1_i = mat1[i]
    bet = mat2[i] + mat1_i * elm1_i2
    den = mat3[i] + mat1_i * elm2_i2 + bet * elm1_i1
    if abs(den) < smallnum:
        raise np.linalg.LinAlgError(f"Singularity in Pentadiagonal Matrix, Row #{i + 1:3d}")
    den = -1.0 / den
    elm3[i] = bet
    elm4[i] = den

    col = rhs.copy()
    col_i2 = col[0] * elm3[0]
    col[0] = col_i2
    col_i1 = (mat22 * col_i2 - col[1]) * elm3[1]
    col[1] = col_i1
    for i in range(2, ntotal):
        col_i = (mat1[i] * col_i2 + elm3[i] * col_i1 - col[i]) * elm4[i]
        col_i2 = col_i1
        col_i1 = col_i
        col[i] = col_i

    i = ntotal - 2
    col_i = col_i2 + elm1[i] * col_i1
    col[i] = col_i
    col_i2 = col_i1
    col_i1 = col_i
    for i in range(ntotal - 3, -1, -1):
        col_i = col[i] + elm1[i] * col_i1 + elm2[i] * col_i2
        col[i] = col_i
        col_i2 = col_i1
        col_i1 = col_i

    lcon = col[0::2].copy()
    mcon = col[1::2].copy()
    return lcon, mcon


def _direct_beam(
    *,
    do_brdf_surface: bool,
    do_surface_leaving: bool,
    do_sl_isotropic: bool,
    fourier: int,
    flux_factor: float,
    x0: float,
    delta_factorm: float,
    albedo: float,
    brdf_f_0m: float,
    slterm_isotropic: float,
    slterm_f_0m: float,
    trans_solar_beam: float,
    do_reflected_directbeam: bool,
) -> tuple[float, float]:
    if not do_reflected_directbeam:
        return 0.0, 0.0

    pi = math.acos(-1.0)
    attn = flux_factor * x0 / delta_factorm / pi * trans_solar_beam
    if do_brdf_surface:
        direct_beam = attn * brdf_f_0m
    else:
        direct_beam = attn * albedo

    if do_surface_leaving:
        helpv = flux_factor / delta_factorm
        if do_sl_isotropic and fourier == 0:
            direct_beam = direct_beam + slterm_isotropic * helpv
        else:
            direct_beam = direct_beam + slterm_f_0m * helpv
    return attn, direct_beam


def _upuser_intensity(
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_level_output: bool,
    nlay: int,
    taylor_order: int,
    layer_pis_cutoffb: int,
    surface_factor: float,
    albedo: float,
    ubrdf_fmb: float,
    fluxmult: float,
    stream_value: float,
    taylor_small: float,
    delta_tau: np.ndarray,
    gamma_p: np.ndarray,
    gamma_m: np.ndarray,
    sigma_p: np.ndarray,
    aterm_save: np.ndarray,
    bterm_save: np.ndarray,
    initial_transb: np.ndarray,
    itrans_usermb: np.ndarray,
    t_delt_usermb: np.ndarray,
    t_delt_mubarb: np.ndarray,
    t_delt_eigennl: float,
    lcon: np.ndarray,
    lcon_xvec1nl: float,
    mcon: np.ndarray,
    mcon_xvec1nl: float,
    wlower1nl: float,
    u_xposb: np.ndarray,
    u_xnegb: np.ndarray,
    hmult_1b: np.ndarray,
    hmult_2b: np.ndarray,
    emult_upb: np.ndarray,
) -> tuple[float, np.ndarray]:
    radlevel = np.zeros(nlay + 1, dtype=float)
    if do_include_surface:
        par = wlower1nl
        hom = lcon_xvec1nl * t_delt_eigennl + mcon_xvec1nl
        idownsurf = (par + hom) * stream_value
        if do_brdf_surface:
            boa_source = surface_factor * idownsurf * ubrdf_fmb
        else:
            boa_source = surface_factor * albedo * idownsurf
    else:
        boa_source = 0.0

    cumsource_old = boa_source
    if do_level_output:
        radlevel[nlay] = fluxmult * cumsource_old

    for n in range(nlay - 1, -1, -1):
        layersource = lcon[n] * u_xposb[n] * hmult_2b[n] + mcon[n] * u_xnegb[n] * hmult_1b[n]
        if (n + 1) <= layer_pis_cutoffb:
            wdel = t_delt_mubarb[n]
            itrans = initial_transb[n]
            gammamn = gamma_m[n]
            emult_upn = emult_upb[n]
            if abs(gammamn) < taylor_small:
                fac2 = wdel * t_delt_usermb[n]
                mult = taylor_series_2(
                    taylor_order,
                    taylor_small,
                    gammamn,
                    sigma_p[n],
                    delta_tau[n],
                    1.0,
                    fac2,
                    1.0,
                )
                sd = itrans_usermb[n] * mult
            else:
                sd = (itrans * hmult_2b[n] - emult_upn) / gammamn
            su = (-itrans * wdel * hmult_1b[n] + emult_upn) / gamma_p[n]
            pmult_ud = sd * aterm_save[n]
            pmult_uu = su * bterm_save[n]
            layersource = layersource + u_xposb[n] * pmult_ud + u_xnegb[n] * pmult_uu

        cumsource_new = layersource + t_delt_usermb[n] * cumsource_old
        if do_level_output:
            radlevel[n] = fluxmult * cumsource_new
        cumsource_old = cumsource_new

    return fluxmult * cumsource_old, radlevel


def _dnuser_intensity(
    do_level_output: bool,
    nlay: int,
    taylor_order: int,
    layer_pis_cutoffb: int,
    fluxmult: float,
    taylor_small: float,
    delta_tau: np.ndarray,
    gamma_p: np.ndarray,
    gamma_m: np.ndarray,
    sigma_m: np.ndarray,
    aterm_save: np.ndarray,
    bterm_save: np.ndarray,
    initial_transb: np.ndarray,
    itrans_usermb: np.ndarray,
    t_delt_usermb: np.ndarray,
    t_delt_mubarb: np.ndarray,
    lcon: np.ndarray,
    mcon: np.ndarray,
    u_xposb: np.ndarray,
    u_xnegb: np.ndarray,
    hmult_1b: np.ndarray,
    hmult_2b: np.ndarray,
    emult_dnb: np.ndarray,
) -> tuple[float, np.ndarray]:
    radlevel = np.zeros(nlay + 1, dtype=float)
    cumsource_old = 0.0
    for n in range(nlay):
        layersource = lcon[n] * u_xnegb[n] * hmult_1b[n] + mcon[n] * u_xposb[n] * hmult_2b[n]
        if (n + 1) <= layer_pis_cutoffb:
            wdel = t_delt_mubarb[n]
            itrans = initial_transb[n]
            gammamn = gamma_m[n]
            emult_dnn = emult_dnb[n]
            if abs(gammamn) < taylor_small:
                mult = taylor_series_2(
                    taylor_order,
                    taylor_small,
                    gammamn,
                    sigma_m[n],
                    delta_tau[n],
                    t_delt_usermb[n],
                    wdel,
                    1.0,
                )
                sd = itrans_usermb[n] * mult
            else:
                sd = (itrans * hmult_1b[n] - emult_dnn) / gammamn
            su = (-itrans * wdel * hmult_2b[n] + emult_dnn) / gamma_p[n]
            pmult_dd = sd * aterm_save[n]
            pmult_du = su * bterm_save[n]
            layersource = layersource + u_xnegb[n] * pmult_dd + u_xposb[n] * pmult_du
        cumsource_new = layersource + t_delt_usermb[n] * cumsource_old
        if do_level_output:
            radlevel[n + 1] = fluxmult * cumsource_new
        cumsource_old = cumsource_new
    return fluxmult * cumsource_old, radlevel


def _upuser_intensity_thermal(
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_level_output: bool,
    nlay: int,
    surface_factor: float,
    albedo: float,
    ubrdf_fm: np.ndarray,
    fluxmult: float,
    stream_value: float,
    t_delt_userm: np.ndarray,
    t_delt_eigennl: float,
    lcon: np.ndarray,
    lcon_xvec1nl: float,
    mcon: np.ndarray,
    mcon_xvec1nl: float,
    wlower1nl: float,
    u_xpos: np.ndarray,
    u_xneg: np.ndarray,
    hmult_1: np.ndarray,
    hmult_2: np.ndarray,
    layer_tsup_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_users = u_xpos.shape[0]
    radlevel = np.zeros((n_users, nlay + 1), dtype=float)
    intensity = np.zeros(n_users, dtype=float)
    boa_source = np.zeros(n_users, dtype=float)
    if do_include_surface:
        par = wlower1nl
        hom = lcon_xvec1nl * t_delt_eigennl + mcon_xvec1nl
        idownsurf = (par + hom) * stream_value
        if do_brdf_surface:
            boa_source = surface_factor * idownsurf * ubrdf_fm
        else:
            boa_source.fill(surface_factor * albedo * idownsurf)
    for um in range(n_users):
        cumsource_old = boa_source[um]
        if do_level_output:
            radlevel[um, nlay] = fluxmult * cumsource_old
        for n in range(nlay - 1, -1, -1):
            layersource = (
                lcon[n] * u_xpos[um, n] * hmult_2[um, n]
                + mcon[n] * u_xneg[um, n] * hmult_1[um, n]
                + layer_tsup_up[um, n]
            )
            cumsource_new = layersource + t_delt_userm[n, um] * cumsource_old
            if do_level_output:
                radlevel[um, n] = fluxmult * cumsource_new
            cumsource_old = cumsource_new
        intensity[um] = fluxmult * cumsource_old
    return intensity, radlevel


def _dnuser_intensity_thermal(
    do_level_output: bool,
    nlay: int,
    fluxmult: float,
    t_delt_userm: np.ndarray,
    lcon: np.ndarray,
    mcon: np.ndarray,
    u_xpos: np.ndarray,
    u_xneg: np.ndarray,
    hmult_1: np.ndarray,
    hmult_2: np.ndarray,
    layer_tsup_dn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_users = u_xpos.shape[0]
    radlevel = np.zeros((n_users, nlay + 1), dtype=float)
    intensity = np.zeros(n_users, dtype=float)
    for um in range(n_users):
        cumsource_old = 0.0
        for n in range(nlay):
            layersource = (
                lcon[n] * u_xneg[um, n] * hmult_1[um, n]
                + mcon[n] * u_xpos[um, n] * hmult_2[um, n]
                + layer_tsup_dn[um, n]
            )
            cumsource_new = layersource + t_delt_userm[n, um] * cumsource_old
            if do_level_output:
                radlevel[um, n + 1] = fluxmult * cumsource_new
            cumsource_old = cumsource_new
        intensity[um] = fluxmult * cumsource_old
    return intensity, radlevel


def _fluxes_solar(
    do_upwelling: bool,
    do_dnwelling: bool,
    do_directbeamb: bool,
    pi4: float,
    stream_value: float,
    fluxfac: float,
    fluxmult: float,
    x0b: float,
    trans_solar_beamb: float,
    lcon_xvec21: float,
    mcon_xvec21: float,
    eigentrans1: float,
    wupper21: float,
    lcon_xvec1nl: float,
    mcon_xvec1nl: float,
    eigentransnl: float,
    wlower1nl: float,
) -> tuple[np.ndarray, np.ndarray]:
    toa = np.zeros(2, dtype=float)
    boa = np.zeros(2, dtype=float)
    pi2 = 0.5 * pi4
    if do_upwelling:
        shom = lcon_xvec21 + mcon_xvec21 * eigentrans1
        quad = fluxmult * (wupper21 + shom)
        toa[0] = 0.5 * quad
        toa[1] = pi2 * stream_value * quad
    if do_dnwelling:
        shom = lcon_xvec1nl * eigentransnl + mcon_xvec1nl
        quad = fluxmult * (wlower1nl + shom)
        boa[0] = 0.5 * quad
        boa[1] = pi2 * stream_value * quad
        if do_directbeamb:
            dmean = fluxfac * trans_solar_beamb / pi4
            dflux = fluxfac * trans_solar_beamb * x0b
            boa[0] += dmean
            boa[1] += dflux
    return toa, boa


def _fluxes_thermal(
    do_upwelling: bool,
    do_dnwelling: bool,
    pi4: float,
    stream_value: float,
    fluxmult: float,
    lcon_xvec21: float,
    mcon_xvec21: float,
    eigentrans1: float,
    wupper21: float,
    lcon_xvec1nl: float,
    mcon_xvec1nl: float,
    eigentransnl: float,
    wlower1nl: float,
) -> tuple[np.ndarray, np.ndarray]:
    toa = np.zeros(2, dtype=float)
    boa = np.zeros(2, dtype=float)
    pi2 = 0.5 * pi4
    if do_upwelling:
        shom = lcon_xvec21 + mcon_xvec21 * eigentrans1
        quad = fluxmult * (wupper21 + shom)
        toa[0] = 0.5 * quad
        toa[1] = pi2 * stream_value * quad
    if do_dnwelling:
        shom = lcon_xvec1nl * eigentransnl + mcon_xvec1nl
        quad = fluxmult * (wlower1nl + shom)
        boa[0] = 0.5 * quad
        boa[1] = pi2 * stream_value * quad
    return toa, boa


def _solve_optimized_thermal(prepared: PreparedInputs, options) -> dict[str, np.ndarray]:
    """Solves the optimized thermal two-stream forward problem.

    Parameters
    ----------
    prepared
        Preprocessed thermal solver inputs.
    options
        Public solver options object.

    Returns
    -------
    dict of ndarray
        Thermal radiance, flux, and optional level-profile outputs.
    """
    if prepared.thermal is None:
        raise ValueError("thermal inputs are required for thermal mode")

    geom = prepared.geometry
    delta_tau, omega_total, asymm_total = _apply_delta_scaling(prepared, options.do_delta_scaling)
    thermal_coeffs = _thermal_setup(delta_tau, prepared.thermal.thermal_bb_input)
    n_users, nlay = thermal_problem_size(prepared)
    solved = _initialize_solution_storage(n_users, nlay, flux_geometry_count=1)

    eigenvalue, eigentrans, xpos, norm_saved = _hom_solution_thermal(
        stream_value=prepared.stream_value,
        pxsq=geom.pxsq[0],
        omega=omega_total,
        asymm=asymm_total,
        delta_tau=delta_tau,
    )
    u_xpos, u_xneg, hmult_1, hmult_2, t_delt_userm = prepare_thermal_postprocessing(
        do_postprocessing=geom.do_postprocessing,
        delta_tau=delta_tau,
        user_secants=geom.user_secants,
        n_users=n_users,
        nlay=nlay,
        build_user_solution=lambda: _hom_user_solution_thermal(
            stream_value=prepared.stream_value,
            user_streams=geom.user_streams,
            xpos=xpos,
            omega=omega_total,
            asymm=asymm_total,
        ),
        build_hmult=lambda t_delt_userm: _hmult_master(
            taylor_order=3,
            taylor_small=1.0e-3,
            delta_tau=delta_tau,
            user_secants=geom.user_secants,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            t_delt_userm=t_delt_userm,
        ),
        make_zero_array=lambda shape: np.zeros(shape, dtype=float),
        exp_outer=lambda left, right: np.exp(-np.outer(left, right)),
    )
    t_c_plus, t_c_minus, tterm_save, t_wupper, t_wlower = _thermal_gf_solution(
        omega=omega_total,
        delta_tau=delta_tau,
        thermcoeffs=thermal_coeffs,
        tcutoff=options.thermal_tcutoff,
        eigenvalue=eigenvalue,
        eigentrans=eigentrans,
        xpos=xpos,
        norm_saved=norm_saved,
    )
    layer_tsup_up, layer_tsup_dn = _thermal_terms(
        do_upwelling=options.do_upwelling,
        do_dnwelling=options.do_dnwelling,
        user_streams=geom.user_streams,
        tcutoff=options.thermal_tcutoff,
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
    do_include_surface = include_thermal_surface_term(
        albedo=prepared.albedo,
        do_brdf_surface=options.do_brdf_surface,
    )
    lcon, mcon = _solve_bvp(
        do_include_surface=do_include_surface,
        do_brdf_surface=options.do_brdf_surface,
        do_include_surface_emission=(prepared.thermal.emissivity != 0.0),
        nlay=prepared.tau_arr.size,
        albedo=prepared.albedo,
        brdf_fm=0.0 if prepared.brdf is None else prepared.brdf.brdf_f[0],
        emissivity=prepared.thermal.emissivity,
        surfbb=prepared.thermal.surfbb,
        surface_factorm=geom.surface_factor[0],
        xpos=xpos,
        eigentrans=eigentrans,
        stream_value=prepared.stream_value,
        direct_beam=0.0,
        wupper=t_wupper,
        wlower=t_wlower,
        bvp_solver=options.bvp_solver,
    )
    boundary_terms = prepare_thermal_boundary_terms(
        lcon=lcon,
        mcon=mcon,
        xpos=xpos,
        eigentrans=eigentrans,
        wlower=t_wlower,
    )
    if options.do_upwelling and geom.do_postprocessing:
        solved["intensity_toa"], solved["radlevel_up"] = _upuser_intensity_thermal(
            do_include_surface=do_include_surface,
            do_brdf_surface=options.do_brdf_surface,
            do_level_output=options.do_level_output,
            nlay=prepared.tau_arr.size,
            surface_factor=geom.surface_factor[0],
            albedo=prepared.albedo,
            ubrdf_fm=np.zeros(n_users, dtype=float)
            if prepared.brdf is None
            else prepared.brdf.ubrdf_f[:, 0],
            fluxmult=geom.delta_factor[0],
            stream_value=prepared.stream_value,
            t_delt_userm=t_delt_userm,
            t_delt_eigennl=boundary_terms["eigentransnl"],
            lcon=lcon,
            lcon_xvec1nl=boundary_terms["lcon_xvec1nl"],
            mcon=mcon,
            mcon_xvec1nl=boundary_terms["mcon_xvec1nl"],
            wlower1nl=boundary_terms["wlower1nl"],
            u_xpos=u_xpos,
            u_xneg=u_xneg,
            hmult_1=hmult_1,
            hmult_2=hmult_2,
            layer_tsup_up=layer_tsup_up,
        )
    if options.do_dnwelling and geom.do_postprocessing:
        solved["intensity_boa"], solved["radlevel_dn"] = _dnuser_intensity_thermal(
            do_level_output=options.do_level_output,
            nlay=prepared.tau_arr.size,
            fluxmult=geom.delta_factor[0],
            t_delt_userm=t_delt_userm,
            lcon=lcon,
            mcon=mcon,
            u_xpos=u_xpos,
            u_xneg=u_xneg,
            hmult_1=hmult_1,
            hmult_2=hmult_2,
            layer_tsup_dn=layer_tsup_dn,
        )
    if geom.do_include_mvout[0]:
        eigentrans1 = eigentrans[0]
        lcon_xvec21 = lcon[0] * xpos[1, 0]
        mcon_xvec21 = mcon[0] * xpos[0, 0]
        wupper21 = t_wupper[1, 0]
        toa, boa = _fluxes_thermal(
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            pi4=geom.pi4,
            stream_value=prepared.stream_value,
            fluxmult=geom.delta_factor[0],
            lcon_xvec21=lcon_xvec21,
            mcon_xvec21=mcon_xvec21,
            eigentrans1=eigentrans1,
            wupper21=wupper21,
            lcon_xvec1nl=boundary_terms["lcon_xvec1nl"],
            mcon_xvec1nl=boundary_terms["mcon_xvec1nl"],
            eigentransnl=boundary_terms["eigentransnl"],
            wlower1nl=boundary_terms["wlower1nl"],
        )
        accumulate_flux_pair(
            fluxes_toa=solved["fluxes_toa"],
            fluxes_boa=solved["fluxes_boa"],
            toa=toa,
            boa=boa,
            index=0,
            fourier=0,
            azmfac=1.0,
        )
    return _finalize_solution_storage(solved, prepared)


def solve_optimized_solar_obs(prepared: PreparedInputs, options) -> dict[str, np.ndarray]:
    """Solves the optimized solar two-stream forward problem.

    Parameters
    ----------
    prepared
        Preprocessed solar solver inputs.
    options
        Public solver options object.

    Returns
    -------
    dict of ndarray
        Solar radiance, flux, and optional level-profile outputs.
    """
    if prepared.source_mode == "thermal":
        return _solve_optimized_thermal(prepared, options)
    geom = prepared.geometry
    delta_tau, omega_total, asymm_total = _apply_delta_scaling(prepared, options.do_delta_scaling)
    misc = prepare_solar_misc(
        do_plane_parallel=options.do_plane_parallel,
        build_plane_parallel=lambda: _qsprep_obs_pp(
            delta_tau=delta_tau,
            average_secant_pp=geom.average_secant_pp,
            user_secants=geom.user_secants,
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            do_postprocessing=geom.do_postprocessing,
            taylor_small=1.0e-3,
            taylor_order=3,
        ),
        build_spherical=lambda: _qsprep_obs(
            delta_tau=delta_tau,
            chapman_factors=geom.chapman_factors,
            user_secants=geom.user_secants,
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            do_postprocessing=geom.do_postprocessing,
            taylor_small=1.0e-3,
            taylor_order=3,
        ),
    )

    ngeoms, nlay = solar_problem_size(prepared)
    solved = _initialize_solution_storage(ngeoms, nlay)

    for fourier in range(geom.n_fouriers + 1):
        pxsq = geom.pxsq[fourier]
        px0x = geom.px0x[:, fourier]
        surface_factor = geom.surface_factor[fourier]
        delta_factor = geom.delta_factor[fourier]
        (
            eigenvalue,
            eigentrans,
            xpos,
            norm_saved,
            u_xpos,
            u_xneg,
            hmult_1,
            hmult_2,
        ) = prepare_solar_fourier_postprocessing(
            build_hom_solution=lambda: _hom_solution_solar(
                fourier=fourier,
                stream_value=prepared.stream_value,
                pxsq=pxsq,
                omega=omega_total,
                asymm=asymm_total,
                delta_tau=delta_tau,
            ),
            build_user_solution=lambda xpos: _hom_user_solution_solar(
                fourier=fourier,
                stream_value=prepared.stream_value,
                px11=geom.px11,
                user_streams=geom.user_streams,
                ulp=geom.ulp,
                xpos=xpos,
                omega=omega_total,
                asymm=asymm_total,
            ),
            build_hmult=lambda eigenvalue, eigentrans: _hmult_master(
                taylor_order=3,
                taylor_small=1.0e-3,
                delta_tau=delta_tau,
                user_secants=geom.user_secants,
                eigenvalue=eigenvalue,
                eigentrans=eigentrans,
                t_delt_userm=misc["t_delt_userm"],
            ),
        )

        do_include_surface = include_surface_term(
            fourier,
            albedo=prepared.albedo,
            do_brdf_surface=options.do_brdf_surface,
        )
        direct_beam_terms = prepare_solar_direct_beam_terms(
            ngeoms=ngeoms,
            do_include_surface=do_include_surface,
            make_zero_vector=lambda size: np.zeros(size, dtype=float),
            compute_direct_beam=lambda ib: _direct_beam(
                do_brdf_surface=options.do_brdf_surface,
                do_surface_leaving=options.do_surface_leaving,
                do_sl_isotropic=options.do_sl_isotropic,
                fourier=fourier,
                flux_factor=prepared.flux_factor,
                x0=geom.x0[ib],
                delta_factorm=delta_factor,
                albedo=prepared.albedo,
                brdf_f_0m=0.0 if prepared.brdf is None else prepared.brdf.brdf_f_0[ib, fourier],
                slterm_isotropic=(
                    0.0
                    if prepared.surface_leaving is None
                    else prepared.surface_leaving.slterm_isotropic[ib]
                ),
                slterm_f_0m=(
                    0.0
                    if prepared.surface_leaving is None
                    else prepared.surface_leaving.slterm_f_0[ib, fourier]
                ),
                trans_solar_beam=misc["trans_solar_beam"][ib],
                do_reflected_directbeam=True,
            )[1],
        )

        for ib in range(ngeoms):
            (
                gamma_m,
                gamma_p,
                aterm,
                bterm,
                wupper,
                wlower,
                lcon,
                mcon,
                boundary_terms,
            ) = prepare_solar_geometry_solution(
                build_gbeam_solution=lambda: _gbeam_solution(
                    do_plane_parallel=options.do_plane_parallel,
                    do_postprocessing=geom.do_postprocessing,
                    taylor_order=3,
                    taylor_small=1.0e-3,
                    delta_tau=delta_tau,
                    fourier=fourier,
                    pi4=geom.pi4,
                    flux_factor=prepared.flux_factor,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    px0xb=px0x[ib],
                    omega=omega_total,
                    asymm=asymm_total,
                    average_secant_ppb=geom.average_secant_pp[ib],
                    average_secantb=misc["average_secant"][:, ib],
                    initial_transb=misc["initial_trans"][:, ib],
                    t_delt_mubarb=misc["t_delt_mubar"][:, ib],
                    xpos=xpos,
                    eigenvalue=eigenvalue,
                    eigentrans=eigentrans,
                    norm_saved=norm_saved,
                ),
                solve_bvp=lambda wupper, wlower: _solve_bvp(
                    do_include_surface=do_include_surface,
                    do_brdf_surface=options.do_brdf_surface,
                    do_include_surface_emission=False,
                    nlay=prepared.tau_arr.size,
                    albedo=prepared.albedo,
                    brdf_fm=0.0 if prepared.brdf is None else prepared.brdf.brdf_f[fourier],
                    emissivity=0.0,
                    surfbb=0.0,
                    surface_factorm=surface_factor,
                    xpos=xpos,
                    eigentrans=eigentrans,
                    stream_value=prepared.stream_value,
                    direct_beam=direct_beam_terms[ib],
                    wupper=wupper,
                    wlower=wlower,
                    bvp_solver=options.bvp_solver,
                ),
                extract_boundary_terms=lambda lcon, mcon, wupper, wlower: {
                    "eigentransnl": eigentrans[-1],
                    "lcon_xvec1nl": lcon[-1] * xpos[0, -1],
                    "mcon_xvec1nl": mcon[-1] * xpos[1, -1],
                    "wlower1nl": wlower[0, -1],
                    "eigentrans1": eigentrans[0],
                    "lcon_xvec21": lcon[0] * xpos[1, 0],
                    "mcon_xvec21": mcon[0] * xpos[0, 0],
                    "wupper21": wupper[1, 0],
                },
            )
            if options.do_upwelling and geom.do_postprocessing:
                intensity_f_up, rad_f_up = _upuser_intensity(
                    do_include_surface=do_include_surface,
                    do_brdf_surface=options.do_brdf_surface,
                    do_level_output=options.do_level_output,
                    nlay=prepared.tau_arr.size,
                    taylor_order=3,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    surface_factor=surface_factor,
                    albedo=prepared.albedo,
                    ubrdf_fmb=0.0 if prepared.brdf is None else prepared.brdf.ubrdf_f[ib, fourier],
                    fluxmult=delta_factor,
                    stream_value=prepared.stream_value,
                    taylor_small=1.0e-3,
                    delta_tau=delta_tau,
                    gamma_p=gamma_p,
                    gamma_m=gamma_m,
                    sigma_p=misc["sigma_p"][:, ib],
                    aterm_save=aterm,
                    bterm_save=bterm,
                    initial_transb=misc["initial_trans"][:, ib],
                    itrans_usermb=misc["itrans_userm"][:, ib],
                    t_delt_usermb=misc["t_delt_userm"][:, ib],
                    t_delt_mubarb=misc["t_delt_mubar"][:, ib],
                    t_delt_eigennl=boundary_terms["eigentransnl"],
                    lcon=lcon,
                    lcon_xvec1nl=boundary_terms["lcon_xvec1nl"],
                    mcon=mcon,
                    mcon_xvec1nl=boundary_terms["mcon_xvec1nl"],
                    wlower1nl=boundary_terms["wlower1nl"],
                    u_xposb=u_xpos[ib, :],
                    u_xnegb=u_xneg[ib, :],
                    hmult_1b=hmult_1[ib, :],
                    hmult_2b=hmult_2[ib, :],
                    emult_upb=misc["emult_up"][:, ib],
                )
                accumulate_scalar_and_levels(
                    scalar_store=solved["intensity_toa"],
                    scalar_value=intensity_f_up,
                    level_store=solved["radlevel_up"],
                    level_value=rad_f_up,
                    index=ib,
                    fourier=fourier,
                    azmfac=geom.azmfac[ib],
                    do_level_output=options.do_level_output,
                )

            if options.do_dnwelling and geom.do_postprocessing:
                intensity_f_dn, rad_f_dn = _dnuser_intensity(
                    do_level_output=options.do_level_output,
                    nlay=prepared.tau_arr.size,
                    taylor_order=3,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    fluxmult=delta_factor,
                    taylor_small=1.0e-3,
                    delta_tau=delta_tau,
                    gamma_p=gamma_p,
                    gamma_m=gamma_m,
                    sigma_m=misc["sigma_m"][:, ib],
                    aterm_save=aterm,
                    bterm_save=bterm,
                    initial_transb=misc["initial_trans"][:, ib],
                    itrans_usermb=misc["itrans_userm"][:, ib],
                    t_delt_usermb=misc["t_delt_userm"][:, ib],
                    t_delt_mubarb=misc["t_delt_mubar"][:, ib],
                    lcon=lcon,
                    mcon=mcon,
                    u_xposb=u_xpos[ib, :],
                    u_xnegb=u_xneg[ib, :],
                    hmult_1b=hmult_1[ib, :],
                    hmult_2b=hmult_2[ib, :],
                    emult_dnb=misc["emult_dn"][:, ib],
                )
                accumulate_scalar_and_levels(
                    scalar_store=solved["intensity_boa"],
                    scalar_value=intensity_f_dn,
                    level_store=solved["radlevel_dn"],
                    level_value=rad_f_dn,
                    index=ib,
                    fourier=fourier,
                    azmfac=geom.azmfac[ib],
                    do_level_output=options.do_level_output,
                )

            if geom.do_include_mvout[fourier]:
                toa, boa = _fluxes_solar(
                    do_upwelling=options.do_upwelling,
                    do_dnwelling=options.do_dnwelling,
                    do_directbeamb=do_include_surface,
                    pi4=geom.pi4,
                    stream_value=prepared.stream_value,
                    fluxfac=prepared.flux_factor,
                    fluxmult=delta_factor,
                    x0b=geom.x0[ib],
                    trans_solar_beamb=misc["trans_solar_beam"][ib],
                    lcon_xvec21=boundary_terms["lcon_xvec21"],
                    mcon_xvec21=boundary_terms["mcon_xvec21"],
                    eigentrans1=boundary_terms["eigentrans1"],
                    wupper21=boundary_terms["wupper21"],
                    lcon_xvec1nl=boundary_terms["lcon_xvec1nl"],
                    mcon_xvec1nl=boundary_terms["mcon_xvec1nl"],
                    eigentransnl=boundary_terms["eigentransnl"],
                    wlower1nl=boundary_terms["wlower1nl"],
                )
                accumulate_flux_pair(
                    fluxes_toa=solved["fluxes_toa"],
                    fluxes_boa=solved["fluxes_boa"],
                    toa=toa,
                    boa=boa,
                    index=ib,
                    fourier=fourier,
                    azmfac=geom.azmfac[ib],
                )

    return _finalize_solution_storage(solved, prepared)
