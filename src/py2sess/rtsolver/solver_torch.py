"""Torch-native two-stream forward solver implementations."""

from __future__ import annotations

import math

from .backend import _load_torch
from ..optical.delta_m_torch import delta_m_scale_optical_properties_torch
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
from .taylor_torch import taylor_series_1_torch, taylor_series_2_torch

torch = _load_torch()
_OPTICAL_THICKNESS_MIN = 1.0e-12


def _as_reference_tensor(value, reference):
    """Converts scalar-like inputs to the dtype/device of a reference tensor."""
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _requires_grad(value) -> bool:
    """Returns true when a tensor-valued control needs the zero branch kept."""
    return isinstance(value, torch.Tensor) and value.requires_grad


def _initialize_torch_solution_storage(
    size: int,
    nlay: int,
    *,
    device,
    dtype,
    flux_geometry_count: int | None = None,
):
    """Allocates the standard torch forward-solver output arrays.

    Parameters
    ----------
    size
        Number of geometries or user angles in the solve.
    nlay
        Number of atmospheric layers.
    device
        Torch device receiving the result tensors.
    flux_geometry_count
        Optional number of flux columns to allocate when it differs from the
        scalar-radiance geometry count.

    Returns
    -------
    dict
        Fresh torch tensors for scalar radiances, fluxes, and level profiles.
    """
    flux_count = size if flux_geometry_count is None else flux_geometry_count
    return {
        "intensity_toa": torch.zeros((size,), dtype=dtype, device=device),
        "intensity_boa": torch.zeros((size,), dtype=dtype, device=device),
        "fluxes_toa": torch.zeros((2, flux_count), dtype=dtype, device=device),
        "fluxes_boa": torch.zeros((2, flux_count), dtype=dtype, device=device),
        "radlevel_up": torch.zeros((size, nlay + 1), dtype=dtype, device=device),
        "radlevel_dn": torch.zeros((size, nlay + 1), dtype=dtype, device=device),
    }


def _finalize_torch_solution_storage(solved, prepared):
    """Applies source-mode-specific output reshaping for torch results.

    Parameters
    ----------
    solved
        Output tensors produced by the solver.
    prepared
        Preprocessed solver inputs.

    Returns
    -------
    dict
        Output tensors with any source-mode-specific reshaping applied.
    """
    if prepared.source_mode == "solar_lat" and prepared.lattice_counts is not None:
        nbeams, nusers, nazms = prepared.lattice_counts
        stride = nusers * nazms
        indices = (
            torch.arange(nbeams, device=solved["fluxes_toa"].device, dtype=torch.long) * stride
        )
        solved["fluxes_toa"] = solved["fluxes_toa"].index_select(1, indices)
        solved["fluxes_boa"] = solved["fluxes_boa"].index_select(1, indices)
    return solved


def solve_optimized_solar_obs_torch(
    prepared,
    options,
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    d2s_scaling,
    flux_factor=None,
    albedo=None,
):
    """Solves the optimized solar two-stream forward problem on torch tensors.

    Parameters
    ----------
    prepared
        Preprocessed solar solver inputs.
    options
        Public solver options object.
    tau_arr, omega_arr, asymm_arr, d2s_scaling
        Torch tensors for the optical inputs used by the native torch path.

    Returns
    -------
    dict
        Solar radiance, flux, and optional level-profile tensors.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if prepared.source_mode not in {"solar_obs", "solar_lat"}:
        raise NotImplementedError(
            "torch-native 2S forward is currently implemented for solar_obs and solar_lat only"
        )

    geom = prepared.geometry
    dtype = tau_arr.dtype
    tau_arr = tau_arr.to(dtype=dtype)
    omega_arr = omega_arr.to(dtype=dtype)
    asymm_arr = asymm_arr.to(dtype=dtype)
    d2s_scaling = d2s_scaling.to(dtype=dtype)
    flux_factor = _as_reference_tensor(
        prepared.flux_factor if flux_factor is None else flux_factor, tau_arr
    )
    albedo = _as_reference_tensor(prepared.albedo if albedo is None else albedo, tau_arr)

    delta_tau, omega_total, asymm_total = apply_delta_scaling_torch(
        tau_arr=tau_arr,
        omega_arr=omega_arr,
        asymm_arr=asymm_arr,
        d2s_scaling=d2s_scaling,
    )
    misc = prepare_solar_misc(
        do_plane_parallel=options.do_plane_parallel,
        build_plane_parallel=lambda: qsprep_obs_pp_torch(
            delta_tau=delta_tau,
            average_secant_pp=torch.as_tensor(
                geom.average_secant_pp, dtype=dtype, device=tau_arr.device
            ),
            user_secants=torch.as_tensor(geom.user_secants, dtype=dtype, device=tau_arr.device),
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            do_postprocessing=geom.do_postprocessing,
            taylor_small=1.0e-3,
            taylor_order=3,
        ),
        build_spherical=lambda: qsprep_obs_torch(
            delta_tau=delta_tau,
            chapman_factors=torch.as_tensor(
                geom.chapman_factors, dtype=dtype, device=tau_arr.device
            ),
            user_secants=torch.as_tensor(geom.user_secants, dtype=dtype, device=tau_arr.device),
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            do_postprocessing=geom.do_postprocessing,
            taylor_small=1.0e-3,
            taylor_order=3,
        ),
    )

    ngeoms, nlay = solar_problem_size(prepared)
    solved = _initialize_torch_solution_storage(ngeoms, nlay, device=tau_arr.device, dtype=dtype)

    for fourier in range(geom.n_fouriers + 1):
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
            build_hom_solution=lambda: hom_solution_solar_torch(
                fourier=fourier,
                stream_value=prepared.stream_value,
                pxsq=geom.pxsq[fourier],
                omega=omega_total,
                asymm=asymm_total,
                delta_tau=delta_tau,
            ),
            build_user_solution=lambda xpos: hom_user_solution_solar_torch(
                fourier=fourier,
                stream_value=prepared.stream_value,
                px11=geom.px11,
                user_streams=torch.as_tensor(geom.user_streams, dtype=dtype, device=tau_arr.device),
                ulp=torch.as_tensor(geom.ulp, dtype=dtype, device=tau_arr.device),
                xpos=xpos,
                omega=omega_total,
                asymm=asymm_total,
            ),
            build_hmult=lambda eigenvalue, eigentrans: hmult_master_torch(
                taylor_order=3,
                taylor_small=1.0e-3,
                delta_tau=delta_tau,
                user_secants=torch.as_tensor(geom.user_secants, dtype=dtype, device=tau_arr.device),
                eigenvalue=eigenvalue,
                eigentrans=eigentrans,
                t_delt_userm=misc["t_delt_userm"],
            ),
        )

        do_include_surface = include_surface_term(
            fourier,
            albedo=prepared.albedo,
            do_brdf_surface=options.do_brdf_surface,
        ) or _requires_grad(albedo)
        direct_beam_terms = prepare_solar_direct_beam_terms(
            ngeoms=ngeoms,
            do_include_surface=do_include_surface,
            make_zero_vector=lambda size: torch.zeros((size,), dtype=dtype, device=tau_arr.device),
            compute_direct_beam=lambda ib: direct_beam_torch(
                do_brdf_surface=options.do_brdf_surface,
                do_surface_leaving=options.do_surface_leaving,
                do_sl_isotropic=options.do_sl_isotropic,
                fourier=fourier,
                flux_factor=flux_factor,
                x0=geom.x0[ib],
                delta_factorm=geom.delta_factor[fourier],
                albedo=albedo,
                brdf_f_0m=(
                    torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
                    if prepared.brdf is None
                    else torch.as_tensor(
                        prepared.brdf.brdf_f_0[ib, fourier],
                        dtype=dtype,
                        device=tau_arr.device,
                    )
                ),
                slterm_isotropic=(
                    torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
                    if prepared.surface_leaving is None
                    else torch.as_tensor(
                        prepared.surface_leaving.slterm_isotropic[ib],
                        dtype=dtype,
                        device=tau_arr.device,
                    )
                ),
                slterm_f_0m=(
                    torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
                    if prepared.surface_leaving is None
                    else torch.as_tensor(
                        prepared.surface_leaving.slterm_f_0[ib, fourier],
                        dtype=dtype,
                        device=tau_arr.device,
                    )
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
                build_gbeam_solution=lambda: gbeam_solution_torch(
                    do_plane_parallel=options.do_plane_parallel,
                    do_postprocessing=geom.do_postprocessing,
                    taylor_order=3,
                    taylor_small=1.0e-3,
                    delta_tau=delta_tau,
                    fourier=fourier,
                    pi4=geom.pi4,
                    flux_factor=flux_factor,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    px0xb=geom.px0x[ib, fourier],
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
                solve_bvp=lambda wupper, wlower: solve_bvp_torch(
                    do_include_surface=do_include_surface,
                    do_brdf_surface=options.do_brdf_surface,
                    do_include_surface_emission=False,
                    nlay=nlay,
                    albedo=albedo,
                    brdf_fm=(
                        torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
                        if prepared.brdf is None
                        else torch.as_tensor(
                            prepared.brdf.brdf_f[fourier], dtype=dtype, device=tau_arr.device
                        )
                    ),
                    emissivity=0.0,
                    surfbb=0.0,
                    surface_factorm=geom.surface_factor[fourier],
                    xpos=xpos,
                    eigentrans=eigentrans,
                    stream_value=prepared.stream_value,
                    direct_beam=direct_beam_terms[ib],
                    wupper=wupper,
                    wlower=wlower,
                )[:2],
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
                intensity_f_up, rad_f_up = upuser_intensity_torch(
                    do_include_surface=do_include_surface,
                    do_brdf_surface=options.do_brdf_surface,
                    do_level_output=options.do_level_output,
                    nlay=nlay,
                    taylor_order=3,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    surface_factor=geom.surface_factor[fourier],
                    albedo=albedo,
                    ubrdf_fmb=(
                        torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
                        if prepared.brdf is None
                        else torch.as_tensor(
                            prepared.brdf.ubrdf_f[ib, fourier], dtype=dtype, device=tau_arr.device
                        )
                    ),
                    fluxmult=geom.delta_factor[fourier],
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
                intensity_f_dn, rad_f_dn = dnuser_intensity_torch(
                    do_level_output=options.do_level_output,
                    nlay=nlay,
                    taylor_order=3,
                    layer_pis_cutoffb=int(misc["layer_pis_cutoff"][ib]),
                    fluxmult=geom.delta_factor[fourier],
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
                toa, boa = fluxes_solar_torch(
                    do_upwelling=options.do_upwelling,
                    do_dnwelling=options.do_dnwelling,
                    do_directbeamb=bool(do_include_surface),
                    pi4=geom.pi4,
                    stream_value=prepared.stream_value,
                    fluxfac=flux_factor,
                    fluxmult=geom.delta_factor[fourier],
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

    return _finalize_torch_solution_storage(solved, prepared)


def apply_delta_scaling_torch(
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    d2s_scaling,
) -> tuple:
    """Applies delta-M scaling on torch tensors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = tau_arr.dtype
    tau_arr = tau_arr.to(dtype=dtype)
    omega_arr = omega_arr.to(dtype=dtype)
    asymm_arr = asymm_arr.to(dtype=dtype)
    d2s_scaling = d2s_scaling.to(dtype=dtype)

    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties_torch(
        tau_arr,
        omega_arr,
        asymm_arr,
        d2s_scaling,
    )
    delta_tau = torch.where(
        delta_tau <= 0.0,
        torch.full_like(delta_tau, _OPTICAL_THICKNESS_MIN),
        delta_tau,
    )
    return delta_tau, omega_total, asymm_total


def hom_solution_solar_torch(
    *,
    fourier: int,
    stream_value: float,
    pxsq: float,
    omega,
    asymm,
    delta_tau,
) -> tuple:
    """Builds the homogeneous solar eigensystem for one Fourier mode."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = omega.dtype
    omega = omega.to(dtype=dtype)
    asymm = asymm.to(dtype=dtype)
    delta_tau = delta_tau.to(dtype=dtype)

    xinv = 1.0 / stream_value
    omega_asymm_3 = 3.0 * omega * asymm
    if fourier == 0:
        ep = omega + pxsq * omega_asymm_3
        em = omega - pxsq * omega_asymm_3
    else:
        ep = omega_asymm_3 * pxsq
        em = omega_asymm_3 * pxsq
    sab = xinv * (((ep + em) * 0.5) - 1.0)
    dab = xinv * (((ep - em) * 0.5) - 1.0)
    eigenvalue = torch.sqrt(sab * dab)
    helpv = eigenvalue * delta_tau
    eigentrans = torch.where(helpv > 88.0, torch.zeros_like(helpv), torch.exp(-helpv))
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    xpos = torch.stack((xpos1, xpos2), dim=0)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos, norm_saved


def hom_user_solution_solar_torch(
    *,
    fourier: int,
    stream_value: float,
    px11: float,
    user_streams,
    ulp,
    xpos,
    omega,
    asymm,
) -> tuple:
    """Builds solar user-angle homogeneous solutions for one Fourier mode."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = omega.dtype
    user_streams = user_streams.to(dtype=dtype)
    ulp = ulp.to(dtype=dtype)
    xpos = xpos.to(dtype=dtype)
    omega = omega.to(dtype=dtype)
    asymm = asymm.to(dtype=dtype)

    nlayers = xpos.shape[1]
    ngeoms = user_streams.shape[0]
    u_xpos = torch.zeros((ngeoms, nlayers), dtype=omega.dtype, device=omega.device)
    u_xneg = torch.zeros((ngeoms, nlayers), dtype=omega.dtype, device=omega.device)
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


def qsprep_obs_pp_torch(
    *,
    delta_tau,
    average_secant_pp,
    user_secants,
    do_upwelling: bool,
    do_dnwelling: bool,
    do_postprocessing: bool,
    taylor_small: float,
    taylor_order: int,
) -> dict:
    """Builds plane-parallel solar transmittance and multiplier inputs."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    average_secant_pp = average_secant_pp.to(dtype=dtype)
    user_secants = user_secants.to(dtype=dtype)

    nlayers = int(delta_tau.shape[0])
    ngeoms = int(average_secant_pp.shape[0])
    do_directbeam = torch.ones(ngeoms, dtype=torch.bool, device=delta_tau.device)
    layer_pis_cutoff = torch.full((ngeoms,), nlayers, dtype=torch.int64, device=delta_tau.device)
    initial_trans = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    average_secant = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    trans_solar_beam = torch.zeros((ngeoms,), dtype=delta_tau.dtype, device=delta_tau.device)
    t_delt_mubar = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    itrans_userm = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    t_delt_userm = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    sigma_p = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    sigma_m = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    emult_up = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    emult_dn = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)

    for ib in range(ngeoms):
        if do_postprocessing:
            usib = user_secants[ib]
        else:
            usib = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
        sb = average_secant_pp[ib]
        cutoff = nlayers
        taugrid = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)

        for n in range(nlayers):
            deltau = delta_tau[n]
            taugridn1 = taugrid
            taugrid = taugrid + deltau
            tauslant = taugrid * sb
            if (n + 1) <= cutoff:
                if bool(tauslant > 88.0):
                    cutoff = n + 1
                initial_n = torch.exp(-taugridn1 * sb)
                spher = deltau * sb
                wdel = torch.zeros_like(spher) if bool(spher > 88.0) else torch.exp(-spher)
            else:
                initial_n = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
                wdel = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
            initial_trans[n, ib] = initial_n
            average_secant[n, ib] = sb
            t_delt_mubar[n, ib] = wdel

            if do_postprocessing:
                itudel = initial_n * usib
                itrans_userm[n, ib] = itudel
                spher = deltau * usib
                udel = torch.zeros_like(spher) if bool(spher > 88.0) else torch.exp(-spher)
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
                        diff = torch.abs(usib - sb)
                        if bool(diff < taylor_small):
                            sd = taylor_series_1_torch(
                                taylor_order, sigma_mn, deltau, wdel, torch.ones_like(wdel)
                            )
                        else:
                            sd = (udel - wdel) / sigma_mn
                        emult_dn[n, ib] = itudel * sd

        layer_pis_cutoff[ib] = cutoff
        if bool(taugrid * sb > 88.0):
            trans_solar_beam[ib] = 0.0
            do_directbeam[ib] = False
        else:
            trans_solar_beam[ib] = torch.exp(-(taugrid * sb))

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


def qsprep_obs_torch(
    *,
    delta_tau,
    chapman_factors,
    user_secants,
    do_upwelling: bool,
    do_dnwelling: bool,
    do_postprocessing: bool,
    taylor_small: float,
    taylor_order: int,
) -> dict:
    """Builds spherical solar transmittance and multiplier inputs."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    chapman_factors = chapman_factors.to(dtype=dtype)
    user_secants = user_secants.to(dtype=dtype)

    nlayers = int(delta_tau.shape[0])
    ngeoms = int(chapman_factors.shape[2])
    do_directbeam = torch.ones(ngeoms, dtype=torch.bool, device=delta_tau.device)
    layer_pis_cutoff = torch.full((ngeoms,), nlayers, dtype=torch.int64, device=delta_tau.device)
    initial_trans = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    average_secant = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    trans_solar_beam = torch.zeros((ngeoms,), dtype=delta_tau.dtype, device=delta_tau.device)
    t_delt_mubar = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    itrans_userm = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    t_delt_userm = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    sigma_p = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    sigma_m = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    emult_up = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)
    emult_dn = torch.zeros((nlayers, ngeoms), dtype=delta_tau.dtype, device=delta_tau.device)

    for ib in range(ngeoms):
        if do_postprocessing:
            usib = user_secants[ib]
        else:
            usib = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
        s_t_0 = torch.ones((), dtype=delta_tau.dtype, device=delta_tau.device)
        s_t_1 = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
        cutoff = nlayers
        tauslant = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)

        for n in range(nlayers):
            deltau = delta_tau[n]
            tauslantn1 = tauslant
            tauslant = torch.dot(delta_tau[: n + 1], chapman_factors[: n + 1, n, ib])

            if (n + 1) <= cutoff:
                if bool(tauslant > 88.0):
                    cutoff = n + 1
                else:
                    s_t_1 = torch.exp(-tauslant)
                sb = (tauslant - tauslantn1) / deltau
                initial_n = s_t_0
                s_t_0 = s_t_1
                spher = deltau * sb
                wdel = torch.zeros_like(spher) if bool(spher > 88.0) else torch.exp(-spher)
            else:
                sb = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
                initial_n = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)
                wdel = torch.zeros((), dtype=delta_tau.dtype, device=delta_tau.device)

            average_secant[n, ib] = sb
            initial_trans[n, ib] = initial_n
            t_delt_mubar[n, ib] = wdel

            if do_postprocessing:
                itudel = initial_n * usib
                itrans_userm[n, ib] = itudel
                spher = deltau * usib
                udel = torch.zeros_like(spher) if bool(spher > 88.0) else torch.exp(-spher)
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
                        diff = torch.abs(usib - sb)
                        if bool(diff < taylor_small):
                            sd = taylor_series_1_torch(
                                taylor_order, sigma_mn, deltau, wdel, torch.ones_like(wdel)
                            )
                        else:
                            sd = (udel - wdel) / sigma_mn
                        emult_dn[n, ib] = itudel * sd

        layer_pis_cutoff[ib] = cutoff
        if bool(tauslant > 88.0):
            trans_solar_beam[ib] = 0.0
            do_directbeam[ib] = False
        else:
            trans_solar_beam[ib] = torch.exp(-tauslant)

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


def hmult_master_torch(
    *,
    taylor_order: int,
    taylor_small: float,
    delta_tau,
    user_secants,
    eigenvalue,
    eigentrans,
    t_delt_userm,
) -> tuple:
    """Builds user-angle homogeneous multipliers on torch tensors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    user_secants = user_secants.to(dtype=dtype)
    eigenvalue = eigenvalue.to(dtype=dtype)
    eigentrans = eigentrans.to(dtype=dtype)
    t_delt_userm = t_delt_userm.to(dtype=dtype)

    nlayers = int(delta_tau.shape[0])
    ngeoms = int(user_secants.shape[0])
    hmult_1 = torch.zeros((ngeoms, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    hmult_2 = torch.zeros((ngeoms, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
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
            if bool(torch.abs(zm) < taylor_small):
                hmult_1[um, n] = taylor_series_1_torch(taylor_order, zm, dn, udel, sm)
            else:
                hmult_1[um, n] = sm * (eigtn - udel) / zm
    return hmult_1, hmult_2


def gbeam_solution_torch(
    *,
    do_plane_parallel: bool,
    do_postprocessing: bool,
    taylor_order: int,
    taylor_small: float,
    delta_tau,
    fourier: int,
    pi4: float,
    flux_factor: float,
    layer_pis_cutoffb: int,
    px0xb: float,
    omega,
    asymm,
    average_secant_ppb: float,
    average_secantb,
    initial_transb,
    t_delt_mubarb,
    xpos,
    eigenvalue,
    eigentrans,
    norm_saved,
) -> tuple:
    """Builds the particular solar beam solution for one geometry."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    omega = omega.to(dtype=dtype)
    asymm = asymm.to(dtype=dtype)
    average_secantb = average_secantb.to(dtype=dtype)
    initial_transb = initial_transb.to(dtype=dtype)
    t_delt_mubarb = t_delt_mubarb.to(dtype=dtype)
    xpos = xpos.to(dtype=dtype)
    eigenvalue = eigenvalue.to(dtype=dtype)
    eigentrans = eigentrans.to(dtype=dtype)
    norm_saved = norm_saved.to(dtype=dtype)

    nlayers = int(delta_tau.shape[0])
    aterm_save = torch.zeros((nlayers,), dtype=delta_tau.dtype, device=delta_tau.device)
    bterm_save = torch.zeros((nlayers,), dtype=delta_tau.dtype, device=delta_tau.device)
    gamma_m = torch.zeros((nlayers,), dtype=delta_tau.dtype, device=delta_tau.device)
    gamma_p = torch.zeros((nlayers,), dtype=delta_tau.dtype, device=delta_tau.device)
    wupper = torch.zeros((2, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    wlower = torch.zeros((2, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    f1 = flux_factor / pi4

    for n in range(layer_pis_cutoffb):
        secbar = (
            average_secantb[n]
            if not do_plane_parallel
            else torch.as_tensor(average_secant_ppb, dtype=delta_tau.dtype, device=delta_tau.device)
        )
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
        if bool(torch.abs(gamma_mn) < taylor_small):
            cfunc = taylor_series_1_torch(
                taylor_order, gamma_mn, delta_tau[n], wdel, torch.ones_like(wdel)
            )
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


def direct_beam_torch(
    *,
    do_brdf_surface: bool,
    do_surface_leaving: bool,
    do_sl_isotropic: bool,
    fourier: int,
    flux_factor: float,
    x0: float,
    delta_factorm: float,
    albedo: float,
    brdf_f_0m,
    slterm_isotropic,
    slterm_f_0m,
    trans_solar_beam,
    do_reflected_directbeam: bool,
) -> tuple:
    """Computes the reflected direct-beam surface source in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    trans_solar_beam = trans_solar_beam.to(dtype=trans_solar_beam.dtype)
    dtype = trans_solar_beam.dtype
    device = trans_solar_beam.device
    zero = torch.zeros((), dtype=dtype, device=device)
    if not do_reflected_directbeam:
        return zero, zero

    brdf_f_0m = torch.as_tensor(brdf_f_0m, dtype=dtype, device=device)
    slterm_isotropic = torch.as_tensor(slterm_isotropic, dtype=dtype, device=device)
    slterm_f_0m = torch.as_tensor(slterm_f_0m, dtype=dtype, device=device)

    attn = flux_factor * x0 / delta_factorm / math.pi * trans_solar_beam
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


def build_bvp_matrix_torch(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    nlay: int,
    albedo: float,
    brdf_fm,
    surface_factorm: float,
    xpos,
    eigentrans,
    stream_value: float,
):
    """Builds the two-stream BVP matrix in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = xpos.dtype
    xpos = xpos.to(dtype=dtype)
    eigentrans = eigentrans.to(dtype=dtype)
    brdf_fm = torch.as_tensor(brdf_fm, dtype=xpos.dtype, device=xpos.device)

    ntotal = 2 * nlay
    mat = torch.zeros((ntotal, ntotal), dtype=xpos.dtype, device=xpos.device)
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


def solve_bvp_torch(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_include_surface_emission: bool,
    nlay: int,
    albedo: float,
    brdf_fm,
    emissivity: float,
    surfbb: float,
    surface_factorm: float,
    xpos,
    eigentrans,
    stream_value: float,
    direct_beam,
    wupper,
    wlower,
) -> tuple:
    """Solves the two-stream BVP system and returns integration constants."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = xpos.dtype
    xpos = xpos.to(dtype=dtype)
    eigentrans = eigentrans.to(dtype=dtype)
    direct_beam = torch.as_tensor(direct_beam, dtype=xpos.dtype, device=xpos.device)
    wupper = wupper.to(dtype=dtype)
    wlower = wlower.to(dtype=dtype)
    brdf_fm = torch.as_tensor(brdf_fm, dtype=xpos.dtype, device=xpos.device)

    mat = build_bvp_matrix_torch(
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
    rhs = torch.zeros((2 * nlay,), dtype=xpos.dtype, device=xpos.device)
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

    sol = torch.linalg.solve(mat, rhs)
    lcon = sol[0::2].clone()
    mcon = sol[1::2].clone()
    return lcon, mcon, mat, rhs


def upuser_intensity_torch(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_level_output: bool,
    nlay: int,
    taylor_order: int,
    layer_pis_cutoffb: int,
    surface_factor: float,
    albedo: float,
    ubrdf_fmb,
    fluxmult: float,
    stream_value: float,
    taylor_small: float,
    delta_tau,
    gamma_p,
    gamma_m,
    sigma_p,
    aterm_save,
    bterm_save,
    initial_transb,
    itrans_usermb,
    t_delt_usermb,
    t_delt_mubarb,
    t_delt_eigennl,
    lcon,
    lcon_xvec1nl,
    mcon,
    mcon_xvec1nl,
    wlower1nl,
    u_xposb,
    u_xnegb,
    hmult_1b,
    hmult_2b,
    emult_upb,
) -> tuple:
    """Computes solar upwelling user-stream intensity in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    dtype = delta_tau.dtype
    device = delta_tau.device
    gamma_p = gamma_p.to(dtype=dtype)
    gamma_m = gamma_m.to(dtype=dtype)
    sigma_p = sigma_p.to(dtype=dtype)
    aterm_save = aterm_save.to(dtype=dtype)
    bterm_save = bterm_save.to(dtype=dtype)
    initial_transb = initial_transb.to(dtype=dtype)
    itrans_usermb = itrans_usermb.to(dtype=dtype)
    t_delt_usermb = t_delt_usermb.to(dtype=dtype)
    t_delt_mubarb = t_delt_mubarb.to(dtype=dtype)
    t_delt_eigennl = torch.as_tensor(t_delt_eigennl, dtype=dtype, device=device)
    lcon = lcon.to(dtype=dtype)
    lcon_xvec1nl = torch.as_tensor(lcon_xvec1nl, dtype=dtype, device=device)
    mcon = mcon.to(dtype=dtype)
    mcon_xvec1nl = torch.as_tensor(mcon_xvec1nl, dtype=dtype, device=device)
    wlower1nl = torch.as_tensor(wlower1nl, dtype=dtype, device=device)
    u_xposb = u_xposb.to(dtype=dtype)
    u_xnegb = u_xnegb.to(dtype=dtype)
    hmult_1b = hmult_1b.to(dtype=dtype)
    hmult_2b = hmult_2b.to(dtype=dtype)
    emult_upb = emult_upb.to(dtype=dtype)
    ubrdf_fmb = torch.as_tensor(ubrdf_fmb, dtype=dtype, device=device)

    radlevel = torch.zeros((nlay + 1,), dtype=dtype, device=device)
    if do_include_surface:
        par = wlower1nl
        hom = lcon_xvec1nl * t_delt_eigennl + mcon_xvec1nl
        idownsurf = (par + hom) * stream_value
        if do_brdf_surface:
            boa_source = surface_factor * idownsurf * ubrdf_fmb
        else:
            boa_source = (
                torch.as_tensor(surface_factor * albedo, dtype=dtype, device=device) * idownsurf
            )
    else:
        boa_source = torch.zeros((), dtype=dtype, device=device)

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
            if bool(torch.abs(gammamn) < taylor_small):
                fac2 = wdel * t_delt_usermb[n]
                mult = taylor_series_2_torch(
                    taylor_order,
                    taylor_small,
                    gammamn,
                    sigma_p[n],
                    delta_tau[n],
                    torch.ones((), dtype=dtype, device=device),
                    fac2,
                    torch.ones((), dtype=dtype, device=device),
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


def fluxes_solar_torch(
    *,
    do_upwelling: bool,
    do_dnwelling: bool,
    do_directbeamb: bool,
    pi4: float,
    stream_value: float,
    fluxfac: float,
    fluxmult: float,
    x0b: float,
    trans_solar_beamb,
    lcon_xvec21,
    mcon_xvec21,
    eigentrans1,
    wupper21,
    lcon_xvec1nl,
    mcon_xvec1nl,
    eigentransnl,
    wlower1nl,
) -> tuple:
    """Computes solar mean-intensity and flux outputs in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    if hasattr(trans_solar_beamb, "dtype"):
        trans_solar_beamb = torch.as_tensor(
            trans_solar_beamb,
            dtype=trans_solar_beamb.dtype,
            device=trans_solar_beamb.device,
        )
    else:
        trans_solar_beamb = torch.as_tensor(trans_solar_beamb, dtype=torch.float64)
    dtype = trans_solar_beamb.dtype
    device = trans_solar_beamb.device
    lcon_xvec21 = torch.as_tensor(lcon_xvec21, dtype=dtype, device=device)
    mcon_xvec21 = torch.as_tensor(mcon_xvec21, dtype=dtype, device=device)
    eigentrans1 = torch.as_tensor(eigentrans1, dtype=dtype, device=device)
    wupper21 = torch.as_tensor(wupper21, dtype=dtype, device=device)
    lcon_xvec1nl = torch.as_tensor(lcon_xvec1nl, dtype=dtype, device=device)
    mcon_xvec1nl = torch.as_tensor(mcon_xvec1nl, dtype=dtype, device=device)
    eigentransnl = torch.as_tensor(eigentransnl, dtype=dtype, device=device)
    wlower1nl = torch.as_tensor(wlower1nl, dtype=dtype, device=device)
    toa = torch.zeros((2,), dtype=dtype, device=device)
    boa = torch.zeros((2,), dtype=dtype, device=device)
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
            boa[0] = boa[0] + dmean
            boa[1] = boa[1] + dflux
    return toa, boa


def thermal_setup_torch(delta_tau, thermal_bb_input):
    """Builds linear-in-optical-depth thermal source coefficients."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    thermal_bb_input = thermal_bb_input.to(dtype=dtype)
    nlayers = int(delta_tau.shape[0])
    thermcoeffs = torch.zeros((2, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    tcn1 = thermal_bb_input[0]
    for n in range(nlayers):
        tcn = thermal_bb_input[n + 1]
        thermcoeffs[0, n] = tcn1
        thermcoeffs[1, n] = (tcn - tcn1) / delta_tau[n]
        tcn1 = tcn
    return thermcoeffs


def hom_solution_thermal_torch(*, stream_value: float, pxsq: float, omega, asymm, delta_tau):
    """Computes thermal homogeneous two-stream solutions in torch."""
    return hom_solution_solar_torch(
        fourier=0,
        stream_value=stream_value,
        pxsq=pxsq,
        omega=omega,
        asymm=asymm,
        delta_tau=delta_tau,
    )


def hom_user_solution_thermal_torch(*, stream_value: float, user_streams, xpos, omega, asymm):
    """Computes thermal homogeneous user-stream solutions in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = omega.dtype
    user_streams = user_streams.to(dtype=dtype)
    xpos = xpos.to(dtype=dtype)
    omega = omega.to(dtype=dtype)
    asymm = asymm.to(dtype=dtype)

    nlayers = xpos.shape[1]
    n_users = user_streams.shape[0]
    u_xpos = torch.zeros((n_users, nlayers), dtype=omega.dtype, device=omega.device)
    u_xneg = torch.zeros((n_users, nlayers), dtype=omega.dtype, device=omega.device)
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
            u_xpos[um, n] = u_help_p0 * omegan + u_help_p1 * omega_mom * mu
            u_xneg[um, n] = u_help_m0 * omegan + u_help_m1 * omega_mom * mu
    return u_xpos, u_xneg


def thermal_gf_solution_torch(
    *,
    omega,
    delta_tau,
    thermcoeffs,
    tcutoff: float,
    eigenvalue,
    eigentrans,
    xpos,
    norm_saved,
):
    """Computes thermal Green's-function particular solutions in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    omega = omega.to(dtype=dtype)
    delta_tau = delta_tau.to(dtype=dtype)
    thermcoeffs = thermcoeffs.to(dtype=dtype)
    eigenvalue = eigenvalue.to(dtype=dtype)
    eigentrans = eigentrans.to(dtype=dtype)
    xpos = xpos.to(dtype=dtype)
    norm_saved = norm_saved.to(dtype=dtype)

    nlayers = int(delta_tau.shape[0])
    t_c_minus = torch.zeros((3, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    t_c_plus = torch.zeros((3, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    tterm_save = torch.zeros((nlayers,), dtype=delta_tau.dtype, device=delta_tau.device)
    t_wupper = torch.zeros((2, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    t_wlower = torch.zeros((2, nlayers), dtype=delta_tau.dtype, device=delta_tau.device)
    for n in range(nlayers):
        if bool(delta_tau[n] <= tcutoff):
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


def thermal_terms_torch(
    *,
    do_upwelling: bool,
    do_dnwelling: bool,
    user_streams,
    tcutoff: float,
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
    """Computes layer thermal source terms for user streams in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    user_streams = user_streams.to(dtype=dtype)
    t_delt_userm = t_delt_userm.to(dtype=dtype)
    delta_tau = delta_tau.to(dtype=dtype)
    u_xpos = u_xpos.to(dtype=dtype)
    u_xneg = u_xneg.to(dtype=dtype)
    hmult_1 = hmult_1.to(dtype=dtype)
    hmult_2 = hmult_2.to(dtype=dtype)
    t_c_plus = t_c_plus.to(dtype=dtype)
    t_c_minus = t_c_minus.to(dtype=dtype)
    tterm_save = tterm_save.to(dtype=dtype)

    n_users, nlayers = u_xpos.shape
    layer_tsup_up = torch.zeros((n_users, nlayers), dtype=u_xpos.dtype, device=u_xpos.device)
    layer_tsup_dn = torch.zeros((n_users, nlayers), dtype=u_xpos.dtype, device=u_xpos.device)
    if do_upwelling:
        for n in range(nlayers):
            if bool(delta_tau[n] <= tcutoff):
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
            if bool(delta_tau[n] <= tcutoff):
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


def upuser_intensity_thermal_torch(
    *,
    do_include_surface: bool,
    do_brdf_surface: bool,
    do_level_output: bool,
    nlay: int,
    surface_factor: float,
    albedo: float,
    ubrdf_fm,
    fluxmult: float,
    stream_value: float,
    t_delt_userm,
    t_delt_eigennl,
    lcon,
    lcon_xvec1nl,
    mcon,
    mcon_xvec1nl,
    wlower1nl,
    u_xpos,
    u_xneg,
    hmult_1,
    hmult_2,
    layer_tsup_up,
):
    """Computes thermal upwelling user-stream intensity in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    u_xpos = u_xpos.to(dtype=u_xpos.dtype)
    dtype = u_xpos.dtype
    device = u_xpos.device
    t_delt_userm = t_delt_userm.to(dtype=dtype)
    lcon = lcon.to(dtype=dtype)
    mcon = mcon.to(dtype=dtype)
    hmult_1 = hmult_1.to(dtype=dtype)
    hmult_2 = hmult_2.to(dtype=dtype)
    layer_tsup_up = layer_tsup_up.to(dtype=dtype)
    u_xneg = u_xneg.to(dtype=dtype)
    ubrdf_fm = ubrdf_fm.to(dtype=dtype)
    lcon_xvec1nl = torch.as_tensor(lcon_xvec1nl, dtype=dtype, device=device)
    mcon_xvec1nl = torch.as_tensor(mcon_xvec1nl, dtype=dtype, device=device)
    wlower1nl = torch.as_tensor(wlower1nl, dtype=dtype, device=device)
    t_delt_eigennl = torch.as_tensor(t_delt_eigennl, dtype=dtype, device=device)

    n_users = u_xpos.shape[0]
    radlevel = torch.zeros((n_users, nlay + 1), dtype=dtype, device=device)
    intensity = torch.zeros((n_users,), dtype=dtype, device=device)
    boa_source = torch.zeros((n_users,), dtype=dtype, device=device)
    if do_include_surface:
        par = wlower1nl
        hom = lcon_xvec1nl * t_delt_eigennl + mcon_xvec1nl
        idownsurf = (par + hom) * stream_value
        if do_brdf_surface:
            boa_source = surface_factor * idownsurf * ubrdf_fm
        else:
            boa_source = torch.ones_like(boa_source) * (surface_factor * albedo * idownsurf)
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


def dnuser_intensity_torch(
    *,
    do_level_output: bool,
    nlay: int,
    taylor_order: int,
    layer_pis_cutoffb: int,
    fluxmult: float,
    taylor_small: float,
    delta_tau,
    gamma_p,
    gamma_m,
    sigma_m,
    aterm_save,
    bterm_save,
    initial_transb,
    itrans_usermb,
    t_delt_usermb,
    t_delt_mubarb,
    lcon,
    mcon,
    u_xposb,
    u_xnegb,
    hmult_1b,
    hmult_2b,
    emult_dnb,
):
    """Computes solar downwelling user-stream intensity in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = delta_tau.dtype
    delta_tau = delta_tau.to(dtype=dtype)
    gamma_p = gamma_p.to(dtype=dtype)
    gamma_m = gamma_m.to(dtype=dtype)
    sigma_m = sigma_m.to(dtype=dtype)
    aterm_save = aterm_save.to(dtype=dtype)
    bterm_save = bterm_save.to(dtype=dtype)
    initial_transb = initial_transb.to(dtype=dtype)
    itrans_usermb = itrans_usermb.to(dtype=dtype)
    t_delt_usermb = t_delt_usermb.to(dtype=dtype)
    t_delt_mubarb = t_delt_mubarb.to(dtype=dtype)
    lcon = lcon.to(dtype=dtype)
    mcon = mcon.to(dtype=dtype)
    u_xposb = u_xposb.to(dtype=dtype)
    u_xnegb = u_xnegb.to(dtype=dtype)
    hmult_1b = hmult_1b.to(dtype=dtype)
    hmult_2b = hmult_2b.to(dtype=dtype)
    emult_dnb = emult_dnb.to(dtype=dtype)

    dtype = delta_tau.dtype
    device = delta_tau.device
    radlevel = torch.zeros((nlay + 1,), dtype=dtype, device=device)
    cumsource_old = torch.zeros((), dtype=dtype, device=device)
    for n in range(nlay):
        layersource = lcon[n] * u_xnegb[n] * hmult_1b[n] + mcon[n] * u_xposb[n] * hmult_2b[n]
        if (n + 1) <= layer_pis_cutoffb:
            wdel = t_delt_mubarb[n]
            itrans = initial_transb[n]
            gammamn = gamma_m[n]
            emult_dnn = emult_dnb[n]
            if bool(torch.abs(gammamn) < taylor_small):
                mult = taylor_series_2_torch(
                    taylor_order,
                    taylor_small,
                    gammamn,
                    sigma_m[n],
                    delta_tau[n],
                    t_delt_usermb[n],
                    wdel,
                    torch.ones_like(wdel),
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


def dnuser_intensity_thermal_torch(
    *,
    do_level_output: bool,
    nlay: int,
    fluxmult: float,
    t_delt_userm,
    lcon,
    mcon,
    u_xpos,
    u_xneg,
    hmult_1,
    hmult_2,
    layer_tsup_dn,
):
    """Computes thermal downwelling user-stream intensity in torch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    u_xpos = u_xpos.to(dtype=u_xpos.dtype)
    dtype = u_xpos.dtype
    device = u_xpos.device
    t_delt_userm = t_delt_userm.to(dtype=dtype)
    lcon = lcon.to(dtype=dtype)
    mcon = mcon.to(dtype=dtype)
    u_xneg = u_xneg.to(dtype=dtype)
    hmult_1 = hmult_1.to(dtype=dtype)
    hmult_2 = hmult_2.to(dtype=dtype)
    layer_tsup_dn = layer_tsup_dn.to(dtype=dtype)

    n_users = u_xpos.shape[0]
    radlevel = torch.zeros((n_users, nlay + 1), dtype=dtype, device=device)
    intensity = torch.zeros((n_users,), dtype=dtype, device=device)
    for um in range(n_users):
        cumsource_old = torch.zeros((), dtype=dtype, device=device)
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


def fluxes_thermal_torch(
    *,
    do_upwelling: bool,
    do_dnwelling: bool,
    pi4: float,
    stream_value: float,
    fluxmult: float,
    lcon_xvec21,
    mcon_xvec21,
    eigentrans1,
    wupper21,
    lcon_xvec1nl,
    mcon_xvec1nl,
    eigentransnl,
    wlower1nl,
):
    """Computes thermal mean-intensity and flux outputs in torch."""
    return fluxes_solar_torch(
        do_upwelling=do_upwelling,
        do_dnwelling=do_dnwelling,
        do_directbeamb=False,
        pi4=pi4,
        stream_value=stream_value,
        fluxfac=0.0,
        fluxmult=fluxmult,
        x0b=0.0,
        trans_solar_beamb=torch.as_tensor(0.0, dtype=lcon_xvec21.dtype, device=lcon_xvec21.device),
        lcon_xvec21=lcon_xvec21,
        mcon_xvec21=mcon_xvec21,
        eigentrans1=eigentrans1,
        wupper21=wupper21,
        lcon_xvec1nl=lcon_xvec1nl,
        mcon_xvec1nl=mcon_xvec1nl,
        eigentransnl=eigentransnl,
        wlower1nl=wlower1nl,
    )


def solve_optimized_thermal_torch(
    prepared,
    options,
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    d2s_scaling,
    thermal_bb_input=None,
    surfbb=None,
    emissivity=None,
    albedo=None,
):
    """Solves the optimized thermal two-stream forward problem on torch tensors.

    Parameters
    ----------
    prepared
        Preprocessed thermal solver inputs.
    options
        Public solver options object.
    tau_arr, omega_arr, asymm_arr, d2s_scaling
        Torch tensors for the optical inputs used by the native torch path.

    Returns
    -------
    dict
        Thermal radiance, flux, and optional level-profile tensors.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if prepared.source_mode != "thermal":
        raise NotImplementedError(
            "torch-native thermal 2S forward is implemented for thermal mode only"
        )
    geom = prepared.geometry
    thermal = prepared.thermal
    if thermal is None:
        raise ValueError("thermal inputs are required")
    dtype = tau_arr.dtype
    tau_arr = tau_arr.to(dtype=dtype)
    omega_arr = omega_arr.to(dtype=dtype)
    asymm_arr = asymm_arr.to(dtype=dtype)
    d2s_scaling = d2s_scaling.to(dtype=dtype)
    thermal_bb_input = (
        torch.as_tensor(thermal.thermal_bb_input, dtype=dtype, device=tau_arr.device)
        if thermal_bb_input is None
        else thermal_bb_input.to(dtype=dtype, device=tau_arr.device)
    )
    surfbb = _as_reference_tensor(thermal.surfbb if surfbb is None else surfbb, tau_arr)
    emissivity = _as_reference_tensor(
        thermal.emissivity if emissivity is None else emissivity, tau_arr
    )
    albedo = _as_reference_tensor(prepared.albedo if albedo is None else albedo, tau_arr)

    delta_tau, omega_total, asymm_total = apply_delta_scaling_torch(
        tau_arr=tau_arr,
        omega_arr=omega_arr,
        asymm_arr=asymm_arr,
        d2s_scaling=d2s_scaling,
    )
    thermal_coeffs = thermal_setup_torch(delta_tau, thermal_bb_input)
    n_users, nlay = thermal_problem_size(prepared)
    solved = _initialize_torch_solution_storage(
        n_users,
        nlay,
        device=tau_arr.device,
        dtype=dtype,
        flux_geometry_count=1,
    )

    eigenvalue, eigentrans, xpos, norm_saved = hom_solution_thermal_torch(
        stream_value=prepared.stream_value,
        pxsq=geom.pxsq[0],
        omega=omega_total,
        asymm=asymm_total,
        delta_tau=delta_tau,
    )
    user_streams_t = torch.as_tensor(geom.user_streams, dtype=dtype, device=tau_arr.device)
    user_secants_t = torch.as_tensor(geom.user_secants, dtype=dtype, device=tau_arr.device)
    u_xpos, u_xneg, hmult_1, hmult_2, t_delt_userm = prepare_thermal_postprocessing(
        do_postprocessing=geom.do_postprocessing,
        delta_tau=delta_tau,
        user_secants=user_secants_t,
        n_users=n_users,
        nlay=nlay,
        build_user_solution=lambda: hom_user_solution_thermal_torch(
            stream_value=prepared.stream_value,
            user_streams=user_streams_t,
            xpos=xpos,
            omega=omega_total,
            asymm=asymm_total,
        ),
        build_hmult=lambda t_delt_userm: hmult_master_torch(
            taylor_order=3,
            taylor_small=1.0e-3,
            delta_tau=delta_tau,
            user_secants=user_secants_t,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            t_delt_userm=t_delt_userm,
        ),
        make_zero_array=lambda shape: torch.zeros(shape, dtype=dtype, device=tau_arr.device),
        exp_outer=lambda left, right: torch.exp(-torch.outer(left, right)),
    )

    t_c_plus, t_c_minus, tterm_save, t_wupper, t_wlower = thermal_gf_solution_torch(
        omega=omega_total,
        delta_tau=delta_tau,
        thermcoeffs=thermal_coeffs,
        tcutoff=options.thermal_tcutoff,
        eigenvalue=eigenvalue,
        eigentrans=eigentrans,
        xpos=xpos,
        norm_saved=norm_saved,
    )
    layer_tsup_up, _layer_tsup_dn = thermal_terms_torch(
        do_upwelling=options.do_upwelling,
        do_dnwelling=options.do_dnwelling,
        user_streams=torch.as_tensor(geom.user_streams, dtype=dtype, device=tau_arr.device),
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
    ) or _requires_grad(albedo)
    do_include_surface_emission = (
        (thermal.emissivity != 0.0) or _requires_grad(surfbb) or _requires_grad(emissivity)
    )
    brdf_fm = torch.tensor(0.0, dtype=dtype, device=tau_arr.device)
    ubrdf_fm = torch.zeros((n_users,), dtype=dtype, device=tau_arr.device)
    if options.do_brdf_surface and prepared.brdf is not None:
        do_include_surface = True
        brdf_fm = torch.as_tensor(prepared.brdf.brdf_f[0], dtype=dtype, device=tau_arr.device)
        ubrdf_fm = torch.as_tensor(prepared.brdf.ubrdf_f[:, 0], dtype=dtype, device=tau_arr.device)
    lcon, mcon, _mat, _rhs = solve_bvp_torch(
        do_include_surface=do_include_surface,
        do_brdf_surface=options.do_brdf_surface,
        do_include_surface_emission=do_include_surface_emission,
        nlay=nlay,
        albedo=albedo,
        brdf_fm=brdf_fm,
        emissivity=emissivity,
        surfbb=surfbb,
        surface_factorm=geom.surface_factor[0],
        xpos=xpos,
        eigentrans=eigentrans,
        stream_value=prepared.stream_value,
        direct_beam=torch.tensor(0.0, dtype=dtype, device=tau_arr.device),
        wupper=t_wupper,
        wlower=t_wlower,
    )
    boundary_terms = prepare_thermal_boundary_terms(
        lcon=lcon,
        mcon=mcon,
        xpos=xpos,
        eigentrans=eigentrans,
        wlower=t_wlower,
    )
    if options.do_upwelling and geom.do_postprocessing:
        solved["intensity_toa"], solved["radlevel_up"] = upuser_intensity_thermal_torch(
            do_include_surface=do_include_surface,
            do_brdf_surface=options.do_brdf_surface,
            do_level_output=options.do_level_output,
            nlay=nlay,
            surface_factor=geom.surface_factor[0],
            albedo=albedo,
            ubrdf_fm=ubrdf_fm,
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
        solved["intensity_boa"], solved["radlevel_dn"] = dnuser_intensity_thermal_torch(
            do_level_output=options.do_level_output,
            nlay=nlay,
            fluxmult=geom.delta_factor[0],
            t_delt_userm=t_delt_userm,
            lcon=lcon,
            mcon=mcon,
            u_xpos=u_xpos,
            u_xneg=u_xneg,
            hmult_1=hmult_1,
            hmult_2=hmult_2,
            layer_tsup_dn=_layer_tsup_dn,
        )
    if geom.do_include_mvout[0]:
        toa, boa = fluxes_thermal_torch(
            do_upwelling=options.do_upwelling,
            do_dnwelling=options.do_dnwelling,
            pi4=geom.pi4,
            stream_value=prepared.stream_value,
            fluxmult=geom.delta_factor[0],
            lcon_xvec21=lcon[0] * xpos[1, 0],
            mcon_xvec21=mcon[0] * xpos[0, 0],
            eigentrans1=eigentrans[0],
            wupper21=t_wupper[1, 0],
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

    return solved
