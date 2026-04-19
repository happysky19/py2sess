"""Batched torch helpers for thermal observation-geometry cases."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .backend import _load_torch
from .bvp_batch_torch import (
    _canonical_torch_bvp_engine,
    default_auto_bvp_context_torch,
    repair_nonfinite_bvp_rows_torch,
    solve_thermal_bvp_batch_torch,
    solve_thermal_block_bvp_batch_torch,
    solve_thermal_dense_bvp_batch_torch,
)
from .fo_solar_obs import _fo_eps_geometry
from .optical_torch import delta_m_scale_optical_properties_torch
from .taylor_torch import taylor_series_1_torch

torch = _load_torch()


@dataclass(frozen=True)
class ThermalBatchTorchResult:
    """Batched thermal endpoint radiances."""

    two_stream_toa: object
    fo_total_up_toa: object

    @property
    def total_toa(self):
        """Returns 2S plus FO upwelling TOA radiance."""
        return self.two_stream_toa + self.fo_total_up_toa


def _as_tensor(value, *, dtype, device):
    """Converts ``value`` to a torch tensor on the requested context."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        # Solver inputs are read-only. Sharing mmap-backed CPU arrays avoids
        # copying full-spectrum cache slices before every torch batch call.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            return torch.as_tensor(value, dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _hom_solution_thermal_batch(*, stream_value: float, pxsq: float, omega, asymm, delta_tau):
    """Builds the thermal homogeneous solution for a wavelength batch."""
    xinv = 1.0 / stream_value
    omega_asymm_3 = 3.0 * omega * asymm
    sab = xinv * (omega - 1.0)
    dab = xinv * (pxsq * omega_asymm_3 - 1.0)
    eigenvalue = torch.sqrt(sab * dab)
    helpv = eigenvalue * delta_tau
    eigentrans = torch.where(helpv > 88.0, torch.zeros_like(helpv), torch.exp(-helpv))
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos1, xpos2, norm_saved


def _thermal_coefficients_batch(delta_tau, thermal_bb_input):
    """Builds linear thermal-source coefficients for a wavelength batch."""
    lower = thermal_bb_input[:, :-1]
    upper = thermal_bb_input[:, 1:]
    return lower, (upper - lower) / delta_tau


def _thermal_green_function_batch(
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
    zero = torch.zeros_like(delta_tau)
    tterm = (1.0 - omega) * (xpos1 + xpos2) / norm_saved
    k1 = 1.0 / eigenvalue
    tcm2 = k1 * therm1
    tcp2 = tcm2
    tcm1 = k1 * (therm0 - tcm2)
    tcp1 = k1 * (therm0 + tcp2)
    sum_p = tcp1 + tcp2 * delta_tau
    tcm0 = -tcm1
    tcp0 = -sum_p
    # These Green-function combinations can subtract two O(1e4) terms to
    # produce an O(1e-9) result in optically thin layers. Rewrite them in
    # terms of ``1 - exp(-k*dt)`` to avoid catastrophic cancellation in float32.
    one_minus_eigentrans = -torch.expm1(-eigenvalue * delta_tau)
    t_gmult_dn = tterm * (tcm1 * one_minus_eigentrans + tcm2 * delta_tau)
    t_gmult_up = tterm * (tcp1 * one_minus_eigentrans - eigentrans * tcp2 * delta_tau)
    t_wupper0 = torch.where(active, t_gmult_up * xpos2, zero)
    t_wupper1 = torch.where(active, t_gmult_up * xpos1, zero)
    t_wlower0 = torch.where(active, t_gmult_dn * xpos1, zero)
    t_wlower1 = torch.where(active, t_gmult_dn * xpos2, zero)
    tterm_save = torch.where(active, tterm, zero)
    return (
        (tcp0, tcp1, tcp2),
        (tcm0, tcm1, tcm2),
        tterm_save,
        (t_wupper0, t_wupper1),
        (t_wlower0, t_wlower1),
    )


def _homogeneous_multipliers_batch(
    *, delta_tau, user_secant: float, eigenvalue, eigentrans, t_delt_userm
):
    """Builds thermal homogeneous multipliers for one user stream."""
    zp = user_secant + eigenvalue
    zm = user_secant - eigenvalue
    hmult_2 = user_secant * (1.0 - eigentrans * t_delt_userm) / zp
    # ``eigentrans - t_delt_userm`` can lose all significant digits in float32
    # for optically thin layers, so compute it through ``expm1`` instead.
    regular = user_secant * t_delt_userm * torch.expm1(zm * delta_tau) / zm
    taylor = taylor_series_1_torch(3, zm, delta_tau, t_delt_userm, user_secant)
    hmult_1 = torch.where(torch.abs(zm) < 1.0e-3, taylor, regular)
    return hmult_1, hmult_2


def _thermal_user_solution_batch(
    *, stream_value: float, user_stream: float, xpos1, xpos2, omega, asymm
):
    """Builds user-angle homogeneous solutions for one thermal user angle."""
    hmu_stream = 0.5 * stream_value
    u_help_p0 = (xpos2 + xpos1) * 0.5
    u_help_p1 = (xpos2 - xpos1) * hmu_stream
    omega_mom = 3.0 * omega * asymm
    u_xpos = u_help_p0 * omega + u_help_p1 * omega_mom * user_stream
    u_xneg = u_help_p0 * omega - u_help_p1 * omega_mom * user_stream
    return u_xpos, u_xneg


def _thermal_layer_sources_up_batch(
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
    # Factor these source combinations to avoid subtracting nearly identical
    # ``1`` and ``exp(-dt/mu)`` terms in float32.
    one_minus_t_delt_userm = -torch.expm1(-delta_tau / user_stream)
    su = tcp0 * hmult_1 + tsgm_uu1 * one_minus_t_delt_userm - tcp2 * delta_tau * t_delt_userm
    sd = tcm0 * hmult_2 + tsgm_ud1 * one_minus_t_delt_userm - tcm2 * delta_tau * t_delt_userm
    source = tterm_save * (u_xpos * sd + u_xneg * su)
    return torch.where(delta_tau > tcutoff, source, torch.zeros_like(source))


def _two_stream_thermal_toa_batch(
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
    pxsq,
    thermal_tcutoff,
    bvp_device=None,
    bvp_dtype=None,
    bvp_engine: str = "auto",
):
    """Computes batched 2S thermal upwelling TOA radiance."""
    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties_torch(
        tau,
        omega,
        asymm,
        scaling,
    )
    therm0, therm1 = _thermal_coefficients_batch(delta_tau, thermal_bb_input)
    eigenvalue, eigentrans, xpos1, xpos2, norm_saved = _hom_solution_thermal_batch(
        stream_value=stream_value,
        pxsq=pxsq,
        omega=omega_total,
        asymm=asymm_total,
        delta_tau=delta_tau,
    )
    user_secant = 1.0 / user_stream
    t_delt_userm = torch.exp(-delta_tau * user_secant)
    u_xpos, u_xneg = _thermal_user_solution_batch(
        stream_value=stream_value,
        user_stream=user_stream,
        xpos1=xpos1,
        xpos2=xpos2,
        omega=omega_total,
        asymm=asymm_total,
    )
    hmult_1, hmult_2 = _homogeneous_multipliers_batch(
        delta_tau=delta_tau,
        user_secant=user_secant,
        eigenvalue=eigenvalue,
        eigentrans=eigentrans,
        t_delt_userm=t_delt_userm,
    )
    t_c_plus, t_c_minus, tterm_save, t_wupper, t_wlower = _thermal_green_function_batch(
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
    layer_tsup_up = _thermal_layer_sources_up_batch(
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
    bvp_engine = _canonical_torch_bvp_engine(bvp_engine)
    if bvp_engine == "pentadiagonal":
        solve_bvp = solve_thermal_bvp_batch_torch
    elif bvp_engine == "block":
        solve_bvp = solve_thermal_block_bvp_batch_torch
    elif bvp_engine == "dense":
        solve_bvp = solve_thermal_dense_bvp_batch_torch
    else:
        solve_bvp = (
            solve_thermal_bvp_batch_torch
            if delta_tau.dtype == torch.float64 or delta_tau.device.type == "mps"
            else solve_thermal_dense_bvp_batch_torch
        )
    surface_factor = 2.0
    lcon, mcon = solve_bvp(
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
        solve_device=bvp_device,
        solve_dtype=bvp_dtype,
    )
    if solve_bvp in {solve_thermal_bvp_batch_torch, solve_thermal_block_bvp_batch_torch}:
        lcon, mcon = repair_nonfinite_bvp_rows_torch(
            lcon=lcon,
            mcon=mcon,
            bad_row_solver=solve_thermal_dense_bvp_batch_torch,
            solver_kwargs={
                "albedo": albedo,
                "emissivity": emissivity,
                "surfbb": surfbb,
                "surface_factor": surface_factor,
                "stream_value": stream_value,
                "xpos1": xpos1,
                "xpos2": xpos2,
                "eigentrans": eigentrans,
                "wupper": t_wupper,
                "wlower": t_wlower,
            },
            solve_device=bvp_device,
            solve_dtype=bvp_dtype,
        )
    wlower0, _wlower1 = t_wlower
    idownsurf = (
        wlower0[:, -1] + lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
    ) * stream_value
    boa_source = surface_factor * albedo * idownsurf
    layersource = lcon * u_xpos * hmult_2 + mcon * u_xneg * hmult_1 + layer_tsup_up
    trans_prefix = torch.cumprod(t_delt_userm, dim=1)
    layer_weights = torch.cat((torch.ones_like(trans_prefix[:, :1]), trans_prefix[:, :-1]), dim=1)
    return torch.sum(layersource * layer_weights, dim=1) + trans_prefix[:, -1] * boa_source


def _fo_thermal_toa_batch(
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
    fo_geometry=None,
):
    """Computes batched FO thermal upwelling TOA radiance."""
    dtype = tau.dtype
    device = tau.device
    deltaus = tau * (1.0 - omega * scaling)
    lower_bb = thermal_bb_input[:, :-1]
    upper_bb = thermal_bb_input[:, 1:]
    single_scatter_scale = 1.0 - omega
    therm0 = lower_bb * single_scatter_scale
    therm1 = ((upper_bb - lower_bb) / deltaus) * single_scatter_scale
    height_t = _as_tensor(heights, dtype=dtype, device=device)
    extinction = deltaus / (height_t[:-1] - height_t[1:])
    geometry = fo_geometry
    if geometry is None:
        geometry = _fo_eps_geometry(
            user_obsgeoms=np.array([[0.0, user_angle_degrees, 0.0]], dtype=float),
            height_grid=np.asarray(heights, dtype=float),
            earth_radius=earth_radius,
            nfine=nfine,
            vsign=1.0,
        )
    do_nadir = bool(geometry["do_nadir"][0])
    if do_nadir:
        xfine = _as_tensor(geometry["xfine"][:, :, 0], dtype=dtype, device=device)
        wfine = _as_tensor(geometry["wfine"][:, :, 0], dtype=dtype, device=device)
        lostrans = torch.where(deltaus < 88.0, torch.exp(-deltaus), torch.zeros_like(deltaus))
        xjkn = xfine.unsqueeze(0) * extinction.unsqueeze(1)
        solution = therm0.unsqueeze(1) + xjkn * therm1.unsqueeze(1)
        sources_up = torch.sum(
            solution * extinction.unsqueeze(1) * torch.exp(-xjkn) * wfine.unsqueeze(0), dim=1
        )
    else:
        rayconv = _as_tensor(geometry["raycon"][0], dtype=dtype, device=device)
        cota = _as_tensor(geometry["cota"][:, 0], dtype=dtype, device=device)
        cot_upper = cota[:-1]
        cot_lower = cota[1:]
        ke = rayconv * extinction
        lostau = ke * (cot_upper - cot_lower)
        lostrans = torch.where(lostau < 88.0, torch.exp(-lostau), torch.zeros_like(lostau))
        xfine = _as_tensor(geometry["xfine"][:, :, 0], dtype=dtype, device=device)
        wfine = _as_tensor(geometry["wfine"][:, :, 0], dtype=dtype, device=device)
        cotfine = _as_tensor(geometry["cotfine"][:, :, 0], dtype=dtype, device=device)
        csqfine = _as_tensor(geometry["csqfine"][:, :, 0], dtype=dtype, device=device)
        xjkn = xfine.unsqueeze(0) * extinction.unsqueeze(1)
        solution = therm0.unsqueeze(1) + xjkn * therm1.unsqueeze(1)
        weight = ke.unsqueeze(1) * csqfine.unsqueeze(0) * wfine.unsqueeze(0)
        optical_path = ke.unsqueeze(1) * (
            cot_upper.unsqueeze(0).unsqueeze(0) - cotfine.unsqueeze(0)
        )
        sources_up = torch.sum(
            solution * weight * torch.exp(-optical_path),
            dim=1,
        )
    trans_prefix = torch.cumprod(lostrans, dim=1)
    layer_weights = torch.cat((torch.ones_like(trans_prefix[:, :1]), trans_prefix[:, :-1]), dim=1)
    cum_atmos = torch.sum(sources_up * layer_weights, dim=1)
    cum_surface = trans_prefix[:, -1] * surfbb * emissivity
    return cum_atmos + cum_surface


def solve_thermal_batch_torch(
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
    dtype=None,
    device=None,
    bvp_device=None,
    bvp_dtype=None,
    bvp_engine: str = "auto",
    fo_geometry=None,
) -> ThermalBatchTorchResult:
    """Solves thermal observation-geometry spectra with torch tensors.

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
    fo_geometry
        Optional precomputed FO geometry. Use this when repeatedly solving
        wavelength chunks with the same geometry.
    dtype, device
        Optional torch dtype and device for converted inputs.
    bvp_device, bvp_dtype
        Optional context for the dense BVP fallback. This is useful for Apple
        MPS development runs, where ``bvp_engine="auto"`` defaults to a CPU
        float64 BVP solve while keeping the outer tensors on MPS.
    bvp_engine
        ``"auto"`` keeps the supported parity-oriented default on CPU while
        routing the current MPS development path through a CPU float64 BVP
        solve. ``"block"`` enables the opt-in CPU block-tridiagonal solver.
        ``"pentadiagonal"`` forces the experimental recurrence and is useful
        for lower precision speed studies.

    Returns
    -------
    ThermalBatchTorchResult
        Batched 2S thermal, FO thermal, and total TOA radiances.

    Notes
    -----
    This helper is narrower than the public scalar API: it covers repeated
    thermal spectra with one thermal user angle, Lambertian surface handling,
    and no BRDF.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    device = torch.device("cpu") if device is None else torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "torch device 'mps' is not available in this process. On macOS this "
            "usually means the current sandbox cannot access Metal/MPS; rerun the "
            "benchmark outside the sandbox/escalated environment, or use the "
            "supported CPU path with device='cpu'."
        )
    dtype = (
        torch.float32
        if dtype is None and device.type == "mps"
        else (torch.float64 if dtype is None else dtype)
    )
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError(
            "torch device 'mps' requires float32 or lower precision; MPS does not support float64"
        )
    bvp_engine = _canonical_torch_bvp_engine(bvp_engine)
    if bvp_engine == "auto":
        bvp_device, bvp_dtype = default_auto_bvp_context_torch(
            device=device,
            bvp_device=bvp_device,
            bvp_dtype=bvp_dtype,
        )
    else:
        bvp_device = None if bvp_device is None else torch.device(bvp_device)
    bvp_dtype = dtype if bvp_dtype is None else bvp_dtype
    tau = _as_tensor(tau_arr, dtype=dtype, device=device)
    omega = _as_tensor(omega_arr, dtype=dtype, device=device)
    asymm = _as_tensor(asymm_arr, dtype=dtype, device=device)
    scaling = _as_tensor(d2s_scaling, dtype=dtype, device=device)
    bb = _as_tensor(thermal_bb_input, dtype=dtype, device=device)
    surfbb_t = _as_tensor(surfbb, dtype=dtype, device=device)
    albedo_t = _as_tensor(albedo, dtype=dtype, device=device)
    emissivity_t = (
        1.0 - albedo_t if emissivity is None else _as_tensor(emissivity, dtype=dtype, device=device)
    )
    user_stream = float(np.cos(np.deg2rad(user_angle_degrees)))
    two_stream = _two_stream_thermal_toa_batch(
        tau=tau,
        omega=omega,
        asymm=asymm,
        scaling=scaling,
        thermal_bb_input=bb,
        surfbb=surfbb_t,
        emissivity=emissivity_t,
        albedo=albedo_t,
        stream_value=stream_value,
        user_stream=user_stream,
        pxsq=stream_value * stream_value,
        thermal_tcutoff=thermal_tcutoff,
        bvp_device=bvp_device,
        bvp_dtype=bvp_dtype,
        bvp_engine=bvp_engine,
    )
    fo = _fo_thermal_toa_batch(
        tau=tau,
        omega=omega,
        scaling=scaling,
        thermal_bb_input=bb,
        surfbb=surfbb_t,
        emissivity=emissivity_t,
        heights=heights,
        user_angle_degrees=user_angle_degrees,
        earth_radius=earth_radius,
        nfine=nfine,
        fo_geometry=fo_geometry,
    )
    return ThermalBatchTorchResult(two_stream_toa=two_stream, fo_total_up_toa=fo)
