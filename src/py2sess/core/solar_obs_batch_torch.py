"""Batched torch helpers for solar observation-geometry cases."""

from __future__ import annotations

import numpy as np
import warnings

from .backend import _load_torch
from .bvp_batch_torch import (
    _canonical_torch_bvp_engine,
    default_auto_bvp_context_torch,
    repair_nonfinite_bvp_rows_torch,
    solve_solar_observation_bvp_batch_torch,
    solve_solar_observation_block_bvp_batch_torch,
    solve_solar_observation_dense_bvp_batch_torch,
)
from .optical_torch import delta_m_scale_optical_properties_torch
from .taylor_torch import taylor_series_1_torch

torch = _load_torch()

MAX_TAU_PATH = 88.0
MAX_TAU_QPATH = 88.0
TAYLOR_SMALL = 1.0e-3
TAYLOR_ORDER = 3


def _exp_cutoff_torch(values, cutoff: float):
    """Returns ``exp(-values)`` with the Fortran optical-depth cutoff."""
    result = torch.exp(-values)
    return torch.where(values > cutoff, torch.zeros_like(result), result)


def _as_tensor(value, *, dtype, device):
    """Converts ``value`` to a torch tensor on the requested context."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        # Solver inputs are read-only. Sharing mmap-backed CPU arrays avoids
        # copying full-spectrum cache slices before every torch batch call.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            return torch.as_tensor(value, dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _taylor_series_2_vectorized_torch(
    order: int, small: float, eps, y, delta, fac1, fac2, sm: float
):
    """Evaluates the second Fortran Taylor multiplier over tensor masks."""
    result = torch.empty_like(eps)
    near = torch.abs(y) < small
    far = ~near

    mterms = order + 2
    eps_near = eps[near]
    y_near = y[near]
    delta_near = delta[near]
    d = [torch.ones_like(eps_near)]
    for m in range(1, mterms + 1):
        d.append(delta_near * d[-1] / float(m))
    power = torch.ones_like(eps_near)
    power2 = torch.ones_like(eps_near)
    mult = d[2].clone()
    for m in range(3, mterms + 1):
        power = power * (eps_near - y_near)
        power2 = power - y_near * power2
        mult = mult + d[m] * power2
    result[near] = mult * fac1[near] * sm

    mterms = order + 1
    eps_far = eps[far]
    y1 = 1.0 / y[far]
    delta_far = delta[far]
    d = [torch.ones_like(eps_far)]
    for m in range(1, mterms + 1):
        d.append(delta_far * d[-1] / float(m))
    ac = [torch.ones_like(eps_far)]
    for _m in range(1, mterms + 1):
        ac.append(y1 * ac[-1])
    cc = [torch.ones_like(eps_far)]
    for m in range(1, mterms + 1):
        total = torch.zeros_like(eps_far)
        for j in range(m + 1):
            total = total + ac[j] * d[m - j]
        cc.append(total)
    term_1 = [fac1[far] * ac[m] - fac2[far] * cc[m] for m in range(mterms + 1)]
    power = torch.ones_like(eps_far)
    mult = term_1[1].clone()
    for m in range(2, mterms + 1):
        power = eps_far * power
        mult = mult + power * term_1[m]
    result[far] = mult * sm * y1

    return result


def _qsprep_obs_batch_torch(delta_tau, chapman, user_secant: float):
    """Builds spherical solar transmittance and multiplier inputs in batch."""
    batch, nlayers = delta_tau.shape
    dtype = delta_tau.dtype
    device = delta_tau.device
    tauslant_all = delta_tau @ chapman
    tauslant_previous = torch.empty_like(tauslant_all)
    tauslant_previous[:, 0] = 0.0
    tauslant_previous[:, 1:] = tauslant_all[:, :-1]
    delta_tauslant = tauslant_all - tauslant_previous
    too_deep = tauslant_all > MAX_TAU_PATH
    if not bool(torch.any(too_deep)):
        average_secant = delta_tauslant / delta_tau
        initial_trans = torch.exp(-tauslant_previous)
        t_delt_mubar = torch.exp(-delta_tauslant)
        itrans_userm = initial_trans * user_secant
        trans_solar_beam = torch.exp(-tauslant_all[:, -1])
        user_spher = delta_tau * user_secant
        t_delt_userm = _exp_cutoff_torch(user_spher, MAX_TAU_PATH)
        sigma_p = average_secant + user_secant
        emult_up = itrans_userm * (1.0 - t_delt_mubar * t_delt_userm) / sigma_p
        return {
            "layer_pis_cutoff": torch.full((batch,), nlayers, dtype=torch.int64, device=device),
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
    has_too_deep = torch.any(too_deep, dim=1)
    first_too_deep = torch.argmax(too_deep.to(torch.int64), dim=1) + 1
    cutoff = torch.where(
        has_too_deep,
        first_too_deep,
        torch.full((batch,), nlayers, dtype=torch.int64, device=device),
    )
    active = torch.arange(1, nlayers + 1, device=device).unsqueeze(0) <= cutoff.unsqueeze(1)
    zero = torch.zeros((batch, nlayers), dtype=dtype, device=device)
    initial_trans_raw = torch.exp(-tauslant_previous)
    initial_trans = torch.where(active, initial_trans_raw, zero)
    average_secant_raw = delta_tauslant / delta_tau
    average_secant = torch.where(active, average_secant_raw, zero)
    t_delt_mubar = torch.where(
        active & (delta_tauslant <= MAX_TAU_PATH), torch.exp(-delta_tauslant), zero
    )
    itrans_userm = initial_trans * user_secant
    trans_solar_beam = _exp_cutoff_torch(tauslant_all[:, -1], MAX_TAU_PATH)
    user_spher = delta_tau * user_secant
    t_delt_userm = _exp_cutoff_torch(user_spher, MAX_TAU_PATH)
    sigma_p_raw = average_secant_raw + user_secant
    sigma_p = torch.where(active, sigma_p_raw, zero)
    emult_up = torch.where(
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


def _hom_solution_solar_batch_torch(
    *,
    fourier: int,
    stream_value: float,
    pxsq: float,
    omega,
    omega_asymm_3,
    delta_tau,
):
    """Builds the solar homogeneous eigensystem for a wavelength batch."""
    xinv = 1.0 / stream_value
    if fourier == 0:
        sab = xinv * (omega - 1.0)
        dab = xinv * (pxsq * omega_asymm_3 - 1.0)
    else:
        sab = xinv * (pxsq * omega_asymm_3 - 1.0)
        dab = -xinv
    eigenvalue = torch.sqrt(sab * dab)
    helpv = eigenvalue * delta_tau
    eigentrans = _exp_cutoff_torch(helpv, MAX_TAU_QPATH)
    difvec = -sab / eigenvalue
    xpos1 = 0.5 * (1.0 + difvec)
    xpos2 = 0.5 * (1.0 - difvec)
    norm_saved = stream_value * (xpos1 * xpos1 - xpos2 * xpos2)
    return eigenvalue, eigentrans, xpos1, xpos2, norm_saved


def _hom_user_solution_solar_batch_torch(
    *,
    fourier: int,
    stream_value: float,
    px11: float,
    user_stream: float,
    ulp: float,
    xpos1,
    xpos2,
    omega,
    omega_asymm_3,
):
    """Builds solar user-angle homogeneous solutions for one observation geometry."""
    hmu_stream = 0.5 * stream_value
    if fourier == 0:
        u_help_p0 = (xpos2 + xpos1) * 0.5
        u_help_p1 = (xpos2 - xpos1) * hmu_stream
        u_xpos = u_help_p0 * omega + u_help_p1 * omega_asymm_3 * user_stream
        u_xneg = u_help_p0 * omega - u_help_p1 * omega_asymm_3 * user_stream
    else:
        u_help_p1 = -(xpos2 + xpos1) * px11 * 0.5
        u_xpos = u_help_p1 * omega_asymm_3 * ulp
        u_xneg = u_xpos
    return u_xpos, u_xneg


def _hmult_master_batch_torch(
    *, delta_tau, user_secant: float, eigenvalue, eigentrans, t_delt_userm
):
    """Builds user-angle homogeneous multipliers for a wavelength batch."""
    zp = user_secant + eigenvalue
    zm = user_secant - eigenvalue
    zudel = eigentrans * t_delt_userm
    hmult_2 = user_secant * (1.0 - zudel) / zp
    hmult_1 = user_secant * (eigentrans - t_delt_userm) / zm
    near = torch.abs(zm) < TAYLOR_SMALL
    return torch.where(
        near,
        taylor_series_1_torch(TAYLOR_ORDER, zm, delta_tau, t_delt_userm, user_secant),
        hmult_1,
    ), hmult_2


def _gbeam_solution_batch_torch(
    *,
    fourier: int,
    pi4: float,
    flux_factor,
    layer_pis_cutoff,
    px0x: float,
    omega,
    omega_asymm_3=None,
    asymm=None,
    average_secant,
    initial_trans,
    t_delt_mubar,
    xpos1,
    xpos2,
    eigenvalue,
    eigentrans,
    norm_saved,
    delta_tau,
    all_layers_active: bool = False,
):
    """Builds solar Green-function beam terms for a wavelength batch."""
    batch, nlayers = delta_tau.shape
    if omega_asymm_3 is None:
        if asymm is None:
            raise TypeError("provide either omega_asymm_3 or asymm")
        omega_asymm_3 = 3.0 * omega * asymm
    dtype = delta_tau.dtype
    device = delta_tau.device
    f1 = flux_factor / pi4
    gamma_p_raw = average_secant + eigenvalue
    gamma_m_raw = average_secant - eigenvalue
    if all_layers_active:
        gamma_p = gamma_p_raw
        gamma_m = gamma_m_raw
        zdel = eigentrans
        wdel = t_delt_mubar
        zwdel = zdel * wdel
        cfunc = (zdel - wdel) / gamma_m_raw
        near = torch.abs(gamma_m_raw) < TAYLOR_SMALL
        cfunc = torch.where(
            near,
            taylor_series_1_torch(TAYLOR_ORDER, gamma_m_raw, delta_tau, wdel, 1.0),
            cfunc,
        )
        dfunc = (1.0 - zwdel) / gamma_p_raw
        if fourier == 0:
            tp = omega + px0x * omega_asymm_3
            tm = omega - px0x * omega_asymm_3
        else:
            tp = px0x * omega_asymm_3
            tm = tp
        dpin = tp * f1[:, None]
        dmin = tm * f1[:, None]
        sum_la = dpin * xpos1 + dmin * xpos2
        sum_lb = dmin * xpos1 + dpin * xpos2
        aterm = sum_la / norm_saved
        bterm = sum_lb / norm_saved
        gfunc_dn = cfunc * aterm * initial_trans
        gfunc_up = dfunc * bterm * initial_trans
        wupper0 = gfunc_up * xpos2
        wupper1 = gfunc_up * xpos1
        wlower0 = gfunc_dn * xpos1
        wlower1 = gfunc_dn * xpos2
        return gamma_m, gamma_p, aterm, bterm, (wupper0, wupper1), (wlower0, wlower1)

    active = torch.arange(1, nlayers + 1, device=device).unsqueeze(0) <= layer_pis_cutoff.unsqueeze(
        1
    )
    zero = torch.zeros((batch, nlayers), dtype=dtype, device=device)
    gamma_p = torch.where(active, gamma_p_raw, zero)
    gamma_m = torch.where(active, gamma_m_raw, zero)

    zdel = eigentrans
    wdel = t_delt_mubar
    zwdel = zdel * wdel
    cfunc = (zdel - wdel) / gamma_m_raw
    near = active & (torch.abs(gamma_m_raw) < TAYLOR_SMALL)
    cfunc = torch.where(
        near,
        taylor_series_1_torch(TAYLOR_ORDER, gamma_m_raw, delta_tau, wdel, 1.0),
        cfunc,
    )
    cfunc = torch.where(active, cfunc, zero)
    dfunc = torch.where(active, (1.0 - zwdel) / gamma_p_raw, zero)

    if fourier == 0:
        tp = omega + px0x * omega_asymm_3
        tm = omega - px0x * omega_asymm_3
    else:
        tp = px0x * omega_asymm_3
        tm = tp
    dpin = tp * f1[:, None]
    dmin = tm * f1[:, None]
    sum_la = dpin * xpos1 + dmin * xpos2
    sum_lb = dmin * xpos1 + dpin * xpos2
    aterm_raw = sum_la / norm_saved
    bterm_raw = sum_lb / norm_saved
    aterm = torch.where(active, aterm_raw, zero)
    bterm = torch.where(active, bterm_raw, zero)

    gfunc_dn = cfunc * aterm * initial_trans
    gfunc_up = dfunc * bterm * initial_trans
    wupper0 = gfunc_up * xpos2
    wupper1 = gfunc_up * xpos1
    wlower0 = gfunc_dn * xpos1
    wlower1 = gfunc_dn * xpos2
    return gamma_m, gamma_p, aterm, bterm, (wupper0, wupper1), (wlower0, wlower1)


def _upuser_intensity_batch_torch(
    *,
    layer_pis_cutoff,
    surface_factor: float,
    albedo,
    fluxmult: float,
    stream_value: float,
    delta_tau,
    gamma_p,
    gamma_m,
    sigma_p,
    aterm,
    bterm,
    initial_trans,
    itrans_userm,
    t_delt_userm,
    t_delt_mubar,
    eigentrans,
    lcon,
    mcon,
    wlower1,
    xpos1,
    xpos2,
    u_xpos,
    u_xneg,
    hmult_1,
    hmult_2,
    emult_up,
    all_layers_active: bool = False,
    return_profile: bool = False,
):
    """Computes upwelling TOA user intensity for a wavelength batch."""
    nlay = delta_tau.shape[1]
    par = wlower1[:, -1]
    hom = lcon[:, -1] * xpos1[:, -1] * eigentrans[:, -1] + mcon[:, -1] * xpos2[:, -1]
    idownsurf = (par + hom) * stream_value
    cumsource = surface_factor * albedo * idownsurf
    layersource = lcon * u_xpos * hmult_2 + mcon * u_xneg * hmult_1
    fac2 = t_delt_mubar * t_delt_userm
    if all_layers_active:
        sd = (initial_trans * hmult_2 - emult_up) / gamma_m
        su = (-initial_trans * t_delt_mubar * hmult_1 + emult_up) / gamma_p
        near = torch.abs(gamma_m) < TAYLOR_SMALL
        if bool(torch.any(near)):
            taylor = _taylor_series_2_vectorized_torch(
                TAYLOR_ORDER,
                TAYLOR_SMALL,
                gamma_m[near],
                sigma_p[near],
                delta_tau[near],
                torch.ones_like(fac2[near]),
                fac2[near],
                1.0,
            )
            sd = sd.clone()
            sd[near] = itrans_userm[near] * taylor
        pmult_ud = sd * aterm
        pmult_uu = su * bterm
        layersource = layersource + u_xpos * pmult_ud + u_xneg * pmult_uu
        if return_profile:
            profile = torch.empty(
                (delta_tau.shape[0], nlay + 1), dtype=delta_tau.dtype, device=delta_tau.device
            )
            profile[:, nlay] = cumsource
            for n in range(nlay - 1, -1, -1):
                cumsource = layersource[:, n] + t_delt_userm[:, n] * cumsource
                profile[:, n] = cumsource
            return fluxmult * profile
        for n in range(nlay - 1, -1, -1):
            cumsource = layersource[:, n] + t_delt_userm[:, n] * cumsource
        return fluxmult * cumsource

    active = torch.arange(1, nlay + 1, device=delta_tau.device).unsqueeze(
        0
    ) <= layer_pis_cutoff.unsqueeze(1)
    zero = torch.zeros_like(gamma_m)
    sd = (initial_trans * hmult_2 - emult_up) / gamma_m
    su = (-initial_trans * t_delt_mubar * hmult_1 + emult_up) / gamma_p
    near = active & (torch.abs(gamma_m) < TAYLOR_SMALL)
    if bool(torch.any(near)):
        taylor = _taylor_series_2_vectorized_torch(
            TAYLOR_ORDER,
            TAYLOR_SMALL,
            gamma_m[near],
            sigma_p[near],
            delta_tau[near],
            torch.ones_like(fac2[near]),
            fac2[near],
            1.0,
        )
        sd = sd.clone()
        sd[near] = itrans_userm[near] * taylor
    pmult_ud = sd * aterm
    pmult_uu = su * bterm
    particulate = u_xpos * pmult_ud + u_xneg * pmult_uu
    layersource = layersource + torch.where(active, particulate, zero)
    if return_profile:
        profile = torch.empty(
            (delta_tau.shape[0], nlay + 1), dtype=delta_tau.dtype, device=delta_tau.device
        )
        profile[:, nlay] = cumsource
        for n in range(nlay - 1, -1, -1):
            cumsource = layersource[:, n] + t_delt_userm[:, n] * cumsource
            profile[:, n] = cumsource
        return fluxmult * profile
    for n in range(nlay - 1, -1, -1):
        cumsource = layersource[:, n] + t_delt_userm[:, n] * cumsource
    return fluxmult * cumsource


def solve_solar_obs_batch_torch(
    *,
    tau,
    omega,
    asymm,
    scaling,
    albedo,
    flux_factor,
    stream_value: float,
    chapman,
    x0: float,
    user_stream: float,
    user_secant: float,
    azmfac: float,
    px11: float,
    pxsq,
    px0x,
    ulp: float,
    dtype=None,
    device=None,
    bvp_device=None,
    bvp_dtype=None,
    bvp_engine: str = "auto",
    return_profile: bool = False,
):
    """Solves batched solar-observation 2S radiance with torch tensors.

    Parameters
    ----------
    bvp_device, bvp_dtype
        Optional context for the dense BVP fallback. This is useful for Apple
        MPS development runs, where ``bvp_engine="auto"`` defaults to a CPU
        float64 BVP solve while keeping the outer tensors on MPS.
    bvp_engine
        ``"auto"`` keeps the supported parity-oriented default on CPU while
        routing the current MPS development path through a CPU float64 BVP
        solve. ``"block"`` enables the opt-in CPU block-tridiagonal solver.
        ``"pentadiagonal"`` forces the experimental recurrence and remains a
        speed-oriented path rather than the Fortran-parity path.
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
    tau_t = _as_tensor(tau, dtype=dtype, device=device)
    omega_t = _as_tensor(omega, dtype=dtype, device=device)
    asymm_t = _as_tensor(asymm, dtype=dtype, device=device)
    scaling_t = _as_tensor(scaling, dtype=dtype, device=device)
    albedo_t = _as_tensor(albedo, dtype=dtype, device=device)
    flux_t = _as_tensor(flux_factor, dtype=dtype, device=device)
    chapman_t = _as_tensor(chapman, dtype=dtype, device=device)
    pxsq_t = _as_tensor(pxsq, dtype=dtype, device=device)
    px0x_t = _as_tensor(px0x, dtype=dtype, device=device)

    delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties_torch(
        tau_t,
        omega_t,
        asymm_t,
        scaling_t,
    )
    misc = _qsprep_obs_batch_torch(delta_tau, chapman_t, user_secant)
    pi4 = 4.0 * np.pi
    omega_asymm_3 = 3.0 * omega_total * asymm_total
    all_layers_active = bool(misc["all_active"])
    total = torch.zeros(tau_t.shape[0], dtype=dtype, device=device)
    total_profile = None
    if return_profile:
        total_profile = torch.zeros(
            (tau_t.shape[0], tau_t.shape[1] + 1), dtype=dtype, device=device
        )

    for fourier in (0, 1):
        surface_factor = 2.0 if fourier == 0 else 1.0
        delta_factor = 1.0 if fourier == 0 else 2.0
        eigenvalue, eigentrans, xpos1, xpos2, norm_saved = _hom_solution_solar_batch_torch(
            fourier=fourier,
            stream_value=stream_value,
            pxsq=float(pxsq_t[fourier]),
            omega=omega_total,
            omega_asymm_3=omega_asymm_3,
            delta_tau=delta_tau,
        )
        u_xpos, u_xneg = _hom_user_solution_solar_batch_torch(
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
        hmult_1, hmult_2 = _hmult_master_batch_torch(
            delta_tau=delta_tau,
            user_secant=user_secant,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            t_delt_userm=misc["t_delt_userm"],
        )
        gamma_m, gamma_p, aterm, bterm, wupper, wlower = _gbeam_solution_batch_torch(
            fourier=fourier,
            pi4=pi4,
            flux_factor=flux_t,
            layer_pis_cutoff=misc["layer_pis_cutoff"],
            px0x=float(px0x_t[fourier]),
            omega=omega_total,
            omega_asymm_3=omega_asymm_3,
            average_secant=misc["average_secant"],
            initial_trans=misc["initial_trans"],
            t_delt_mubar=misc["t_delt_mubar"],
            xpos1=xpos1,
            xpos2=xpos2,
            eigenvalue=eigenvalue,
            eigentrans=eigentrans,
            norm_saved=norm_saved,
            delta_tau=delta_tau,
            all_layers_active=all_layers_active,
        )
        if fourier == 0:
            direct_beam = flux_t * x0 / delta_factor / np.pi * misc["trans_solar_beam"] * albedo_t
            bvp_albedo = albedo_t
        else:
            direct_beam = torch.zeros_like(albedo_t)
            bvp_albedo = torch.zeros_like(albedo_t)
        if bvp_engine == "pentadiagonal":
            solve_bvp = solve_solar_observation_bvp_batch_torch
        elif bvp_engine == "block":
            solve_bvp = solve_solar_observation_block_bvp_batch_torch
        elif bvp_engine == "dense":
            solve_bvp = solve_solar_observation_dense_bvp_batch_torch
        else:
            solve_bvp = (
                solve_solar_observation_bvp_batch_torch
                if delta_tau.dtype == torch.float64 or device.type == "mps"
                else solve_solar_observation_dense_bvp_batch_torch
            )
        lcon, mcon = solve_bvp(
            albedo=bvp_albedo,
            direct_beam=direct_beam,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=wupper,
            wlower=wlower,
            solve_device=bvp_device,
            solve_dtype=bvp_dtype,
        )
        if solve_bvp in {
            solve_solar_observation_bvp_batch_torch,
            solve_solar_observation_block_bvp_batch_torch,
        }:
            lcon, mcon = repair_nonfinite_bvp_rows_torch(
                lcon=lcon,
                mcon=mcon,
                bad_row_solver=solve_solar_observation_dense_bvp_batch_torch,
                solver_kwargs={
                    "albedo": bvp_albedo,
                    "direct_beam": direct_beam,
                    "surface_factor": surface_factor,
                    "stream_value": stream_value,
                    "xpos1": xpos1,
                    "xpos2": xpos2,
                    "eigentrans": eigentrans,
                    "wupper": wupper,
                    "wlower": wlower,
                },
                solve_device=bvp_device,
                solve_dtype=bvp_dtype,
            )
        contribution = _upuser_intensity_batch_torch(
            layer_pis_cutoff=misc["layer_pis_cutoff"],
            surface_factor=surface_factor,
            albedo=bvp_albedo,
            fluxmult=delta_factor,
            stream_value=stream_value,
            delta_tau=delta_tau,
            gamma_p=gamma_p,
            gamma_m=gamma_m,
            sigma_p=misc["sigma_p"],
            aterm=aterm,
            bterm=bterm,
            initial_trans=misc["initial_trans"],
            itrans_userm=misc["itrans_userm"],
            t_delt_userm=misc["t_delt_userm"],
            t_delt_mubar=misc["t_delt_mubar"],
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
            emult_up=misc["emult_up"],
            all_layers_active=all_layers_active,
            return_profile=return_profile,
        )
        if return_profile:
            total_profile = contribution if fourier == 0 else total_profile + azmfac * contribution
            total = total_profile[:, 0]
        else:
            total = contribution if fourier == 0 else total + azmfac * contribution
    return total_profile if return_profile else total
