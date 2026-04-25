"""Torch-native first-order solar observation-geometry solvers."""

from __future__ import annotations

import math

from .backend import _load_torch
from .fo_solar_obs import FoSolarObsResult, _find_sunpaths_direct, _fo_eps_geometry
from ..optical.delta_m_torch import (
    default_delta_m_truncation_factor_torch,
    validate_delta_m_truncation_factor_torch,
)

torch = _load_torch()


def _fo_work_dtype(value):
    """Returns the internal FO working dtype for the current torch device."""
    return torch.float32 if value.device.type == "mps" else torch.float64


def _phase_function_hg_torch(mu, asymmetry, n_moments: int):
    if n_moments < 0:
        raise ValueError("n_moments must be non-negative")
    if n_moments == 0:
        return torch.ones_like(mu)
    if bool((torch.abs(asymmetry) >= 1.0).any()):
        raise ValueError("g must satisfy -1 < g < 1 for Henyey-Greenstein scattering")
    denominator = 1.0 + asymmetry * asymmetry - 2.0 * asymmetry * mu
    return (1.0 - asymmetry * asymmetry) / torch.pow(denominator, 1.5)


def _torch_context_from_values(*, dtype, device, values):
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    resolved_dtype = dtype
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    for value in values:
        if torch.is_tensor(value):
            if resolved_dtype is None:
                resolved_dtype = value.dtype
            if device is None:
                resolved_device = value.device
            break
    if resolved_dtype is None:
        resolved_dtype = torch.get_default_dtype()
    return resolved_dtype, resolved_device


def _as_tensor(value, *, dtype, device):
    if torch.is_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(dtype=dtype, device=device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _normalize_solar_obs_angles_torch(angles, *, dtype, device):
    arr = _as_tensor(angles, dtype=dtype, device=device)
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("angles must be finite")
    if arr.ndim == 1 and int(arr.numel()) == 3:
        return arr.reshape(1, 3)
    if arr.ndim != 2 or int(arr.shape[1]) != 3:
        raise ValueError("angles must have shape (3,) or (ngeom, 3)")
    return arr


def _solar_obs_scattering_cosines_torch(angles, *, dtype, device):
    geoms = _normalize_solar_obs_angles_torch(angles, dtype=dtype, device=device)
    deg2rad = torch.as_tensor(math.pi / 180.0, dtype=dtype, device=device)
    sza = geoms[:, 0] * deg2rad
    vza = geoms[:, 1] * deg2rad
    raz = geoms[:, 2] * deg2rad
    mu1 = torch.cos(vza)
    cosscat = -(torch.cos(vza) * torch.cos(sza)) + torch.sin(vza) * torch.sin(sza) * torch.cos(raz)
    overhead = torch.isclose(geoms[:, 0], torch.zeros_like(geoms[:, 0]))
    if bool(overhead.any()):
        nadir_limb = torch.isclose(mu1, torch.zeros_like(mu1))
        overhead_values = torch.where(nadir_limb, torch.zeros_like(mu1), -mu1)
        cosscat = torch.where(overhead, overhead_values, cosscat)
    return cosscat


def fo_scatter_term_henyey_greenstein_torch(
    *,
    ssa,
    g,
    angles,
    delta_m_truncation_factor=None,
    n_moments: int = 5000,
    dtype=None,
    device=None,
):
    """Build differentiable solar FO scatter terms for HG phase functions.

    Any positive ``n_moments`` uses the closed-form HG phase function;
    ``n_moments=0`` selects isotropic scattering.
    """
    if int(n_moments) < 0:
        raise ValueError("n_moments must be non-negative")
    dtype, device = _torch_context_from_values(
        dtype=dtype,
        device=device,
        values=(ssa, g, angles, delta_m_truncation_factor),
    )
    ssa_t = _as_tensor(ssa, dtype=dtype, device=device)
    g_t = _as_tensor(g, dtype=dtype, device=device)
    if ssa_t.ndim == 0 or g_t.ndim == 0:
        raise ValueError("ssa and g must have a layer axis")
    if delta_m_truncation_factor is None:
        scaling_t = default_delta_m_truncation_factor_torch(g_t)
    else:
        scaling_t = _as_tensor(delta_m_truncation_factor, dtype=dtype, device=device)
    try:
        ssa_b, g_b, scaling_b = torch.broadcast_tensors(ssa_t, g_t, scaling_t)
    except RuntimeError as exc:
        raise ValueError(
            "ssa, g, and delta_m_truncation_factor must be broadcast-compatible"
        ) from exc
    if not bool(torch.isfinite(ssa_b).all() and torch.isfinite(g_b).all()):
        raise ValueError("ssa and g must be finite")
    validate_delta_m_truncation_factor_torch(scaling_b, ssa_b)
    denominator = 1.0 - scaling_b * ssa_b
    eps = torch.as_tensor(torch.finfo(dtype).eps, dtype=dtype, device=device)
    if bool((torch.abs(denominator) <= eps).any()):
        raise ValueError("1 - delta_m_truncation_factor * ssa is too close to zero")

    cosscat = _solar_obs_scattering_cosines_torch(angles, dtype=dtype, device=device)
    mu = cosscat.reshape((1,) * (ssa_b.ndim - 1) + (1, int(cosscat.numel())))
    phase = _phase_function_hg_torch(mu, g_b.unsqueeze(-1), int(n_moments))
    exact = phase * (ssa_b / denominator).unsqueeze(-1)
    if int(cosscat.numel()) == 1:
        exact = exact[..., 0]
    return exact.contiguous()


def solve_fo_solar_obs_plane_parallel_torch(
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    user_obsgeoms,
    d2s_scaling,
    albedo: float,
    flux_factor: float = 1.0,
    n_moments: int = 5000,
    exact_scatter=None,
) -> FoSolarObsResult:
    """Solves FO solar observation geometry in plane-parallel torch mode."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if user_obsgeoms is None:
        raise ValueError("user_obsgeoms are required for FO solar observation geometry")
    if user_obsgeoms.ndim != 2 or user_obsgeoms.shape[1] != 3:
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")

    work_dtype = _fo_work_dtype(tau_arr)
    tau_arr = tau_arr.to(dtype=work_dtype)
    omega_arr = omega_arr.to(dtype=work_dtype)
    asymm_arr = asymm_arr.to(dtype=work_dtype)
    user_obsgeoms = user_obsgeoms.to(dtype=work_dtype)
    d2s_scaling = d2s_scaling.to(dtype=work_dtype)

    deg2rad = math.pi / 180.0
    mu0 = torch.cos(user_obsgeoms[:, 0] * deg2rad)
    mu1 = torch.cos(user_obsgeoms[:, 1] * deg2rad)
    vza = user_obsgeoms[:, 1] * deg2rad
    sza = user_obsgeoms[:, 0] * deg2rad
    azm = user_obsgeoms[:, 2] * deg2rad
    cosscat = -(torch.cos(vza) * torch.cos(sza)) + torch.sin(vza) * torch.sin(sza) * torch.cos(azm)
    do_nadir = torch.isclose(user_obsgeoms[:, 1], torch.zeros_like(user_obsgeoms[:, 1]))

    nlayers = int(tau_arr.shape[0])
    ngeoms = int(user_obsgeoms.shape[0])
    if exact_scatter is not None:
        exact_scatter = exact_scatter.to(dtype=work_dtype)
        if exact_scatter.ndim == 1:
            exact_scatter = exact_scatter[:, None]
        if exact_scatter.shape != (nlayers, ngeoms):
            raise ValueError(
                "exact_scatter must have shape (n_layers,) or (n_layers, n_geometries)"
            )
    flux = 0.25 * flux_factor / math.pi
    intensity_ss = []
    intensity_db = []

    for v in range(ngeoms):
        phase_terms = []
        for n in range(nlayers):
            if exact_scatter is not None:
                phase_terms.append(exact_scatter[n, v])
            else:
                phase = _phase_function_hg_torch(cosscat[v], asymm_arr[n], n_moments)
                tms = omega_arr[n] / (1.0 - d2s_scaling[n] * omega_arr[n])
                phase_terms.append(phase * tms)
        phase_terms = torch.stack(phase_terms)

        attenuations = [torch.ones((), dtype=tau_arr.dtype, device=tau_arr.device)]
        cumtau = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        for n in range(nlayers):
            cumtau = cumtau + tau_arr[n]
            sumd = cumtau / mu0[v]
            attenuations.append(torch.exp(-sumd) if bool(sumd < 88.0) else torch.zeros_like(sumd))

        lostrans_up = []
        sources_up = []
        solutions = []
        factor1 = []
        factor2 = []
        attn_prev = attenuations[0]
        if bool(torch.isclose(mu1[v], torch.zeros_like(mu1[v]))):
            for n in range(nlayers):
                solutions.append(phase_terms[n] * attn_prev)
                attn_prev = attenuations[n + 1]
                factor1.append(torch.zeros_like(attn_prev))
                factor2.append(torch.zeros_like(attn_prev))
                lostrans_up.append(torch.zeros_like(attn_prev))
        else:
            for n in range(nlayers):
                lostau = tau_arr[n] / mu1[v]
                solutions.append(phase_terms[n] * attn_prev)
                lostrans = torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                lostrans_up.append(lostrans)
                attn = attenuations[n + 1]
                factor1.append(
                    attn / attn_prev
                    if not bool(torch.isclose(attn_prev, torch.zeros_like(attn_prev)))
                    else torch.zeros_like(attn)
                )
                factor2.append(mu1[v] / mu0[v])
                attn_prev = attn

        for n in range(nlayers):
            if bool(torch.isclose(mu1[v], torch.zeros_like(mu1[v]))):
                sources_up.append(solutions[n])
            else:
                multiplier = (1.0 - factor1[n] * lostrans_up[n]) / (factor2[n] + 1.0)
                sources_up.append(solutions[n] * multiplier)

        cumsource_up = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        cumsource_db = 4.0 * mu0[v] * albedo * attenuations[-1]
        for n in range(nlayers - 1, -1, -1):
            cumsource_db = lostrans_up[n] * cumsource_db
            cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
        intensity_ss.append(flux * cumsource_up)
        intensity_db.append(flux * cumsource_db)

    intensity_ss = torch.stack(intensity_ss)
    intensity_db = torch.stack(intensity_db)
    return FoSolarObsResult(
        intensity_total=intensity_ss + intensity_db,
        intensity_ss=intensity_ss,
        intensity_db=intensity_db,
        mu0=mu0,
        mu1=mu1,
        cosscat=cosscat,
        do_nadir=do_nadir,
    )


def solve_fo_solar_obs_rps_torch(
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    user_obsgeoms,
    d2s_scaling,
    height_grid,
    earth_radius: float,
    albedo: float,
    flux_factor: float = 1.0,
    n_moments: int = 5000,
    exact_scatter=None,
) -> FoSolarObsResult:
    """Solves FO solar observation geometry in regular pseudo-spherical torch mode."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if user_obsgeoms is None:
        raise ValueError("user_obsgeoms are required for FO solar observation geometry")
    if user_obsgeoms.ndim != 2 or user_obsgeoms.shape[1] != 3:
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")
    if height_grid is None:
        raise ValueError("height_grid is required for RPS solar FO")

    work_dtype = _fo_work_dtype(tau_arr)
    tau_arr = tau_arr.to(dtype=work_dtype)
    omega_arr = omega_arr.to(dtype=work_dtype)
    asymm_arr = asymm_arr.to(dtype=work_dtype)
    user_obsgeoms = user_obsgeoms.to(dtype=work_dtype)
    d2s_scaling = d2s_scaling.to(dtype=work_dtype)
    height_grid = height_grid.to(dtype=work_dtype)

    deg2rad = math.pi / 180.0
    nlayers = int(tau_arr.shape[0])
    ngeoms = int(user_obsgeoms.shape[0])
    if exact_scatter is not None:
        exact_scatter = exact_scatter.to(dtype=work_dtype)
        if exact_scatter.ndim == 1:
            exact_scatter = exact_scatter[:, None]
        if exact_scatter.shape != (nlayers, ngeoms):
            raise ValueError(
                "exact_scatter must have shape (n_layers,) or (n_layers, n_geometries)"
            )
    heights_np = height_grid.detach().cpu().numpy()
    radii_np = earth_radius + heights_np
    flux = 0.25 * flux_factor / math.pi

    mu0 = torch.cos(user_obsgeoms[:, 0] * deg2rad)
    mu1 = torch.cos(user_obsgeoms[:, 1] * deg2rad)
    vza = user_obsgeoms[:, 1] * deg2rad
    sza = user_obsgeoms[:, 0] * deg2rad
    azm = user_obsgeoms[:, 2] * deg2rad
    cosscat = -(torch.cos(vza) * torch.cos(sza)) + torch.sin(vza) * torch.sin(sza) * torch.cos(azm)
    do_nadir = torch.isclose(user_obsgeoms[:, 1], torch.zeros_like(user_obsgeoms[:, 1]))

    intensity_ss = []
    intensity_db = []

    for v in range(ngeoms):
        phase_terms = []
        for n in range(nlayers):
            if exact_scatter is not None:
                phase_terms.append(exact_scatter[n, v])
            else:
                phase = _phase_function_hg_torch(cosscat[v], asymm_arr[n], n_moments)
                tms = omega_arr[n] / (1.0 - d2s_scaling[n] * omega_arr[n])
                phase_terms.append(phase * tms)
        phase_terms = torch.stack(phase_terms)

        theta_boa_r = float(user_obsgeoms[v, 0].detach().cpu().item()) * deg2rad
        stheta_boa = (
            1.0
            if math.isclose(float(user_obsgeoms[v, 0].detach().cpu().item()), 90.0)
            else math.sin(theta_boa_r)
        )
        do_overhead_sun = math.isclose(float(user_obsgeoms[v, 0].detach().cpu().item()), 0.0)

        sunpaths_cols = []
        for n in range(1, nlayers + 1):
            sunpaths_local = _find_sunpaths_direct(
                do_zero_sun_boa=do_overhead_sun,
                radstart=float(radii_np[n]),
                radii=radii_np,
                theta_start=theta_boa_r,
                sin_theta_start=stheta_boa,
                n=n,
            )
            col = torch.zeros(nlayers, dtype=tau_arr.dtype, device=tau_arr.device)
            if n > 0:
                col[:n] = torch.tensor(
                    sunpaths_local[:n], dtype=tau_arr.dtype, device=tau_arr.device
                )
            sunpaths_cols.append(col)

        attenuations = [torch.ones((), dtype=tau_arr.dtype, device=tau_arr.device)]
        suntau = []
        for n in range(nlayers):
            sumd = torch.dot(
                (tau_arr / (height_grid[:-1] - height_grid[1:]))[: n + 1], sunpaths_cols[n][: n + 1]
            )
            suntau.append(sumd)
            attenuations.append(torch.exp(-sumd) if bool(sumd < 88.0) else torch.zeros_like(sumd))

        solutions = []
        sources_up = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]
        lostrans_up = []
        factor1 = []
        factor2 = []
        nstart = nlayers
        mu1v = mu1[v]
        attn_prev = attenuations[0]
        suntaun1 = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        if bool(torch.isclose(mu1v, torch.zeros_like(mu1v))):
            for n in range(nlayers):
                solutions.append(phase_terms[n] * attn_prev)
                attn_prev = attenuations[n + 1]
                lostrans_up.append(torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device))
                factor1.append(torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device))
                factor2.append(torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device))
        else:
            for n in range(nlayers):
                lostau = tau_arr[n] / mu1v
                solutions.append(phase_terms[n] * attn_prev)
                lostrans = torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                lostrans_up.append(lostrans)
                attn = attenuations[n + 1]
                suntaun = suntau[n]
                if not bool(torch.isclose(attn_prev, torch.zeros_like(attn_prev))):
                    factor1.append(attn / attn_prev)
                    factor2.append((suntaun - suntaun1) / lostau)
                    nstart = n + 1
                else:
                    factor1.append(torch.zeros_like(attn))
                    factor2.append(torch.zeros_like(attn))
                attn_prev = attn
                suntaun1 = suntaun

        for n in range(nlayers - 1, nstart, -1):
            sources_up[n] = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        for n in range(nstart - 1, -1, -1):
            if bool(torch.isclose(mu1v, torch.zeros_like(mu1v))):
                sources_up[n] = solutions[n]
            else:
                multiplier = (1.0 - factor1[n] * lostrans_up[n]) / (factor2[n] + 1.0)
                sources_up[n] = solutions[n] * multiplier

        cumsource_up = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        cumsource_db = 4.0 * mu0[v] * albedo * attenuations[-1]
        for n in range(nlayers - 1, -1, -1):
            cumsource_db = lostrans_up[n] * cumsource_db
            cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
        intensity_ss.append(flux * cumsource_up)
        intensity_db.append(flux * cumsource_db)

    intensity_ss = torch.stack(intensity_ss)
    intensity_db = torch.stack(intensity_db)
    return FoSolarObsResult(
        intensity_total=intensity_ss + intensity_db,
        intensity_ss=intensity_ss,
        intensity_db=intensity_db,
        mu0=mu0,
        mu1=mu1,
        cosscat=cosscat,
        do_nadir=do_nadir,
    )


def solve_fo_solar_obs_eps_torch(
    *,
    tau_arr,
    omega_arr,
    asymm_arr,
    user_obsgeoms,
    d2s_scaling,
    height_grid,
    earth_radius: float,
    albedo: float,
    flux_factor: float = 1.0,
    n_moments: int = 5000,
    nfine: int = 3,
    exact_scatter=None,
) -> FoSolarObsResult:
    """Solves FO solar observation geometry in enhanced pseudo-spherical torch mode."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if user_obsgeoms is None:
        raise ValueError("user_obsgeoms are required for FO solar observation geometry")
    if user_obsgeoms.ndim != 2 or user_obsgeoms.shape[1] != 3:
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")
    if height_grid is None:
        raise ValueError("height_grid is required for EPS solar FO")

    work_dtype = _fo_work_dtype(tau_arr)
    tau_arr = tau_arr.to(dtype=work_dtype)
    omega_arr = omega_arr.to(dtype=work_dtype)
    asymm_arr = asymm_arr.to(dtype=work_dtype)
    user_obsgeoms = user_obsgeoms.to(dtype=work_dtype)
    d2s_scaling = d2s_scaling.to(dtype=work_dtype)
    height_grid = height_grid.to(dtype=work_dtype)

    nlayers = int(tau_arr.shape[0])
    ngeoms = int(user_obsgeoms.shape[0])
    if exact_scatter is not None:
        exact_scatter = exact_scatter.to(dtype=work_dtype)
        if exact_scatter.ndim == 1:
            exact_scatter = exact_scatter[:, None]
        if exact_scatter.shape != (nlayers, ngeoms):
            raise ValueError(
                "exact_scatter must have shape (n_layers,) or (n_layers, n_geometries)"
            )
    flux = 0.25 * flux_factor / math.pi
    extinction = tau_arr / (height_grid[:-1] - height_grid[1:])
    deltaus = tau_arr

    geometry_np = _fo_eps_geometry(
        user_obsgeoms=user_obsgeoms.detach().cpu().numpy(),
        height_grid=height_grid.detach().cpu().numpy(),
        earth_radius=earth_radius,
        nfine=nfine,
        vsign=1.0,
    )
    geom = {}
    for key, value in geometry_np.items():
        if hasattr(value, "dtype") and str(value.dtype) in {"bool", "bool_"}:
            geom[key] = torch.tensor(value, device=tau_arr.device)
        else:
            geom[key] = torch.tensor(value, dtype=tau_arr.dtype, device=tau_arr.device)

    mu0 = geom["mu0"]
    cosscat = geom["cosscat"]
    do_nadir = geom["do_nadir"]
    mu1 = torch.cos(user_obsgeoms[:, 1] * (math.pi / 180.0))

    intensity_ss = []
    intensity_db = []

    for v in range(ngeoms):
        phase_terms = []
        for n in range(nlayers):
            if exact_scatter is not None:
                phase_terms.append(exact_scatter[n, v])
            else:
                phase = _phase_function_hg_torch(cosscat[v], asymm_arr[n], n_moments)
                tms = omega_arr[n] / (1.0 - d2s_scaling[n] * omega_arr[n])
                phase_terms.append(phase * tms)
        phase_terms = torch.stack(phase_terms)

        ntrav_nl = int(geom["ntraversenl"][v].item())
        sunpathsnl = geom["sunpathsnl"][:, v]
        total_tau = torch.dot(extinction[:ntrav_nl], sunpathsnl[:ntrav_nl])
        attenuations_nl = (
            torch.exp(-total_tau) if bool(total_tau < 88.0) else torch.zeros_like(total_tau)
        )
        sources_up = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]
        lostrans_up = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]

        if bool(do_nadir[v].item()):
            for n in range(nlayers):
                lostrans_up[n] = torch.exp(-deltaus[n])
            for n in range(nlayers, 0, -1):
                kn = extinction[n - 1]
                layer_sum = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
                nfine_layer = int(geom["nfinedivs"][n - 1, v].item())
                for j in range(nfine_layer):
                    ntrav = int(geom["ntraversefine"][j, n - 1, v].item())
                    fine_tau = torch.dot(
                        extinction[:ntrav], geom["sunpathsfine"][:ntrav, j, n - 1, v]
                    )
                    attenuation = (
                        torch.exp(-fine_tau)
                        if bool(fine_tau < 88.0)
                        else torch.zeros_like(fine_tau)
                    )
                    solution = phase_terms[n - 1] * attenuation
                    layer_sum = (
                        layer_sum
                        + solution
                        * torch.exp(-geom["xfine"][j, n - 1, v] * kn)
                        * geom["wfine"][j, n - 1, v]
                    )
                sources_up[n - 1] = layer_sum * kn
        else:
            cot_1 = geom["cota"][nlayers, v]
            rayconv = geom["raycon"][v]
            for n in range(nlayers, 0, -1):
                cot_2 = geom["cota"][n - 1, v]
                ke = rayconv * extinction[n - 1]
                lostrans_up[n - 1] = torch.exp(-ke * (cot_2 - cot_1))
                nfine_layer = int(geom["nfinedivs"][n - 1, v].item())
                layer_sum = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
                for j in range(nfine_layer):
                    tran = torch.exp(-ke * (cot_2 - geom["cotfine"][j, n - 1, v]))
                    ntrav = int(geom["ntraversefine"][j, n - 1, v].item())
                    fine_tau = torch.dot(
                        extinction[:ntrav], geom["sunpathsfine"][:ntrav, j, n - 1, v]
                    )
                    attenuation = (
                        torch.exp(-fine_tau)
                        if bool(fine_tau < 88.0)
                        else torch.zeros_like(fine_tau)
                    )
                    solution = phase_terms[n - 1] * attenuation
                    layer_sum = (
                        layer_sum
                        + solution
                        * geom["csqfine"][j, n - 1, v]
                        * tran
                        * geom["wfine"][j, n - 1, v]
                    )
                sources_up[n - 1] = layer_sum * ke
                cot_1 = cot_2

        cumsource_up = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        cumsource_db = 4.0 * mu0[v] * albedo * attenuations_nl
        for n in range(nlayers - 1, -1, -1):
            cumsource_db = lostrans_up[n] * cumsource_db
            cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
        intensity_ss.append(flux * cumsource_up)
        intensity_db.append(flux * cumsource_db)

    intensity_ss = torch.stack(intensity_ss)
    intensity_db = torch.stack(intensity_db)
    return FoSolarObsResult(
        intensity_total=intensity_ss + intensity_db,
        intensity_ss=intensity_ss,
        intensity_db=intensity_db,
        mu0=mu0,
        mu1=mu1,
        cosscat=cosscat,
        do_nadir=do_nadir,
    )
