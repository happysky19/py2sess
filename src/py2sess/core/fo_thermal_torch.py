"""Torch-native first-order thermal observation-geometry solver."""

from __future__ import annotations

from .backend import _load_torch
from .fo_thermal import FoThermalResult
from .fo_solar_obs import _fo_eps_geometry

torch = _load_torch()


def solve_fo_thermal_torch(
    *,
    tau_arr,
    omega_arr,
    d2s_scaling,
    user_angles,
    thermal_bb_input,
    surfbb: float,
    emissivity: float,
    do_plane_parallel: bool,
    do_optical_deltam_scaling: bool = True,
    do_source_deltam_scaling: bool = True,
    height_grid=None,
    earth_radius: float = 6371.0,
    nfine: int = 3,
) -> FoThermalResult:
    """Solves FO thermal observation geometry using native torch operations."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")

    dtype = tau_arr.dtype
    tau_arr = tau_arr.to(dtype=dtype)
    omega_arr = omega_arr.to(dtype=dtype)
    d2s_scaling = d2s_scaling.to(dtype=dtype)
    user_angles = user_angles.to(dtype=dtype)
    thermal_bb_input = thermal_bb_input.to(dtype=dtype)
    if height_grid is not None:
        height_grid = height_grid.to(dtype=dtype)

    nlayers = int(tau_arr.shape[0])
    mu1 = torch.cos(user_angles * (torch.pi / 180.0))
    bb_input = thermal_bb_input
    user_emissivity = torch.full_like(mu1, float(emissivity))
    deltaus_all = (
        tau_arr * (1.0 - omega_arr * d2s_scaling) if do_optical_deltam_scaling else tau_arr
    )

    tcom0 = []
    tcom1 = []
    bb_inputn1 = bb_input[0]
    for n in range(nlayers):
        tms = 1.0 - omega_arr[n]
        if do_source_deltam_scaling:
            tms = tms / (1.0 - omega_arr[n] * d2s_scaling[n])
        bb_inputn = bb_input[n + 1]
        deltaus = deltaus_all[n]
        thermcoeffs = (bb_inputn - bb_inputn1) / deltaus
        tcom0.append(bb_inputn1 * tms)
        tcom1.append(thermcoeffs * tms)
        bb_inputn1 = bb_inputn
    tcom0 = torch.stack(tcom0)
    tcom1 = torch.stack(tcom1)

    intensity_atmos_up_toa = []
    intensity_surface_toa = []
    intensity_total_up_toa = []
    intensity_atmos_dn_toa = []
    intensity_atmos_up_boa = []
    intensity_surface_boa = []
    intensity_total_up_boa = []
    intensity_atmos_dn_boa = []

    if do_plane_parallel:
        geometry = None
        extinction = None
    else:
        if height_grid is None:
            raise ValueError("height_grid is required for EPS thermal FO")
        extinction = deltaus_all / (height_grid[:-1] - height_grid[1:])
        dummy_obsgeoms = torch.stack(
            (
                torch.zeros_like(user_angles),
                user_angles,
                torch.zeros_like(user_angles),
            ),
            dim=1,
        )
        geometry_np = _fo_eps_geometry(
            user_obsgeoms=dummy_obsgeoms.detach().cpu().numpy(),
            height_grid=height_grid.detach().cpu().numpy(),
            earth_radius=earth_radius,
            nfine=nfine,
            vsign=1.0,
        )
        geometry = {}
        for key, value in geometry_np.items():
            if hasattr(value, "dtype") and str(value.dtype) in {"bool", "bool_"}:
                geometry[key] = torch.tensor(value, device=tau_arr.device)
            else:
                geometry[key] = torch.tensor(value, dtype=tau_arr.dtype, device=tau_arr.device)

    for v, mu1v in enumerate(mu1):
        lostrans_up = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]
        sources_up = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]
        lostrans_dn = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]
        sources_dn = [
            torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device) for _ in range(nlayers)
        ]

        if do_plane_parallel:
            for n in range(nlayers):
                lostau = deltaus_all[n] / mu1v
                lostrans = torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                lostrans_up[n] = lostrans
                t_mult_up1 = tcom0[n] + tcom1[n] * mu1v
                t_mult_up0 = -t_mult_up1 - tcom1[n] * deltaus_all[n]
                sources_up[n] = t_mult_up0 * lostrans + t_mult_up1
        else:
            assert geometry is not None and extinction is not None and height_grid is not None
            do_nadir = bool(geometry["do_nadir"][v].item())
            rayconv = geometry["raycon"][v]
            cot_1 = (
                geometry["cota"][nlayers, v]
                if not do_nadir
                else torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
            )
            for n in range(nlayers, 0, -1):
                kn = extinction[n - 1]
                nj = int(geometry["nfinedivs"][n - 1, v].item())
                if do_nadir:
                    lostau = deltaus_all[n - 1]
                    lostrans_up[n - 1] = (
                        torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                    )
                    for j in range(nj):
                        xjkn = geometry["xfine"][j, n - 1, v] * kn
                        solution = tcom0[n - 1] + xjkn * tcom1[n - 1]
                        wtrans = kn * torch.exp(-xjkn) * geometry["wfine"][j, n - 1, v]
                        sources_up[n - 1] = sources_up[n - 1] + solution * wtrans
                else:
                    cot_2 = geometry["cota"][n - 1, v]
                    ke = rayconv * kn
                    lostau = ke * (cot_2 - cot_1)
                    lostrans_up[n - 1] = (
                        torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                    )
                    for j in range(nj):
                        xjkn = geometry["xfine"][j, n - 1, v] * kn
                        tran = torch.exp(-ke * (cot_2 - geometry["cotfine"][j, n - 1, v]))
                        solution = tcom0[n - 1] + xjkn * tcom1[n - 1]
                        wtrans = (
                            ke
                            * tran
                            * geometry["csqfine"][j, n - 1, v]
                            * geometry["wfine"][j, n - 1, v]
                        )
                        sources_up[n - 1] = sources_up[n - 1] + solution * wtrans
                    cot_1 = cot_2

        cumsource_up = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        cumsource_surface = (
            torch.as_tensor(float(surfbb), dtype=tau_arr.dtype, device=tau_arr.device)
            * user_emissivity[v]
        )
        for n in range(nlayers - 1, -1, -1):
            cumsource_surface = lostrans_up[n] * cumsource_surface
            cumsource_up = lostrans_up[n] * cumsource_up + sources_up[n]
        intensity_atmos_up_toa.append(cumsource_up)
        intensity_surface_toa.append(cumsource_surface)
        intensity_total_up_toa.append(cumsource_up + cumsource_surface)
        intensity_atmos_up_boa.append(torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device))
        intensity_surface_boa.append(
            torch.as_tensor(float(surfbb), dtype=tau_arr.dtype, device=tau_arr.device)
            * user_emissivity[v]
        )
        intensity_total_up_boa.append(intensity_surface_boa[-1])

        if do_plane_parallel:
            for n in range(nlayers - 1, -1, -1):
                lostau = deltaus_all[n] / mu1v
                lostrans = torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                lostrans_dn[n] = lostrans
                t_mult_dn1 = tcom0[n] - tcom1[n] * mu1v
                sources_dn[n] = -t_mult_dn1 * lostrans + t_mult_dn1 + tcom1[n] * deltaus_all[n]
        else:
            assert geometry is not None and extinction is not None and height_grid is not None
            do_nadir = bool(geometry["do_nadir"][v].item())
            rayconv = geometry["raycon"][v]
            cot_1 = (
                geometry["cota"][nlayers, v]
                if not do_nadir
                else torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
            )
            for n in range(nlayers, 0, -1):
                kn = extinction[n - 1]
                nj = int(geometry["nfinedivs"][n - 1, v].item())
                if do_nadir:
                    lostau = deltaus_all[n - 1]
                    lostrans_dn[n - 1] = (
                        torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                    )
                    radiin = earth_radius + height_grid[n]
                    radiin1 = earth_radius + height_grid[n - 1]
                    rdiff = radiin1 - radiin
                    for j in range(nj):
                        xfjnv = geometry["xfine"][j, n - 1, v]
                        solution = tcom0[n - 1] + xfjnv * kn * tcom1[n - 1]
                        wtrans = (
                            kn * torch.exp(-kn * (rdiff - xfjnv)) * geometry["wfine"][j, n - 1, v]
                        )
                        sources_dn[n - 1] = sources_dn[n - 1] + solution * wtrans
                else:
                    cot_2 = geometry["cota"][n - 1, v]
                    ke = rayconv * kn
                    lostau = ke * (cot_2 - cot_1)
                    lostrans_dn[n - 1] = (
                        torch.exp(-lostau) if bool(lostau < 88.0) else torch.zeros_like(lostau)
                    )
                    for j in range(nj):
                        tran = torch.exp(-ke * (geometry["cotfine"][j, n - 1, v] - cot_1))
                        solution = tcom0[n - 1] + geometry["xfine"][j, n - 1, v] * kn * tcom1[n - 1]
                        wtrans = (
                            ke
                            * tran
                            * geometry["csqfine"][j, n - 1, v]
                            * geometry["wfine"][j, n - 1, v]
                        )
                        sources_dn[n - 1] = sources_dn[n - 1] + solution * wtrans
                    cot_1 = cot_2

        cumsource_dn = torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device)
        for n in range(nlayers):
            cumsource_dn = sources_dn[n] + lostrans_dn[n] * cumsource_dn
        intensity_atmos_dn_toa.append(torch.zeros((), dtype=tau_arr.dtype, device=tau_arr.device))
        intensity_atmos_dn_boa.append(cumsource_dn)

    return FoThermalResult(
        intensity_atmos_up_toa=torch.stack(intensity_atmos_up_toa),
        intensity_surface_toa=torch.stack(intensity_surface_toa),
        intensity_total_up_toa=torch.stack(intensity_total_up_toa),
        intensity_atmos_dn_toa=torch.stack(intensity_atmos_dn_toa),
        intensity_atmos_up_boa=torch.stack(intensity_atmos_up_boa),
        intensity_surface_boa=torch.stack(intensity_surface_boa),
        intensity_total_up_boa=torch.stack(intensity_total_up_boa),
        intensity_atmos_dn_boa=torch.stack(intensity_atmos_dn_boa),
        mu1=mu1,
    )
