"""Torch phase-function preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..rtsolver.backend import _load_torch
from .delta_m_torch import validate_delta_m_truncation_factor_torch

torch = None


@dataclass(frozen=True)
class TwoStreamPhaseTorchInputs:
    """Torch two-stream phase inputs derived from Rayleigh/aerosol fractions."""

    g: Any
    delta_m_truncation_factor: Any


def _require_torch():
    global torch
    torch = _load_torch()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    return torch


def _context(*values, dtype=None, device=None):
    torch_module = _require_torch()
    resolved_device = torch_module.device("cpu") if device is None else torch_module.device(device)
    resolved_dtype = dtype
    for value in values:
        if torch_module.is_tensor(value):
            if resolved_dtype is None:
                resolved_dtype = value.dtype
            if device is None:
                resolved_device = value.device
            break
    if resolved_dtype is None:
        resolved_dtype = torch_module.float64
    return resolved_dtype, resolved_device


def _as_tensor(value, *, dtype, device):
    torch_module = _require_torch()
    if torch_module.is_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(dtype=dtype, device=device)
        return value
    return torch_module.as_tensor(value, dtype=dtype, device=device)


def ssa_from_optical_depth_torch(total_tau, scattering_tau, *, dtype=None, device=None):
    """Returns differentiable single-scattering albedo from optical depths."""
    dtype, device = _context(total_tau, scattering_tau, dtype=dtype, device=device)
    total = _as_tensor(total_tau, dtype=dtype, device=device)
    scattering = _as_tensor(scattering_tau, dtype=dtype, device=device)
    total_b, scattering_b = torch.broadcast_tensors(total, scattering)
    return torch.where(total_b > 0.0, scattering_b / total_b, torch.zeros_like(total_b))


def aerosol_interp_fraction_torch(wavelengths, *, reverse: bool = False, dtype=None, device=None):
    """Returns the Fortran endpoint interpolation fraction for aerosol moments."""
    dtype, device = _context(wavelengths, dtype=dtype, device=device)
    grid = _as_tensor(wavelengths, dtype=dtype, device=device)
    if grid.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if int(grid.numel()) == 0:
        raise ValueError("wavelengths must not be empty")
    if not bool(torch.isfinite(grid).all()):
        raise ValueError("wavelengths must be finite")
    span = grid[-1] - grid[0]
    if float(span.detach().cpu()) == 0.0:
        return torch.zeros_like(grid)
    values = torch.flip(grid, dims=(0,)) if reverse else grid
    return (values - grid[0]) / span


def _broadcast_leading(name: str, value, shape: tuple[int, ...], *, dtype, device):
    arr = _as_tensor(value, dtype=dtype, device=device)
    if not bool(torch.isfinite(arr).all()):
        raise ValueError(f"{name} must be finite")
    try:
        return torch.broadcast_to(arr, shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to spectral shape {shape}") from exc


def _aerosol_moments(aerosol_moments, *, dtype, device):
    moments = _as_tensor(aerosol_moments, dtype=dtype, device=device)
    if moments.ndim != 3 or int(moments.shape[0]) != 2 or int(moments.shape[1]) < 3:
        raise ValueError("aerosol_moments must have shape (2, nmom + 1, naerosol)")
    if not bool(torch.isfinite(moments).all()):
        raise ValueError("aerosol_moments must be finite")
    return moments


def _aerosol_phase_endpoints(moments, cos_scatter):
    p_lm2 = torch.ones_like(cos_scatter)
    endpoint_phase = moments[:, 0, :, None] * p_lm2

    p_lm1 = cos_scatter
    endpoint_phase = endpoint_phase + moments[:, 1, :, None] * p_lm1
    for ell in range(2, int(moments.shape[1])):
        p_l = ((2 * ell - 1) * cos_scatter * p_lm1 - (ell - 1) * p_lm2) / ell
        endpoint_phase = endpoint_phase + moments[:, ell, :, None] * p_l
        p_lm2, p_lm1 = p_lm1, p_l
    return endpoint_phase


def _aerosol_moment(
    *,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
    moment_index: int,
):
    moment = aerosol_moments[0, moment_index] + aerosol_interp_fraction[..., None] * (
        aerosol_moments[1, moment_index] - aerosol_moments[0, moment_index]
    )
    return torch.einsum("...la,...a->...l", aerosol_fraction, moment)


def build_two_stream_phase_inputs_torch(
    *,
    ssa,
    depol,
    rayleigh_fraction,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
    dtype=None,
    device=None,
) -> TwoStreamPhaseTorchInputs:
    """Builds differentiable two-stream ``g`` and delta-M truncation factor."""
    dtype, device = _context(
        ssa,
        depol,
        rayleigh_fraction,
        aerosol_fraction,
        aerosol_moments,
        aerosol_interp_fraction,
        dtype=dtype,
        device=device,
    )
    ssa_t = _as_tensor(ssa, dtype=dtype, device=device)
    ray_t = _as_tensor(rayleigh_fraction, dtype=dtype, device=device)
    ssa_b, ray_b = torch.broadcast_tensors(ssa_t, ray_t)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if not bool(torch.isfinite(ssa_b).all() and torch.isfinite(ray_b).all()):
        raise ValueError("ssa and rayleigh_fraction must be finite")

    lead_shape = tuple(ssa_b.shape[:-1])
    depol_b = _broadcast_leading("depol", depol, lead_shape, dtype=dtype, device=device)
    fac_b = _broadcast_leading(
        "aerosol_interp_fraction", aerosol_interp_fraction, lead_shape, dtype=dtype, device=device
    )
    moments = _aerosol_moments(aerosol_moments, dtype=dtype, device=device)
    aer_frac = _as_tensor(aerosol_fraction, dtype=dtype, device=device)
    if not bool(torch.isfinite(aer_frac).all()):
        raise ValueError("aerosol_fraction must be finite")
    if int(moments.shape[2]) != int(aer_frac.shape[-1]):
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = torch.broadcast_to(aer_frac, tuple(ssa_b.shape) + (int(moments.shape[2]),))

    ray2mom = (1.0 - depol_b) / (2.0 + depol_b)
    moment1 = _aerosol_moment(
        aerosol_fraction=aer_frac_b,
        aerosol_moments=moments,
        aerosol_interp_fraction=fac_b,
        moment_index=1,
    )
    moment2 = ray_b * ray2mom[..., None] + _aerosol_moment(
        aerosol_fraction=aer_frac_b,
        aerosol_moments=moments,
        aerosol_interp_fraction=fac_b,
        moment_index=2,
    )
    g = moment1 / 3.0
    truncation_factor = moment2 / 5.0
    validate_delta_m_truncation_factor_torch(truncation_factor, ssa_b)
    return TwoStreamPhaseTorchInputs(g=g, delta_m_truncation_factor=truncation_factor)


def _normalize_angles(angles, *, dtype, device):
    arr = _as_tensor(angles, dtype=dtype, device=device)
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("angles must be finite")
    if arr.ndim == 1 and int(arr.numel()) == 3:
        return arr.reshape(1, 3)
    if arr.ndim == 2 and int(arr.shape[1]) == 3:
        return arr
    raise ValueError("angles must have shape (3,) or (ngeom, 3)")


def _solar_obs_scattering_cosines(angles, *, dtype, device):
    geoms = _normalize_angles(angles, dtype=dtype, device=device)
    deg2rad = torch.as_tensor(torch.pi / 180.0, dtype=dtype, device=device)
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


def build_solar_fo_scatter_term_torch(
    *,
    ssa,
    depol,
    rayleigh_fraction,
    aerosol_fraction,
    aerosol_moments,
    aerosol_interp_fraction,
    angles,
    delta_m_truncation_factor,
    dtype=None,
    device=None,
):
    """Builds differentiable Fortran-style solar FO exact-scatter source terms."""
    dtype, device = _context(
        ssa,
        depol,
        rayleigh_fraction,
        aerosol_fraction,
        aerosol_moments,
        aerosol_interp_fraction,
        angles,
        delta_m_truncation_factor,
        dtype=dtype,
        device=device,
    )
    ssa_t = _as_tensor(ssa, dtype=dtype, device=device)
    ray_t = _as_tensor(rayleigh_fraction, dtype=dtype, device=device)
    factor_t = _as_tensor(delta_m_truncation_factor, dtype=dtype, device=device)
    ssa_b, ray_b, factor_b = torch.broadcast_tensors(ssa_t, ray_t, factor_t)
    if ssa_b.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    if not bool(torch.isfinite(ssa_b).all() and torch.isfinite(ray_b).all()):
        raise ValueError("ssa and rayleigh_fraction must be finite")
    validate_delta_m_truncation_factor_torch(factor_b, ssa_b)

    lead_shape = tuple(ssa_b.shape[:-1])
    depol_b = _broadcast_leading("depol", depol, lead_shape, dtype=dtype, device=device)
    fac_b = _broadcast_leading(
        "aerosol_interp_fraction", aerosol_interp_fraction, lead_shape, dtype=dtype, device=device
    )
    cos_scatter = _solar_obs_scattering_cosines(angles, dtype=dtype, device=device)
    delta = 2.0 * (1.0 - depol_b) / (2.0 + depol_b)
    raypf = delta[..., None] * 0.75 * (1.0 + cos_scatter * cos_scatter) + 1.0 - delta[..., None]
    phase_total = ray_b[..., :, None] * raypf[..., None, :]

    moments = _aerosol_moments(aerosol_moments, dtype=dtype, device=device)
    aer_frac = _as_tensor(aerosol_fraction, dtype=dtype, device=device)
    if not bool(torch.isfinite(aer_frac).all()):
        raise ValueError("aerosol_fraction must be finite")
    if int(moments.shape[2]) != int(aer_frac.shape[-1]):
        raise ValueError("aerosol_fraction and aerosol_moments disagree on naerosol")
    aer_frac_b = torch.broadcast_to(aer_frac, tuple(ssa_b.shape) + (int(moments.shape[2]),))
    aerosol_phase_endpoints = _aerosol_phase_endpoints(moments, cos_scatter)
    aerosol_phase = aerosol_phase_endpoints[0] + fac_b[..., None, None] * (
        aerosol_phase_endpoints[1] - aerosol_phase_endpoints[0]
    )
    phase_total = phase_total + torch.einsum("...la,...ag->...lg", aer_frac_b, aerosol_phase)

    scatter = phase_total * ssa_b[..., None] / (1.0 - factor_b[..., None] * ssa_b[..., None])
    if int(cos_scatter.numel()) == 1:
        return scatter[..., 0]
    return scatter
