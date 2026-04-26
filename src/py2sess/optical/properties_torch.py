"""Torch optical-property builders from component optical depths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .phase_torch import _as_tensor, _context, ssa_from_optical_depth_torch


@dataclass(frozen=True)
class LayerOpticalPropertiesTorch:
    """Torch layer RT inputs derived from component optical depths."""

    tau: Any
    ssa: Any
    rayleigh_fraction: Any
    aerosol_fraction: Any


def _finite_nonnegative(name: str, value) -> None:
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if bool((value < 0.0).any()):
        raise ValueError(f"{name} must be nonnegative")


def _broadcast_to_shape(name: str, value, shape: tuple[int, ...], *, dtype, device):
    tensor = _as_tensor(value, dtype=dtype, device=device)
    _finite_nonnegative(name, tensor)
    try:
        return tensor.broadcast_to(shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to shape {shape}") from exc


def _aerosol_tensor(name: str, value, layer_shape: tuple[int, ...], *, dtype, device):
    tensor = _as_tensor(value, dtype=dtype, device=device)
    _finite_nonnegative(name, tensor)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must include an aerosol axis")
    if tuple(tensor.shape) == layer_shape:
        raise ValueError(
            f"{name} must include an aerosol axis; use shape {layer_shape + (1,)} "
            "for a single aerosol component"
        )
    target = layer_shape + (int(tensor.shape[-1]),)
    try:
        return tensor.broadcast_to(target)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to shape {target}") from exc


def _aerosol_property_tensor(
    name: str,
    value,
    layer_shape: tuple[int, ...],
    naerosol: int,
    *,
    dtype,
    device,
):
    tensor = _as_tensor(value, dtype=dtype, device=device)
    _finite_nonnegative(name, tensor)
    if tensor.ndim == 1 and int(tensor.shape[0]) == naerosol:
        tensor = tensor.reshape((1,) * len(layer_shape) + (naerosol,))
    target = layer_shape + (naerosol,)
    try:
        return tensor.broadcast_to(target)
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to shape {target}") from exc


def _resolve_layer_shape(torch_module, *values) -> tuple[int, ...]:
    shapes = []
    for value in values:
        if value is None:
            continue
        shapes.append(tuple(value) if isinstance(value, tuple) else tuple(value.shape))
    layer_shape = tuple(torch_module.broadcast_shapes(*shapes)) if shapes else ()
    if not layer_shape:
        raise ValueError("optical-depth inputs must include a layer axis")
    return layer_shape


def _resolve_aerosol_scattering(
    *,
    aerosol_extinction_tau,
    aerosol_scattering_tau,
    aerosol_single_scattering_albedo,
    layer_shape: tuple[int, ...],
    dtype,
    device,
):
    if aerosol_scattering_tau is not None and aerosol_single_scattering_albedo is not None:
        raise ValueError(
            "pass only one of aerosol_scattering_tau or aerosol_single_scattering_albedo"
        )
    if aerosol_scattering_tau is not None:
        scattering = _aerosol_tensor(
            "aerosol_scattering_tau",
            aerosol_scattering_tau,
            layer_shape,
            dtype=dtype,
            device=device,
        )
    elif aerosol_single_scattering_albedo is not None:
        aerosol_ssa = _aerosol_property_tensor(
            "aerosol_single_scattering_albedo",
            aerosol_single_scattering_albedo,
            layer_shape,
            int(aerosol_extinction_tau.shape[-1]),
            dtype=dtype,
            device=device,
        )
        if bool((aerosol_ssa > 1.0).any()):
            raise ValueError("aerosol_single_scattering_albedo must be <= 1")
        scattering = aerosol_extinction_tau * aerosol_ssa
    else:
        raise ValueError(
            "aerosol_extinction_tau requires aerosol_scattering_tau or "
            "aerosol_single_scattering_albedo"
        )
    if bool((scattering > aerosol_extinction_tau + 1.0e-14).any()):
        raise ValueError("aerosol_scattering_tau must not exceed aerosol_extinction_tau")
    return scattering


def build_layer_optical_properties_torch(
    *,
    gas_absorption_tau=0.0,
    rayleigh_scattering_tau=0.0,
    aerosol_extinction_tau=None,
    aerosol_scattering_tau=None,
    aerosol_single_scattering_albedo=None,
    dtype=None,
    device=None,
) -> LayerOpticalPropertiesTorch:
    """Builds differentiable RT optical inputs from component optical depths."""
    import torch

    dtype, device = _context(
        gas_absorption_tau,
        rayleigh_scattering_tau,
        aerosol_extinction_tau,
        aerosol_scattering_tau,
        aerosol_single_scattering_albedo,
        dtype=dtype,
        device=device,
    )
    aerosol_leading = None
    if aerosol_extinction_tau is not None:
        aerosol_ext = _as_tensor(aerosol_extinction_tau, dtype=dtype, device=device)
        if aerosol_ext.ndim == 0:
            raise ValueError("aerosol_extinction_tau must include an aerosol axis")
        aerosol_leading = tuple(aerosol_ext.shape[:-1])
    else:
        if aerosol_scattering_tau is not None or aerosol_single_scattering_albedo is not None:
            raise ValueError("aerosol scattering inputs require aerosol_extinction_tau")
        aerosol_ext = None

    gas_raw = _as_tensor(gas_absorption_tau, dtype=dtype, device=device)
    ray_raw = _as_tensor(rayleigh_scattering_tau, dtype=dtype, device=device)
    layer_shape = _resolve_layer_shape(torch, gas_raw, ray_raw, aerosol_leading)
    gas_tau = _broadcast_to_shape(
        "gas_absorption_tau",
        gas_raw,
        layer_shape,
        dtype=dtype,
        device=device,
    )
    ray_tau = _broadcast_to_shape(
        "rayleigh_scattering_tau",
        ray_raw,
        layer_shape,
        dtype=dtype,
        device=device,
    )

    if aerosol_ext is None:
        aerosol_ext_b = torch.zeros(layer_shape + (0,), dtype=dtype, device=device)
        aerosol_scat_b = aerosol_ext_b
    else:
        aerosol_ext_b = _aerosol_tensor(
            "aerosol_extinction_tau",
            aerosol_ext,
            layer_shape,
            dtype=dtype,
            device=device,
        )
        aerosol_scat_b = _resolve_aerosol_scattering(
            aerosol_extinction_tau=aerosol_ext_b,
            aerosol_scattering_tau=aerosol_scattering_tau,
            aerosol_single_scattering_albedo=aerosol_single_scattering_albedo,
            layer_shape=layer_shape,
            dtype=dtype,
            device=device,
        )

    total_tau = gas_tau + ray_tau + aerosol_ext_b.sum(dim=-1)
    scattering_tau = ray_tau + aerosol_scat_b.sum(dim=-1)
    ssa = ssa_from_optical_depth_torch(total_tau, scattering_tau, dtype=dtype, device=device)
    rayleigh_fraction = ssa_from_optical_depth_torch(
        scattering_tau,
        ray_tau,
        dtype=dtype,
        device=device,
    )
    aerosol_fraction = torch.where(
        scattering_tau[..., None] > 0.0,
        aerosol_scat_b / scattering_tau[..., None],
        torch.zeros_like(aerosol_scat_b),
    )
    return LayerOpticalPropertiesTorch(
        tau=total_tau,
        ssa=ssa,
        rayleigh_fraction=rayleigh_fraction,
        aerosol_fraction=aerosol_fraction,
    )
