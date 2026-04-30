"""Optical-property builders from component optical depths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phase import ssa_from_optical_depth


@dataclass(frozen=True)
class LayerOpticalProperties:
    """Layer RT inputs derived from component optical depths."""

    tau: np.ndarray
    ssa: np.ndarray
    rayleigh_fraction: np.ndarray
    aerosol_fraction: np.ndarray


def _finite_nonnegative(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if np.any(value < 0.0):
        raise ValueError(f"{name} must be nonnegative")


def _broadcast_to_shape(
    name: str,
    value,
    shape: tuple[int, ...],
    *,
    validate_inputs: bool,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if validate_inputs:
        _finite_nonnegative(name, arr)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {shape}") from exc


def _aerosol_array(
    name: str,
    value,
    layer_shape: tuple[int, ...],
    *,
    validate_inputs: bool,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if validate_inputs:
        _finite_nonnegative(name, arr)
    if arr.ndim == 0:
        raise ValueError(f"{name} must include an aerosol axis")
    if arr.shape == layer_shape:
        raise ValueError(
            f"{name} must include an aerosol axis; use shape {layer_shape + (1,)} "
            "for a single aerosol component"
        )
    target = layer_shape + (arr.shape[-1],)
    try:
        return np.broadcast_to(arr, target)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {target}") from exc


def _aerosol_property_array(
    name: str,
    value,
    layer_shape: tuple[int, ...],
    naerosol: int,
    *,
    validate_inputs: bool,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if validate_inputs:
        _finite_nonnegative(name, arr)
    if arr.ndim == 1 and arr.shape[0] == naerosol:
        arr = arr.reshape((1,) * len(layer_shape) + (naerosol,))
    target = layer_shape + (naerosol,)
    try:
        return np.broadcast_to(arr, target)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {target}") from exc


def _resolve_layer_shape(*values) -> tuple[int, ...]:
    shapes = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, tuple):
            shapes.append(value)
        else:
            shapes.append(np.asarray(value, dtype=float).shape)
    layer_shape = np.broadcast_shapes(*shapes) if shapes else ()
    if not layer_shape:
        raise ValueError("optical-depth inputs must include a layer axis")
    return layer_shape


def sum_component_axis(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 0:
        return np.zeros(values.shape[:-1], dtype=values.dtype)
    if values.shape[-1] <= 8:
        total = values[..., 0].copy()
        for index in range(1, values.shape[-1]):
            total += values[..., index]
        return total
    return np.sum(values, axis=-1)


def _resolve_aerosol_scattering(
    *,
    aerosol_extinction_tau: np.ndarray,
    aerosol_scattering_tau,
    aerosol_single_scattering_albedo,
    layer_shape: tuple[int, ...],
    validate_inputs: bool,
) -> np.ndarray:
    if aerosol_scattering_tau is not None and aerosol_single_scattering_albedo is not None:
        raise ValueError(
            "pass only one of aerosol_scattering_tau or aerosol_single_scattering_albedo"
        )
    if aerosol_scattering_tau is not None:
        scattering = _aerosol_array(
            "aerosol_scattering_tau",
            aerosol_scattering_tau,
            layer_shape,
            validate_inputs=validate_inputs,
        )
    elif aerosol_single_scattering_albedo is not None:
        aerosol_ssa = _aerosol_property_array(
            "aerosol_single_scattering_albedo",
            aerosol_single_scattering_albedo,
            layer_shape,
            aerosol_extinction_tau.shape[-1],
            validate_inputs=validate_inputs,
        )
        if validate_inputs and np.any(aerosol_ssa > 1.0):
            raise ValueError("aerosol_single_scattering_albedo must be <= 1")
        scattering = aerosol_extinction_tau * aerosol_ssa
    else:
        raise ValueError(
            "aerosol_extinction_tau requires aerosol_scattering_tau or "
            "aerosol_single_scattering_albedo"
        )
    if validate_inputs and np.any(scattering > aerosol_extinction_tau + 1.0e-14):
        raise ValueError("aerosol_scattering_tau must not exceed aerosol_extinction_tau")
    return scattering


def build_layer_optical_properties(
    *,
    absorption_tau=None,
    gas_absorption_tau=None,
    rayleigh_scattering_tau=0.0,
    aerosol_extinction_tau=None,
    aerosol_scattering_tau=None,
    aerosol_single_scattering_albedo=None,
    validate_inputs: bool = True,
) -> LayerOpticalProperties:
    """Build RT optical inputs from layer optical-depth components."""
    if absorption_tau is not None and gas_absorption_tau is not None:
        raise ValueError("pass only one of absorption_tau or gas_absorption_tau")
    if absorption_tau is None:
        absorption_tau = 0.0 if gas_absorption_tau is None else gas_absorption_tau

    if aerosol_extinction_tau is not None:
        aerosol_ext = np.asarray(aerosol_extinction_tau, dtype=float)
        if aerosol_ext.ndim == 0:
            raise ValueError("aerosol_extinction_tau must include an aerosol axis")
        aerosol_leading = aerosol_ext.shape[:-1]
    elif aerosol_scattering_tau is not None:
        aerosol_scat = np.asarray(aerosol_scattering_tau, dtype=float)
        if aerosol_scat.ndim == 0:
            raise ValueError("aerosol_scattering_tau must include an aerosol axis")
        aerosol_ext = None
        aerosol_leading = aerosol_scat.shape[:-1]
    else:
        aerosol_ext = None
        aerosol_leading = None
        if aerosol_single_scattering_albedo is not None:
            raise ValueError("aerosol_single_scattering_albedo requires aerosol_extinction_tau")

    layer_shape = _resolve_layer_shape(
        absorption_tau,
        rayleigh_scattering_tau,
        aerosol_leading,
    )
    absorption = _broadcast_to_shape(
        "absorption_tau",
        absorption_tau,
        layer_shape,
        validate_inputs=validate_inputs,
    )
    ray_tau = _broadcast_to_shape(
        "rayleigh_scattering_tau",
        rayleigh_scattering_tau,
        layer_shape,
        validate_inputs=validate_inputs,
    )

    if aerosol_ext is None:
        if aerosol_scattering_tau is None:
            aerosol_ext_b = np.zeros(layer_shape + (0,), dtype=float)
            aerosol_scat_b = aerosol_ext_b
        else:
            aerosol_scat_b = _aerosol_array(
                "aerosol_scattering_tau",
                aerosol_scattering_tau,
                layer_shape,
                validate_inputs=validate_inputs,
            )
            aerosol_ext_b = aerosol_scat_b
    else:
        aerosol_ext_b = _aerosol_array(
            "aerosol_extinction_tau",
            aerosol_ext,
            layer_shape,
            validate_inputs=validate_inputs,
        )
        aerosol_scat_b = _resolve_aerosol_scattering(
            aerosol_extinction_tau=aerosol_ext_b,
            aerosol_scattering_tau=aerosol_scattering_tau,
            aerosol_single_scattering_albedo=aerosol_single_scattering_albedo,
            layer_shape=layer_shape,
            validate_inputs=validate_inputs,
        )

    aerosol_ext_sum = sum_component_axis(aerosol_ext_b)
    aerosol_scat_sum = (
        aerosol_ext_sum if aerosol_scat_b is aerosol_ext_b else sum_component_axis(aerosol_scat_b)
    )
    total_tau = absorption + ray_tau + aerosol_ext_sum
    scattering_tau = ray_tau + aerosol_scat_sum
    ssa = ssa_from_optical_depth(total_tau, scattering_tau)
    rayleigh_fraction = ssa_from_optical_depth(scattering_tau, ray_tau)
    scattering_positive = scattering_tau > 0.0
    aerosol_fraction = np.empty_like(aerosol_scat_b, dtype=float)
    np.divide(
        aerosol_scat_b,
        scattering_tau[..., None],
        out=aerosol_fraction,
        where=scattering_positive[..., None],
    )
    if not np.all(scattering_positive):
        aerosol_fraction[~scattering_positive] = 0.0
    return LayerOpticalProperties(
        tau=total_tau,
        ssa=ssa,
        rayleigh_fraction=rayleigh_fraction,
        aerosol_fraction=aerosol_fraction,
    )
