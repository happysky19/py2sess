"""Differentiable torch helpers for thermal source construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.backend import _load_torch
from .planck import (
    _BOLTZMANN_CONSTANT,
    _CM_TO_METERS,
    _LIGHT_SPEED,
    _MICRONS_TO_METERS,
    _PLANCK_CONSTANT,
)


@dataclass(frozen=True)
class ThermalSourceTorchInputs:
    """Torch blackbody source inputs derived from temperature tensors."""

    planck: Any
    surface_planck: Any


def _require_torch():
    """Returns torch or raises when the optional dependency is unavailable."""
    torch = _load_torch()
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch thermal source helpers require PyTorch")
    return torch


def _context_from_values(*, dtype, device, values):
    """Returns a dtype/device context, preferring existing tensor inputs."""
    torch = _require_torch()
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
    """Converts a value to a torch tensor without detaching tensor inputs."""
    torch = _require_torch()
    if torch.is_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(dtype=dtype, device=device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _validate_positive_tensor(name: str, value):
    """Rejects non-finite or non-positive numeric inputs before solving."""
    torch = _require_torch()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if not bool((value > 0.0).all()):
        raise ValueError(f"{name} must be strictly positive")


def planck_radiance_wavelength_torch(
    temperature_k,
    wavelength_microns,
    *,
    dtype=None,
    device=None,
    validate: bool = False,
):
    """Evaluates wavelength-form Planck radiance with torch autograd support.

    The formula and SI units match :func:`py2sess.planck_radiance_wavelength`.
    Inputs follow normal torch broadcasting rules. Runtime validation is off by
    default to avoid synchronization overhead inside optimization loops.
    """
    torch = _require_torch()
    dtype, device = _context_from_values(
        dtype=dtype,
        device=device,
        values=(temperature_k, wavelength_microns),
    )
    temperature = _as_tensor(temperature_k, dtype=dtype, device=device)
    wavelength = _as_tensor(wavelength_microns, dtype=dtype, device=device) * _MICRONS_TO_METERS
    if validate:
        _validate_positive_tensor("temperature_k", temperature)
        _validate_positive_tensor("wavelength_microns", wavelength)
    h = torch.as_tensor(_PLANCK_CONSTANT, dtype=dtype, device=device)
    c = torch.as_tensor(_LIGHT_SPEED, dtype=dtype, device=device)
    k = torch.as_tensor(_BOLTZMANN_CONSTANT, dtype=dtype, device=device)
    exponent = h * c / (wavelength * k * temperature)
    numerator = 2.0 * h * c * c
    denominator = wavelength**5 * torch.expm1(exponent)
    return numerator / denominator


def planck_radiance_wavenumber_torch(
    temperature_k,
    wavenumber_cm_inv,
    *,
    dtype=None,
    device=None,
    validate: bool = False,
):
    """Evaluates wavenumber-form Planck radiance with torch autograd support.

    The formula and SI units match :func:`py2sess.planck_radiance_wavenumber`.
    Inputs follow normal torch broadcasting rules. Runtime validation is off by
    default to avoid synchronization overhead inside optimization loops.
    """
    torch = _require_torch()
    dtype, device = _context_from_values(
        dtype=dtype,
        device=device,
        values=(temperature_k, wavenumber_cm_inv),
    )
    temperature = _as_tensor(temperature_k, dtype=dtype, device=device)
    wavenumber = _as_tensor(wavenumber_cm_inv, dtype=dtype, device=device) / _CM_TO_METERS
    if validate:
        _validate_positive_tensor("temperature_k", temperature)
        _validate_positive_tensor("wavenumber_cm_inv", wavenumber)
    h = torch.as_tensor(_PLANCK_CONSTANT, dtype=dtype, device=device)
    c = torch.as_tensor(_LIGHT_SPEED, dtype=dtype, device=device)
    k = torch.as_tensor(_BOLTZMANN_CONSTANT, dtype=dtype, device=device)
    exponent = h * c * wavenumber / (k * temperature)
    numerator = 2.0 * h * c * c * wavenumber**3
    denominator = torch.expm1(exponent)
    return numerator / denominator


def _profile_spectral_grid(spectral_coordinate, level_temperature):
    """Broadcasts a spectral grid against a one-dimensional level profile."""
    if level_temperature.ndim != 1:
        raise ValueError("level_temperature_k must be one-dimensional")
    if spectral_coordinate.ndim == 0:
        return spectral_coordinate, level_temperature
    spectral = spectral_coordinate.reshape(spectral_coordinate.shape + (1,))
    levels = level_temperature.reshape((1,) * spectral_coordinate.ndim + level_temperature.shape)
    return spectral, levels


def thermal_source_from_temperature_profile_torch(
    level_temperature_k,
    surface_temperature_k,
    *,
    wavelength_microns=None,
    wavenumber_cm_inv=None,
    dtype=None,
    device=None,
    validate: bool = False,
) -> ThermalSourceTorchInputs:
    """Builds differentiable torch thermal source inputs from temperatures.

    Exactly one spectral coordinate must be provided. If the spectral coordinate
    is a vector, ``planck`` has shape ``(n_spectral, n_levels)`` and
    ``surface_planck`` has shape ``(n_spectral,)``.
    """
    provided = sum(value is not None for value in (wavelength_microns, wavenumber_cm_inv))
    if provided != 1:
        raise ValueError("Specify exactly one spectral coordinate")
    dtype, device = _context_from_values(
        dtype=dtype,
        device=device,
        values=(level_temperature_k, surface_temperature_k, wavelength_microns, wavenumber_cm_inv),
    )
    level_temperature = _as_tensor(level_temperature_k, dtype=dtype, device=device)
    surface_temperature = _as_tensor(surface_temperature_k, dtype=dtype, device=device)
    if validate:
        _validate_positive_tensor("level_temperature_k", level_temperature)
        _validate_positive_tensor("surface_temperature_k", surface_temperature)
    if wavelength_microns is not None:
        spectral = _as_tensor(wavelength_microns, dtype=dtype, device=device)
        spectral_grid, level_grid = _profile_spectral_grid(spectral, level_temperature)
        planck = planck_radiance_wavelength_torch(
            level_grid,
            spectral_grid,
            dtype=dtype,
            device=device,
            validate=validate,
        )
        surface_planck = planck_radiance_wavelength_torch(
            surface_temperature,
            spectral,
            dtype=dtype,
            device=device,
            validate=validate,
        )
    else:
        spectral = _as_tensor(wavenumber_cm_inv, dtype=dtype, device=device)
        spectral_grid, level_grid = _profile_spectral_grid(spectral, level_temperature)
        planck = planck_radiance_wavenumber_torch(
            level_grid,
            spectral_grid,
            dtype=dtype,
            device=device,
            validate=validate,
        )
        surface_planck = planck_radiance_wavenumber_torch(
            surface_temperature,
            spectral,
            dtype=dtype,
            device=device,
            validate=validate,
        )
    return ThermalSourceTorchInputs(planck=planck, surface_planck=surface_planck)
