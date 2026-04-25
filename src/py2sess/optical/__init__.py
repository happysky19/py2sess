"""Optical-property and source helpers for py2sess."""

from .brdf_solar_obs import solar_obs_brdf_from_kernels
from .brdf_thermal import thermal_brdf_from_kernels
from .delta_m import (
    default_delta_m_truncation_factor,
    delta_m_scale_optical_properties,
    validate_delta_m_truncation_factor,
)
from .planck import (
    ThermalSourceInputs,
    planck_radiance_wavelength,
    planck_radiance_wavenumber,
    planck_radiance_wavenumber_band,
    thermal_source_from_temperature_profile,
)
from .planck_torch import (
    ThermalSourceTorchInputs,
    planck_radiance_wavelength_torch,
    planck_radiance_wavenumber_torch,
    thermal_source_from_temperature_profile_torch,
)
from .surface_leaving import (
    SurfaceLeavingCoefficients,
    morcasiwat_reflectance,
    seawater_refractive_index,
    surface_leaving_from_water,
)

__all__ = [
    "default_delta_m_truncation_factor",
    "delta_m_scale_optical_properties",
    "validate_delta_m_truncation_factor",
    "ThermalSourceInputs",
    "ThermalSourceTorchInputs",
    "planck_radiance_wavelength",
    "planck_radiance_wavelength_torch",
    "planck_radiance_wavenumber",
    "planck_radiance_wavenumber_torch",
    "planck_radiance_wavenumber_band",
    "thermal_source_from_temperature_profile",
    "thermal_source_from_temperature_profile_torch",
    "SurfaceLeavingCoefficients",
    "morcasiwat_reflectance",
    "seawater_refractive_index",
    "surface_leaving_from_water",
    "solar_obs_brdf_from_kernels",
    "thermal_brdf_from_kernels",
]
