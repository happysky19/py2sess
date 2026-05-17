"""Optical-property and source helpers for py2sess."""

from .brdf_solar_obs import solar_obs_brdf_from_kernels
from .brdf_thermal import thermal_brdf_from_kernels
from .delta_m import (
    default_delta_m_truncation_factor,
    delta_m_scale_optical_properties,
    validate_delta_m_truncation_factor,
)
from .phase import (
    TwoStreamPhaseInputs,
    aerosol_interp_fraction,
    build_solar_fo_scatter_term,
    build_two_stream_phase_inputs,
    ssa_from_optical_depth,
)
from .phase_torch import (
    TwoStreamPhaseTorchInputs,
    aerosol_interp_fraction_torch,
    build_solar_fo_scatter_term_torch,
    build_two_stream_phase_inputs_torch,
    ssa_from_optical_depth_torch,
)
from .properties import LayerOpticalProperties, build_layer_optical_properties
from .properties_torch import LayerOpticalPropertiesTorch, build_layer_optical_properties_torch
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
from .solar_reference import (
    ASTRONOMICAL_UNIT_M,
    IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
    IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2,
    OcoSolarModel,
    ToonSolarReference,
    solar_planck_irradiance_w_m2_um,
    solar_planck_continuum_ratio,
)

__all__ = [
    "default_delta_m_truncation_factor",
    "delta_m_scale_optical_properties",
    "validate_delta_m_truncation_factor",
    "TwoStreamPhaseInputs",
    "TwoStreamPhaseTorchInputs",
    "aerosol_interp_fraction",
    "aerosol_interp_fraction_torch",
    "build_solar_fo_scatter_term",
    "build_solar_fo_scatter_term_torch",
    "build_two_stream_phase_inputs",
    "build_two_stream_phase_inputs_torch",
    "ssa_from_optical_depth",
    "ssa_from_optical_depth_torch",
    "LayerOpticalProperties",
    "LayerOpticalPropertiesTorch",
    "build_layer_optical_properties",
    "build_layer_optical_properties_torch",
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
    "ASTRONOMICAL_UNIT_M",
    "IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K",
    "IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2",
    "OcoSolarModel",
    "ToonSolarReference",
    "solar_planck_irradiance_w_m2_um",
    "solar_planck_continuum_ratio",
    "solar_obs_brdf_from_kernels",
    "thermal_brdf_from_kernels",
]
