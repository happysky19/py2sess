"""Core numerical helpers exported for advanced py2sess users."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PreparedGeometry",
    "PreparedInputs",
    "ThermalSourceInputs",
    "ThermalSourceTorchInputs",
    "ThermalBatchNumpyResult",
    "fo_scatter_term_henyey_greenstein",
    "fo_scatter_term_henyey_greenstein_torch",
    "planck_radiance_wavelength",
    "planck_radiance_wavelength_torch",
    "planck_radiance_wavenumber",
    "planck_radiance_wavenumber_torch",
    "planck_radiance_wavenumber_band",
    "precompute_fo_thermal_geometry_numpy",
    "prepare_inputs",
    "solve_solar_observation_bvp_batch",
    "solve_fo_solar_obs_eps_batch_torch",
    "solve_solar_obs_batch_numpy",
    "solve_solar_obs_batch_torch",
    "solve_thermal_bvp_batch",
    "solve_thermal_batch_numpy",
    "thermal_source_from_temperature_profile",
    "thermal_source_from_temperature_profile_torch",
    "SurfaceLeavingCoefficients",
    "morcasiwat_reflectance",
    "seawater_refractive_index",
    "surface_leaving_from_water",
]

_LAZY_EXPORTS = {
    "PreparedGeometry": (".preprocess", "PreparedGeometry"),
    "PreparedInputs": (".preprocess", "PreparedInputs"),
    "ThermalSourceInputs": (".thermal_source", "ThermalSourceInputs"),
    "ThermalSourceTorchInputs": (".thermal_source_torch", "ThermalSourceTorchInputs"),
    "ThermalBatchNumpyResult": (".thermal_batch_numpy", "ThermalBatchNumpyResult"),
    "fo_scatter_term_henyey_greenstein": (
        ".fo_solar_obs",
        "fo_scatter_term_henyey_greenstein",
    ),
    "fo_scatter_term_henyey_greenstein_torch": (
        ".fo_solar_obs_torch",
        "fo_scatter_term_henyey_greenstein_torch",
    ),
    "planck_radiance_wavelength": (".thermal_source", "planck_radiance_wavelength"),
    "planck_radiance_wavelength_torch": (
        ".thermal_source_torch",
        "planck_radiance_wavelength_torch",
    ),
    "planck_radiance_wavenumber": (".thermal_source", "planck_radiance_wavenumber"),
    "planck_radiance_wavenumber_torch": (
        ".thermal_source_torch",
        "planck_radiance_wavenumber_torch",
    ),
    "planck_radiance_wavenumber_band": (".thermal_source", "planck_radiance_wavenumber_band"),
    "precompute_fo_thermal_geometry_numpy": (
        ".thermal_batch_numpy",
        "precompute_fo_thermal_geometry_numpy",
    ),
    "prepare_inputs": (".preprocess", "prepare_inputs"),
    "solve_solar_observation_bvp_batch": (".bvp_batch", "solve_solar_observation_bvp_batch"),
    "solve_fo_solar_obs_eps_batch_torch": (
        ".fo_solar_obs_batch_torch",
        "solve_fo_solar_obs_eps_batch_torch",
    ),
    "solve_solar_obs_batch_numpy": (".solar_obs_batch_numpy", "solve_solar_obs_batch_numpy"),
    "solve_solar_obs_batch_torch": (".solar_obs_batch_torch", "solve_solar_obs_batch_torch"),
    "solve_thermal_bvp_batch": (".bvp_batch", "solve_thermal_bvp_batch"),
    "solve_thermal_batch_numpy": (".thermal_batch_numpy", "solve_thermal_batch_numpy"),
    "thermal_source_from_temperature_profile": (
        ".thermal_source",
        "thermal_source_from_temperature_profile",
    ),
    "thermal_source_from_temperature_profile_torch": (
        ".thermal_source_torch",
        "thermal_source_from_temperature_profile_torch",
    ),
    "SurfaceLeavingCoefficients": (".surface_leaving", "SurfaceLeavingCoefficients"),
    "morcasiwat_reflectance": (".surface_leaving", "morcasiwat_reflectance"),
    "seawater_refractive_index": (".surface_leaving", "seawater_refractive_index"),
    "surface_leaving_from_water": (".surface_leaving", "surface_leaving_from_water"),
}


def __getattr__(name: str):
    """Lazily resolves package-level exports on first access."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'py2sess.core' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
