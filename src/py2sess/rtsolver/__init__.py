"""Radiative-transfer kernels and numerical helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PreparedGeometry",
    "PreparedInputs",
    "ThermalBatchNumpyResult",
    "fo_scatter_term_henyey_greenstein",
    "fo_scatter_term_henyey_greenstein_torch",
    "precompute_fo_thermal_geometry_numpy",
    "prepare_inputs",
    "solve_solar_observation_bvp_batch",
    "solve_fo_solar_obs_eps_batch_torch",
    "solve_solar_obs_batch_numpy",
    "solve_solar_obs_batch_torch",
    "solve_thermal_bvp_batch",
    "solve_thermal_batch_numpy",
]

_LAZY_EXPORTS = {
    "PreparedGeometry": (".preprocess", "PreparedGeometry"),
    "PreparedInputs": (".preprocess", "PreparedInputs"),
    "ThermalBatchNumpyResult": (".thermal_batch_numpy", "ThermalBatchNumpyResult"),
    "fo_scatter_term_henyey_greenstein": (
        ".fo_solar_obs",
        "fo_scatter_term_henyey_greenstein",
    ),
    "fo_scatter_term_henyey_greenstein_torch": (
        ".fo_solar_obs_torch",
        "fo_scatter_term_henyey_greenstein_torch",
    ),
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
}


def __getattr__(name: str):
    """Lazily resolves package-level exports on first access."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'py2sess.rtsolver' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
