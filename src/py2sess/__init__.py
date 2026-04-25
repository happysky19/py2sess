"""Public package exports for the Python 2S-ESS forward-model port.

The exports are loaded lazily so importing a NumPy-only core module does not
also import the high-level solver API or optional torch backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .api import TwoStreamEss, TwoStreamEssBatchResult, TwoStreamEssOptions, TwoStreamEssResult
    from .rtsolver.fo_solar_obs import FoSolarObsResult, fo_scatter_term_henyey_greenstein
    from .rtsolver.fo_solar_obs_torch import fo_scatter_term_henyey_greenstein_torch
    from .rtsolver.fo_thermal import FoThermalResult
    from .optical.surface_leaving import (
        SurfaceLeavingCoefficients,
        morcasiwat_reflectance,
        seawater_refractive_index,
        surface_leaving_from_water,
    )
    from .optical.planck import (
        ThermalSourceInputs,
        planck_radiance_wavelength,
        planck_radiance_wavenumber,
        planck_radiance_wavenumber_band,
        thermal_source_from_temperature_profile,
    )
    from .optical.planck_torch import (
        ThermalSourceTorchInputs,
        planck_radiance_wavelength_torch,
        planck_radiance_wavenumber_torch,
        thermal_source_from_temperature_profile_torch,
    )
    from .retrieval import (
        NoiseModel,
        OptimalEstimationProblem,
        OptimalEstimationResult,
        RetrievalDiagnostics,
        evaluate_jacobian,
        retrieval_diagnostics,
        solve_optimal_estimation,
    )
    from .reference_cases import (
        TirBenchmarkCase,
        UvBenchmarkCase,
        load_tir_benchmark_case,
        load_uv_benchmark_case,
    )

__all__ = [
    "TwoStreamEss",
    "TwoStreamEssOptions",
    "TwoStreamEssResult",
    "TwoStreamEssBatchResult",
    "FoSolarObsResult",
    "FoThermalResult",
    "fo_scatter_term_henyey_greenstein",
    "fo_scatter_term_henyey_greenstein_torch",
    "ThermalSourceInputs",
    "ThermalSourceTorchInputs",
    "planck_radiance_wavelength",
    "planck_radiance_wavelength_torch",
    "planck_radiance_wavenumber",
    "planck_radiance_wavenumber_torch",
    "planck_radiance_wavenumber_band",
    "thermal_source_from_temperature_profile",
    "thermal_source_from_temperature_profile_torch",
    "TirBenchmarkCase",
    "UvBenchmarkCase",
    "load_tir_benchmark_case",
    "load_uv_benchmark_case",
    "SurfaceLeavingCoefficients",
    "morcasiwat_reflectance",
    "seawater_refractive_index",
    "surface_leaving_from_water",
    "NoiseModel",
    "OptimalEstimationProblem",
    "OptimalEstimationResult",
    "RetrievalDiagnostics",
    "evaluate_jacobian",
    "retrieval_diagnostics",
    "solve_optimal_estimation",
]


def __getattr__(name: str):
    """Lazily resolves public package exports."""
    if name in {
        "TwoStreamEss",
        "TwoStreamEssOptions",
        "TwoStreamEssResult",
        "TwoStreamEssBatchResult",
    }:
        from . import api

        value = getattr(api, name)
    elif name == "FoSolarObsResult":
        from .rtsolver.fo_solar_obs import FoSolarObsResult as value
    elif name == "fo_scatter_term_henyey_greenstein":
        from .rtsolver.fo_solar_obs import fo_scatter_term_henyey_greenstein as value
    elif name == "fo_scatter_term_henyey_greenstein_torch":
        from .rtsolver.fo_solar_obs_torch import fo_scatter_term_henyey_greenstein_torch as value
    elif name == "FoThermalResult":
        from .rtsolver.fo_thermal import FoThermalResult as value
    elif name in {
        "ThermalSourceInputs",
        "ThermalSourceTorchInputs",
        "planck_radiance_wavelength",
        "planck_radiance_wavelength_torch",
        "planck_radiance_wavenumber",
        "planck_radiance_wavenumber_torch",
        "planck_radiance_wavenumber_band",
        "thermal_source_from_temperature_profile",
        "thermal_source_from_temperature_profile_torch",
    }:
        from importlib import import_module

        torch_names = {
            "ThermalSourceTorchInputs",
            "planck_radiance_wavelength_torch",
            "planck_radiance_wavenumber_torch",
            "thermal_source_from_temperature_profile_torch",
        }
        module = import_module(
            ".optical.planck_torch" if name in torch_names else ".optical.planck",
            __name__,
        )

        value = getattr(module, name)
    elif name in {
        "TirBenchmarkCase",
        "UvBenchmarkCase",
        "load_tir_benchmark_case",
        "load_uv_benchmark_case",
    }:
        from . import reference_cases

        value = getattr(reference_cases, name)
    elif name in {
        "SurfaceLeavingCoefficients",
        "morcasiwat_reflectance",
        "seawater_refractive_index",
        "surface_leaving_from_water",
    }:
        from .optical import surface_leaving

        value = getattr(surface_leaving, name)
    elif name in {
        "NoiseModel",
        "OptimalEstimationProblem",
        "OptimalEstimationResult",
        "RetrievalDiagnostics",
        "evaluate_jacobian",
        "retrieval_diagnostics",
        "solve_optimal_estimation",
    }:
        from . import retrieval

        value = getattr(retrieval, name)
    else:
        raise AttributeError(f"module 'py2sess' has no attribute {name!r}")
    globals()[name] = value
    return value
