"""Public package exports for the Python 2S-ESS forward-model port.

The exports are loaded lazily so importing a NumPy-only core module does not
also import the high-level solver API or optional torch backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .api import TwoStreamEss, TwoStreamEssOptions, TwoStreamEssResult
    from .core.fo_solar_obs import FoSolarObsResult
    from .core.fo_thermal import FoThermalResult
    from .core.surface_leaving import (
        SurfaceLeavingCoefficients,
        morcasiwat_reflectance,
        seawater_refractive_index,
        surface_leaving_from_water,
    )
    from .core.thermal_source import (
        ThermalSourceInputs,
        planck_radiance_wavelength,
        planck_radiance_wavenumber,
        planck_radiance_wavenumber_band,
        thermal_source_from_temperature_profile,
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
    "FoSolarObsResult",
    "FoThermalResult",
    "ThermalSourceInputs",
    "planck_radiance_wavelength",
    "planck_radiance_wavenumber",
    "planck_radiance_wavenumber_band",
    "thermal_source_from_temperature_profile",
    "TirBenchmarkCase",
    "UvBenchmarkCase",
    "load_tir_benchmark_case",
    "load_uv_benchmark_case",
    "SurfaceLeavingCoefficients",
    "morcasiwat_reflectance",
    "seawater_refractive_index",
    "surface_leaving_from_water",
]


def __getattr__(name: str):
    """Lazily resolves public package exports."""
    if name in {"TwoStreamEss", "TwoStreamEssOptions", "TwoStreamEssResult"}:
        from . import api

        value = getattr(api, name)
    elif name == "FoSolarObsResult":
        from .core.fo_solar_obs import FoSolarObsResult as value
    elif name == "FoThermalResult":
        from .core.fo_thermal import FoThermalResult as value
    elif name in {
        "ThermalSourceInputs",
        "planck_radiance_wavelength",
        "planck_radiance_wavenumber",
        "planck_radiance_wavenumber_band",
        "thermal_source_from_temperature_profile",
    }:
        from .core import thermal_source

        value = getattr(thermal_source, name)
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
        from .core import surface_leaving

        value = getattr(surface_leaving, name)
    else:
        raise AttributeError(f"module 'py2sess' has no attribute {name!r}")
    globals()[name] = value
    return value
