"""Benchmark reference adapters for optional external solvers."""

from __future__ import annotations

__all__ = [
    "KINETICS_FLXOUT_COLUMNS",
    "PYDISORT_FLUX_CHANNELS",
    "kinetics_flux_to_py2sess",
    "parse_kinetics_flux_table",
    "pydisort_flux_to_py2sess",
    "rayleigh_phase_moments",
    "run_pydisort_absorbing_solar_flux",
    "run_pydisort_solar_flux",
    "solar_isotropic_single_scatter_flux",
    "solar_isotropic_twostream_single_scatter_flux",
    "solar_rayleigh_single_scatter_flux",
    "solar_rayleigh_twostream_single_scatter_flux",
]


def __getattr__(name: str):
    if name in {
        "PYDISORT_FLUX_CHANNELS",
        "pydisort_flux_to_py2sess",
        "rayleigh_phase_moments",
        "run_pydisort_absorbing_solar_flux",
        "run_pydisort_solar_flux",
        "solar_isotropic_single_scatter_flux",
        "solar_isotropic_twostream_single_scatter_flux",
        "solar_rayleigh_single_scatter_flux",
        "solar_rayleigh_twostream_single_scatter_flux",
    }:
        from . import flux_references

        value = getattr(flux_references, name)
    elif name in {
        "KINETICS_FLXOUT_COLUMNS",
        "kinetics_flux_to_py2sess",
        "parse_kinetics_flux_table",
    }:
        from . import kinetics_flux

        value = getattr(kinetics_flux, name)
    else:
        raise AttributeError(f"module 'py2sess.benchmarks' has no attribute {name!r}")
    globals()[name] = value
    return value
