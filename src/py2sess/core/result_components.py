"""Helpers for building grouped result views."""

from __future__ import annotations

from typing import Any


def build_solar_components(
    *,
    intensity_toa,
    intensity_boa,
    fo_intensity_total,
    fo_intensity_ss,
    fo_intensity_db,
    combined_intensity_toa,
) -> dict[str, Any]:
    """Builds the grouped solar component view for a result object."""
    return {
        "twostream_toa": intensity_toa,
        "twostream_boa": intensity_boa,
        "fo_total": fo_intensity_total,
        "fo_ss": fo_intensity_ss,
        "fo_db": fo_intensity_db,
        "combined_toa": combined_intensity_toa,
    }


def build_thermal_components(
    *,
    intensity_toa,
    intensity_boa,
    fo_mu1,
    fo_thermal_atmos_up_toa,
    fo_thermal_surface_toa,
    fo_thermal_total_up_toa,
    fo_thermal_atmos_dn_toa,
    fo_thermal_atmos_up_boa,
    fo_thermal_surface_boa,
    fo_thermal_total_up_boa,
    fo_thermal_atmos_dn_boa,
) -> dict[str, Any]:
    """Builds the grouped thermal component view for a result object."""
    return {
        "twostream_toa": intensity_toa,
        "twostream_boa": intensity_boa,
        "fo_mu1": fo_mu1,
        "fo_toa_up": {
            "atmosphere": fo_thermal_atmos_up_toa,
            "surface": fo_thermal_surface_toa,
            "total": fo_thermal_total_up_toa,
        },
        "fo_toa_down_atmosphere": fo_thermal_atmos_dn_toa,
        "fo_boa_up": {
            "atmosphere": fo_thermal_atmos_up_boa,
            "surface": fo_thermal_surface_boa,
            "total": fo_thermal_total_up_boa,
        },
        "fo_boa_down_atmosphere": fo_thermal_atmos_dn_boa,
    }
