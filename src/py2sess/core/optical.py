"""Compatibility imports for optical-property helpers."""

from ..optical.delta_m import (
    default_delta_m_truncation_factor,
    delta_m_scale_optical_properties,
    validate_delta_m_truncation_factor,
)

__all__ = [
    "default_delta_m_truncation_factor",
    "delta_m_scale_optical_properties",
    "validate_delta_m_truncation_factor",
]
