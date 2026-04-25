"""Compatibility imports for torch optical-property helpers."""

from ..optical.delta_m_torch import (
    default_delta_m_truncation_factor_torch,
    delta_m_scale_optical_properties_torch,
    validate_delta_m_truncation_factor_torch,
)

__all__ = [
    "default_delta_m_truncation_factor_torch",
    "delta_m_scale_optical_properties_torch",
    "validate_delta_m_truncation_factor_torch",
]
