"""Helpers for reshaping lattice-style outputs."""

from __future__ import annotations

from typing import Any


def lattice_shape(lattice_counts: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """Returns the lattice shape or raises when it is unavailable."""
    if lattice_counts is None:
        raise ValueError("lattice_counts are not available for this result")
    return lattice_counts


def reshape_lattice_array(values, lattice_counts: tuple[int, int, int] | None):
    """Reshapes a flat lattice result into beam/angle/azimuth form."""
    return values.reshape(lattice_shape(lattice_counts))


def add_lattice_axes(
    mapping: dict[str, Any],
    lattice_axes,
) -> dict[str, Any]:
    """Attaches lattice axis arrays to a reshaped result mapping."""
    if lattice_axes is None:
        return mapping

    mapping["beam_szas"] = lattice_axes[0]
    mapping["user_angles"] = lattice_axes[1]
    mapping["user_relazms"] = lattice_axes[2]
    return mapping
