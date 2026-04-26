#!/usr/bin/env python3
"""Create a local benchmark bundle that keeps only runtime inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


_ABSORPTION_KEYS = ("absorption_tau", "gas_absorption_tau")
_AEROSOL_EXTINCTION_KEY = "aerosol_extinction_tau"
_AEROSOL_SCATTERING_KEY = "aerosol_scattering_tau"
_AEROSOL_SSA_KEY = "aerosol_single_scattering_albedo"
_RAYLEIGH_KEY = "rayleigh_scattering_tau"

_UV_DIRECT_LAYER_KEYS = {"tau", "omega"}
_UV_DUMPED_OPTICS_KEYS = {"asymm", "scaling", "fo_exact_scatter"}
_UV_GEOMETRY_KEYS = {
    "chapman",
    "x0",
    "user_stream",
    "user_secant",
    "azmfac",
    "px11",
    "pxsq",
    "px0x",
    "ulp",
}

_TIR_DIRECT_LAYER_KEYS = {"tau_arr", "omega_arr"}
_TIR_DUMPED_OPTICS_KEYS = {"asymm_arr", "d2s_scaling"}
_TIR_DIRECT_SOURCE_KEYS = {"thermal_bb_input", "surfbb"}
_TIR_TEMPERATURE_SOURCE_KEYS = {"level_temperature_k", "surface_temperature_k"}
_TIR_SOURCE_COORDINATE_KEYS = {
    "wavenumber_band_cm_inv",
    "wavenumber_cm_inv",
    "wavelength_microns",
}
_TIR_AEROSOL_COORDINATE_KEYS = {"wavenumber_cm_inv", "wavelength_microns"}


def _looks_like_row_index(values: np.ndarray) -> bool:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        return False
    return np.allclose(grid, np.arange(1, grid.size + 1, dtype=float), rtol=0.0, atol=1.0e-12)


def _has_layer_components(keys: set[str]) -> bool:
    has_absorption = any(key in keys for key in _ABSORPTION_KEYS)
    if not has_absorption or _RAYLEIGH_KEY not in keys:
        return False
    if _AEROSOL_EXTINCTION_KEY in keys:
        return _AEROSOL_SCATTERING_KEY in keys or _AEROSOL_SSA_KEY in keys
    return True


def _has_phase_inputs(keys: set[str], *, layer_components: bool) -> bool:
    required = {"depol", "aerosol_moments"}
    if not layer_components:
        required.update({"rayleigh_fraction", "aerosol_fraction"})
    return required.issubset(keys)


def _can_derive_uv_aerosol_interp(arrays: dict[str, np.ndarray]) -> bool:
    if "wavelengths" not in arrays:
        return False
    return not _looks_like_row_index(arrays["wavelengths"])


def _can_derive_tir_aerosol_interp(arrays: dict[str, np.ndarray]) -> bool:
    if "wavelengths" in arrays and not _looks_like_row_index(arrays["wavelengths"]):
        return True
    return any(key in arrays for key in _TIR_AEROSOL_COORDINATE_KEYS)


def minimal_bundle_arrays(kind: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy with dumped fields removed when Python can rebuild them."""
    if kind not in {"uv", "tir"}:
        raise ValueError("kind must be 'uv' or 'tir'")

    keys = set(arrays)
    layer_components = _has_layer_components(keys)
    phase_inputs = _has_phase_inputs(keys, layer_components=layer_components)
    keep = dict(arrays)

    if kind == "uv":
        for key in _UV_GEOMETRY_KEYS:
            keep.pop(key, None)
        if layer_components:
            for key in _UV_DIRECT_LAYER_KEYS | {"rayleigh_fraction", "aerosol_fraction"}:
                keep.pop(key, None)
        if phase_inputs:
            for key in _UV_DUMPED_OPTICS_KEYS:
                keep.pop(key, None)
            if _can_derive_uv_aerosol_interp(keep):
                keep.pop("aerosol_interp_fraction", None)
        return keep

    if layer_components:
        for key in _TIR_DIRECT_LAYER_KEYS | {"rayleigh_fraction", "aerosol_fraction"}:
            keep.pop(key, None)
    if phase_inputs:
        for key in _TIR_DUMPED_OPTICS_KEYS:
            keep.pop(key, None)
        if _can_derive_tir_aerosol_interp(keep):
            keep.pop("aerosol_interp_fraction", None)
    if _TIR_TEMPERATURE_SOURCE_KEYS.issubset(keys) and keys.intersection(
        _TIR_SOURCE_COORDINATE_KEYS
    ):
        for key in _TIR_DIRECT_SOURCE_KEYS:
            keep.pop(key, None)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["uv", "tir"])
    parser.add_argument("input", type=Path, help="Input enriched benchmark bundle.")
    parser.add_argument("output", type=Path, help="Output runtime-minimal benchmark bundle.")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)

    with np.load(args.input) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    minimal = minimal_bundle_arrays(args.kind, arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **minimal)
    removed = sorted(set(arrays) - set(minimal))
    print(f"wrote {args.output}")
    print(f"kept {len(minimal)} arrays, removed {len(removed)} arrays")
    if removed:
        print("removed: " + ", ".join(removed))


if __name__ == "__main__":
    main()
