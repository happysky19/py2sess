#!/usr/bin/env python3
"""Add physical optical-preprocessing inputs to a local full-spectrum bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def _read_aerosol_moments(handle, n_moments: int) -> np.ndarray:
    moments = np.empty((2, n_moments + 1, 5), dtype=np.float64)
    for _ in range(n_moments + 1):
        values = handle.readline().split()
        if len(values) != 11:
            raise ValueError("invalid aerosol-moment row in Fortran dump")
        index = int(values[0])
        moments[0, index] = [float(value) for value in values[1:6]]
        moments[1, index] = [float(value) for value in values[6:11]]
    return moments


def _reverse_endpoint_interp_fraction(wavelengths: np.ndarray) -> np.ndarray:
    grid = np.asarray(wavelengths, dtype=np.float64)
    if not np.all(np.isfinite(grid)):
        raise ValueError("wavelengths must be finite")
    span = grid[-1] - grid[0]
    if span == 0.0:
        return np.zeros_like(grid)
    return (grid[::-1] - grid[0]) / span


def _infer_tir_profile_path(dump_path: Path) -> Path:
    match = re.match(
        r"Dump_(?P<location>[^_]+)_(?P<date>[^_]+)_(?P<time>[^.]+)\.dat(?:_.+)?$",
        dump_path.name,
    )
    if match is None:
        raise ValueError("could not infer TIR profile file from dump name; pass --profile-file")
    profile_name = (
        f"Profiles_{match.group('location')}_20067{match.group('date')}_{match.group('time')}.dat"
    )
    return (dump_path.parent / "../../geocape_data/Profile_Data" / profile_name).resolve()


def _parse_tir_profile(profile_path: Path, *, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    surface_temperature = None
    rows: list[list[float]] = []
    with profile_path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("surfaceTemperature"):
                surface_temperature = float(stripped.split("=", maxsplit=1)[1])
                continue
            values = stripped.split()
            if values and values[0].isdigit():
                rows.append([float(value) for value in values])

    if surface_temperature is None:
        raise ValueError(f"missing surface temperature in {profile_path}")
    if len(rows) < n_layers + 1:
        raise ValueError(
            f"profile {profile_path} has {len(rows)} levels; expected at least {n_layers + 1}"
        )

    profile_temperature = np.array([row[2] for row in rows], dtype=np.float64)
    level_temperature = profile_temperature[-(n_layers + 1) :][::-1]
    if not np.all(np.isfinite(level_temperature)) or np.any(level_temperature <= 0.0):
        raise ValueError(f"profile {profile_path} contains invalid level temperatures")
    return (
        level_temperature,
        np.array([surface_temperature], dtype=np.float64),
    )


def _add_rt_equivalent_components(
    arrays: dict[str, np.ndarray],
    *,
    tau_key: str,
    ssa_key: str,
) -> None:
    required = (tau_key, ssa_key, "rayleigh_fraction", "aerosol_fraction")
    if not all(key in arrays for key in required):
        return
    if {"absorption_tau", "rayleigh_scattering_tau", "aerosol_scattering_tau"}.issubset(arrays):
        return

    tau = np.asarray(arrays[tau_key], dtype=np.float64)
    ssa = np.asarray(arrays[ssa_key], dtype=np.float64)
    rayleigh_fraction = np.asarray(arrays["rayleigh_fraction"], dtype=np.float64)
    aerosol_fraction = np.asarray(arrays["aerosol_fraction"], dtype=np.float64)
    if tau.shape != ssa.shape or tau.shape != rayleigh_fraction.shape:
        raise ValueError("tau, ssa, and rayleigh_fraction shapes must match")
    if aerosol_fraction.shape[:-1] != tau.shape:
        raise ValueError("aerosol_fraction must have shape tau.shape + (n_aerosol,)")
    if np.any(ssa < 0.0) or np.any(ssa > 1.0):
        raise ValueError("ssa must satisfy 0 <= ssa <= 1")

    scattering_tau = tau * ssa
    absorption_tau = tau - scattering_tau
    negative_roundoff = absorption_tau < 0.0
    if np.any(negative_roundoff):
        tolerance = 1.0e-12 * np.maximum(np.abs(tau), np.abs(scattering_tau))
        unrecoverable = negative_roundoff & (np.abs(absorption_tau) > tolerance)
        if np.any(unrecoverable):
            raise ValueError("derived absorption_tau is negative")
        absorption_tau = np.where(negative_roundoff, 0.0, absorption_tau)
    if np.any(absorption_tau < 0.0):  # pragma: no cover
        raise ValueError("derived absorption_tau is negative")

    arrays.setdefault("absorption_tau", absorption_tau)
    arrays.setdefault("rayleigh_scattering_tau", scattering_tau * rayleigh_fraction)
    arrays.setdefault("aerosol_scattering_tau", scattering_tau[..., None] * aerosol_fraction)


def _copy_bundle_with_optics(
    *,
    kind: str,
    bundle_path: Path,
    output_path: Path,
    optics: dict[str, np.ndarray],
) -> None:
    with np.load(bundle_path) as source:
        arrays = {key: np.array(source[key]) for key in source.files}
    n_rows = int(np.asarray(arrays["wavelengths"]).shape[0])
    if int(optics["depol"].shape[0]) != n_rows:
        if "selected_indices" not in arrays:
            raise ValueError("bundle row count does not match dump and has no selected_indices")
        indices = np.asarray(arrays["selected_indices"], dtype=int) - 1
        optics = {
            key: (value[indices] if value.shape[:1] == (int(optics["depol"].shape[0]),) else value)
            for key, value in optics.items()
        }
    arrays.update(optics)
    if kind == "uv":
        _add_rt_equivalent_components(arrays, tau_key="tau", ssa_key="omega")
    else:
        _add_rt_equivalent_components(arrays, tau_key="tau_arr", ssa_key="omega_arr")
    np.savez_compressed(output_path, **arrays)


def _parse_uv_dump(dump_path: Path) -> dict[str, np.ndarray]:
    with dump_path.open("r") as handle:
        header = handle.readline().split()
        if len(header) < 6:
            raise ValueError("invalid UV dump header")
        n_layers = int(header[0])
        n_rows = int(header[1])
        n_moments = int(header[2])
        for _ in range(n_layers + 1):
            handle.readline()
        aerosol_moments = _read_aerosol_moments(handle, n_moments)

        wavelengths = np.empty(n_rows, dtype=np.float64)
        depol = np.empty(n_rows, dtype=np.float64)
        rayleigh_fraction = np.empty((n_rows, n_layers), dtype=np.float64)
        aerosol_fraction = np.empty((n_rows, n_layers, 5), dtype=np.float64)

        for row in range(n_rows):
            spec = handle.readline().split()
            if len(spec) < 6:
                raise ValueError(f"invalid UV spectral row {row + 1}")
            wavelengths[row] = float(spec[1])
            depol[row] = float(spec[4])
            for layer in range(n_layers):
                values = handle.readline().split()
                if len(values) < 10:
                    raise ValueError(f"invalid UV layer row {row + 1}, layer {layer + 1}")
                rayleigh_fraction[row, layer] = float(values[4])
                aerosol_fraction[row, layer] = [float(value) for value in values[5:10]]
            if (row + 1) % 50_000 == 0:
                print(f"  parsed UV rows: {row + 1}/{n_rows}", flush=True)

    return {
        "wavelengths": wavelengths,
        "depol": depol,
        "rayleigh_fraction": rayleigh_fraction,
        "aerosol_fraction": aerosol_fraction,
        "aerosol_moments": aerosol_moments,
        "aerosol_interp_fraction": _reverse_endpoint_interp_fraction(wavelengths),
    }


def _parse_tir_dump(
    dump_path: Path,
    *,
    profile_file: Path | None = None,
) -> dict[str, np.ndarray]:
    with dump_path.open("r") as handle:
        header = handle.readline().split()
        if len(header) < 6:
            raise ValueError("invalid TIR dump header")
        n_layers = int(header[0])
        n_rows = int(header[1])
        n_moments = int(header[2])
        for _ in range(n_layers + 1):
            handle.readline()
        aerosol_moments = _read_aerosol_moments(handle, n_moments)

        wavelengths = np.empty(n_rows, dtype=np.float64)
        wavenumber = np.empty(n_rows, dtype=np.float64)
        depol = np.empty(n_rows, dtype=np.float64)
        rayleigh_fraction = np.empty((n_rows, n_layers), dtype=np.float64)
        aerosol_fraction = np.empty((n_rows, n_layers, 5), dtype=np.float64)

        for row in range(n_rows):
            spec = handle.readline().split()
            if len(spec) < 7:
                raise ValueError(f"invalid TIR spectral row {row + 1}")
            wavelengths[row] = float(spec[1])
            wavenumber[row] = float(spec[2])
            depol[row] = float(spec[4])
            for layer in range(n_layers):
                values = handle.readline().split()
                if len(values) < 10:
                    raise ValueError(f"invalid TIR layer row {row + 1}, layer {layer + 1}")
                rayleigh_fraction[row, layer] = float(values[4])
                aerosol_fraction[row, layer] = [float(value) for value in values[5:10]]
            if (row + 1) % 50_000 == 0:
                print(f"  parsed TIR rows: {row + 1}/{n_rows}", flush=True)

    profile_path = _infer_tir_profile_path(dump_path) if profile_file is None else profile_file
    if not profile_path.is_file():
        raise FileNotFoundError(f"TIR profile file not found: {profile_path}; pass --profile-file")
    level_temperature, surface_temperature = _parse_tir_profile(
        profile_path,
        n_layers=n_layers,
    )

    return {
        "wavelengths": wavelengths,
        "wavenumber_cm_inv": wavenumber,
        "wavenumber_band_cm_inv": np.column_stack((wavenumber - 0.5, wavenumber + 0.5)),
        "level_temperature_k": level_temperature,
        "surface_temperature_k": surface_temperature,
        "depol": depol,
        "rayleigh_fraction": rayleigh_fraction,
        "aerosol_fraction": aerosol_fraction,
        "aerosol_moments": aerosol_moments,
        "aerosol_interp_fraction": _reverse_endpoint_interp_fraction(wavelengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("uv", "tir"))
    parser.add_argument("dump", type=Path, help="Original Fortran text dump.")
    parser.add_argument("bundle", type=Path, help="Existing local benchmark bundle.")
    parser.add_argument("output", type=Path, help="Output enriched benchmark bundle.")
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=None,
        help="TIR GEOCAPE profile file. Defaults to the path implied by the dump name.",
    )
    args = parser.parse_args()

    if args.output.resolve() == args.bundle.resolve():
        raise ValueError("output must be a different path; keep the original bundle intact")
    if not args.dump.is_file():
        raise FileNotFoundError(args.dump)
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)

    optics = (
        _parse_uv_dump(args.dump)
        if args.kind == "uv"
        else _parse_tir_dump(args.dump, profile_file=args.profile_file)
    )
    _copy_bundle_with_optics(
        kind=args.kind,
        bundle_path=args.bundle,
        output_path=args.output,
        optics=optics,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
