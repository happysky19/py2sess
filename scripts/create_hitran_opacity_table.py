#!/usr/bin/env python3
"""Create a pressure-temperature gas cross-section table from HITRAN lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from py2sess.optical.hitran import (
    hitran_cross_sections,
    load_hitran_partition_functions,
    read_hitran_lines,
)
from py2sess.optical.scene_io import load_profile_text, load_scene_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile", type=Path, help="Profile text file for an exact profile table."
    )
    parser.add_argument("--scene", type=Path, help="Scene YAML used with --profile.")
    parser.add_argument("--hitran-dir", type=Path)
    parser.add_argument("--gas", action="append", help="Gas name; repeat per gas.")
    parser.add_argument("--pressure-hpa", type=float, nargs="+")
    parser.add_argument("--temperature-k", type=float, nargs="+")
    parser.add_argument("--wavenumber-file", type=Path)
    parser.add_argument("--wavenumber-start", type=float)
    parser.add_argument("--wavenumber-step", type=float)
    parser.add_argument("--wavenumber-count", type=int)
    parser.add_argument("--spectral-limit", type=int)
    parser.add_argument("--fwhm", type=float, default=0.0)
    args = parser.parse_args()

    try:
        if args.profile is not None or args.scene is not None:
            gases, hitran_dir, wavenumber, pressure, temperature, table = _profile_table(args)
        else:
            gases, hitran_dir, wavenumber, pressure, temperature, table = _lookup_table(args)
    except ValueError as exc:
        parser.error(str(exc))
    _write_table(args.output, gases, wavenumber, pressure, temperature, table)
    print(f"wrote {args.output} from {hitran_dir}")


def _profile_table(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.profile is None or args.scene is None:
        raise ValueError("--profile and --scene must be passed together")
    scene = load_scene_yaml(args.scene)
    gases = tuple(args.gas or _scene_gases(scene))
    if not gases:
        raise ValueError("scene must define gases or pass --gas")
    hitran_dir = args.hitran_dir or _scene_hitran_dir(scene, args.scene.parent)
    wavenumber = _scene_wavenumber(scene, args.scene.parent)
    if args.spectral_limit is not None:
        wavenumber = wavenumber[: args.spectral_limit]
    profile = load_profile_text(args.profile)
    table = _cross_sections(
        hitran_dir=hitran_dir,
        gases=gases,
        wavenumber=wavenumber,
        pressure_atm=profile.pressure_hpa / 1013.25,
        temperature_k=profile.temperature_k,
        fwhm=args.fwhm,
    )
    return gases, hitran_dir, wavenumber, profile.pressure_hpa, profile.temperature_k, table


def _lookup_table(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [
        name
        for name in ("hitran_dir", "gas", "pressure_hpa", "temperature_k")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError("missing required lookup-table option(s): " + ", ".join(missing))
    wavenumber = _wavenumber_grid(args)
    pressure = np.asarray(args.pressure_hpa, dtype=float)
    temperature = np.asarray(args.temperature_k, dtype=float)
    pgrid, tgrid = np.meshgrid(pressure / 1013.25, temperature, indexing="ij")
    flat = _cross_sections(
        hitran_dir=args.hitran_dir,
        gases=tuple(args.gas),
        wavenumber=wavenumber,
        pressure_atm=pgrid.reshape(-1),
        temperature_k=tgrid.reshape(-1),
        fwhm=args.fwhm,
    )
    table = flat.reshape((len(args.gas), wavenumber.size, pressure.size, temperature.size))
    return tuple(args.gas), args.hitran_dir, wavenumber, pressure, temperature, table


def _cross_sections(
    *,
    hitran_dir: Path,
    gases: tuple[str, ...],
    wavenumber: np.ndarray,
    pressure_atm: np.ndarray,
    temperature_k: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    partition = load_hitran_partition_functions(hitran_dir)
    table = np.empty((len(gases), wavenumber.size, pressure_atm.size), dtype=float)
    for index, gas in enumerate(gases):
        lines = read_hitran_lines(hitran_dir, gas, wavenumber)
        table[index] = hitran_cross_sections(
            hitran_dir=hitran_dir,
            molecule=gas,
            spectral_grid=wavenumber,
            pressure_atm=pressure_atm,
            temperature_k=temperature_k,
            fwhm=fwhm,
            partition_functions=partition,
            lines=lines,
        )
    return table


def _wavenumber_grid(args: argparse.Namespace) -> np.ndarray:
    if args.wavenumber_file is not None:
        grid = np.loadtxt(args.wavenumber_file, dtype=float)
    else:
        missing = [
            name
            for name in ("wavenumber_start", "wavenumber_step", "wavenumber_count")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError("pass --wavenumber-file or start/step/count")
        grid = args.wavenumber_start + args.wavenumber_step * np.arange(args.wavenumber_count)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("wavenumber grid must be strictly increasing")
    return grid


def _scene_wavenumber(scene: dict, base_dir: Path) -> np.ndarray:
    spectral = scene.get("spectral", {})
    if not isinstance(spectral, dict):
        raise ValueError("scene spectral section must be a mapping")
    if "wavenumber_cm_inv" in spectral:
        grid = _array_spec(spectral["wavenumber_cm_inv"], base_dir)
    elif "wavelengths_nm" in spectral:
        grid = 1.0e7 / _array_spec(spectral["wavelengths_nm"], base_dir)
    elif "wavelength_microns" in spectral:
        grid = 1.0e4 / _array_spec(spectral["wavelength_microns"], base_dir)
    else:
        raise ValueError("scene spectral section must define a HITRAN-compatible grid")
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("spectral grid must contain at least two points")
    if np.any(np.diff(grid) < 0.0):
        grid = grid[::-1]
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("wavenumber grid must be unique")
    return grid


def _array_spec(spec, base_dir: Path) -> np.ndarray:
    if isinstance(spec, dict):
        if {"start", "step", "count"}.issubset(spec):
            return float(spec["start"]) + float(spec["step"]) * np.arange(int(spec["count"]))
        if "path" in spec:
            path = _resolve_path(spec["path"], base_dir)
            return np.load(path) if path.suffix == ".npy" else np.loadtxt(path, dtype=float)
        if "value" in spec:
            return np.asarray(spec["value"], dtype=float)
    return np.asarray(spec, dtype=float)


def _scene_gases(scene: dict) -> tuple[str, ...]:
    gases = scene.get("gases", ())
    if isinstance(gases, str):
        return (gases,)
    return tuple(str(gas) for gas in gases)


def _scene_hitran_dir(scene: dict, base_dir: Path) -> Path:
    opacity = scene.get("opacity", {})
    gas_cfg = opacity.get("gas_cross_sections", {}) if isinstance(opacity, dict) else {}
    hitran = gas_cfg.get("hitran", {}) if isinstance(gas_cfg, dict) else {}
    if not isinstance(hitran, dict) or "path" not in hitran:
        raise ValueError("scene opacity.gas_cross_sections.hitran.path is required")
    return _resolve_path(hitran["path"], base_dir)


def _resolve_path(value, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def _write_table(
    output: Path,
    gases: tuple[str, ...],
    wavenumber: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    cross_section: np.ndarray,
) -> None:
    names = _gas_name_chars(gases)
    with netcdf_file(output, "w") as data:
        data.createDimension("gas", len(gases))
        data.createDimension("spectral", wavenumber.size)
        data.createDimension("name_strlen", names.shape[1])
        data.createVariable("gas_names", "c", ("gas", "name_strlen"))[:] = names
        data.createVariable("wavenumber_cm_inv", "d", ("spectral",))[:] = wavenumber
        if cross_section.ndim == 3:
            data.createDimension("level", pressure_hpa.size)
            data.createVariable("pressure_hpa", "d", ("level",))[:] = pressure_hpa
            data.createVariable("temperature_k", "d", ("level",))[:] = temperature_k
            data.createVariable("cross_section", "d", ("gas", "spectral", "level"))[:] = (
                cross_section
            )
        else:
            data.createDimension("pressure", pressure_hpa.size)
            data.createDimension("temperature", temperature_k.size)
            data.createVariable("pressure_hpa", "d", ("pressure",))[:] = pressure_hpa
            data.createVariable("temperature_k", "d", ("temperature",))[:] = temperature_k
            data.createVariable(
                "cross_section",
                "d",
                ("gas", "spectral", "pressure", "temperature"),
            )[:] = cross_section


def _gas_name_chars(gases: tuple[str, ...]) -> np.ndarray:
    width = max(len(gas) for gas in gases)
    names = np.full((len(gases), width), b" ", dtype="S1")
    for index, gas in enumerate(gases):
        names[index, : len(gas)] = np.frombuffer(gas.encode(), dtype="S1")
    return names


if __name__ == "__main__":
    main()
