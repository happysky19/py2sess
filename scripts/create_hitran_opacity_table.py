#!/usr/bin/env python3
"""Create a NetCDF gas cross-section table from HITRAN lines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tempfile

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
    parser.add_argument("--workers", type=int, default=1, help="Parallel gas workers.")
    args = parser.parse_args()

    try:
        inputs = _table_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.workers < 1:
        parser.error("--workers must be positive")
    with netcdf_file(args.output, "w") as data:
        xsec_var = _create_table_file(data, inputs)
        _write_cross_sections(args.output, inputs, xsec_var, fwhm=args.fwhm, workers=args.workers)
    print(f"wrote {args.output} from {inputs['hitran_dir']}")


def _table_inputs(args: argparse.Namespace) -> dict[str, object]:
    if args.spectral_limit is not None and args.spectral_limit < 2:
        raise ValueError("--spectral-limit must be at least 2")
    if args.profile is not None or args.scene is not None:
        if args.profile is None or args.scene is None:
            raise ValueError("--profile and --scene must be passed together")
        scene = load_scene_yaml(args.scene)
        gases = tuple(args.gas or _scene_gases(scene))
        if not gases:
            raise ValueError("scene must define gases or pass --gas")
        profile = load_profile_text(args.profile)
        wavenumber = _scene_wavenumber(scene, args.scene.parent)
        if args.spectral_limit is not None:
            wavenumber = wavenumber[: args.spectral_limit]
        return {
            "gases": gases,
            "hitran_dir": args.hitran_dir or _scene_hitran_dir(scene, args.scene.parent),
            "wavenumber": wavenumber,
            "pressure_hpa": profile.pressure_hpa,
            "temperature_k": profile.temperature_k,
            "pressure_atm": profile.pressure_hpa / 1013.25,
            "temperature_calc": profile.temperature_k,
            "is_lookup": False,
        }

    missing = [
        name
        for name in ("hitran_dir", "gas", "pressure_hpa", "temperature_k")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError("missing required lookup-table option(s): " + ", ".join(missing))
    pressure = np.asarray(args.pressure_hpa, dtype=float)
    temperature = np.asarray(args.temperature_k, dtype=float)
    pgrid, tgrid = np.meshgrid(pressure / 1013.25, temperature, indexing="ij")
    return {
        "gases": tuple(args.gas),
        "hitran_dir": args.hitran_dir,
        "wavenumber": _wavenumber_grid(args),
        "pressure_hpa": pressure,
        "temperature_k": temperature,
        "pressure_atm": pgrid.reshape(-1),
        "temperature_calc": tgrid.reshape(-1),
        "is_lookup": True,
    }


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
    if args.spectral_limit is not None:
        grid = np.asarray(grid, dtype=float).reshape(-1)[: args.spectral_limit]
    return _increasing_grid(grid)


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
    if grid.size >= 2 and np.any(np.diff(grid) < 0.0):
        grid = grid[::-1]
    return _increasing_grid(grid)


def _increasing_grid(values) -> np.ndarray:
    grid = np.asarray(values, dtype=float).reshape(-1)
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("wavenumber grid must be strictly increasing")
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
    return path if path.is_absolute() else base_dir / path


def _create_table_file(data, inputs: dict[str, object]):
    gases = inputs["gases"]
    wavenumber = inputs["wavenumber"]
    pressure = inputs["pressure_hpa"]
    temperature = inputs["temperature_k"]
    names = _gas_name_chars(gases)
    data.createDimension("gas", len(gases))
    data.createDimension("spectral", wavenumber.size)
    data.createDimension("name_strlen", names.shape[1])
    data.createVariable("gas_names", "c", ("gas", "name_strlen"))[:] = names
    data.createVariable("wavenumber_cm_inv", "d", ("spectral",))[:] = wavenumber
    if not inputs["is_lookup"]:
        data.createDimension("level", pressure.size)
        data.createVariable("pressure_hpa", "d", ("level",))[:] = pressure
        data.createVariable("temperature_k", "d", ("level",))[:] = temperature
        return data.createVariable("cross_section", "d", ("gas", "spectral", "level"))
    data.createDimension("pressure", pressure.size)
    data.createDimension("temperature", temperature.size)
    data.createVariable("pressure_hpa", "d", ("pressure",))[:] = pressure
    data.createVariable("temperature_k", "d", ("temperature",))[:] = temperature
    return data.createVariable(
        "cross_section",
        "d",
        ("gas", "spectral", "pressure", "temperature"),
    )


def _write_cross_sections(
    output: Path,
    inputs: dict[str, object],
    xsec_var,
    *,
    fwhm: float,
    workers: int,
) -> None:
    gases = tuple(inputs["gases"])
    if workers == 1 or len(gases) == 1:
        partition = load_hitran_partition_functions(inputs["hitran_dir"])
        for index, gas in enumerate(gases):
            xsec_var[index] = _compute_gas_cross_section(inputs, gas, fwhm, partition).reshape(
                xsec_var.shape[1:]
            )
        return

    worker_count = min(int(workers), len(gases))
    with tempfile.TemporaryDirectory(prefix=f"{output.stem}_", dir=output.parent) as tmpdir:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _compute_gas_cross_section_file,
                    inputs,
                    index,
                    gas,
                    fwhm,
                    Path(tmpdir) / f"gas_{index}.npy",
                )
                for index, gas in enumerate(gases)
            ]
            for future in as_completed(futures):
                index, gas, path = future.result()
                xsec_var[index] = np.load(path, mmap_mode="r").reshape(xsec_var.shape[1:])
                print(f"  wrote gas {index + 1}/{len(gases)}: {gas}", flush=True)


def _compute_gas_cross_section_file(
    inputs: dict[str, object],
    index: int,
    gas: str,
    fwhm: float,
    output: Path,
) -> tuple[int, str, str]:
    partition = load_hitran_partition_functions(inputs["hitran_dir"])
    np.save(output, _compute_gas_cross_section(inputs, gas, fwhm, partition))
    return index, gas, str(output)


def _compute_gas_cross_section(
    inputs: dict[str, object],
    gas: str,
    fwhm: float,
    partition: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    lines = read_hitran_lines(inputs["hitran_dir"], gas, inputs["wavenumber"])
    return hitran_cross_sections(
        hitran_dir=inputs["hitran_dir"],
        molecule=gas,
        spectral_grid=inputs["wavenumber"],
        pressure_atm=inputs["pressure_atm"],
        temperature_k=inputs["temperature_calc"],
        fwhm=fwhm,
        partition_functions=partition,
        lines=lines,
    )


def _gas_name_chars(gases: tuple[str, ...]) -> np.ndarray:
    width = max(len(gas) for gas in gases)
    names = np.full((len(gases), width), b" ", dtype="S1")
    for index, gas in enumerate(gases):
        names[index, : len(gas)] = np.frombuffer(gas.encode(), dtype="S1")
    return names


if __name__ == "__main__":
    main()
