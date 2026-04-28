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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--hitran-dir", type=Path, required=True)
    parser.add_argument("--gas", action="append", required=True, help="Gas name; repeat per gas.")
    parser.add_argument("--pressure-hpa", type=float, nargs="+", required=True)
    parser.add_argument("--temperature-k", type=float, nargs="+", required=True)
    parser.add_argument("--wavenumber-file", type=Path)
    parser.add_argument("--wavenumber-start", type=float)
    parser.add_argument("--wavenumber-step", type=float)
    parser.add_argument("--wavenumber-count", type=int)
    parser.add_argument("--fwhm", type=float, default=0.0)
    args = parser.parse_args()

    try:
        wavenumber = _wavenumber_grid(args)
    except ValueError as exc:
        parser.error(str(exc))
    pressure = np.asarray(args.pressure_hpa, dtype=float)
    temperature = np.asarray(args.temperature_k, dtype=float)
    pgrid, tgrid = np.meshgrid(pressure / 1013.25, temperature, indexing="ij")

    partition = load_hitran_partition_functions(args.hitran_dir)
    table = np.empty((len(args.gas), wavenumber.size, pressure.size, temperature.size), dtype=float)
    for index, gas in enumerate(args.gas):
        lines = read_hitran_lines(args.hitran_dir, gas, wavenumber)
        xsec = hitran_cross_sections(
            hitran_dir=args.hitran_dir,
            molecule=gas,
            spectral_grid=wavenumber,
            pressure_atm=pgrid.reshape(-1),
            temperature_k=tgrid.reshape(-1),
            fwhm=args.fwhm,
            partition_functions=partition,
            lines=lines,
        )
        table[index] = xsec.reshape(wavenumber.size, pressure.size, temperature.size)

    _write_table(args.output, args.gas, wavenumber, pressure, temperature, table)
    print(f"wrote {args.output}")


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


def _write_table(
    output: Path,
    gases: list[str],
    wavenumber: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    cross_section: np.ndarray,
) -> None:
    names = _gas_name_chars(gases)
    with netcdf_file(output, "w") as data:
        data.createDimension("gas", len(gases))
        data.createDimension("spectral", wavenumber.size)
        data.createDimension("pressure", pressure_hpa.size)
        data.createDimension("temperature", temperature_k.size)
        data.createDimension("name_strlen", names.shape[1])
        data.createVariable("gas_names", "c", ("gas", "name_strlen"))[:] = names
        data.createVariable("wavenumber_cm_inv", "d", ("spectral",))[:] = wavenumber
        data.createVariable("pressure_hpa", "d", ("pressure",))[:] = pressure_hpa
        data.createVariable("temperature_k", "d", ("temperature",))[:] = temperature_k
        data.createVariable(
            "cross_section",
            "d",
            ("gas", "spectral", "pressure", "temperature"),
        )[:] = cross_section


def _gas_name_chars(gases: list[str]) -> np.ndarray:
    width = max(len(gas) for gas in gases)
    names = np.full((len(gases), width), b" ", dtype="S1")
    for index, gas in enumerate(gases):
        names[index, : len(gas)] = np.frombuffer(gas.encode(), dtype="S1")
    return names


if __name__ == "__main__":
    main()
