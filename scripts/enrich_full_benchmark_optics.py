#!/usr/bin/env python3
"""Add physical phase-mixing inputs to a local full-spectrum benchmark bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _copy_bundle_with_optics(
    *,
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


def _parse_tir_dump(dump_path: Path) -> dict[str, np.ndarray]:
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
        depol = np.empty(n_rows, dtype=np.float64)
        rayleigh_fraction = np.empty((n_rows, n_layers), dtype=np.float64)
        aerosol_fraction = np.empty((n_rows, n_layers, 5), dtype=np.float64)

        for row in range(n_rows):
            spec = handle.readline().split()
            if len(spec) < 7:
                raise ValueError(f"invalid TIR spectral row {row + 1}")
            wavelengths[row] = float(spec[1])
            depol[row] = float(spec[4])
            for layer in range(n_layers):
                values = handle.readline().split()
                if len(values) < 11:
                    raise ValueError(f"invalid TIR layer row {row + 1}, layer {layer + 1}")
                rayleigh_fraction[row, layer] = float(values[4])
                aerosol_fraction[row, layer] = [float(value) for value in values[5:10]]
            if (row + 1) % 50_000 == 0:
                print(f"  parsed TIR rows: {row + 1}/{n_rows}", flush=True)

    return {
        "wavelengths": wavelengths,
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
    args = parser.parse_args()

    if args.output.resolve() == args.bundle.resolve():
        raise ValueError("output must be a different path; keep the original bundle intact")
    if not args.dump.is_file():
        raise FileNotFoundError(args.dump)
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)

    optics = _parse_uv_dump(args.dump) if args.kind == "uv" else _parse_tir_dump(args.dump)
    _copy_bundle_with_optics(bundle_path=args.bundle, output_path=args.output, optics=optics)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
