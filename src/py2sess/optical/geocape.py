"""Readers for GEOCAPE optical input tables used by the benchmark scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_AEROSOL_AGGREGATES = ("organic", "seasalt-acc", "seasalt-coa", "soot", "sulfate")
_SELECT_WAVELENGTHS_MICRONS = (0.3, 0.4, 0.6, 0.999)


@dataclass(frozen=True)
class GeocapeAerosolTables:
    """Aerosol optical tables read from GEOCAPE ``SSprops`` files."""

    wavelengths_microns: np.ndarray
    bulk_iops: np.ndarray
    moments: np.ndarray


def gas_cross_section_from_table(
    path: str | Path,
    wavelengths_nm,
    *,
    wavelength_column: int = 0,
    value_column: int = 1,
    skiprows: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """Interpolate a two-column gas cross-section table onto ``wavelengths_nm``."""
    grid = np.asarray(wavelengths_nm, dtype=float)
    if grid.ndim != 1:
        raise ValueError("wavelengths_nm must be one-dimensional")
    if not np.all(np.isfinite(grid)):
        raise ValueError("wavelengths_nm must be finite")

    columns = np.loadtxt(
        path,
        dtype=float,
        skiprows=skiprows,
        usecols=(wavelength_column, value_column),
    )
    if columns.ndim != 2 or columns.shape[1] != 2:
        raise ValueError("gas cross-section table must have at least two numeric columns")
    order = np.argsort(columns[:, 0])
    table_wavelengths = columns[order, 0]
    values = columns[order, 1] * float(scale)
    if np.any(np.diff(table_wavelengths) <= 0.0):
        raise ValueError("gas cross-section wavelengths must be unique")
    if grid.min() < table_wavelengths[0] or grid.max() > table_wavelengths[-1]:
        raise ValueError("gas cross-section table does not cover the scene spectral grid")

    return np.interp(grid, table_wavelengths, values)


def gas_cross_sections_from_tables(
    *,
    wavelengths_nm,
    gas_names: tuple[str, ...],
    tables: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Build ``(nspec, ngas)`` gas cross sections from per-species table specs."""
    grid = np.asarray(wavelengths_nm, dtype=float)
    columns = []
    normalized = {name.upper(): spec for name, spec in tables.items()}
    for gas in gas_names:
        spec = normalized.get(gas.upper())
        if spec is None:
            raise ValueError(f"gas_cross_sections.tables is missing {gas}")
        if "path" not in spec:
            raise ValueError(f"gas_cross_sections.tables.{gas} requires path")
        columns.append(
            gas_cross_section_from_table(
                spec["path"],
                grid,
                wavelength_column=int(spec.get("wavelength_column", 0)),
                value_column=int(spec.get("value_column", 1)),
                skiprows=int(spec.get("skiprows", 0)),
                scale=float(spec.get("scale", 1.0)),
            )
        )
    if not columns:
        return np.zeros((grid.shape[0], 0), dtype=float)
    return np.column_stack(columns)


def load_geocape_solar_flux(
    path: str | Path,
    wavenumber_cm_inv,
    *,
    scale: float = 1.0e4,
) -> np.ndarray:
    """Read the GEOCAPE solar spectrum and interpolate it onto wavenumber."""
    wavenumber = _finite_1d("wavenumber_cm_inv", wavenumber_cm_inv)
    table = np.loadtxt(path, dtype=float, skiprows=2)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("solar spectrum table must have at least two columns")
    return np.interp(wavenumber, table[:, 0], table[:, 1]) * float(scale)


def load_geocape_surface_albedo(path: str | Path, wavenumber_cm_inv) -> np.ndarray:
    """Read ASTER emissivity and return Lambertian albedo = 1 - emissivity."""
    wavenumber = _finite_1d("wavenumber_cm_inv", wavenumber_cm_inv)
    table = np.loadtxt(path, dtype=float, skiprows=10, max_rows=386)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("surface emissivity table must have at least two columns")
    emissivity = np.interp(wavenumber, table[:, 0], table[:, 1])
    return 1.0 - emissivity


def load_geocape_aerosol_loadings(
    files: list[str | Path],
    *,
    n_layers: int,
    select_index: int,
    active_layers: int = 50,
) -> np.ndarray:
    """Read GEOCAPE per-component aerosol loading files."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if select_index < 1:
        raise ValueError("select_index is 1-based and must be positive")
    if active_layers <= 0:
        raise ValueError("active_layers must be positive")
    out = np.zeros((n_layers, len(files)), dtype=float)
    active = min(active_layers, n_layers)
    for component, path in enumerate(files):
        data = np.loadtxt(path, dtype=float)
        data = np.atleast_2d(data)
        if not np.all(np.isfinite(data)) or np.any(data < 0.0):
            raise ValueError(f"{path} aerosol loading values must be finite and nonnegative")
        if data.shape[0] < active:
            raise ValueError(f"{path} has fewer than {active} aerosol loading rows")
        if data.shape[1] < select_index:
            raise ValueError(f"{path} does not contain select_index {select_index}")
        out[n_layers - active :, component] = data[:active, select_index - 1][::-1]
    return out


def _finite_1d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return arr


def geocape_select_wavelength_microns(select_index: int) -> float:
    """Return the GEOCAPE loading reference wavelength for a 1-based index."""
    if select_index < 1 or select_index > len(_SELECT_WAVELENGTHS_MICRONS):
        raise ValueError("select_index must be one of 1, 2, 3, or 4")
    return _SELECT_WAVELENGTHS_MICRONS[select_index - 1]


def load_geocape_aerosol_tables(
    ssprops_dir: str | Path,
    *,
    first_wavelength_microns: float,
    last_wavelength_microns: float,
    aggregates: tuple[str, ...] = _DEFAULT_AEROSOL_AGGREGATES,
    moment_cutoff: float = 1.0e-5,
    max_moments: int | None = None,
) -> GeocapeAerosolTables:
    """Read GEOCAPE aerosol bulk IOPs and endpoint phase moments."""
    root = Path(ssprops_dir)
    bulk_columns = []
    moment_columns = []
    wavelengths = None
    for aggregate in aggregates:
        base = root / aggregate / "70"
        mie = np.loadtxt(base / "mie3_1.mie", dtype=float)
        if mie.ndim != 2 or mie.shape[1] < 3:
            raise ValueError(f"{base / 'mie3_1.mie'} must have at least three columns")
        if wavelengths is None:
            wavelengths = mie[:, 0]
            if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0.0):
                raise ValueError("aerosol SSprops wavelengths must be finite and increasing")
        elif not np.allclose(wavelengths, mie[:, 0]):
            raise ValueError("aerosol bulk IOP wavelength grids must match")
        if not np.all(np.isfinite(mie[:, 1:3])) or np.any(mie[:, 1:3] < 0.0):
            raise ValueError("aerosol bulk IOP values must be finite and nonnegative")
        bulk_columns.append(np.column_stack((mie[:, 2], mie[:, 1])))
        moments = _read_moment_file(base / "mie3_1.mom", max_moments=max_moments)
        if moments.shape[0] != mie.shape[0]:
            raise ValueError("aerosol moment and bulk IOP wavelength counts must match")
        moment_columns.append(moments)

    if wavelengths is None:
        raise ValueError("at least one aerosol aggregate is required")
    bulk = np.stack(bulk_columns, axis=-1).transpose(1, 0, 2)
    moments = _endpoint_moments(
        wavelengths=wavelengths,
        aggregate_moments=moment_columns,
        first_wavelength=float(first_wavelength_microns),
        last_wavelength=float(last_wavelength_microns),
        moment_cutoff=float(moment_cutoff),
    )
    return GeocapeAerosolTables(
        wavelengths_microns=np.asarray(wavelengths, dtype=float),
        bulk_iops=bulk,
        moments=moments,
    )


def _read_moment_file(path: Path, *, max_moments: int | None) -> np.ndarray:
    blocks: list[np.ndarray] = []
    current: list[tuple[int, float]] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("wavelength"):
                if current is not None:
                    blocks.append(_moment_block(current, max_moments=max_moments))
                current = []
                continue
            if stripped.startswith("l="):
                parts = stripped.replace("=", " ").split()
                if len(parts) >= 3 and current is not None:
                    current.append((int(parts[1]), float(parts[2])))
        if current is not None:
            blocks.append(_moment_block(current, max_moments=max_moments))
    if not blocks:
        raise ValueError(f"{path} contains no aerosol moment blocks")
    width = max(block.shape[0] for block in blocks)
    values = np.zeros((len(blocks), width), dtype=float)
    for index, block in enumerate(blocks):
        values[index, : block.shape[0]] = block
    return values


def _moment_block(rows: list[tuple[int, float]], *, max_moments: int | None) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=float)
    width = max(index for index, _ in rows) + 1
    if max_moments is not None:
        width = min(width, max_moments + 1)
    out = np.zeros(width, dtype=float)
    for index, value in rows:
        if index < width:
            out[index] = value
    return out


def _endpoint_moments(
    *,
    wavelengths: np.ndarray,
    aggregate_moments: list[np.ndarray],
    first_wavelength: float,
    last_wavelength: float,
    moment_cutoff: float,
) -> np.ndarray:
    endpoints = (first_wavelength, last_wavelength)
    raw: list[list[np.ndarray]] = []
    width = 0
    for endpoint in endpoints:
        endpoint_rows = []
        lower, upper, weight_lower, weight_upper = _interp_indices(wavelengths, endpoint)
        for moments in aggregate_moments:
            value = weight_lower * moments[lower] + weight_upper * moments[upper]
            value = _truncate_moments(value, cutoff=moment_cutoff)
            width = max(width, value.shape[0])
            endpoint_rows.append(value)
        raw.append(endpoint_rows)

    out = np.zeros((2, width, len(aggregate_moments)), dtype=float)
    for endpoint_index, endpoint_rows in enumerate(raw):
        for aggregate_index, value in enumerate(endpoint_rows):
            out[endpoint_index, : value.shape[0], aggregate_index] = value
    return out


def _interp_indices(grid: np.ndarray, value: float) -> tuple[int, int, float, float]:
    if value < grid[0] or value > grid[-1]:
        raise ValueError("aerosol wavelength is out of SSprops table bounds")
    if value == grid[0]:
        return 0, 0, 1.0, 0.0
    if value == grid[-1]:
        last = grid.shape[0] - 1
        return last, last, 1.0, 0.0
    index = int(np.searchsorted(grid, value, side="right"))
    lower = index - 1
    upper = index
    span = grid[upper] - grid[lower]
    weight_lower = (grid[upper] - value) / span
    return lower, upper, weight_lower, 1.0 - weight_lower


def _truncate_moments(values: np.ndarray, *, cutoff: float) -> np.ndarray:
    if cutoff <= 0.0:
        return values
    stop = values.shape[0]
    for index, value in enumerate(values):
        if abs(value) <= cutoff:
            stop = index
            break
    return values[:stop]
