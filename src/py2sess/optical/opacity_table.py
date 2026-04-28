"""Pressure-temperature gas cross-section table reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


def gas_cross_sections_from_table3d(
    *,
    path: str | Path,
    gas_names: tuple[str, ...],
    pressure_hpa,
    temperature_k,
    spectral: dict[str, np.ndarray],
) -> np.ndarray:
    """Interpolate a gas cross-section table to profile levels.

    The returned array has shape ``(nspec, nlevel, ngas)`` and uses the same
    units as the table. This mirrors the direct HITRAN path: integration to
    layer optical depth happens later in ``gas_absorption_tau_from_cross_sections``.
    """
    table = _read_table(path)
    target_spectral = _target_spectral_axis(table, spectral)
    xsec = _align_table_axes(table, gas_names=gas_names)
    xsec = _interp_spectral(table["spectral"], xsec, target_spectral)
    return _interp_pressure_temperature(
        pressure_axis=np.asarray(table["pressure_hpa"], dtype=float),
        temperature_axis=np.asarray(table["temperature_k"], dtype=float),
        cross_section=xsec,
        pressure_hpa=np.asarray(pressure_hpa, dtype=float),
        temperature_k=np.asarray(temperature_k, dtype=float),
    )


def _read_table(path: str | Path) -> dict[str, np.ndarray | tuple[str, ...]]:
    with netcdf_file(Path(path), "r", mmap=False) as data:
        arrays = {key: np.array(variable.data).copy() for key, variable in data.variables.items()}
    return _normalize_table(arrays)


def _normalize_table(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray | tuple[str, ...]]:
    xsec = _first(arrays, "cross_section", "gas_cross_sections")
    pressure = _first(arrays, "pressure_hpa", "pressure")
    temperature = _first(arrays, "temperature_k", "temperature")
    spectral_name, spectral = _spectral_array(arrays)
    table: dict[str, np.ndarray | tuple[str, ...]] = {
        "cross_section": np.asarray(xsec, dtype=float),
        "pressure_hpa": np.asarray(pressure, dtype=float),
        "temperature_k": np.asarray(temperature, dtype=float),
        "spectral_name": np.array(spectral_name),
        "spectral": np.asarray(spectral, dtype=float),
    }
    if "gas_names" in arrays:
        table["gas_names"] = _decode_names(arrays["gas_names"])
    return table


def _first(arrays: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in arrays:
            return arrays[name]
    raise ValueError(f"opacity table is missing {'/'.join(names)}")


def _spectral_array(arrays: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    for name in ("wavenumber_cm_inv", "wavelength_nm", "wavelength_microns"):
        if name in arrays:
            return name, arrays[name]
    raise ValueError(
        "opacity table requires wavenumber_cm_inv, wavelength_nm, or wavelength_microns"
    )


def _decode_names(value) -> tuple[str, ...]:
    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "U"} and arr.ndim == 1:
        return tuple(
            str(item.decode() if isinstance(item, bytes) else item).strip() for item in arr
        )
    if arr.dtype.kind in {"S", "U"} and arr.ndim == 2:
        names = []
        for row in arr:
            text = b"".join(row).decode() if row.dtype.kind == "S" else "".join(row)
            names.append(text.strip())
        return tuple(names)
    return tuple(str(item).strip() for item in arr.reshape(-1))


def _target_spectral_axis(
    table: dict[str, np.ndarray | tuple[str, ...]],
    spectral: dict[str, np.ndarray],
) -> np.ndarray:
    name = str(np.asarray(table["spectral_name"]).item())
    if name in spectral:
        return np.asarray(spectral[name], dtype=float)
    if name == "wavenumber_cm_inv":
        return 1.0e7 / np.asarray(spectral["wavelengths"], dtype=float)
    if name == "wavelength_nm":
        return np.asarray(spectral["wavelengths"], dtype=float)
    if name == "wavelength_microns":
        return np.asarray(spectral["wavelengths"], dtype=float) / 1000.0
    raise ValueError(f"unsupported opacity table spectral axis {name!r}")


def _align_table_axes(
    table: dict[str, np.ndarray | tuple[str, ...]], *, gas_names: tuple[str, ...]
) -> np.ndarray:
    xsec = np.asarray(table["cross_section"], dtype=float)
    if xsec.ndim != 4:
        raise ValueError("cross_section must have shape (gas, spectral, pressure, temperature)")
    if not np.all(np.isfinite(xsec)) or np.any(xsec < 0.0):
        raise ValueError("cross_section must be finite and nonnegative")
    if table.get("gas_names") is not None:
        return _reorder_gases(xsec, table, gas_names=gas_names)
    if xsec.shape[0] == len(gas_names):
        return xsec
    raise ValueError("cross_section gas axis does not match scene gases")


def _reorder_gases(
    xsec: np.ndarray,
    table: dict[str, np.ndarray | tuple[str, ...]],
    *,
    gas_names: tuple[str, ...],
) -> np.ndarray:
    table_names = table.get("gas_names")
    if table_names is None:
        return xsec
    if len(table_names) != xsec.shape[0]:
        raise ValueError("gas_names length must match the cross_section gas axis")
    name_to_index = {name.upper(): index for index, name in enumerate(table_names)}
    try:
        order = [name_to_index[name.upper()] for name in gas_names]
    except KeyError as exc:
        raise ValueError(f"opacity table is missing gas {exc.args[0]}") from exc
    return xsec[order]


def _interp_spectral(axis: np.ndarray, xsec: np.ndarray, target: np.ndarray) -> np.ndarray:
    axis = _increasing_axis(axis, "opacity table spectral axis")
    target = np.asarray(target, dtype=float)
    if target.shape == axis.shape and np.allclose(target, axis, rtol=0.0, atol=1.0e-12):
        return xsec
    if np.any(target < axis[0]) or np.any(target > axis[-1]):
        raise ValueError("scene spectral grid extends outside opacity table")
    lower, upper, weight = _bracket(axis, target, "spectral axis")
    weight = weight[np.newaxis, :, np.newaxis, np.newaxis]
    return (1.0 - weight) * xsec[:, lower, :, :] + weight * xsec[:, upper, :, :]


def _interp_pressure_temperature(
    *,
    pressure_axis: np.ndarray,
    temperature_axis: np.ndarray,
    cross_section: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    pressure_values = _increasing_axis(pressure_axis, "opacity table pressure_hpa")
    temp_grid = _increasing_axis(temperature_axis, "opacity table temperature_k")
    if np.any(pressure_values <= 0.0):
        raise ValueError("opacity table pressure_hpa must be positive")
    if np.any(temp_grid <= 0.0):
        raise ValueError("opacity table temperature_k must be positive")
    pressure_grid = np.log(pressure_values)
    target_pressure = np.log(_profile_axis(pressure_hpa, "pressure_hpa"))
    target_temp = _profile_axis(temperature_k, "temperature_k")
    if target_pressure.shape != target_temp.shape:
        raise ValueError("pressure_hpa and temperature_k must have the same shape")

    p0, p1, wp = _bracket(pressure_grid, target_pressure, "pressure_hpa")
    t0, t1, wt = _bracket(temp_grid, target_temp, "temperature_k")
    c00 = cross_section[:, :, p0, t0]
    c10 = cross_section[:, :, p1, t0]
    c01 = cross_section[:, :, p0, t1]
    c11 = cross_section[:, :, p1, t1]
    wp = wp[np.newaxis, np.newaxis, :]
    wt = wt[np.newaxis, np.newaxis, :]
    out = (
        (1.0 - wp) * (1.0 - wt) * c00
        + wp * (1.0 - wt) * c10
        + (1.0 - wp) * wt * c01
        + wp * wt * c11
    )
    return np.moveaxis(out, 0, -1)


def _increasing_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _profile_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(axis)) or np.any(axis <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return axis


def _bracket(
    grid: np.ndarray, values: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(values < grid[0]) or np.any(values > grid[-1]):
        raise ValueError(f"profile {name} extends outside opacity table")
    if grid.size == 1:
        if np.any(values != grid[0]):
            raise ValueError(f"profile {name} extends outside opacity table")
        zeros = np.zeros(values.shape, dtype=int)
        return zeros, zeros, np.zeros(values.shape, dtype=float)
    upper = np.searchsorted(grid, values, side="right")
    upper = np.clip(upper, 1, grid.size - 1)
    lower = upper - 1
    denom = grid[upper] - grid[lower]
    weight = np.where(denom == 0.0, 0.0, (values - grid[lower]) / denom)
    exact_upper = values == grid[-1]
    lower = np.where(exact_upper, grid.size - 1, lower)
    upper = np.where(exact_upper, grid.size - 1, upper)
    weight = np.where(exact_upper, 0.0, weight)
    return lower, upper, weight
