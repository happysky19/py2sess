"""Utilities for building thermal source inputs from temperature profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VectorLike = np.ndarray | list[float] | tuple[float, ...]
_ScalarOrArrayLike = float | np.ndarray | list[float] | tuple[float, ...]

_PLANCK_CONSTANT = 6.62607015e-34
_LIGHT_SPEED = 2.99792458e8
_BOLTZMANN_CONSTANT = 1.380649e-23
_MICRONS_TO_METERS = 1.0e-6
_CM_TO_METERS = 1.0e-2

_FORTRAN_C2 = 1.438786
_FORTRAN_SIGMA_OVER_PI = 1.80491891383e-8
_FORTRAN_PLANCK_CONC = 1.5398973382e-1
_FORTRAN_PLANCK_EPSIL = 1.0e-8
_FORTRAN_PLANCK_VMAX = 32.0
_FORTRAN_PLANCK_NSIMPSON = 25
_FORTRAN_PLANCK_CRITERION = 1.0e-10
_FORTRAN_PLANCK_VCUT = 1.5
_FORTRAN_PLANCK_VCP = (10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0)
_ROW_BAND_CHUNK_SIZE = 16384
_ROW_BAND_UNIFORM_CHUNK_SIZE = 32768


@dataclass(frozen=True)
class ThermalSourceInputs:
    """Thermal source inputs for public ``forward`` calls."""

    planck: np.ndarray
    surface_planck: float | np.ndarray


def _as_float_array(name: str, values: _VectorLike) -> np.ndarray:
    """Convert input values to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validate_temperature(
    name: str,
    temperature_k: _ScalarOrArrayLike,
) -> np.ndarray:
    """Validate temperature inputs in Kelvin."""

    array = np.asarray(temperature_k, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive in Kelvin")
    return array


def _validate_positive_coordinate(name: str, value: _ScalarOrArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _profile_spectral_grid(spectral_coordinate: np.ndarray, level_temperature: np.ndarray):
    if level_temperature.ndim != 1:
        raise ValueError("level_temperature_k must be one-dimensional")
    if spectral_coordinate.ndim == 0:
        return spectral_coordinate, level_temperature
    spectral = spectral_coordinate.reshape(spectral_coordinate.shape + (1,))
    levels = level_temperature.reshape((1,) * spectral_coordinate.ndim + level_temperature.shape)
    return spectral, levels


def _scalar_or_array(value) -> float | np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array


def planck_radiance_wavelength(
    temperature_k: _ScalarOrArrayLike,
    wavelength_microns: _ScalarOrArrayLike,
) -> np.ndarray:
    """Evaluate the Planck function in wavelength form."""

    temperature = _validate_temperature("temperature_k", temperature_k)
    wavelength = _validate_positive_coordinate("wavelength_microns", wavelength_microns)
    wavelength_m = wavelength * _MICRONS_TO_METERS
    exponent = (_PLANCK_CONSTANT * _LIGHT_SPEED) / (
        wavelength_m * _BOLTZMANN_CONSTANT * temperature
    )
    numerator = 2.0 * _PLANCK_CONSTANT * _LIGHT_SPEED**2
    denominator = wavelength_m**5 * np.expm1(exponent)
    return numerator / denominator


def planck_radiance_wavenumber(
    temperature_k: _ScalarOrArrayLike,
    wavenumber_cm_inv: _ScalarOrArrayLike,
) -> np.ndarray:
    """Evaluate the Planck function in wavenumber form."""

    temperature = _validate_temperature("temperature_k", temperature_k)
    wavenumber = _validate_positive_coordinate("wavenumber_cm_inv", wavenumber_cm_inv)
    wavenumber_m_inv = wavenumber / _CM_TO_METERS
    exponent = (_PLANCK_CONSTANT * _LIGHT_SPEED * wavenumber_m_inv) / (
        _BOLTZMANN_CONSTANT * temperature
    )
    numerator = 2.0 * _PLANCK_CONSTANT * _LIGHT_SPEED**2 * wavenumber_m_inv**3
    denominator = np.expm1(exponent)
    return numerator / denominator


def _fortran_planck_polynomial(x_value: float) -> float:
    """Evaluate the low-wavenumber polynomial used by 2S-ESS Fortran."""

    a1 = 3.33333333333e-1
    a2 = -1.25e-1
    a3 = 1.66666666667e-2
    a4 = -1.98412698413e-4
    a5 = 3.67430922986e-6
    a6 = -7.51563251563e-8
    x_sq = x_value * x_value
    return (
        _FORTRAN_PLANCK_CONC
        * x_sq
        * x_value
        * (a1 + x_value * (a2 + x_value * (a3 + x_sq * (a4 + x_sq * (a5 + x_sq * a6)))))
    )


def _fortran_planck_exponential(x_value: float) -> float:
    """Evaluate the high-wavenumber exponential series used by 2S-ESS Fortran."""

    max_terms = 0
    while True:
        max_terms += 1
        if x_value >= _FORTRAN_PLANCK_VCP[max_terms - 1]:
            break
    exp_minus_x = np.exp(-x_value)
    factor = 1.0
    total = 0.0
    for term in range(1, max_terms + 1):
        mv = term * x_value
        factor *= exp_minus_x
        total += factor * (6.0 + mv * (6.0 + mv * (3.0 + mv))) * float(term) ** -4.0
    return total * _FORTRAN_PLANCK_CONC


def _fortran_planck_band_scalar(
    temperature_k: float,
    wavenumber_low_cm_inv: float,
    wavenumber_high_cm_inv: float,
) -> float:
    """Evaluate the Fortran 2S-ESS band-integrated Planck helper for one temperature."""

    gamma = _FORTRAN_C2 / temperature_k
    x_low = gamma * wavenumber_low_cm_inv
    x_high = gamma * wavenumber_high_cm_inv
    scaling = _FORTRAN_SIGMA_OVER_PI * temperature_k**4

    if (
        x_low > _FORTRAN_PLANCK_EPSIL
        and x_high < _FORTRAN_PLANCK_VMAX
        and (wavenumber_high_cm_inv - wavenumber_low_cm_inv) / wavenumber_high_cm_inv < 1.0e-2
    ):
        interval = x_high - x_low
        planck_low = x_low**3 / np.expm1(x_low)
        planck_high = x_high**3 / np.expm1(x_high)
        endpoints = planck_low + planck_high
        previous = endpoints * 0.5 * interval
        for n in range(1, _FORTRAN_PLANCK_NSIMPSON + 1):
            step = 0.5 * interval / float(n)
            value = endpoints
            for k in range(1, 2 * n):
                x_current = x_low + float(k) * step
                factor = float(2 * (1 + (k % 2)))
                value += factor * x_current**3 / np.expm1(x_current)
            value *= step / 3.0
            if abs((value - previous) / value) <= _FORTRAN_PLANCK_CRITERION:
                return float(scaling * value * _FORTRAN_PLANCK_CONC)
            previous = value
        raise RuntimeError("Fortran-compatible Simpson Planck integration did not converge")

    small_values = 0
    polynomial = [0.0, 0.0]
    exponential = [0.0, 0.0]
    for index, x_value in enumerate((x_low, x_high)):
        if x_value < _FORTRAN_PLANCK_VCUT:
            small_values += 1
            polynomial[index] = _fortran_planck_polynomial(x_value)
        else:
            exponential[index] = _fortran_planck_exponential(x_value)

    if small_values == 2:
        value = polynomial[1] - polynomial[0]
    elif small_values == 1:
        value = 1.0 - polynomial[0] - exponential[1]
    else:
        value = exponential[0] - exponential[1]
    return float(scaling * value)


def _validate_wavenumber_band(
    wavenumber_low_cm_inv: _ScalarOrArrayLike,
    wavenumber_high_cm_inv: _ScalarOrArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(wavenumber_low_cm_inv, dtype=np.float64)
    high = np.asarray(wavenumber_high_cm_inv, dtype=np.float64)
    low, high = np.broadcast_arrays(low, high)
    if not np.all(np.isfinite(low)) or np.any(low < 0.0):
        raise ValueError("wavenumber_low_cm_inv must be non-negative and finite")
    if not np.all(np.isfinite(high)) or np.any(high <= low):
        raise ValueError(
            "wavenumber_high_cm_inv must be finite and greater than wavenumber_low_cm_inv"
        )
    return low, high


def _fortran_planck_band_simpson_vectorized(
    x_low: np.ndarray,
    x_high: np.ndarray,
    scaling: np.ndarray,
) -> np.ndarray:
    interval = x_high - x_low
    f_low = x_low**3 / np.expm1(x_low)
    f_high = x_high**3 / np.expm1(x_high)
    endpoints = f_low + f_high
    previous = endpoints * 0.5 * interval
    result = np.empty_like(x_low, dtype=np.float64)

    mid = x_low + 0.5 * interval
    value = (endpoints + 4.0 * mid**3 / np.expm1(mid)) * interval / 6.0
    converged = _simpson_converged(value, previous)
    if np.any(converged):
        result[converged] = scaling[converged] * value[converged] * _FORTRAN_PLANCK_CONC
    if np.all(converged):
        return result

    active = ~converged
    q1 = x_low[active] + 0.25 * interval[active]
    q3 = x_low[active] + 0.75 * interval[active]
    value2 = (
        (
            endpoints[active]
            + 4.0 * q1**3 / np.expm1(q1)
            + 2.0 * mid[active] ** 3 / np.expm1(mid[active])
            + 4.0 * q3**3 / np.expm1(q3)
        )
        * interval[active]
        / 12.0
    )
    converged2 = _simpson_converged(value2, value[active])
    active_positions = np.flatnonzero(active)
    if np.any(converged2):
        result[active_positions[converged2]] = (
            scaling[active][converged2] * value2[converged2] * _FORTRAN_PLANCK_CONC
        )
    if np.all(converged2):
        return result

    keep_positions = active_positions[~converged2]
    active_x_low = x_low[keep_positions]
    active_interval = interval[keep_positions]
    active_endpoints = endpoints[keep_positions]
    active_previous = value2[~converged2]
    active_scaling = scaling[keep_positions]

    for n in range(3, _FORTRAN_PLANCK_NSIMPSON + 1):
        step = 0.5 * active_interval / float(n)
        value = active_endpoints.copy()
        for k in range(1, 2 * n):
            x_current = active_x_low + float(k) * step
            factor = float(2 * (1 + (k % 2)))
            value += factor * x_current**3 / np.expm1(x_current)
        value *= step / 3.0
        converged = _simpson_converged(value, active_previous)
        if np.any(converged):
            result[keep_positions[converged]] = (
                active_scaling[converged] * value[converged] * _FORTRAN_PLANCK_CONC
            )
        if not np.any(converged):
            active_previous = value
            continue
        keep = ~converged
        if not np.any(keep):
            return result
        keep_positions = keep_positions[keep]
        active_x_low = active_x_low[keep]
        active_interval = active_interval[keep]
        active_endpoints = active_endpoints[keep]
        active_previous = value[keep]
        active_scaling = active_scaling[keep]

    raise RuntimeError("Fortran-compatible Simpson Planck integration did not converge")


def _simpson_converged(value: np.ndarray, previous: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(value), np.finfo(np.float64).tiny)
    return np.abs(value - previous) / scale <= _FORTRAN_PLANCK_CRITERION


def _fortran_planck_band_vectorized(
    temperature_k: np.ndarray,
    wavenumber_low_cm_inv: np.ndarray,
    wavenumber_high_cm_inv: np.ndarray,
) -> np.ndarray:
    temperature, low, high = np.broadcast_arrays(
        temperature_k,
        wavenumber_low_cm_inv,
        wavenumber_high_cm_inv,
    )
    gamma = _FORTRAN_C2 / temperature
    x_low = gamma * low
    x_high = gamma * high
    scaling = _FORTRAN_SIGMA_OVER_PI * temperature**4
    narrow = (
        (x_low > _FORTRAN_PLANCK_EPSIL)
        & (x_high < _FORTRAN_PLANCK_VMAX)
        & ((high - low) / high < 1.0e-2)
    )

    result = np.empty(temperature.shape, dtype=np.float64)
    if np.any(narrow):
        result.ravel()[narrow.ravel()] = _fortran_planck_band_simpson_vectorized(
            x_low[narrow].ravel(),
            x_high[narrow].ravel(),
            scaling[narrow].ravel(),
        )
    if np.any(~narrow):
        flat_result = result.ravel()
        flat_temperature = temperature.ravel()
        flat_low = low.ravel()
        flat_high = high.ravel()
        for index in np.flatnonzero((~narrow).ravel()):
            flat_result[index] = _fortran_planck_band_scalar(
                float(flat_temperature[index]),
                float(flat_low[index]),
                float(flat_high[index]),
            )
    return result


def planck_radiance_wavenumber_band(
    temperature_k: _ScalarOrArrayLike,
    wavenumber_low_cm_inv: _ScalarOrArrayLike,
    wavenumber_high_cm_inv: _ScalarOrArrayLike,
) -> np.ndarray:
    """Evaluate the Fortran-compatible band-integrated Planck function."""

    temperature = _validate_temperature("temperature_k", temperature_k)
    low, high = _validate_wavenumber_band(wavenumber_low_cm_inv, wavenumber_high_cm_inv)
    if low.ndim == 0 and high.ndim == 0:
        result = np.empty(temperature.shape, dtype=np.float64)
        flat_temperature = np.ravel(temperature)
        flat_result = np.ravel(result)
        for index, temp in enumerate(flat_temperature):
            flat_result[index] = _fortran_planck_band_scalar(
                float(temp),
                float(low),
                float(high),
            )
        return result
    return _fortran_planck_band_vectorized(temperature, low, high)


def _row_band_thermal_source(
    level_temperature: np.ndarray,
    surface_temperature: np.ndarray,
    wavenumber_band_cm_inv: np.ndarray,
) -> ThermalSourceInputs:
    bands = np.asarray(wavenumber_band_cm_inv, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[1] != 2:
        raise ValueError("wavenumber_band_cm_inv must have shape (n_spectral, 2)")
    low, high = _validate_wavenumber_band(bands[:, 0], bands[:, 1])
    n_rows = bands.shape[0]
    if surface_temperature.ndim > 1:
        raise ValueError("surface_temperature_k must be scalar or one-dimensional")
    if surface_temperature.ndim == 1 and surface_temperature.size not in {1, n_rows}:
        raise ValueError("surface_temperature_k must be scalar or have shape (n_spectral,)")

    uniform_source = _row_band_thermal_source_uniform_grid(
        level_temperature,
        surface_temperature,
        low,
        high,
    )
    if uniform_source is not None:
        return uniform_source

    planck = np.empty((n_rows, level_temperature.size), dtype=np.float64)
    surface_planck = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, _ROW_BAND_CHUNK_SIZE):
        stop = min(start + _ROW_BAND_CHUNK_SIZE, n_rows)
        row_slice = slice(start, stop)
        low_chunk = low[row_slice]
        high_chunk = high[row_slice]
        planck[row_slice] = planck_radiance_wavenumber_band(
            level_temperature.reshape(1, -1),
            low_chunk.reshape(-1, 1),
            high_chunk.reshape(-1, 1),
        )
        surface_chunk = (
            surface_temperature
            if surface_temperature.ndim == 0 or surface_temperature.size == 1
            else surface_temperature[row_slice]
        )
        surface_planck[row_slice] = planck_radiance_wavenumber_band(
            surface_chunk,
            low_chunk,
            high_chunk,
        )
    return ThermalSourceInputs(planck=planck, surface_planck=surface_planck)


def _uniform_band_offsets(
    low: np.ndarray, high: np.ndarray
) -> tuple[float, float, np.ndarray] | None:
    if low.size < 2:
        return None
    steps = np.diff(low)
    widths = high - low
    step = float(np.median(steps))
    width = float(np.median(widths))
    if step <= 0.0 or width <= 0.0:
        return None
    if not np.allclose(steps, step, rtol=1.0e-10, atol=1.0e-12):
        return None
    if not np.allclose(widths, width, rtol=1.0e-10, atol=1.0e-12):
        return None
    offsets = np.rint(np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * width / step).astype(int)
    if np.any(offsets < 0):
        return None
    if not np.allclose(offsets * step, np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * width):
        return None
    return step, width, offsets


def _row_band_thermal_source_uniform_grid(
    level_temperature: np.ndarray,
    surface_temperature: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> ThermalSourceInputs | None:
    uniform = _uniform_band_offsets(low, high)
    if uniform is None:
        return None
    if surface_temperature.ndim == 1 and surface_temperature.size > 1:
        return None
    step, width, offsets = uniform
    temperatures = np.concatenate(
        (
            level_temperature,
            np.asarray(surface_temperature, dtype=np.float64).reshape(-1)[:1],
        )
    )
    if not _uniform_band_path_is_narrow(temperatures, low, high):
        return None

    n_rows = low.size
    values = np.empty((n_rows, temperatures.size), dtype=np.float64)
    for start in range(0, n_rows, _ROW_BAND_UNIFORM_CHUNK_SIZE):
        stop = min(start + _ROW_BAND_UNIFORM_CHUNK_SIZE, n_rows)
        chunk = _uniform_band_chunk(
            temperatures=temperatures,
            first_low=float(low[0]),
            step=step,
            width=width,
            offsets=offsets,
            start=start,
            stop=stop,
        )
        if chunk is None:
            return None
        values[start:stop] = chunk
    return ThermalSourceInputs(planck=values[:, :-1], surface_planck=values[:, -1])


def _uniform_band_path_is_narrow(
    temperatures: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> bool:
    gamma = _FORTRAN_C2 / temperatures
    x_low_min = float(np.min(gamma) * np.min(low))
    x_high_max = float(np.max(gamma) * np.max(high))
    relative_width_max = float(np.max((high - low) / high))
    return (
        x_low_min > _FORTRAN_PLANCK_EPSIL
        and x_high_max < _FORTRAN_PLANCK_VMAX
        and relative_width_max < 1.0e-2
    )


def _uniform_band_chunk(
    *,
    temperatures: np.ndarray,
    first_low: float,
    step: float,
    width: float,
    offsets: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray | None:
    n_rows = stop - start
    wavenumber = first_low + step * np.arange(start, stop + int(offsets[-1]), dtype=np.float64)
    gamma = _FORTRAN_C2 / temperatures
    interval = width * gamma
    x_grid = wavenumber[:, np.newaxis] * gamma[np.newaxis, :]
    f_grid = x_grid**3 / np.expm1(x_grid)
    endpoint = f_grid[:n_rows] + f_grid[offsets[4] : offsets[4] + n_rows]
    midpoint = f_grid[offsets[2] : offsets[2] + n_rows]
    previous = endpoint * 0.5 * interval
    value1 = (endpoint + 4.0 * midpoint) * interval / 6.0
    value2 = (
        (
            endpoint
            + 4.0 * f_grid[offsets[1] : offsets[1] + n_rows]
            + 2.0 * midpoint
            + 4.0 * f_grid[offsets[3] : offsets[3] + n_rows]
        )
        * interval
        / 12.0
    )
    converged1 = np.abs((value1 - previous) / value1) <= _FORTRAN_PLANCK_CRITERION
    converged2 = np.abs((value2 - value1) / value2) <= _FORTRAN_PLANCK_CRITERION
    if not np.all(converged1 | converged2):
        return None
    value = np.where(converged1, value1, value2)
    scaling = _FORTRAN_SIGMA_OVER_PI * temperatures**4
    return value * scaling[np.newaxis, :] * _FORTRAN_PLANCK_CONC


def thermal_source_from_temperature_profile(
    level_temperature_k: _VectorLike,
    surface_temperature_k: _ScalarOrArrayLike,
    *,
    wavelength_microns: _ScalarOrArrayLike | None = None,
    wavenumber_cm_inv: _ScalarOrArrayLike | None = None,
    wavenumber_band_cm_inv: _ScalarOrArrayLike | None = None,
) -> ThermalSourceInputs:
    """Build ``planck`` and ``surface_planck`` from one spectral coordinate."""

    level_temperature = _as_float_array("level_temperature_k", level_temperature_k)
    _validate_temperature("level_temperature_k", level_temperature)
    surface_temperature = _validate_temperature("surface_temperature_k", surface_temperature_k)

    provided = sum(
        value is not None
        for value in (wavelength_microns, wavenumber_cm_inv, wavenumber_band_cm_inv)
    )
    if provided != 1:
        raise ValueError("Specify exactly one spectral coordinate")

    if wavelength_microns is not None:
        spectral = _validate_positive_coordinate("wavelength_microns", wavelength_microns)
        spectral_grid, level_grid = _profile_spectral_grid(spectral, level_temperature)
        planck = planck_radiance_wavelength(level_grid, spectral_grid)
        surface_planck = planck_radiance_wavelength(surface_temperature, spectral)
    elif wavenumber_cm_inv is not None:
        spectral = _validate_positive_coordinate("wavenumber_cm_inv", wavenumber_cm_inv)
        spectral_grid, level_grid = _profile_spectral_grid(spectral, level_temperature)
        planck = planck_radiance_wavenumber(level_grid, spectral_grid)
        surface_planck = planck_radiance_wavenumber(surface_temperature, spectral)
    else:
        band = np.asarray(wavenumber_band_cm_inv, dtype=np.float64)
        if band.shape == (2,):
            low, high = band
            planck = planck_radiance_wavenumber_band(level_temperature, low, high)
            surface_planck = planck_radiance_wavenumber_band(surface_temperature, low, high)
        else:
            return _row_band_thermal_source(level_temperature, surface_temperature, band)

    return ThermalSourceInputs(planck=planck, surface_planck=_scalar_or_array(surface_planck))
