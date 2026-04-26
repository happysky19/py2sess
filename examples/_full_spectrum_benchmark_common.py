"""Shared helpers for the full-spectrum benchmark examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from py2sess.optical.properties import build_layer_optical_properties


LAYER_OPTICAL_ABSORPTION_KEYS = ("absorption_tau", "gas_absorption_tau")
LAYER_OPTICAL_RAYLEIGH_KEY = "rayleigh_scattering_tau"
LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY = "aerosol_extinction_tau"
LAYER_OPTICAL_AEROSOL_SCATTERING_KEY = "aerosol_scattering_tau"
LAYER_OPTICAL_AEROSOL_SSA_KEY = "aerosol_single_scattering_albedo"


@dataclass(frozen=True)
class BenchmarkRow:
    """Timing and accuracy summary for one backend run."""

    backend: str
    wavelengths: int
    layers: int
    chunk_size: int
    wall_seconds: float
    rt_seconds: float
    fo_seconds: float | None = None
    two_stream_seconds: float | None = None
    max_abs_diff: float | None = None
    max_rel_diff_pct: float | None = None

    @property
    def rows_per_second_rt(self) -> float:
        """Returns RT-only throughput in spectral rows per second."""
        if self.rt_seconds <= 0.0:
            return 0.0
        return self.wavelengths / self.rt_seconds


def bundle_keys(path: Path) -> set[str]:
    """Returns array names stored in a benchmark bundle."""
    with np.load(path) as data:
        return set(data.files)


def load_bundle(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Loads selected benchmark bundle arrays into memory."""
    with np.load(path) as data:
        names = data.files if keys is None else [key for key in keys if key in data.files]
        return {key: np.array(data[key]) for key in names}


def select_layer_optical_keys(
    available: set[str],
    *,
    total_key: str,
    ssa_key: str,
) -> tuple[str, ...]:
    """Selects direct or component optical-depth inputs for benchmark loading."""
    absorption_key = next(
        (key for key in LAYER_OPTICAL_ABSORPTION_KEYS if key in available),
        None,
    )
    if absorption_key is None or LAYER_OPTICAL_RAYLEIGH_KEY not in available:
        return (total_key, ssa_key)

    keys = [absorption_key, LAYER_OPTICAL_RAYLEIGH_KEY]
    if LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY in available:
        if LAYER_OPTICAL_AEROSOL_SCATTERING_KEY in available:
            keys.extend(
                (
                    LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY,
                    LAYER_OPTICAL_AEROSOL_SCATTERING_KEY,
                )
            )
        elif LAYER_OPTICAL_AEROSOL_SSA_KEY in available:
            keys.extend((LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY, LAYER_OPTICAL_AEROSOL_SSA_KEY))
        else:
            return (total_key, ssa_key)
    elif LAYER_OPTICAL_AEROSOL_SCATTERING_KEY in available:
        keys.append(LAYER_OPTICAL_AEROSOL_SCATTERING_KEY)
    return tuple(keys)


def layer_optical_keys_are_components(keys: tuple[str, ...]) -> bool:
    """Returns true when selected optical keys are component optical depths."""
    return any(key in keys for key in LAYER_OPTICAL_ABSORPTION_KEYS)


def require_python_generated_layer_optical_inputs(
    keys: tuple[str, ...],
    *,
    total_key: str,
    ssa_key: str,
    label: str,
) -> None:
    """Reject direct layer optical inputs in strict generated-input mode."""
    if keys == (total_key, ssa_key):
        raise ValueError(
            f"{label} strict generated-input mode requires component optical-depth "
            "fields such as absorption_tau and rayleigh_scattering_tau"
        )


def prepare_layer_optical_properties(
    bundle: dict[str, np.ndarray],
    *,
    total_key: str,
    ssa_key: str,
) -> tuple[dict[str, np.ndarray], float, str]:
    """Builds ``tau``/``ssa`` and scattering fractions when components exist."""
    if LAYER_OPTICAL_RAYLEIGH_KEY not in bundle:
        return bundle, 0.0, "bundle"
    absorption_key = next((key for key in LAYER_OPTICAL_ABSORPTION_KEYS if key in bundle), None)
    if absorption_key is None:
        return bundle, 0.0, "bundle"

    start = time.perf_counter()
    props = build_layer_optical_properties(
        absorption_tau=bundle[absorption_key],
        rayleigh_scattering_tau=bundle[LAYER_OPTICAL_RAYLEIGH_KEY],
        aerosol_extinction_tau=bundle.get(LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY),
        aerosol_scattering_tau=bundle.get(LAYER_OPTICAL_AEROSOL_SCATTERING_KEY),
        aerosol_single_scattering_albedo=bundle.get(LAYER_OPTICAL_AEROSOL_SSA_KEY),
    )
    prepared = dict(bundle)
    prepared[total_key] = props.tau
    prepared[ssa_key] = props.ssa
    prepared["rayleigh_fraction"] = props.rayleigh_fraction
    prepared["aerosol_fraction"] = props.aerosol_fraction
    return prepared, time.perf_counter() - start, "python-generated from component optical depths"


def require_keys(bundle: dict[str, np.ndarray], keys: tuple[str, ...], *, label: str) -> None:
    """Raises an error when a benchmark bundle is missing required arrays."""
    missing = [key for key in keys if key not in bundle]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"{label} bundle is missing required arrays: {missing_text}")


def recommended_chunk_size(
    *,
    total_rows: int,
    nlayers: int,
    backend: str,
    workload: str,
) -> int:
    """Returns a shape-based serial chunk size for the benchmark examples."""
    if total_rows <= 0:
        return 0
    nlay = max(int(nlayers), 1)
    if workload == "solar_obs":
        row_floats = (48 if backend == "torch" else 40) * nlay + 64
        target_bytes = 512 * 1024 * 1024 if backend == "torch" else 1024 * 1024 * 1024
    elif workload == "thermal":
        row_floats = (6 if backend == "torch" else 4) * nlay + 32
        target_bytes = 384 * 1024 * 1024
    else:  # pragma: no cover
        raise ValueError(f"unsupported workload: {workload}")
    chunk = target_bytes // (8 * row_floats)
    granularity = 2000 if backend == "torch" else 1000
    minimum = granularity
    chunk = max(minimum, int(chunk))
    chunk = min(total_rows, ((chunk + granularity - 1) // granularity) * granularity)
    return max(1, chunk)


def print_problem_header(
    *,
    title: str,
    bundle_path: Path,
    wavelengths: int,
    layers: int,
    load_seconds: float | None = None,
    note: str | None = None,
) -> None:
    """Prints the benchmark header."""
    print(title)
    print(f"  bundle: {bundle_path}")
    print(f"  wavelengths: {wavelengths}")
    print(f"  layers: {layers}")
    if load_seconds is not None:
        print(f"  load (s): {load_seconds:.3f}")
    if note is not None:
        print(f"  note: {note}")


def _format_optional(value: float | None) -> str:
    """Formats optional floating-point columns for compact output tables."""
    return f"{'-':>10}" if value is None else f"{value:10.3f}"


def _format_accuracy(value: float | None) -> str:
    """Formats optional accuracy columns using scientific notation."""
    return f"{'-':>14}" if value is None else f"{value:14.6e}"


def scalar_value(value: np.ndarray | float | int) -> float:
    """Returns a scalar value from a scalar-like array or Python number."""
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def looks_like_row_index(values: np.ndarray) -> bool:
    """Returns true when a coordinate array is just 1-based row numbers."""
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        return False
    row_numbers = np.arange(1, grid.size + 1, dtype=float)
    return np.allclose(grid, row_numbers, rtol=0.0, atol=1.0e-12)


def accuracy_summary(value: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Returns the max absolute diff and max relative diff in percent."""
    abs_diff = float(np.max(np.abs(value - reference)))
    scale = np.maximum(np.abs(reference), 1.0e-15)
    rel_diff_pct = float(np.max(np.abs(value - reference) / scale) * 100.0)
    return abs_diff, rel_diff_pct


def print_rows(rows: list[BenchmarkRow]) -> None:
    """Prints the benchmark summary table."""
    has_accuracy = any(
        row.max_abs_diff is not None or row.max_rel_diff_pct is not None for row in rows
    )
    backend_width = max(18, *(len(row.backend) for row in rows))
    header = (
        f"{'backend':<{backend_width}} {'wall (s)':>10} {'rt (s)':>10} "
        f"{'fo (s)':>10} {'2s (s)':>10} {'#wavelength/s':>14} {'chunk':>8}"
    )
    if has_accuracy:
        header += f" {'max abs diff':>14} {'max rel (%)':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row.backend:<{backend_width}} "
            f"{row.wall_seconds:10.3f} "
            f"{row.rt_seconds:10.3f} "
            f"{_format_optional(row.fo_seconds)} "
            f"{_format_optional(row.two_stream_seconds)} "
            f"{row.rows_per_second_rt:14.0f} "
            f"{row.chunk_size:8d}"
        )
        if has_accuracy:
            line += (
                f" {_format_accuracy(row.max_abs_diff)} {_format_accuracy(row.max_rel_diff_pct)}"
            )
        print(line)
