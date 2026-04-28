"""Shared helpers for the full-spectrum benchmark examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from py2sess.optical.properties import (
    build_layer_optical_properties,
    sum_component_axis,
)
from py2sess.optical.phase import ssa_from_optical_depth
from py2sess.optical.scene import (
    atmospheric_profile_from_levels,
    build_scene_layer_optical_properties,
    build_scene_layer_optical_properties_from_gas_tau,
)


SCENE_LAYER_CROSS_SECTION_REQUIRED_KEYS = ("pressure_hpa", "temperature_k", "gas_cross_sections")
SCENE_LAYER_GAS_TAU_REQUIRED_KEYS = ("pressure_hpa", "temperature_k", "gas_absorption_tau")
SCENE_LAYER_OPTIONAL_KEYS = (
    "gas_vmr",
    "heights",
    "surface_altitude_m",
    "co2_ppmv",
    "opacity_wavelengths",
    "aerosol_loadings",
    "aerosol_wavelengths_microns",
    "aerosol_bulk_iops",
    "aerosol_select_wavelength_microns",
)
LAYER_OPTICAL_ABSORPTION_KEYS = ("absorption_tau", "gas_absorption_tau")
LAYER_OPTICAL_RAYLEIGH_KEY = "rayleigh_scattering_tau"
LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY = "aerosol_extinction_tau"
LAYER_OPTICAL_AEROSOL_SCATTERING_KEY = "aerosol_scattering_tau"
LAYER_OPTICAL_AEROSOL_SSA_KEY = "aerosol_single_scattering_albedo"


@dataclass(frozen=True)
class BenchmarkRow:
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
        if self.rt_seconds <= 0.0:
            return 0.0
        return self.wavelengths / self.rt_seconds


def input_store_kind(path: Path) -> str:
    if path.is_dir():
        return "array-directory"
    return "npz"


def input_keys(path: Path) -> set[str]:
    if path.is_dir():
        return {entry.stem for entry in path.glob("*.npy") if entry.is_file()}
    with np.load(path) as data:
        return set(data.files)


def load_input_arrays(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Loads selected benchmark input arrays.

    Directory inputs store one ``.npy`` file per array and are opened with
    memory mapping. Legacy ``.npz`` inputs are still read into memory.
    """
    if path.is_dir():
        names = (
            sorted(input_keys(path))
            if keys is None
            else [key for key in keys if (path / f"{key}.npy").is_file()]
        )
        return {key: np.load(path / f"{key}.npy", mmap_mode="r") for key in names}
    with np.load(path) as data:
        names = data.files if keys is None else [key for key in keys if key in data.files]
        return {key: np.array(data[key]) for key in names}


def require_directory_input_store(path: Path, *, label: str) -> None:
    if not path.is_dir():
        raise ValueError(
            f"{label} strict generated-input mode requires an array-directory input store, "
            "not a legacy .npz bundle"
        )


def select_layer_optical_keys(
    available: set[str],
    *,
    total_key: str,
    ssa_key: str,
) -> tuple[str, ...]:
    absorption_key = next(
        (key for key in LAYER_OPTICAL_ABSORPTION_KEYS if key in available),
        None,
    )
    if absorption_key is None or LAYER_OPTICAL_RAYLEIGH_KEY not in available:
        if set(SCENE_LAYER_GAS_TAU_REQUIRED_KEYS).issubset(available):
            return SCENE_LAYER_GAS_TAU_REQUIRED_KEYS + tuple(
                key for key in SCENE_LAYER_OPTIONAL_KEYS if key in available
            )
        if set(SCENE_LAYER_CROSS_SECTION_REQUIRED_KEYS).issubset(available):
            return SCENE_LAYER_CROSS_SECTION_REQUIRED_KEYS + tuple(
                key for key in SCENE_LAYER_OPTIONAL_KEYS if key in available
            )
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
    return any(key in keys for key in LAYER_OPTICAL_ABSORPTION_KEYS)


def layer_optical_keys_are_scene(keys: tuple[str, ...]) -> bool:
    keyset = set(keys)
    return set(SCENE_LAYER_GAS_TAU_REQUIRED_KEYS).issubset(keyset) or set(
        SCENE_LAYER_CROSS_SECTION_REQUIRED_KEYS
    ).issubset(keyset)


def require_python_generated_layer_optical_inputs(
    keys: tuple[str, ...],
    *,
    total_key: str,
    ssa_key: str,
    label: str,
) -> None:
    if keys == (total_key, ssa_key):
        raise ValueError(
            f"{label} strict generated-input mode requires component optical-depth "
            "fields such as absorption_tau and rayleigh_scattering_tau"
        )


def select_phase_optical_keys(
    available: set[str],
    *,
    label: str,
    use_dumped_derived_optics: bool,
    layer_optical_generates_fractions: bool,
    layer_optical_from_scene: bool,
    require_python_generated_inputs: bool,
    required_fraction_keys: tuple[str, ...],
    dumped_keys: tuple[str, ...],
    aerosol_interp_key: str,
    aerosol_coordinate_keys: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if require_python_generated_inputs and use_dumped_derived_optics:
        raise ValueError(
            f"{label} strict generated-input mode cannot be combined with "
            "--use-dumped-derived-optics"
        )
    if layer_optical_from_scene:
        required_physical = ("aerosol_moments",)
    elif layer_optical_generates_fractions:
        required_physical = ("depol", "aerosol_moments")
    else:
        required_physical = required_fraction_keys

    if not use_dumped_derived_optics and set(required_physical).issubset(available):
        keys = list(required_physical)
        if aerosol_interp_key in available:
            keys.append(aerosol_interp_key)
        else:
            keys.extend(key for key in aerosol_coordinate_keys if key in available)
        return tuple(keys)
    if require_python_generated_inputs:
        missing = ", ".join(key for key in required_physical if key not in available)
        raise ValueError(
            f"{label} strict generated-input mode requires physical phase inputs"
            + (f": {missing}" if missing else "")
        )
    return dumped_keys


def prepare_layer_optical_properties(
    bundle: dict[str, np.ndarray],
    *,
    total_key: str,
    ssa_key: str,
    validate_inputs: bool = True,
    include_fractions: bool = True,
) -> tuple[dict[str, np.ndarray], float, str]:
    """Builds ``tau``/``ssa`` and scattering fractions when components exist."""
    absorption_key = next((key for key in LAYER_OPTICAL_ABSORPTION_KEYS if key in bundle), None)
    if absorption_key is not None and LAYER_OPTICAL_RAYLEIGH_KEY in bundle:
        start = time.perf_counter()
        if not include_fractions:
            aerosol_ext = bundle.get(LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY)
            aerosol_scat = bundle.get(LAYER_OPTICAL_AEROSOL_SCATTERING_KEY)
            if aerosol_ext is None or aerosol_scat is not None:
                absorption = np.asarray(bundle[absorption_key], dtype=float)
                rayleigh = np.asarray(bundle[LAYER_OPTICAL_RAYLEIGH_KEY], dtype=float)
                aerosol_scat_arr = (
                    None if aerosol_scat is None else np.asarray(aerosol_scat, dtype=float)
                )
                aerosol_ext_arr = (
                    None if aerosol_ext is None else np.asarray(aerosol_ext, dtype=float)
                )
                if validate_inputs:
                    for name, value in (
                        (absorption_key, absorption),
                        (LAYER_OPTICAL_RAYLEIGH_KEY, rayleigh),
                        (LAYER_OPTICAL_AEROSOL_SCATTERING_KEY, aerosol_scat_arr),
                        (LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY, aerosol_ext_arr),
                    ):
                        if value is None:
                            continue
                        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
                            raise ValueError(f"{name} must be finite and nonnegative")
                    if aerosol_ext_arr is not None and aerosol_scat_arr is not None:
                        if np.any(aerosol_scat_arr > aerosol_ext_arr + 1.0e-14):
                            raise ValueError(
                                "aerosol_scattering_tau must not exceed aerosol_extinction_tau"
                            )
                aerosol_scat_sum = (
                    0.0 if aerosol_scat_arr is None else sum_component_axis(aerosol_scat_arr)
                )
                aerosol_ext_sum = (
                    aerosol_scat_sum
                    if aerosol_ext_arr is None
                    else sum_component_axis(aerosol_ext_arr)
                )
                total_tau = absorption + rayleigh + aerosol_ext_sum
                scattering_tau = rayleigh + aerosol_scat_sum
                ssa = ssa_from_optical_depth(total_tau, scattering_tau)
                prepared = dict(bundle)
                prepared[total_key] = total_tau
                prepared[ssa_key] = ssa
                prepared["_scattering_tau"] = scattering_tau
                return (
                    prepared,
                    time.perf_counter() - start,
                    "python-generated from component optical depths",
                )
        props = build_layer_optical_properties(
            absorption_tau=bundle[absorption_key],
            rayleigh_scattering_tau=bundle[LAYER_OPTICAL_RAYLEIGH_KEY],
            aerosol_extinction_tau=bundle.get(LAYER_OPTICAL_AEROSOL_EXTINCTION_KEY),
            aerosol_scattering_tau=bundle.get(LAYER_OPTICAL_AEROSOL_SCATTERING_KEY),
            aerosol_single_scattering_albedo=bundle.get(LAYER_OPTICAL_AEROSOL_SSA_KEY),
            validate_inputs=validate_inputs,
        )
        prepared = dict(bundle)
        prepared[total_key] = props.tau
        prepared[ssa_key] = props.ssa
        prepared["rayleigh_fraction"] = props.rayleigh_fraction
        prepared["aerosol_fraction"] = props.aerosol_fraction
        return (
            prepared,
            time.perf_counter() - start,
            "python-generated from component optical depths",
        )

    if set(SCENE_LAYER_GAS_TAU_REQUIRED_KEYS).issubset(bundle) or set(
        SCENE_LAYER_CROSS_SECTION_REQUIRED_KEYS
    ).issubset(bundle):
        start = time.perf_counter()
        profile = atmospheric_profile_from_levels(
            pressure_hpa=bundle["pressure_hpa"],
            temperature_k=bundle["temperature_k"],
            gas_vmr=bundle.get("gas_vmr"),
            heights_km=bundle.get("heights"),
            surface_altitude_m=scalar_value(bundle.get("surface_altitude_m", 0.0)),
        )
        aerosol_kwargs = {}
        if "aerosol_loadings" in bundle:
            aerosol_kwargs = {
                "aerosol_loadings": bundle["aerosol_loadings"],
                "aerosol_wavelengths_microns": bundle.get("aerosol_wavelengths_microns"),
                "aerosol_bulk_iops": bundle.get("aerosol_bulk_iops"),
                "aerosol_select_wavelength_microns": scalar_value(
                    bundle.get("aerosol_select_wavelength_microns", 0.4)
                ),
            }
        if "gas_absorption_tau" in bundle:
            scene = build_scene_layer_optical_properties_from_gas_tau(
                wavelengths_nm=bundle.get("opacity_wavelengths", bundle["wavelengths"]),
                profile=profile,
                gas_absorption_tau=bundle["gas_absorption_tau"],
                co2_ppmv=scalar_value(bundle.get("co2_ppmv", 385.0)),
                **aerosol_kwargs,
            )
        else:
            scene = build_scene_layer_optical_properties(
                wavelengths_nm=bundle.get("opacity_wavelengths", bundle["wavelengths"]),
                profile=profile,
                gas_cross_sections=bundle["gas_cross_sections"],
                co2_ppmv=scalar_value(bundle.get("co2_ppmv", 385.0)),
                **aerosol_kwargs,
            )
        prepared = dict(bundle)
        prepared["heights"] = profile.heights_km
        prepared[total_key] = scene.layer.tau
        prepared[ssa_key] = scene.layer.ssa
        prepared["absorption_tau"] = scene.gas_absorption_tau
        prepared["rayleigh_scattering_tau"] = scene.rayleigh_scattering_tau
        prepared["aerosol_extinction_tau"] = scene.aerosol_extinction_tau
        prepared["aerosol_scattering_tau"] = scene.aerosol_scattering_tau
        prepared["rayleigh_fraction"] = scene.layer.rayleigh_fraction
        prepared["aerosol_fraction"] = scene.layer.aerosol_fraction
        prepared["depol"] = scene.depol
        return (
            prepared,
            time.perf_counter() - start,
            "python-generated from scene/profile inputs",
        )

    return bundle, 0.0, "direct input"


def require_keys(bundle: dict[str, np.ndarray], keys: tuple[str, ...], *, label: str) -> None:
    missing = [key for key in keys if key not in bundle]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"{label} input store is missing required arrays: {missing_text}")


def public_bvp_solver(engine: str) -> str:
    if engine == "block":
        return "banded"
    if engine == "pentadiagonal":
        return "pentadiag"
    return "scipy"


def slice_spectral_rows(
    bundle: dict[str, np.ndarray],
    keys: tuple[str, ...],
    start: int,
    stop: int,
    *,
    optional_keys: tuple[str, ...] = ("ref_total",),
) -> dict[str, np.ndarray]:
    chunk = {key: bundle[key][start:stop] for key in keys}
    for key in optional_keys:
        if key in bundle:
            chunk[key] = bundle[key][start:stop]
    return chunk


def trim_spectral_rows(
    bundle: dict[str, np.ndarray],
    keys: tuple[str, ...],
    row_count: int,
    stop: int,
) -> dict[str, np.ndarray]:
    prepared = dict(bundle)
    for key in keys:
        if key in prepared and np.asarray(prepared[key]).shape[:1] == (row_count,):
            prepared[key] = prepared[key][:stop]
    return prepared


def recommended_chunk_size(
    *,
    total_rows: int,
    nlayers: int,
    backend: str,
    workload: str,
) -> int:
    if total_rows <= 0:
        return 0
    nlay = max(int(nlayers), 1)
    if workload == "solar_obs":
        row_floats = (48 if backend == "torch" else 40) * nlay + 64
        target_mib = 512 if backend == "torch" else 1400
        target_bytes = target_mib * 1024 * 1024
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
    input_path: Path,
    input_kind: str,
    wavelengths: int,
    layers: int,
    load_seconds: float | None = None,
    note: str | None = None,
) -> None:
    print(title)
    print(f"  input: {input_path}")
    print(f"  input kind: {input_kind}")
    print(f"  wavelengths: {wavelengths}")
    print(f"  layers: {layers}")
    if load_seconds is not None:
        print(f"  load (s): {load_seconds:.3f}")
    if note is not None:
        print(f"  note: {note}")


def print_preprocessing_summary(rows: tuple[tuple[str, str, float], ...]) -> None:
    for label, mode, seconds in rows:
        print(f"  {label}: {mode}, {seconds:.3f} s")
    total = sum(seconds for _, _, seconds in rows)
    labels = " + ".join(label for label, _, _ in rows)
    print(f"  preprocessing total: {total:.3f} s ({labels})")


def _format_optional(value: float | None) -> str:
    return f"{'-':>10}" if value is None else f"{value:10.3f}"


def _format_accuracy(value: float | None) -> str:
    return f"{'-':>14}" if value is None else f"{value:14.6e}"


def scalar_value(value: np.ndarray | float | int) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def looks_like_row_index(values: np.ndarray) -> bool:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        return False
    row_numbers = np.arange(1, grid.size + 1, dtype=float)
    return np.allclose(grid, row_numbers, rtol=0.0, atol=1.0e-12)


def accuracy_summary(value: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    abs_diff = float(np.max(np.abs(value - reference)))
    scale = np.maximum(np.abs(reference), 1.0e-15)
    rel_diff_pct = float(np.max(np.abs(value - reference) / scale) * 100.0)
    return abs_diff, rel_diff_pct


def print_rows(rows: list[BenchmarkRow]) -> None:
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
