"""Benchmark full-spectrum TIR input arrays with NumPy and optional PyTorch."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _full_spectrum_benchmark_common import (
    accuracy_summary,
    add_common_benchmark_arguments,
    BenchmarkRow,
    benchmark_input_source,
    load_input_arrays,
    load_packaged_reference_total,
    looks_like_row_index,
    layer_optical_keys_are_components,
    layer_optical_keys_are_scene,
    print_preprocessing_summary,
    print_problem_header,
    prepare_layer_optical_properties,
    print_rows,
    public_bvp_solver,
    recommended_chunk_size,
    require_python_generated_layer_optical_inputs,
    require_keys,
    scalar_value,
    select_layer_optical_keys,
    select_phase_optical_keys,
    slice_spectral_rows,
    trim_spectral_rows,
    validate_scene_input_args,
)
from py2sess import (
    thermal_source_from_temperature_profile,
    TwoStreamEss,
    TwoStreamEssOptions,
)
from py2sess.optical.phase import (
    aerosol_interp_fraction,
    build_two_stream_phase_inputs,
    build_two_stream_phase_inputs_from_scattering_tau,
)
from py2sess.rtsolver import precompute_fo_thermal_geometry_numpy
from py2sess.rtsolver.backend import has_torch
from py2sess.rtsolver.thermal_batch_numpy import _fo_thermal_toa, _two_stream_thermal_toa


_TIR_REQUIRED_PHYSICAL_OPTICS_KEYS = (
    "depol",
    "rayleigh_fraction",
    "aerosol_fraction",
    "aerosol_moments",
)
_TIR_COMPONENT_PHASE_KEYS = (
    "depol",
    "rayleigh_scattering_tau",
    "aerosol_scattering_tau",
    "aerosol_moments",
)

_TIR_AEROSOL_INTERP_KEY = "aerosol_interp_fraction"

_TIR_DUMPED_OPTICS_KEYS = ("asymm_arr", "d2s_scaling")

_TIR_BASE_KEYS = (
    "wavelengths",
    "heights",
    "user_angle",
    "albedo",
)

_TIR_OPTIONAL_KEYS = ("stream_value", "emissivity")

_TIR_DIRECT_SOURCE_KEYS = ("thermal_bb_input", "surfbb")
_TIR_TEMPERATURE_SOURCE_KEYS = ("level_temperature_k", "surface_temperature_k")
_TIR_SOURCE_COORDINATE_KEYS = (
    "wavenumber_band_cm_inv",
    "wavenumber_cm_inv",
    "wavelength_microns",
)
_TIR_AEROSOL_COORDINATE_KEYS = ("wavenumber_cm_inv", "wavelength_microns")

_TIR_CHUNK_KEYS = (
    "tau",
    "ssa",
    "g",
    "delta_m_truncation_factor",
    "planck",
    "surface_planck",
    "albedo",
    "emissivity",
)
_TIR_LIMIT_KEYS = (
    _TIR_BASE_KEYS
    + _TIR_DIRECT_SOURCE_KEYS
    + _TIR_SOURCE_COORDINATE_KEYS
    + _TIR_DUMPED_OPTICS_KEYS
    + (
        "emissivity",
        "surface_temperature_k",
        "depol",
        "rayleigh_fraction",
        "aerosol_fraction",
        "aerosol_interp_fraction",
    )
)


def _positive_coordinate(name: str, values: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} must be finite")
    if np.any(coordinate <= 0.0):
        raise ValueError(f"{name} must be positive")
    return coordinate


def _tir_aerosol_interp_fraction(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if _TIR_AEROSOL_INTERP_KEY in bundle:
        return bundle[_TIR_AEROSOL_INTERP_KEY], "python-generated"

    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    coordinate_name = "wavelengths"
    if "wavelength_microns" in bundle and looks_like_row_index(wavelengths):
        wavelengths = _positive_coordinate("wavelength_microns", bundle["wavelength_microns"])
        coordinate_name = "wavelength_microns"
    elif "wavenumber_cm_inv" in bundle and looks_like_row_index(wavelengths):
        wavenumber = _positive_coordinate("wavenumber_cm_inv", bundle["wavenumber_cm_inv"])
        wavelengths = 10000.0 / wavenumber
        coordinate_name = "wavenumber_cm_inv"

    if looks_like_row_index(wavelengths):
        raise ValueError(
            "TIR aerosol interpolation requires aerosol_interp_fraction or a "
            "physical spectral coordinate when wavelengths contains row indices"
        )
    _positive_coordinate(coordinate_name, wavelengths)
    return (
        aerosol_interp_fraction(wavelengths, reverse=True),
        f"python-generated (aerosol interpolation from {coordinate_name})",
    )


def _select_source_keys(
    available: set[str],
    *,
    use_dumped_thermal_source: bool,
    require_python_generated_inputs: bool = False,
) -> tuple[str, ...]:
    has_temperature = set(_TIR_TEMPERATURE_SOURCE_KEYS).issubset(available)
    coordinate = next((key for key in _TIR_SOURCE_COORDINATE_KEYS if key in available), None)
    if require_python_generated_inputs and use_dumped_thermal_source:
        raise ValueError(
            "TIR strict generated-input mode cannot be combined with --use-dumped-thermal-source"
        )
    if not use_dumped_thermal_source and has_temperature and coordinate is not None:
        return _TIR_TEMPERATURE_SOURCE_KEYS + (coordinate,)
    if require_python_generated_inputs:
        missing = [key for key in _TIR_TEMPERATURE_SOURCE_KEYS if key not in available]
        if coordinate is None:
            missing.append("wavenumber_band_cm_inv, wavenumber_cm_inv, or wavelength_microns")
        missing_text = ", ".join(missing)
        raise ValueError(
            "TIR strict generated-input mode requires temperature-based thermal source inputs"
            + (f": {missing_text}" if missing_text else "")
        )
    return _TIR_DIRECT_SOURCE_KEYS


def _prepare_optics(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    has_component_phase = all(key in bundle for key in _TIR_COMPONENT_PHASE_KEYS)
    has_fraction_phase = all(key in bundle for key in _TIR_REQUIRED_PHYSICAL_OPTICS_KEYS)
    if use_dumped_derived_optics or not (has_component_phase or has_fraction_phase):
        require_keys(bundle, _TIR_DUMPED_OPTICS_KEYS, label="TIR dumped optical")
        prepared = dict(bundle)
        prepared["g"] = bundle["asymm_arr"]
        prepared["delta_m_truncation_factor"] = bundle["d2s_scaling"]
        mode = "dumped-derived"
        if not use_dumped_derived_optics and not (has_component_phase or has_fraction_phase):
            mode = "dumped-derived (physical optical inputs unavailable)"
        return prepared, 0.0, mode

    start = time.perf_counter()
    fac, mode = _tir_aerosol_interp_fraction(bundle)
    if has_component_phase:
        optics = build_two_stream_phase_inputs_from_scattering_tau(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_scattering_tau=bundle["rayleigh_scattering_tau"],
            aerosol_scattering_tau=bundle["aerosol_scattering_tau"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            scattering_tau=bundle.get("_scattering_tau"),
            validate_inputs=False,
        )
    else:
        optics = build_two_stream_phase_inputs(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_fraction=bundle["rayleigh_fraction"],
            aerosol_fraction=bundle["aerosol_fraction"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            validate_inputs=False,
        )
    prepared = dict(bundle)
    prepared[_TIR_AEROSOL_INTERP_KEY] = fac
    prepared["g"] = optics.g
    prepared["delta_m_truncation_factor"] = optics.delta_m_truncation_factor
    return prepared, time.perf_counter() - start, mode


def _prepare_thermal_source(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_thermal_source: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    source_coordinates = [key for key in _TIR_SOURCE_COORDINATE_KEYS if key in bundle]
    has_temperature = all(key in bundle for key in _TIR_TEMPERATURE_SOURCE_KEYS)
    if not use_dumped_thermal_source and has_temperature and source_coordinates:
        start = time.perf_counter()
        coordinate_name = source_coordinates[0]
        kwargs = {coordinate_name: bundle[coordinate_name]}
        source = thermal_source_from_temperature_profile(
            bundle["level_temperature_k"],
            bundle["surface_temperature_k"],
            **kwargs,
        )
        prepared = dict(bundle)
        prepared["planck"] = np.asarray(source.planck, dtype=float)
        prepared["surface_planck"] = np.asarray(source.surface_planck, dtype=float)
        _validate_thermal_source_shapes(prepared)
        return prepared, time.perf_counter() - start, f"temperature ({coordinate_name})"

    require_keys(bundle, _TIR_DIRECT_SOURCE_KEYS, label="TIR thermal source")
    prepared = dict(bundle)
    prepared["planck"] = bundle["thermal_bb_input"]
    prepared["surface_planck"] = bundle["surfbb"]
    _validate_thermal_source_shapes(prepared)
    return prepared, 0.0, "bundle"


def _validate_thermal_source_shapes(bundle: dict[str, np.ndarray]) -> None:
    n_rows, n_layers = bundle["tau"].shape
    planck = np.asarray(bundle["planck"], dtype=float)
    surface = np.asarray(bundle["surface_planck"], dtype=float)
    expected_planck = (n_rows, n_layers + 1)
    if planck.shape == (n_layers + 1,) and n_rows == 1:
        bundle["planck"] = planck.reshape(1, n_layers + 1)
    elif planck.shape != expected_planck:
        raise ValueError(f"thermal source planck must have shape {expected_planck}")
    if surface.shape == () and n_rows == 1:
        bundle["surface_planck"] = surface.reshape(1)
    elif surface.shape != (n_rows,):
        raise ValueError(f"surface_planck must have shape ({n_rows},)")


def _prepare_surface(bundle: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], str]:
    if "emissivity" in bundle:
        return bundle, "direct input"
    prepared = dict(bundle)
    prepared["emissivity"] = 1.0 - np.asarray(bundle["albedo"], dtype=float)
    return prepared, "1 - albedo"


def _prepare_geometry(bundle: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    heights = np.asarray(bundle["heights"], dtype=float)
    user_angle = scalar_value(bundle["user_angle"])
    prepared = dict(bundle)
    prepared["fo_geometry"] = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    prepared["user_stream"] = np.array([np.cos(np.deg2rad(user_angle))], dtype=float)
    return prepared, time.perf_counter() - start


def benchmark_numpy(
    bundle: dict[str, np.ndarray],
    *,
    wavelengths: int,
    chunk_size: int,
    numpy_bvp_engine: str,
) -> BenchmarkRow:
    """Runs the NumPy TIR full-spectrum benchmark."""
    wall_start = time.perf_counter()
    heights = np.asarray(bundle["heights"], dtype=float)
    user_angle = scalar_value(bundle["user_angle"])
    geometry = bundle["fo_geometry"]
    user_stream = scalar_value(bundle["user_stream"])
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    checksum = 0.0
    max_abs_diff = None
    max_rel_diff_pct = None
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, _TIR_CHUNK_KEYS, start, stop)
        t0 = time.perf_counter()
        two_stream = _two_stream_thermal_toa(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            asymm=chunk["g"],
            scaling=chunk["delta_m_truncation_factor"],
            thermal_bb_input=chunk["planck"],
            surfbb=chunk["surface_planck"],
            emissivity=chunk["emissivity"],
            albedo=chunk["albedo"],
            stream_value=0.5,
            user_stream=user_stream,
            thermal_tcutoff=1.0e-8,
            bvp_engine=numpy_bvp_engine,
        )
        two_stream_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        fo = _fo_thermal_toa(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            scaling=chunk["delta_m_truncation_factor"],
            thermal_bb_input=chunk["planck"],
            surfbb=chunk["surface_planck"],
            emissivity=chunk["emissivity"],
            heights=heights,
            user_angle_degrees=user_angle,
            earth_radius=6371.0,
            nfine=3,
            geometry=geometry,
        )
        fo_seconds += time.perf_counter() - t0
        total = two_stream + fo
        checksum += float(np.sum(total))
        if "ref_total" in chunk:
            chunk_abs_diff, chunk_rel_diff_pct = accuracy_summary(total, chunk["ref_total"])
            max_abs_diff = (
                chunk_abs_diff if max_abs_diff is None else max(max_abs_diff, chunk_abs_diff)
            )
            max_rel_diff_pct = (
                chunk_rel_diff_pct
                if max_rel_diff_pct is None
                else max(max_rel_diff_pct, chunk_rel_diff_pct)
            )
    wall_seconds = time.perf_counter() - wall_start
    rt_seconds = fo_seconds + two_stream_seconds
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    return BenchmarkRow(
        backend="numpy",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=rt_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def benchmark_numpy_forward(
    bundle: dict[str, np.ndarray],
    *,
    wavelengths: int,
    numpy_bvp_engine: str,
    output_levels: bool,
) -> BenchmarkRow:
    """Runs the public NumPy ``TwoStreamEss.forward`` full-spectrum path."""
    wall_start = time.perf_counter()
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=int(bundle["tau"].shape[1]),
            mode="thermal",
            bvp_solver=public_bvp_solver(numpy_bvp_engine),
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau"],
        ssa=bundle["ssa"],
        g=bundle["g"],
        z=bundle["heights"],
        angles=scalar_value(bundle["user_angle"]),
        stream=0.5,
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["delta_m_truncation_factor"],
        planck=bundle["planck"],
        surface_planck=bundle["surface_planck"],
        emissivity=bundle["emissivity"],
        include_fo=True,
    )
    total = np.asarray(result.radiance_total, dtype=float)
    checksum = float(np.sum(total))
    wall_seconds = time.perf_counter() - wall_start
    max_abs_diff = None
    max_rel_diff_pct = None
    if "ref_total" in bundle:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(total, bundle["ref_total"])
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=int(bundle["tau"].shape[1]),
        backend="numpy",
        workload="thermal",
    )
    return BenchmarkRow(
        backend="numpy-forward-levels" if output_levels else "numpy-forward",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=wall_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def benchmark_torch(
    bundle: dict[str, np.ndarray],
    *,
    wavelengths: int,
    chunk_size: int,
    torch_dtype_name: str,
    torch_device_name: str,
    torch_threads: int,
    torch_bvp_engine: str,
) -> BenchmarkRow:
    """Runs the PyTorch TIR full-spectrum benchmark."""
    if not has_torch():
        raise RuntimeError("PyTorch is not installed")
    import torch

    from py2sess.rtsolver.thermal_batch_torch import (
        _as_tensor,
        _fo_thermal_toa_batch,
        _two_stream_thermal_toa_batch,
    )

    wall_start = time.perf_counter()
    device = torch.device(torch_device_name)
    dtype = {"float64": torch.float64, "float32": torch.float32}[torch_dtype_name]
    torch.set_num_threads(torch_threads)
    with torch.no_grad():
        probe = torch.ones(16, dtype=dtype, device=device)
        _ = (probe + 1.0).sum().item()

    heights = np.asarray(bundle["heights"], dtype=float)
    user_angle = scalar_value(bundle["user_angle"])
    geometry = bundle["fo_geometry"]
    user_stream = scalar_value(bundle["user_stream"])
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    checksum = 0.0
    max_abs_diff = None
    max_rel_diff_pct = None
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, _TIR_CHUNK_KEYS, start, stop)
        with torch.no_grad():
            tau_t = _as_tensor(chunk["tau"], dtype=dtype, device=device)
            omega_t = _as_tensor(chunk["ssa"], dtype=dtype, device=device)
            asymm_t = _as_tensor(chunk["g"], dtype=dtype, device=device)
            scaling_t = _as_tensor(chunk["delta_m_truncation_factor"], dtype=dtype, device=device)
            bb_t = _as_tensor(chunk["planck"], dtype=dtype, device=device)
            surfbb_t = _as_tensor(chunk["surface_planck"], dtype=dtype, device=device)
            albedo_t = _as_tensor(chunk["albedo"], dtype=dtype, device=device)
            emissivity_t = _as_tensor(chunk["emissivity"], dtype=dtype, device=device)
            t0 = time.perf_counter()
            two_stream = _two_stream_thermal_toa_batch(
                tau=tau_t,
                omega=omega_t,
                asymm=asymm_t,
                scaling=scaling_t,
                thermal_bb_input=bb_t,
                surfbb=surfbb_t,
                emissivity=emissivity_t,
                albedo=albedo_t,
                stream_value=0.5,
                user_stream=user_stream,
                pxsq=0.25,
                thermal_tcutoff=1.0e-8,
                bvp_device=None,
                bvp_dtype=dtype,
                bvp_engine=torch_bvp_engine,
            )
            two_stream_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            fo = _fo_thermal_toa_batch(
                tau=tau_t,
                omega=omega_t,
                scaling=scaling_t,
                thermal_bb_input=bb_t,
                surfbb=surfbb_t,
                emissivity=emissivity_t,
                heights=heights,
                user_angle_degrees=user_angle,
                earth_radius=6371.0,
                nfine=3,
                fo_geometry=geometry,
            )
            fo_seconds += time.perf_counter() - t0
            total = two_stream + fo
            checksum += float(total.sum().item())
            if "ref_total" in chunk:
                chunk_abs_diff, chunk_rel_diff_pct = accuracy_summary(
                    total.detach().cpu().numpy(),
                    np.asarray(chunk["ref_total"], dtype=float),
                )
                max_abs_diff = (
                    chunk_abs_diff if max_abs_diff is None else max(max_abs_diff, chunk_abs_diff)
                )
                max_rel_diff_pct = (
                    chunk_rel_diff_pct
                    if max_rel_diff_pct is None
                    else max(max_rel_diff_pct, chunk_rel_diff_pct)
                )
    wall_seconds = time.perf_counter() - wall_start
    rt_seconds = fo_seconds + two_stream_seconds
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    return BenchmarkRow(
        backend=f"torch-{torch_device_name}-{torch_dtype_name}",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=rt_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def benchmark_torch_forward(
    bundle: dict[str, np.ndarray],
    *,
    wavelengths: int,
    torch_dtype_name: str,
    torch_device_name: str,
    torch_threads: int,
    torch_bvp_engine: str,
    output_levels: bool,
) -> BenchmarkRow:
    """Runs the public torch ``TwoStreamEss.forward`` full-spectrum path."""
    if not has_torch():
        raise RuntimeError("PyTorch is not installed")
    import torch

    torch.set_num_threads(torch_threads)
    wall_start = time.perf_counter()
    dtype = {"float64": torch.float64, "float32": torch.float32}[torch_dtype_name]
    device = torch.device(torch_device_name)
    with torch.no_grad():
        probe = torch.ones(16, dtype=dtype, device=device)
        _ = (probe + 1.0).sum().item()
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=int(bundle["tau"].shape[1]),
            mode="thermal",
            backend="torch",
            bvp_solver=public_bvp_solver(torch_bvp_engine),
            torch_device=torch_device_name,
            torch_dtype=torch_dtype_name,
            torch_enable_grad=False,
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau"],
        ssa=bundle["ssa"],
        g=bundle["g"],
        z=bundle["heights"],
        angles=scalar_value(bundle["user_angle"]),
        stream=0.5,
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["delta_m_truncation_factor"],
        planck=bundle["planck"],
        surface_planck=bundle["surface_planck"],
        emissivity=bundle["emissivity"],
        include_fo=True,
    )
    total = result.radiance_total.detach().cpu().numpy()
    checksum = float(np.sum(total))
    wall_seconds = time.perf_counter() - wall_start
    max_abs_diff = None
    max_rel_diff_pct = None
    if "ref_total" in bundle:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(total, bundle["ref_total"])
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=int(bundle["tau"].shape[1]),
        backend="torch",
        workload="thermal",
    )
    return BenchmarkRow(
        backend=(
            f"torch-{torch_device_name}-{torch_dtype_name}-forward-levels"
            if output_levels
            else f"torch-{torch_device_name}-{torch_dtype_name}-forward"
        ),
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=wall_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def main() -> None:
    """Runs the full-spectrum TIR benchmark example."""
    parser = argparse.ArgumentParser()
    add_common_benchmark_arguments(
        parser,
        input_help="Path to a TIR runtime array directory, or a legacy full-spectrum .npz bundle.",
        torch_bvp_choices=("auto", "block", "pentadiagonal"),
    )
    parser.add_argument(
        "--use-dumped-thermal-source",
        action="store_true",
        help="Use stored thermal_bb_input/surfbb instead of temperature-based source generation.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    scene_mode = validate_scene_input_args(
        parser,
        args,
        forbidden_scene_flags=(
            ("use_dumped_derived_optics", "--use-dumped-derived-optics"),
            ("use_dumped_thermal_source", "--use-dumped-thermal-source"),
        ),
    )

    bundle, available, input_path, input_kind = benchmark_input_source(
        args,
        kind="tir",
        label="TIR",
        scene_mode=scene_mode,
    )

    layer_optical_keys = select_layer_optical_keys(
        available,
        total_key="tau_arr",
        ssa_key="omega_arr",
    )
    layer_from_scene = layer_optical_keys_are_scene(layer_optical_keys)
    layer_from_components = layer_optical_keys_are_components(layer_optical_keys)
    base_keys = tuple(key for key in _TIR_BASE_KEYS if key != "heights" or not layer_from_scene)
    if args.require_python_generated_inputs:
        require_python_generated_layer_optical_inputs(
            layer_optical_keys,
            total_key="tau_arr",
            ssa_key="omega_arr",
            label="TIR",
        )
    optical_keys = select_phase_optical_keys(
        available,
        label="TIR",
        use_dumped_derived_optics=args.use_dumped_derived_optics,
        layer_optical_generates_fractions=layer_from_scene or layer_from_components,
        layer_optical_from_scene=layer_from_scene,
        require_python_generated_inputs=args.require_python_generated_inputs,
        required_fraction_keys=_TIR_REQUIRED_PHYSICAL_OPTICS_KEYS,
        dumped_keys=_TIR_DUMPED_OPTICS_KEYS,
        aerosol_interp_key=_TIR_AEROSOL_INTERP_KEY,
        aerosol_coordinate_keys=_TIR_AEROSOL_COORDINATE_KEYS,
    )
    source_keys = _select_source_keys(
        available,
        use_dumped_thermal_source=args.use_dumped_thermal_source,
        require_python_generated_inputs=args.require_python_generated_inputs,
    )
    if not scene_mode:
        bundle = load_input_arrays(
            args.input,
            keys=base_keys + _TIR_OPTIONAL_KEYS + layer_optical_keys + optical_keys + source_keys,
        )
    require_keys(
        bundle,
        base_keys + layer_optical_keys + optical_keys + source_keys,
        label="TIR",
    )
    if layer_from_scene:
        row_count_key = "wavelengths"
    elif layer_from_components:
        row_count_key = layer_optical_keys[0]
    else:
        row_count_key = "tau_arr"
    total_rows = int(bundle[row_count_key].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = trim_spectral_rows(
        bundle,
        _TIR_LIMIT_KEYS + layer_optical_keys,
        total_rows,
        wavelengths,
    )
    if not scene_mode and args.input.name == "tir_benchmark_fixture.npz":
        bundle["ref_total"] = load_packaged_reference_total("tir_reference_outputs.npz")[
            :wavelengths
        ]
    load_seconds = time.perf_counter() - load_start

    bundle, layer_optical_seconds, layer_optical_mode = prepare_layer_optical_properties(
        bundle,
        total_key="tau_arr",
        ssa_key="omega_arr",
        validate_inputs=not args.require_python_generated_inputs,
    )
    bundle["tau"] = bundle["tau_arr"]
    bundle["ssa"] = bundle["omega_arr"]
    bundle, geometry_seconds = _prepare_geometry(bundle)
    bundle, optical_seconds, optical_mode = _prepare_optics(
        bundle,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )
    bundle, source_seconds, source_mode = _prepare_thermal_source(
        bundle,
        use_dumped_thermal_source=args.use_dumped_thermal_source,
    )
    bundle, emissivity_mode = _prepare_surface(bundle)

    print_problem_header(
        title="TIR full-spectrum benchmark",
        input_path=input_path,
        input_kind=input_kind,
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        load_seconds=load_seconds,
        note="RT time is FO + 2S. wall time excludes load/preprocessing.",
    )
    print_preprocessing_summary(
        (
            ("layer optical properties", layer_optical_mode, layer_optical_seconds),
            ("geometry preprocessing", "python-generated", geometry_seconds),
            ("optical preprocessing", optical_mode, optical_seconds),
            ("thermal source", source_mode, source_seconds),
        )
    )
    print(f"  emissivity: {emissivity_mode}")

    rows: list[BenchmarkRow] = []
    if args.backend in {"numpy", "both"}:
        chunk_size = args.chunk_size or recommended_chunk_size(
            total_rows=wavelengths,
            nlayers=int(bundle["tau"].shape[1]),
            backend="numpy",
            workload="thermal",
        )
        rows.append(
            benchmark_numpy(
                bundle,
                wavelengths=wavelengths,
                chunk_size=chunk_size,
                numpy_bvp_engine=args.numpy_bvp_engine,
            )
        )
        rows.append(
            benchmark_numpy_forward(
                bundle,
                wavelengths=wavelengths,
                numpy_bvp_engine=args.numpy_bvp_engine,
                output_levels=args.output_levels,
            )
        )
    if args.backend in {"torch", "both"}:
        chunk_size = args.chunk_size or recommended_chunk_size(
            total_rows=wavelengths,
            nlayers=int(bundle["tau"].shape[1]),
            backend="torch",
            workload="thermal",
        )
        rows.append(
            benchmark_torch(
                bundle,
                wavelengths=wavelengths,
                chunk_size=chunk_size,
                torch_dtype_name=args.torch_dtype,
                torch_device_name=args.torch_device,
                torch_threads=args.torch_threads,
                torch_bvp_engine=args.torch_bvp_engine,
            )
        )
        rows.append(
            benchmark_torch_forward(
                bundle,
                wavelengths=wavelengths,
                torch_dtype_name=args.torch_dtype,
                torch_device_name=args.torch_device,
                torch_threads=args.torch_threads,
                torch_bvp_engine=args.torch_bvp_engine,
                output_levels=args.output_levels,
            )
        )
    print_rows(rows)


if __name__ == "__main__":
    main()
