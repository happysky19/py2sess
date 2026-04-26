"""Benchmark a full-spectrum TIR bundle with NumPy and optional PyTorch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _full_spectrum_benchmark_common import (
    accuracy_summary,
    BenchmarkRow,
    bundle_keys,
    load_bundle,
    looks_like_row_index,
    print_problem_header,
    print_rows,
    recommended_chunk_size,
    require_keys,
    scalar_value,
)
from py2sess import (
    thermal_source_from_temperature_profile,
    TwoStreamEss,
    TwoStreamEssOptions,
)
from py2sess.optical.phase import aerosol_interp_fraction, build_two_stream_phase_inputs
from py2sess.rtsolver import precompute_fo_thermal_geometry_numpy
from py2sess.rtsolver.backend import has_torch
from py2sess.rtsolver.thermal_batch_numpy import _fo_thermal_toa, _two_stream_thermal_toa


def _public_bvp_solver(engine: str) -> str:
    """Maps low-level benchmark BVP names to public option names."""
    if engine == "block":
        return "banded"
    if engine == "pentadiagonal":
        return "pentadiag"
    return "scipy"


_TIR_REQUIRED_PHYSICAL_OPTICS_KEYS = (
    "depol",
    "rayleigh_fraction",
    "aerosol_fraction",
    "aerosol_moments",
)

_TIR_AEROSOL_INTERP_KEY = "aerosol_interp_fraction"

_TIR_DUMPED_OPTICS_KEYS = ("asymm_arr", "d2s_scaling")

_TIR_BASE_KEYS = (
    "wavelengths",
    "heights",
    "user_angle",
    "tau_arr",
    "omega_arr",
    "albedo",
)

_TIR_OPTIONAL_KEYS = ("stream_value", "ref_total", "emissivity")

_TIR_DIRECT_SOURCE_KEYS = ("thermal_bb_input", "surfbb")
_TIR_TEMPERATURE_SOURCE_KEYS = ("level_temperature_k", "surface_temperature_k")
_TIR_SOURCE_COORDINATE_KEYS = ("wavenumber_cm_inv", "wavelength_microns")


def _has_keys(bundle: dict[str, np.ndarray], keys: tuple[str, ...]) -> bool:
    return all(key in bundle for key in keys)


def _select_optical_keys(
    available: set[str],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[str, ...]:
    has_physical_optics = set(_TIR_REQUIRED_PHYSICAL_OPTICS_KEYS).issubset(available)
    if not use_dumped_derived_optics and has_physical_optics:
        keys = list(_TIR_REQUIRED_PHYSICAL_OPTICS_KEYS)
        if _TIR_AEROSOL_INTERP_KEY in available:
            keys.append(_TIR_AEROSOL_INTERP_KEY)
        else:
            keys.extend(key for key in _TIR_SOURCE_COORDINATE_KEYS if key in available)
        return tuple(keys)
    return _TIR_DUMPED_OPTICS_KEYS


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
) -> tuple[str, ...]:
    has_temperature = set(_TIR_TEMPERATURE_SOURCE_KEYS).issubset(available)
    coordinates = tuple(key for key in _TIR_SOURCE_COORDINATE_KEYS if key in available)
    if not use_dumped_thermal_source and has_temperature and len(coordinates) >= 1:
        return _TIR_TEMPERATURE_SOURCE_KEYS + coordinates
    return _TIR_DIRECT_SOURCE_KEYS


def _prepare_optics(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    if use_dumped_derived_optics or not _has_keys(bundle, _TIR_REQUIRED_PHYSICAL_OPTICS_KEYS):
        require_keys(bundle, _TIR_DUMPED_OPTICS_KEYS, label="TIR dumped optical")
        mode = "dumped-derived"
        if not use_dumped_derived_optics and not _has_keys(
            bundle, _TIR_REQUIRED_PHYSICAL_OPTICS_KEYS
        ):
            mode = "dumped-derived (physical optical inputs unavailable)"
        return bundle, 0.0, mode

    start = time.perf_counter()
    fac, mode = _tir_aerosol_interp_fraction(bundle)
    optics = build_two_stream_phase_inputs(
        ssa=bundle["omega_arr"],
        depol=bundle["depol"],
        rayleigh_fraction=bundle["rayleigh_fraction"],
        aerosol_fraction=bundle["aerosol_fraction"],
        aerosol_moments=bundle["aerosol_moments"],
        aerosol_interp_fraction=fac,
    )
    prepared = dict(bundle)
    prepared[_TIR_AEROSOL_INTERP_KEY] = fac
    prepared["asymm_arr"] = optics.g
    prepared["d2s_scaling"] = optics.delta_m_truncation_factor
    return prepared, time.perf_counter() - start, mode


def _prepare_thermal_source(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_thermal_source: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    source_coordinates = [key for key in _TIR_SOURCE_COORDINATE_KEYS if key in bundle]
    has_temperature = _has_keys(bundle, _TIR_TEMPERATURE_SOURCE_KEYS)
    if not use_dumped_thermal_source and has_temperature and source_coordinates:
        if len(source_coordinates) != 1:
            raise ValueError(
                "temperature-based thermal source requires exactly one of "
                "wavenumber_cm_inv or wavelength_microns"
            )
        start = time.perf_counter()
        coordinate_name = source_coordinates[0]
        kwargs = {coordinate_name: bundle[coordinate_name]}
        source = thermal_source_from_temperature_profile(
            bundle["level_temperature_k"],
            bundle["surface_temperature_k"],
            **kwargs,
        )
        prepared = dict(bundle)
        prepared["thermal_bb_input"] = np.asarray(source.planck, dtype=float)
        prepared["surfbb"] = np.asarray(source.surface_planck, dtype=float)
        _validate_thermal_source_shapes(prepared)
        return prepared, time.perf_counter() - start, f"temperature ({coordinate_name})"

    require_keys(bundle, _TIR_DIRECT_SOURCE_KEYS, label="TIR thermal source")
    return bundle, 0.0, "bundle"


def _validate_thermal_source_shapes(bundle: dict[str, np.ndarray]) -> None:
    n_rows, n_layers = bundle["tau_arr"].shape
    planck = np.asarray(bundle["thermal_bb_input"], dtype=float)
    surface = np.asarray(bundle["surfbb"], dtype=float)
    expected_planck = (n_rows, n_layers + 1)
    if planck.shape == (n_layers + 1,) and n_rows == 1:
        bundle["thermal_bb_input"] = planck.reshape(1, n_layers + 1)
    elif planck.shape != expected_planck:
        raise ValueError(f"thermal source planck must have shape {expected_planck}")
    if surface.shape == () and n_rows == 1:
        bundle["surfbb"] = surface.reshape(1)
    elif surface.shape != (n_rows,):
        raise ValueError(f"surface_planck must have shape ({n_rows},)")


def _prepare_surface(bundle: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], str]:
    if "emissivity" in bundle:
        return bundle, "bundle"
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


def _slice_rows(bundle: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Returns one spectral chunk from a TIR benchmark bundle."""
    chunk = {
        "tau_arr": bundle["tau_arr"][start:stop],
        "omega_arr": bundle["omega_arr"][start:stop],
        "asymm_arr": bundle["asymm_arr"][start:stop],
        "d2s_scaling": bundle["d2s_scaling"][start:stop],
        "thermal_bb_input": bundle["thermal_bb_input"][start:stop],
        "surfbb": bundle["surfbb"][start:stop],
        "albedo": bundle["albedo"][start:stop],
        "emissivity": bundle["emissivity"][start:stop],
    }
    if "ref_total" in bundle:
        chunk["ref_total"] = bundle["ref_total"][start:stop]
    return chunk


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
        chunk = _slice_rows(bundle, start, stop)
        t0 = time.perf_counter()
        two_stream = _two_stream_thermal_toa(
            tau=chunk["tau_arr"],
            omega=chunk["omega_arr"],
            asymm=chunk["asymm_arr"],
            scaling=chunk["d2s_scaling"],
            thermal_bb_input=chunk["thermal_bb_input"],
            surfbb=chunk["surfbb"],
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
            tau=chunk["tau_arr"],
            omega=chunk["omega_arr"],
            scaling=chunk["d2s_scaling"],
            thermal_bb_input=chunk["thermal_bb_input"],
            surfbb=chunk["surfbb"],
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
        layers=int(bundle["tau_arr"].shape[1]),
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
            nlyr=int(bundle["tau_arr"].shape[1]),
            mode="thermal",
            bvp_solver=_public_bvp_solver(numpy_bvp_engine),
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau_arr"],
        ssa=bundle["omega_arr"],
        g=bundle["asymm_arr"],
        z=bundle["heights"],
        angles=scalar_value(bundle["user_angle"]),
        stream=0.5,
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["d2s_scaling"],
        planck=bundle["thermal_bb_input"],
        surface_planck=bundle["surfbb"],
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
        nlayers=int(bundle["tau_arr"].shape[1]),
        backend="numpy",
        workload="thermal",
    )
    return BenchmarkRow(
        backend="numpy-forward-levels" if output_levels else "numpy-forward",
        wavelengths=wavelengths,
        layers=int(bundle["tau_arr"].shape[1]),
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
        chunk = _slice_rows(bundle, start, stop)
        with torch.no_grad():
            tau_t = _as_tensor(chunk["tau_arr"], dtype=dtype, device=device)
            omega_t = _as_tensor(chunk["omega_arr"], dtype=dtype, device=device)
            asymm_t = _as_tensor(chunk["asymm_arr"], dtype=dtype, device=device)
            scaling_t = _as_tensor(chunk["d2s_scaling"], dtype=dtype, device=device)
            bb_t = _as_tensor(chunk["thermal_bb_input"], dtype=dtype, device=device)
            surfbb_t = _as_tensor(chunk["surfbb"], dtype=dtype, device=device)
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
        layers=int(bundle["tau_arr"].shape[1]),
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
            nlyr=int(bundle["tau_arr"].shape[1]),
            mode="thermal",
            backend="torch",
            bvp_solver=_public_bvp_solver(torch_bvp_engine),
            torch_device=torch_device_name,
            torch_dtype=torch_dtype_name,
            torch_enable_grad=False,
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau_arr"],
        ssa=bundle["omega_arr"],
        g=bundle["asymm_arr"],
        z=bundle["heights"],
        angles=scalar_value(bundle["user_angle"]),
        stream=0.5,
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["d2s_scaling"],
        planck=bundle["thermal_bb_input"],
        surface_planck=bundle["surfbb"],
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
        nlayers=int(bundle["tau_arr"].shape[1]),
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
        layers=int(bundle["tau_arr"].shape[1]),
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=wall_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def main() -> None:
    """Runs the full-spectrum TIR benchmark example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle", type=Path, help="Path to a full-spectrum TIR benchmark bundle (.npz)."
    )
    parser.add_argument("--backend", choices=["numpy", "torch", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Optional spectral-row limit.")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Optional chunk size override."
    )
    parser.add_argument("--numpy-bvp-engine", choices=["auto", "block"], default="auto")
    parser.add_argument(
        "--torch-bvp-engine", choices=["auto", "block", "pentadiagonal"], default="auto"
    )
    parser.add_argument("--torch-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--torch-dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output-levels",
        action="store_true",
        help="Benchmark the public forward profile path instead of endpoint-only output.",
    )
    parser.add_argument(
        "--use-dumped-derived-optics",
        action="store_true",
        help="Use stored g and delta-M factor instead of Python preprocessing.",
    )
    parser.add_argument(
        "--use-dumped-thermal-source",
        action="store_true",
        help="Use stored thermal_bb_input/surfbb instead of temperature-based source generation.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    available = bundle_keys(args.bundle)
    optical_keys = _select_optical_keys(
        available,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )
    source_keys = _select_source_keys(
        available,
        use_dumped_thermal_source=args.use_dumped_thermal_source,
    )
    bundle = load_bundle(
        args.bundle,
        keys=_TIR_BASE_KEYS + _TIR_OPTIONAL_KEYS + optical_keys + source_keys,
    )
    require_keys(
        bundle,
        _TIR_BASE_KEYS + optical_keys + source_keys,
        label="TIR",
    )
    total_rows = int(bundle["tau_arr"].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = dict(bundle)
    bundle["wavelengths"] = bundle["wavelengths"][:wavelengths]
    bundle["tau_arr"] = bundle["tau_arr"][:wavelengths]
    bundle["omega_arr"] = bundle["omega_arr"][:wavelengths]
    bundle["albedo"] = bundle["albedo"][:wavelengths]
    if "emissivity" in bundle:
        bundle["emissivity"] = bundle["emissivity"][:wavelengths]
    for key in _TIR_DIRECT_SOURCE_KEYS + ("surface_temperature_k",) + _TIR_SOURCE_COORDINATE_KEYS:
        if key in bundle and np.asarray(bundle[key]).shape[:1] == (total_rows,):
            bundle[key] = bundle[key][:wavelengths]
    for key in _TIR_DUMPED_OPTICS_KEYS:
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    for key in ("depol", "rayleigh_fraction", "aerosol_fraction", "aerosol_interp_fraction"):
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    if "ref_total" in bundle:
        bundle["ref_total"] = bundle["ref_total"][:wavelengths]
    load_seconds = time.perf_counter() - load_start

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
        bundle_path=args.bundle,
        wavelengths=wavelengths,
        layers=int(bundle["tau_arr"].shape[1]),
        load_seconds=load_seconds,
        note=(
            "RT time is FO + 2S. wall time (s) excludes bundle load and printed "
            "preprocessing, but includes backend-local overhead such as tensor "
            "conversion, PyTorch warmup, and checksum reduction."
        ),
    )
    print(f"  geometry preprocessing: python-generated, {geometry_seconds:.3f} s")
    print(f"  optical preprocessing: {optical_mode}, {optical_seconds:.3f} s")
    print(f"  thermal source: {source_mode}, {source_seconds:.3f} s")
    print(
        "  preprocessing total: "
        f"{geometry_seconds + optical_seconds + source_seconds:.3f} s "
        "(geometry + optical + thermal source)"
    )
    print(f"  emissivity: {emissivity_mode}")

    rows: list[BenchmarkRow] = []
    if args.backend in {"numpy", "both"}:
        chunk_size = args.chunk_size or recommended_chunk_size(
            total_rows=wavelengths,
            nlayers=int(bundle["tau_arr"].shape[1]),
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
            nlayers=int(bundle["tau_arr"].shape[1]),
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
