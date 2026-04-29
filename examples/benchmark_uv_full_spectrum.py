"""Benchmark full-spectrum UV input arrays with NumPy and optional PyTorch."""

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
from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.optical.phase import (
    aerosol_interp_fraction,
    build_solar_phase_inputs_from_scattering_tau,
    build_solar_fo_scatter_term,
    build_two_stream_phase_inputs,
)
from py2sess.rtsolver.backend import has_torch
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors
from py2sess.rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy


_UV_REQUIRED_PHYSICAL_OPTICS_KEYS = (
    "depol",
    "rayleigh_fraction",
    "aerosol_fraction",
    "aerosol_moments",
)
_UV_COMPONENT_PHASE_KEYS = (
    "depol",
    "rayleigh_scattering_tau",
    "aerosol_scattering_tau",
    "aerosol_moments",
)

_UV_AEROSOL_INTERP_KEY = "aerosol_interp_fraction"

_UV_DUMPED_OPTICS_KEYS = ("asymm", "scaling", "fo_exact_scatter")

_UV_BASE_KEYS = (
    "wavelengths",
    "user_obsgeom",
    "heights",
    "albedo",
    "flux_factor",
)

_UV_OPTIONAL_KEYS = ("stream_value",)

_UV_CHUNK_KEYS = (
    "tau",
    "ssa",
    "g",
    "delta_m_truncation_factor",
    "albedo",
    "flux_factor",
    "fo_scatter_term",
)
_UV_LIMIT_KEYS = (
    _UV_BASE_KEYS
    + _UV_DUMPED_OPTICS_KEYS
    + (
        "depol",
        "rayleigh_fraction",
        "aerosol_fraction",
        "aerosol_interp_fraction",
    )
)


def _uv_aerosol_interp_fraction(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    if looks_like_row_index(wavelengths):
        raise ValueError(
            "UV aerosol interpolation requires aerosol_interp_fraction or physical "
            "wavelengths when wavelengths contains row indices"
        )
    return (
        aerosol_interp_fraction(wavelengths, reverse=True),
        "python-generated (aerosol interpolation from wavelengths)",
    )


def _prepare_geometry(bundle: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    geoms = np.asarray(bundle["user_obsgeom"], dtype=float)
    if geoms.ndim == 1 and geoms.size == 3:
        geoms = geoms.reshape(1, 3)
    if geoms.ndim != 2 or geoms.shape[1] != 3:
        raise ValueError("user_obsgeom must have shape (3,) or (n_geometry, 3)")
    if geoms.shape[0] != 1:
        raise ValueError("the low-level UV benchmark path supports exactly one geometry")

    sza, vza, raz = geoms[0]
    if not (0.0 <= sza < 90.0):
        raise ValueError("solar zenith angle must satisfy 0 <= sza < 90")
    if not (0.0 <= vza < 90.0):
        raise ValueError("viewing zenith angle must satisfy 0 <= vza < 90")
    if not (0.0 <= raz <= 360.0):
        raise ValueError("relative azimuth angle must satisfy 0 <= raz <= 360")
    deg_to_rad = np.pi / 180.0
    x0 = np.array([np.cos(sza * deg_to_rad)], dtype=float)
    user_stream = np.array([np.cos(vza * deg_to_rad)], dtype=float)
    user_secant = 1.0 / user_stream
    azmfac = np.array([np.cos(raz * deg_to_rad)], dtype=float)
    stream_value = scalar_value(bundle["stream_value"])
    px11, pxsq, px0x, ulp = auxgeom_solar_obs(
        x0=x0,
        user_streams=user_stream,
        stream_value=stream_value,
        do_postprocessing=True,
    )

    prepared = dict(bundle)
    prepared["user_obsgeom"] = geoms
    prepared["chapman"] = chapman_factors(
        np.asarray(bundle["heights"], dtype=float),
        6371.0,
        float(sza),
    )
    prepared["x0"] = x0
    prepared["user_stream"] = user_stream
    prepared["user_secant"] = user_secant
    prepared["azmfac"] = azmfac
    prepared["px11"] = np.array([px11], dtype=float)
    prepared["pxsq"] = pxsq
    prepared["px0x"] = px0x[0]
    prepared["ulp"] = ulp
    return prepared, time.perf_counter() - start


def _prepare_optics(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    has_component_phase = all(key in bundle for key in _UV_COMPONENT_PHASE_KEYS)
    has_fraction_phase = all(key in bundle for key in _UV_REQUIRED_PHYSICAL_OPTICS_KEYS)
    if use_dumped_derived_optics or not (has_component_phase or has_fraction_phase):
        require_keys(bundle, _UV_DUMPED_OPTICS_KEYS, label="UV dumped optical")
        prepared = dict(bundle)
        prepared["g"] = bundle["asymm"]
        prepared["delta_m_truncation_factor"] = bundle["scaling"]
        prepared["fo_scatter_term"] = bundle["fo_exact_scatter"]
        mode = "dumped-derived"
        if not use_dumped_derived_optics and not (has_component_phase or has_fraction_phase):
            mode = "dumped-derived (physical optical inputs unavailable)"
        return prepared, 0.0, mode

    start = time.perf_counter()
    if "aerosol_interp_fraction" in bundle:
        fac = bundle["aerosol_interp_fraction"]
        mode = "python-generated"
    else:
        fac, mode = _uv_aerosol_interp_fraction(bundle)

    if has_component_phase:
        optics = build_solar_phase_inputs_from_scattering_tau(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_scattering_tau=bundle["rayleigh_scattering_tau"],
            aerosol_scattering_tau=bundle["aerosol_scattering_tau"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            angles=bundle["user_obsgeom"],
            scattering_tau=bundle.get("_scattering_tau"),
            validate_inputs=False,
        )
        fo_scatter = optics.fo_scatter_term
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
        fo_scatter = build_solar_fo_scatter_term(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_fraction=bundle["rayleigh_fraction"],
            aerosol_fraction=bundle["aerosol_fraction"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            angles=bundle["user_obsgeom"],
            delta_m_truncation_factor=optics.delta_m_truncation_factor,
            validate_inputs=False,
        )
    prepared = dict(bundle)
    prepared["aerosol_interp_fraction"] = fac
    prepared["g"] = optics.g
    prepared["delta_m_truncation_factor"] = optics.delta_m_truncation_factor
    prepared["fo_scatter_term"] = fo_scatter
    return prepared, time.perf_counter() - start, mode


def benchmark_numpy(
    bundle: dict[str, np.ndarray],
    *,
    wavelengths: int,
    chunk_size: int,
    numpy_bvp_engine: str,
) -> BenchmarkRow:
    """Runs the NumPy UV full-spectrum benchmark (FO + 2S)."""
    wall_start = time.perf_counter()
    fo_precomputed = fo_solar_obs_batch_precompute(
        user_obsgeom=np.asarray(bundle["user_obsgeom"], dtype=float),
        heights=np.asarray(bundle["heights"], dtype=float),
        earth_radius=6371.0,
        nfine=3,
    )
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    checksum = 0.0
    max_abs_diff = None
    max_rel_diff_pct = None
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, _UV_CHUNK_KEYS, start, stop)

        t0 = time.perf_counter()
        fo_total = solve_fo_solar_obs_eps_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            scaling=chunk["delta_m_truncation_factor"],
            albedo=chunk["albedo"],
            flux_factor=chunk["flux_factor"],
            exact_scatter=chunk["fo_scatter_term"],
            precomputed=fo_precomputed,
        )
        fo_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        two_stream = solve_solar_obs_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            asymm=chunk["g"],
            scaling=chunk["delta_m_truncation_factor"],
            albedo=chunk["albedo"],
            flux_factor=chunk["flux_factor"],
            stream_value=scalar_value(bundle["stream_value"]),
            chapman=bundle["chapman"],
            x0=scalar_value(bundle["x0"]),
            user_stream=scalar_value(bundle["user_stream"]),
            user_secant=scalar_value(bundle["user_secant"]),
            azmfac=scalar_value(bundle["azmfac"]),
            px11=scalar_value(bundle["px11"]),
            pxsq=bundle["pxsq"],
            px0x=bundle["px0x"],
            ulp=scalar_value(bundle["ulp"]),
            bvp_engine=numpy_bvp_engine,
        )
        two_stream_seconds += time.perf_counter() - t0
        total = two_stream + fo_total
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
            mode="solar",
            bvp_solver=public_bvp_solver(numpy_bvp_engine),
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau"],
        ssa=bundle["ssa"],
        g=bundle["g"],
        z=bundle["heights"],
        angles=bundle["user_obsgeom"],
        stream=scalar_value(bundle["stream_value"]),
        fbeam=bundle["flux_factor"],
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["delta_m_truncation_factor"],
        include_fo=True,
        fo_scatter_term=bundle["fo_scatter_term"],
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
        workload="solar_obs",
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
    """Runs the PyTorch UV full-spectrum benchmark (FO + 2S)."""
    if not has_torch():
        raise RuntimeError("PyTorch is not installed")
    import torch

    from py2sess.rtsolver.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch
    from py2sess.rtsolver.solar_obs_batch_torch import solve_solar_obs_batch_torch

    wall_start = time.perf_counter()
    device = torch.device(torch_device_name)
    dtype = {"float64": torch.float64, "float32": torch.float32}[torch_dtype_name]
    torch.set_num_threads(torch_threads)
    with torch.no_grad():
        probe = torch.ones(16, dtype=dtype, device=device)
        _ = (probe + 1.0).sum().item()

    fo_seconds = 0.0
    two_stream_seconds = 0.0
    checksum = 0.0
    max_abs_diff = None
    max_rel_diff_pct = None
    fo_precomputed = fo_solar_obs_batch_precompute(
        user_obsgeom=np.asarray(bundle["user_obsgeom"], dtype=float),
        heights=np.asarray(bundle["heights"], dtype=float),
        earth_radius=6371.0,
        nfine=3,
    )
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, _UV_CHUNK_KEYS, start, stop)

        t0 = time.perf_counter()
        with torch.no_grad():
            fo_total = solve_fo_solar_obs_eps_batch_torch(
                tau=chunk["tau"],
                omega=chunk["ssa"],
                scaling=chunk["delta_m_truncation_factor"],
                albedo=chunk["albedo"],
                flux_factor=chunk["flux_factor"],
                exact_scatter=chunk["fo_scatter_term"],
                precomputed=fo_precomputed,
                dtype=dtype,
                device=device,
            )
        fo_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            two_stream = solve_solar_obs_batch_torch(
                tau=chunk["tau"],
                omega=chunk["ssa"],
                asymm=chunk["g"],
                scaling=chunk["delta_m_truncation_factor"],
                albedo=chunk["albedo"],
                flux_factor=chunk["flux_factor"],
                stream_value=scalar_value(bundle["stream_value"]),
                chapman=bundle["chapman"],
                x0=scalar_value(bundle["x0"]),
                user_stream=scalar_value(bundle["user_stream"]),
                user_secant=scalar_value(bundle["user_secant"]),
                azmfac=scalar_value(bundle["azmfac"]),
                px11=scalar_value(bundle["px11"]),
                pxsq=bundle["pxsq"],
                px0x=bundle["px0x"],
                ulp=scalar_value(bundle["ulp"]),
                dtype=dtype,
                device=device,
                bvp_engine=torch_bvp_engine,
            )
        two_stream_seconds += time.perf_counter() - t0
        total_torch = two_stream + fo_total
        checksum += float(total_torch.sum().item())
        total = total_torch.detach().cpu().numpy()
        if "ref_total" in chunk:
            chunk_abs_diff, chunk_rel_diff_pct = accuracy_summary(
                total,
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
    bvp_solver = public_bvp_solver(torch_bvp_engine)
    wall_start = time.perf_counter()
    dtype = {"float64": torch.float64, "float32": torch.float32}[torch_dtype_name]
    device = torch.device(torch_device_name)
    with torch.no_grad():
        probe = torch.ones(16, dtype=dtype, device=device)
        _ = (probe + 1.0).sum().item()
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=int(bundle["tau"].shape[1]),
            mode="solar",
            backend="torch",
            bvp_solver=bvp_solver,
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
        angles=bundle["user_obsgeom"],
        stream=scalar_value(bundle["stream_value"]),
        fbeam=bundle["flux_factor"],
        albedo=bundle["albedo"],
        delta_m_truncation_factor=bundle["delta_m_truncation_factor"],
        include_fo=True,
        fo_scatter_term=bundle["fo_scatter_term"],
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
        workload="solar_obs",
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
    """Runs the full-spectrum UV benchmark example."""
    parser = argparse.ArgumentParser()
    add_common_benchmark_arguments(
        parser,
        input_help="Path to a UV runtime array directory, or a legacy full-spectrum .npz bundle.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    scene_mode = validate_scene_input_args(
        parser,
        args,
        forbidden_scene_flags=(("use_dumped_derived_optics", "--use-dumped-derived-optics"),),
    )

    bundle, available, input_path, input_kind = benchmark_input_source(
        args,
        kind="uv",
        label="UV",
        scene_mode=scene_mode,
    )

    layer_optical_keys = select_layer_optical_keys(
        available,
        total_key="tau",
        ssa_key="omega",
    )
    layer_from_scene = layer_optical_keys_are_scene(layer_optical_keys)
    layer_from_components = layer_optical_keys_are_components(layer_optical_keys)
    base_keys = tuple(key for key in _UV_BASE_KEYS if key != "heights" or not layer_from_scene)
    if args.require_python_generated_inputs:
        require_python_generated_layer_optical_inputs(
            layer_optical_keys,
            total_key="tau",
            ssa_key="omega",
            label="UV",
        )
    optical_keys = select_phase_optical_keys(
        available,
        label="UV",
        use_dumped_derived_optics=args.use_dumped_derived_optics,
        layer_optical_generates_fractions=layer_from_scene or layer_from_components,
        layer_optical_from_scene=layer_from_scene,
        require_python_generated_inputs=args.require_python_generated_inputs,
        required_fraction_keys=_UV_REQUIRED_PHYSICAL_OPTICS_KEYS,
        dumped_keys=_UV_DUMPED_OPTICS_KEYS,
        aerosol_interp_key=_UV_AEROSOL_INTERP_KEY,
    )
    if not scene_mode:
        bundle = load_input_arrays(
            args.input,
            keys=base_keys + _UV_OPTIONAL_KEYS + layer_optical_keys + optical_keys,
        )
    require_keys(
        bundle,
        base_keys + layer_optical_keys + optical_keys,
        label="UV",
    )
    if layer_from_scene:
        row_count_key = "wavelengths"
    elif layer_from_components:
        row_count_key = layer_optical_keys[0]
    else:
        row_count_key = "tau"
    total_rows = int(bundle[row_count_key].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = trim_spectral_rows(
        bundle,
        _UV_LIMIT_KEYS + layer_optical_keys,
        total_rows,
        wavelengths,
    )
    if not scene_mode and args.input.name == "uv_benchmark_fixture.npz":
        bundle["ref_total"] = load_packaged_reference_total("uv_reference_outputs.npz")[
            :wavelengths
        ]
    if "stream_value" not in bundle:
        bundle["stream_value"] = np.array([1.0 / np.sqrt(3.0)], dtype=float)
    load_seconds = time.perf_counter() - load_start

    bundle, layer_optical_seconds, layer_optical_mode = prepare_layer_optical_properties(
        bundle,
        total_key="tau",
        ssa_key="omega",
        validate_inputs=not args.require_python_generated_inputs,
    )
    bundle["ssa"] = bundle["omega"]
    bundle, geometry_seconds = _prepare_geometry(bundle)
    bundle, optical_seconds, optical_mode = _prepare_optics(
        bundle,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )

    print_problem_header(
        title="UV full-spectrum benchmark",
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
        )
    )

    rows: list[BenchmarkRow] = []
    if args.backend in {"numpy", "both"}:
        chunk_size = args.chunk_size or recommended_chunk_size(
            total_rows=wavelengths,
            nlayers=int(bundle["tau"].shape[1]),
            backend="numpy",
            workload="solar_obs",
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
            workload="solar_obs",
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
