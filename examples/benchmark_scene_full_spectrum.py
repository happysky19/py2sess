"""Benchmark a full-spectrum scene through the public scene API."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _full_spectrum_benchmark_common import (
    accuracy_summary,
    add_common_benchmark_arguments,
    BenchmarkRow,
    print_preprocessing_summary,
    print_problem_header,
    print_rows,
    public_bvp_solver,
    recommended_chunk_size,
    scalar_value,
    slice_spectral_rows,
)
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy
from py2sess.rtsolver.thermal_batch_numpy import _fo_thermal_toa, _two_stream_thermal_toa
from py2sess.rtsolver.thermal_batch_numpy import precompute_fo_thermal_geometry_numpy
from py2sess.rtsolver.backend import has_torch
from py2sess.scene import load_scene


SOLAR_COMPONENT_KEYS = (
    "tau",
    "ssa",
    "g",
    "delta_m_truncation_factor",
    "albedo",
    "fbeam",
    "fo_scatter_term",
)
THERMAL_COMPONENT_KEYS = (
    "tau",
    "ssa",
    "g",
    "delta_m_truncation_factor",
    "planck",
    "surface_planck",
    "albedo",
    "emissivity",
)


def _workload(mode: str) -> str:
    return "solar_obs" if mode == "solar" else "thermal"


def _run_forward(scene, *, args: argparse.Namespace, backend: str) -> BenchmarkRow:
    if backend == "torch":
        if not has_torch():
            raise RuntimeError("PyTorch is not installed")
        import torch

        torch.set_num_threads(args.torch_threads)
        option_overrides = {
            "backend": "torch",
            "bvp_solver": public_bvp_solver(args.torch_bvp_engine),
            "torch_device": args.torch_device,
            "torch_dtype": args.torch_dtype,
            "torch_enable_grad": False,
            "output_levels": args.output_levels,
        }
    else:
        option_overrides = {
            "backend": "numpy",
            "bvp_solver": public_bvp_solver(args.numpy_bvp_engine),
            "output_levels": args.output_levels,
        }

    inputs = scene.to_forward_inputs()
    wavelengths = int(inputs.wavelengths.shape[0]) if inputs.wavelengths is not None else 1
    layers = int(np.asarray(inputs.kwargs["tau"]).shape[-1])
    start = time.perf_counter()
    result = scene.forward(**option_overrides, include_fo=True)
    wall_seconds = time.perf_counter() - start
    radiance = result.radiance_total
    if hasattr(radiance, "detach"):
        radiance = radiance.detach().cpu().numpy()
    radiance = np.asarray(radiance, dtype=float)
    checksum = float(np.sum(radiance))
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    max_abs_diff = None
    max_rel_diff_pct = None
    if inputs.reference_total is not None:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(radiance, inputs.reference_total)
    chunk_size = args.chunk_size or recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend=backend,
        workload=_workload(inputs.mode),
    )
    label = backend
    if backend == "torch":
        label = f"torch-{args.torch_device}-{args.torch_dtype}"
    if args.output_levels:
        label += "-levels"
    return BenchmarkRow(
        backend=f"{label}-scene-forward",
        wavelengths=wavelengths,
        layers=layers,
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=wall_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def _run_numpy_component_timing(scene, *, args: argparse.Namespace) -> BenchmarkRow:
    inputs = scene.to_forward_inputs()
    if inputs.mode == "solar":
        return _run_numpy_solar_component_timing(scene, args=args)
    return _run_numpy_thermal_component_timing(scene, args=args)


def _run_numpy_solar_component_timing(scene, *, args: argparse.Namespace) -> BenchmarkRow:
    inputs = scene.to_forward_inputs()
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = args.chunk_size or recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="numpy",
        workload="solar_obs",
    )
    heights = np.asarray(kwargs["z"], dtype=float)
    angles = np.asarray(kwargs["angles"], dtype=float)
    stream = float(kwargs.get("stream", 1.0 / np.sqrt(3.0)))
    geoms = angles.reshape(1, 3) if angles.ndim == 1 else angles
    if geoms.shape[0] != 1:
        raise ValueError("--component-timing supports one solar geometry")
    sza, vza, raz = geoms[0]
    x0 = np.array([np.cos(np.deg2rad(sza))], dtype=float)
    user_stream = np.array([np.cos(np.deg2rad(vza))], dtype=float)
    from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors

    px11, pxsq, px0x, ulp = auxgeom_solar_obs(
        x0=x0,
        user_streams=user_stream,
        stream_value=stream,
        do_postprocessing=True,
    )
    chapman = chapman_factors(heights, 6371.0, float(sza))
    azmfac = np.array([np.cos(np.deg2rad(raz))], dtype=float)
    fo_precomputed = fo_solar_obs_batch_precompute(
        user_obsgeom=geoms,
        heights=heights,
        earth_radius=6371.0,
        nfine=3,
    )
    bundle = dict(kwargs)
    if inputs.reference_total is not None:
        bundle["ref_total"] = inputs.reference_total
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    wall_start = time.perf_counter()
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
        t0 = time.perf_counter()
        fo = solve_fo_solar_obs_eps_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            scaling=chunk["delta_m_truncation_factor"],
            albedo=chunk["albedo"],
            flux_factor=chunk["fbeam"],
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
            flux_factor=chunk["fbeam"],
            stream_value=stream,
            chapman=chapman,
            x0=scalar_value(x0),
            user_stream=scalar_value(user_stream),
            user_secant=1.0 / scalar_value(user_stream),
            azmfac=scalar_value(azmfac),
            px11=px11,
            pxsq=pxsq,
            px0x=px0x[0],
            ulp=scalar_value(ulp),
            bvp_engine=args.numpy_bvp_engine,
        )
        two_stream_seconds += time.perf_counter() - t0
        total_parts.append(two_stream + fo)
    total = np.concatenate(total_parts)
    return _component_row(
        label="numpy-components",
        total=total,
        reference=inputs.reference_total,
        wall_seconds=time.perf_counter() - wall_start,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        wavelengths=wavelengths,
        layers=layers,
        chunk_size=chunk_size,
    )


def _run_numpy_thermal_component_timing(scene, *, args: argparse.Namespace) -> BenchmarkRow:
    inputs = scene.to_forward_inputs()
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = args.chunk_size or recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="numpy",
        workload="thermal",
    )
    heights = np.asarray(kwargs["z"], dtype=float)
    user_angle = scalar_value(kwargs["angles"])
    user_stream = float(np.cos(np.deg2rad(user_angle)))
    stream = float(kwargs.get("stream", 0.5))
    geometry = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    bundle = dict(kwargs)
    if inputs.reference_total is not None:
        bundle["ref_total"] = inputs.reference_total
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    wall_start = time.perf_counter()
    for start in range(0, wavelengths, chunk_size):
        stop = min(start + chunk_size, wavelengths)
        chunk = slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
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
            stream_value=stream,
            user_stream=user_stream,
            thermal_tcutoff=1.0e-8,
            bvp_engine=args.numpy_bvp_engine,
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
        total_parts.append(two_stream + fo)
    total = np.concatenate(total_parts)
    return _component_row(
        label="numpy-components",
        total=total,
        reference=inputs.reference_total,
        wall_seconds=time.perf_counter() - wall_start,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        wavelengths=wavelengths,
        layers=layers,
        chunk_size=chunk_size,
    )


def _component_row(
    *,
    label: str,
    total: np.ndarray,
    reference: np.ndarray | None,
    wall_seconds: float,
    fo_seconds: float,
    two_stream_seconds: float,
    wavelengths: int,
    layers: int,
    chunk_size: int,
) -> BenchmarkRow:
    max_abs_diff = None
    max_rel_diff_pct = None
    if reference is not None:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(total, reference)
    checksum = float(np.sum(total))
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
    return BenchmarkRow(
        backend=label,
        wavelengths=wavelengths,
        layers=layers,
        chunk_size=chunk_size,
        wall_seconds=wall_seconds,
        rt_seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_benchmark_arguments(
        parser,
        torch_bvp_choices=("auto", "block", "pentadiagonal"),
    )
    parser.add_argument(
        "--component-timing",
        action="store_true",
        help="Also run NumPy low-level kernels to report separate FO and 2S timing.",
    )
    args = parser.parse_args()
    if args.component_timing and args.backend == "torch":
        raise ValueError("--component-timing currently reports NumPy component kernels")
    if args.component_timing and args.output_levels:
        raise ValueError("--component-timing reports endpoint kernels; omit --output-levels")

    load_start = time.perf_counter()
    scene = load_scene(
        profile=args.profile,
        config=args.scene,
        spectral_limit=args.limit,
        strict_runtime_inputs=args.require_python_generated_inputs,
    )
    load_seconds = time.perf_counter() - load_start
    inputs = scene.to_forward_inputs()
    wavelengths = int(inputs.wavelengths.shape[0]) if inputs.wavelengths is not None else 1
    layers = int(np.asarray(inputs.kwargs["tau"]).shape[-1])

    print_problem_header(
        title=f"{inputs.mode.upper()} full-spectrum scene benchmark",
        input_path=args.scene,
        input_kind="profile+scene",
        wavelengths=wavelengths,
        layers=layers,
        load_seconds=load_seconds,
        note="RT time is public scene.forward(). wall time excludes load/preprocessing.",
    )
    print_preprocessing_summary(
        tuple(
            (label, inputs.timing_modes.get(label, "python-generated"), seconds)
            for label, seconds in inputs.timings.items()
        )
    )

    rows: list[BenchmarkRow] = []
    if args.backend in {"numpy", "both"}:
        rows.append(_run_forward(scene, args=args, backend="numpy"))
        if args.component_timing:
            rows.append(_run_numpy_component_timing(scene, args=args))
    if args.backend in {"torch", "both"}:
        rows.append(_run_forward(scene, args=args, backend="torch"))
    print_rows(rows)


if __name__ == "__main__":
    main()
