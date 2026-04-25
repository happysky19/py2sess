"""Benchmark a full-spectrum TIR bundle with NumPy and optional PyTorch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _full_spectrum_benchmark_common import (
    accuracy_summary,
    BenchmarkRow,
    load_bundle,
    print_problem_header,
    print_rows,
    recommended_chunk_size,
    require_keys,
    scalar_value,
)
from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.optical.phase import build_two_stream_phase_inputs
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


_TIR_PHYSICAL_OPTICS_KEYS = (
    "depol",
    "rayleigh_fraction",
    "aerosol_fraction",
    "aerosol_moments",
    "aerosol_interp_fraction",
)

_TIR_DUMPED_OPTICS_KEYS = ("asymm_arr", "d2s_scaling")


def _has_keys(bundle: dict[str, np.ndarray], keys: tuple[str, ...]) -> bool:
    return all(key in bundle for key in keys)


def _prepare_optics(
    bundle: dict[str, np.ndarray],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    if use_dumped_derived_optics or not _has_keys(bundle, _TIR_PHYSICAL_OPTICS_KEYS):
        require_keys(bundle, _TIR_DUMPED_OPTICS_KEYS, label="TIR dumped optical")
        mode = "dumped-derived"
        if not use_dumped_derived_optics and not _has_keys(bundle, _TIR_PHYSICAL_OPTICS_KEYS):
            mode = "dumped-derived (physical optical inputs unavailable)"
        return bundle, 0.0, mode

    start = time.perf_counter()
    optics = build_two_stream_phase_inputs(
        ssa=bundle["omega_arr"],
        depol=bundle["depol"],
        rayleigh_fraction=bundle["rayleigh_fraction"],
        aerosol_fraction=bundle["aerosol_fraction"],
        aerosol_moments=bundle["aerosol_moments"],
        aerosol_interp_fraction=bundle["aerosol_interp_fraction"],
    )
    prepared = dict(bundle)
    prepared["asymm_arr"] = optics.g
    prepared["d2s_scaling"] = optics.delta_m_truncation_factor
    return prepared, time.perf_counter() - start, "python-generated"


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
    geometry = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    user_stream = float(np.cos(np.deg2rad(user_angle)))
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
            emissivity=1.0 - chunk["albedo"],
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
            emissivity=1.0 - chunk["albedo"],
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
        setup_seconds=wall_seconds - rt_seconds,
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
        emissivity=1.0 - bundle["albedo"],
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
        setup_seconds=0.0,
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
    geometry = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    user_stream = float(np.cos(np.deg2rad(user_angle)))
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
            emissivity_t = 1.0 - albedo_t
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
        setup_seconds=wall_seconds - rt_seconds,
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
        emissivity=1.0 - bundle["albedo"],
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
        setup_seconds=0.0,
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
    args = parser.parse_args()

    load_start = time.perf_counter()
    bundle = load_bundle(args.bundle)
    require_keys(
        bundle,
        (
            "wavelengths",
            "heights",
            "user_angle",
            "tau_arr",
            "omega_arr",
            "thermal_bb_input",
            "surfbb",
            "albedo",
        ),
        label="TIR",
    )
    total_rows = int(bundle["tau_arr"].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = dict(bundle)
    bundle["wavelengths"] = bundle["wavelengths"][:wavelengths]
    bundle["tau_arr"] = bundle["tau_arr"][:wavelengths]
    bundle["omega_arr"] = bundle["omega_arr"][:wavelengths]
    bundle["thermal_bb_input"] = bundle["thermal_bb_input"][:wavelengths]
    bundle["surfbb"] = bundle["surfbb"][:wavelengths]
    bundle["albedo"] = bundle["albedo"][:wavelengths]
    for key in _TIR_DUMPED_OPTICS_KEYS:
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    for key in ("depol", "rayleigh_fraction", "aerosol_fraction", "aerosol_interp_fraction"):
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    if "ref_total" in bundle:
        bundle["ref_total"] = bundle["ref_total"][:wavelengths]
    bundle, optical_seconds, optical_mode = _prepare_optics(
        bundle,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )
    load_seconds = time.perf_counter() - load_start

    print_problem_header(
        title="TIR full-spectrum benchmark",
        bundle_path=args.bundle,
        wavelengths=wavelengths,
        layers=int(bundle["tau_arr"].shape[1]),
        load_seconds=load_seconds,
        note=(
            "RT time is FO + 2S. wall time (s) excludes bundle load but includes "
            "backend-local setup such as FO geometry precompute, tensor conversion, "
            "PyTorch warmup, and checksum reduction."
        ),
    )
    print(f"  optical preprocessing: {optical_mode}, {optical_seconds:.3f} s")

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
