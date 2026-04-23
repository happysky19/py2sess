"""Benchmark a full-spectrum UV bundle with NumPy and optional PyTorch."""

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
from py2sess.core.backend import has_torch
from py2sess.core.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.core.solar_obs_batch_numpy import solve_solar_obs_batch_numpy


def _public_bvp_solver(engine: str) -> str:
    """Maps low-level benchmark BVP names to public option names."""
    return "banded" if engine == "block" else "scipy"


def _slice_rows(bundle: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Returns one spectral chunk from a UV benchmark bundle."""
    chunk = {
        "tau": bundle["tau"][start:stop],
        "omega": bundle["omega"][start:stop],
        "asymm": bundle["asymm"][start:stop],
        "scaling": bundle["scaling"][start:stop],
        "albedo": bundle["albedo"][start:stop],
        "flux_factor": bundle["flux_factor"][start:stop],
        "fo_exact_scatter": bundle["fo_exact_scatter"][start:stop],
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
        chunk = _slice_rows(bundle, start, stop)

        t0 = time.perf_counter()
        fo_total = solve_fo_solar_obs_eps_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["omega"],
            scaling=chunk["scaling"],
            albedo=chunk["albedo"],
            flux_factor=chunk["flux_factor"],
            exact_scatter=chunk["fo_exact_scatter"],
            precomputed=fo_precomputed,
        )
        fo_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        two_stream = solve_solar_obs_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["omega"],
            asymm=chunk["asymm"],
            scaling=chunk["scaling"],
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
    _ = checksum
    return BenchmarkRow(
        backend="numpy",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
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
            nlyr=int(bundle["tau"].shape[1]),
            mode="solar",
            bvp_solver=_public_bvp_solver(numpy_bvp_engine),
            output_levels=output_levels,
        )
    )
    result = solver.forward(
        tau=bundle["tau"],
        ssa=bundle["omega"],
        g=bundle["asymm"],
        z=bundle["heights"],
        angles=bundle["user_obsgeom"],
        stream=scalar_value(bundle["stream_value"]),
        fbeam=bundle["flux_factor"],
        albedo=bundle["albedo"],
        delta_m_scaling=bundle["scaling"],
        include_fo=True,
        fo_exact_scatter=bundle["fo_exact_scatter"],
    )
    total = np.asarray(result.radiance_total, dtype=float)
    checksum = float(np.sum(total))
    wall_seconds = time.perf_counter() - wall_start
    max_abs_diff = None
    max_rel_diff_pct = None
    if "ref_total" in bundle:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(total, bundle["ref_total"])
    _ = checksum
    return BenchmarkRow(
        backend="numpy-forward-levels" if output_levels else "numpy-forward",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=wavelengths,
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
    """Runs the PyTorch UV full-spectrum benchmark (FO + 2S)."""
    if not has_torch():
        raise RuntimeError("PyTorch is not installed")
    import torch

    from py2sess.core.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch
    from py2sess.core.solar_obs_batch_torch import solve_solar_obs_batch_torch

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
        chunk = _slice_rows(bundle, start, stop)

        t0 = time.perf_counter()
        with torch.no_grad():
            fo_total = solve_fo_solar_obs_eps_batch_torch(
                tau=chunk["tau"],
                omega=chunk["omega"],
                scaling=chunk["scaling"],
                albedo=chunk["albedo"],
                flux_factor=chunk["flux_factor"],
                exact_scatter=chunk["fo_exact_scatter"],
                precomputed=fo_precomputed,
                dtype=dtype,
                device=device,
            )
        fo_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            two_stream = solve_solar_obs_batch_torch(
                tau=chunk["tau"],
                omega=chunk["omega"],
                asymm=chunk["asymm"],
                scaling=chunk["scaling"],
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
    _ = checksum
    return BenchmarkRow(
        backend=f"torch-{torch_device_name}-{torch_dtype_name}",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
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
    bvp_solver = _public_bvp_solver(torch_bvp_engine)
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
        ssa=bundle["omega"],
        g=bundle["asymm"],
        z=bundle["heights"],
        angles=bundle["user_obsgeom"],
        stream=scalar_value(bundle["stream_value"]),
        fbeam=bundle["flux_factor"],
        albedo=bundle["albedo"],
        delta_m_scaling=bundle["scaling"],
        include_fo=True,
        fo_exact_scatter=bundle["fo_exact_scatter"],
    )
    total = result.radiance_total.detach().cpu().numpy()
    checksum = float(np.sum(total))
    wall_seconds = time.perf_counter() - wall_start
    max_abs_diff = None
    max_rel_diff_pct = None
    if "ref_total" in bundle:
        max_abs_diff, max_rel_diff_pct = accuracy_summary(total, bundle["ref_total"])
    _ = checksum
    return BenchmarkRow(
        backend=(
            f"torch-{torch_device_name}-{torch_dtype_name}-forward-levels"
            if output_levels
            else f"torch-{torch_device_name}-{torch_dtype_name}-forward"
        ),
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        chunk_size=wavelengths,
        wall_seconds=wall_seconds,
        rt_seconds=wall_seconds,
        setup_seconds=0.0,
        max_abs_diff=max_abs_diff,
        max_rel_diff_pct=max_rel_diff_pct,
    )


def main() -> None:
    """Runs the full-spectrum UV benchmark example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle", type=Path, help="Path to a full-spectrum UV benchmark bundle (.npz)."
    )
    parser.add_argument("--backend", choices=["numpy", "torch", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Optional spectral-row limit.")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Optional chunk size override."
    )
    parser.add_argument("--numpy-bvp-engine", choices=["auto", "block"], default="auto")
    parser.add_argument("--torch-bvp-engine", choices=["auto", "block"], default="auto")
    parser.add_argument("--torch-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--torch-dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output-levels",
        action="store_true",
        help="Benchmark the public forward profile path instead of endpoint-only output.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    bundle = load_bundle(args.bundle)
    require_keys(
        bundle,
        (
            "wavelengths",
            "user_obsgeom",
            "heights",
            "tau",
            "omega",
            "asymm",
            "scaling",
            "albedo",
            "flux_factor",
            "fo_exact_scatter",
            "chapman",
            "x0",
            "user_stream",
            "user_secant",
            "azmfac",
            "px11",
            "pxsq",
            "px0x",
            "ulp",
        ),
        label="UV",
    )
    total_rows = int(bundle["tau"].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = dict(bundle)
    bundle["wavelengths"] = bundle["wavelengths"][:wavelengths]
    bundle["tau"] = bundle["tau"][:wavelengths]
    bundle["omega"] = bundle["omega"][:wavelengths]
    bundle["asymm"] = bundle["asymm"][:wavelengths]
    bundle["scaling"] = bundle["scaling"][:wavelengths]
    bundle["albedo"] = bundle["albedo"][:wavelengths]
    bundle["flux_factor"] = bundle["flux_factor"][:wavelengths]
    bundle["fo_exact_scatter"] = bundle["fo_exact_scatter"][:wavelengths]
    if "ref_total" in bundle:
        bundle["ref_total"] = bundle["ref_total"][:wavelengths]
    if "stream_value" not in bundle:
        bundle["stream_value"] = np.array([1.0 / np.sqrt(3.0)], dtype=float)
    load_seconds = time.perf_counter() - load_start

    print_problem_header(
        title="UV full-spectrum benchmark",
        bundle_path=args.bundle,
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
        load_seconds=load_seconds,
        note=(
            "This example benchmarks the UV FO + 2S full-spectrum path. "
            "wall time (s) excludes bundle load but includes backend-local setup such as "
            "PyTorch warmup, tensor conversion, and checksum reduction."
        ),
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
