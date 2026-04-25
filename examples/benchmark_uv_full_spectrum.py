"""Benchmark a full-spectrum UV bundle with NumPy and optional PyTorch."""

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
    print_problem_header,
    print_rows,
    recommended_chunk_size,
    require_keys,
    scalar_value,
)
from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.optical.phase import build_solar_fo_scatter_term, build_two_stream_phase_inputs
from py2sess.rtsolver.backend import has_torch
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors
from py2sess.rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy


def _public_bvp_solver(engine: str) -> str:
    """Maps low-level benchmark BVP names to public option names."""
    return "banded" if engine == "block" else "scipy"


_UV_PHYSICAL_OPTICS_KEYS = (
    "depol",
    "rayleigh_fraction",
    "aerosol_fraction",
    "aerosol_moments",
    "aerosol_interp_fraction",
)

_UV_DUMPED_OPTICS_KEYS = ("asymm", "scaling", "fo_exact_scatter")

_UV_BASE_KEYS = (
    "wavelengths",
    "user_obsgeom",
    "heights",
    "tau",
    "omega",
    "albedo",
    "flux_factor",
)

_UV_OPTIONAL_KEYS = ("stream_value", "ref_total")


def _has_keys(bundle: dict[str, np.ndarray], keys: tuple[str, ...]) -> bool:
    return all(key in bundle for key in keys)


def _select_optical_keys(
    available: set[str],
    *,
    use_dumped_derived_optics: bool,
) -> tuple[str, ...]:
    if not use_dumped_derived_optics and set(_UV_PHYSICAL_OPTICS_KEYS).issubset(available):
        return _UV_PHYSICAL_OPTICS_KEYS
    return _UV_DUMPED_OPTICS_KEYS


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
    if use_dumped_derived_optics or not _has_keys(bundle, _UV_PHYSICAL_OPTICS_KEYS):
        require_keys(bundle, _UV_DUMPED_OPTICS_KEYS, label="UV dumped optical")
        mode = "dumped-derived"
        if not use_dumped_derived_optics and not _has_keys(bundle, _UV_PHYSICAL_OPTICS_KEYS):
            mode = "dumped-derived (physical optical inputs unavailable)"
        return bundle, 0.0, mode

    start = time.perf_counter()
    optics = build_two_stream_phase_inputs(
        ssa=bundle["omega"],
        depol=bundle["depol"],
        rayleigh_fraction=bundle["rayleigh_fraction"],
        aerosol_fraction=bundle["aerosol_fraction"],
        aerosol_moments=bundle["aerosol_moments"],
        aerosol_interp_fraction=bundle["aerosol_interp_fraction"],
    )
    fo_scatter = build_solar_fo_scatter_term(
        ssa=bundle["omega"],
        depol=bundle["depol"],
        rayleigh_fraction=bundle["rayleigh_fraction"],
        aerosol_fraction=bundle["aerosol_fraction"],
        aerosol_moments=bundle["aerosol_moments"],
        aerosol_interp_fraction=bundle["aerosol_interp_fraction"],
        angles=bundle["user_obsgeom"],
        delta_m_truncation_factor=optics.delta_m_truncation_factor,
    )
    prepared = dict(bundle)
    prepared["asymm"] = optics.g
    prepared["scaling"] = optics.delta_m_truncation_factor
    prepared["fo_exact_scatter"] = fo_scatter
    return prepared, time.perf_counter() - start, "python-generated"


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
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
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
        delta_m_truncation_factor=bundle["scaling"],
        include_fo=True,
        fo_scatter_term=bundle["fo_exact_scatter"],
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
    chunk_size = min(wavelengths, 30_000)
    return BenchmarkRow(
        backend="numpy-forward-levels" if output_levels else "numpy-forward",
        wavelengths=wavelengths,
        layers=int(bundle["tau"].shape[1]),
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
    if not np.isfinite(checksum):
        raise RuntimeError("benchmark output checksum is not finite")
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
        delta_m_truncation_factor=bundle["scaling"],
        include_fo=True,
        fo_scatter_term=bundle["fo_exact_scatter"],
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
    chunk_size = min(wavelengths, 30_000)
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
    parser.add_argument(
        "--use-dumped-derived-optics",
        action="store_true",
        help="Use stored g, delta-M factor, and FO scatter instead of Python preprocessing.",
    )
    args = parser.parse_args()

    load_start = time.perf_counter()
    available = bundle_keys(args.bundle)
    optical_keys = _select_optical_keys(
        available,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )
    bundle = load_bundle(
        args.bundle,
        keys=_UV_BASE_KEYS + _UV_OPTIONAL_KEYS + optical_keys,
    )
    require_keys(
        bundle,
        _UV_BASE_KEYS + optical_keys,
        label="UV",
    )
    total_rows = int(bundle["tau"].shape[0])
    wavelengths = total_rows if args.limit is None else min(int(args.limit), total_rows)
    bundle = dict(bundle)
    bundle["wavelengths"] = bundle["wavelengths"][:wavelengths]
    bundle["tau"] = bundle["tau"][:wavelengths]
    bundle["omega"] = bundle["omega"][:wavelengths]
    bundle["albedo"] = bundle["albedo"][:wavelengths]
    bundle["flux_factor"] = bundle["flux_factor"][:wavelengths]
    for key in _UV_DUMPED_OPTICS_KEYS:
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    for key in ("depol", "rayleigh_fraction", "aerosol_fraction", "aerosol_interp_fraction"):
        if key in bundle:
            bundle[key] = bundle[key][:wavelengths]
    if "ref_total" in bundle:
        bundle["ref_total"] = bundle["ref_total"][:wavelengths]
    if "stream_value" not in bundle:
        bundle["stream_value"] = np.array([1.0 / np.sqrt(3.0)], dtype=float)
    bundle, geometry_seconds = _prepare_geometry(bundle)
    bundle, optical_seconds, optical_mode = _prepare_optics(
        bundle,
        use_dumped_derived_optics=args.use_dumped_derived_optics,
    )
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
    print(f"  geometry preprocessing: python-generated, {geometry_seconds:.3f} s")
    print(f"  optical preprocessing: {optical_mode}, {optical_seconds:.3f} s")

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
