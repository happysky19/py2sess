#!/usr/bin/env python3
"""Smoke, parity, and speed checks for py2sess native CUDA builds.

Run from the repository root after building the extension with
PY2SESS_BUILD_CUDA=1. The generated inputs are synthetic and stay inside the
pyharp-style TwoStreamEssNative module boundary:

    prop(nwave, ncol, nlyr, nprop) -> flux(nwave, ncol, nlyr + 1, 2)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import torch


def _dtype(name: str) -> torch.dtype:
    return {"float64": torch.float64, "float32": torch.float32}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_checksum(tensor: torch.Tensor) -> float:
    flat = tensor.reshape(-1)
    stride = max(1, flat.numel() // 1024)
    return float(flat[::stride].sum().detach().cpu().item())


def _make_options(native: Any) -> Any:
    options = native.TwoStreamEssNativeOptions()
    options.stream_value = 0.5
    options.x0 = 1.0
    options.user_stream = 1.0
    options.user_secant = 1.0
    options.azmfac = 1.0
    options.px11 = 1.0
    options.ulp = 0.0
    return options


def _make_prop(
    *,
    nwave: int,
    ncol: int,
    nlay: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    prop = torch.empty((nwave, ncol, nlay, 4), dtype=dtype, device=device)
    row = torch.arange(nwave * ncol, dtype=dtype, device=device).reshape(nwave, ncol, 1)
    layer = torch.linspace(0.0, 1.0, nlay, dtype=dtype, device=device).reshape(1, 1, nlay)
    row_mod = row.remainder(29.0)

    prop[..., 0] = 0.004 + 0.050 * layer + 0.00005 * row_mod
    prop[..., 1] = 0.08 + 0.18 * layer
    prop[..., 2] = 0.04 + 0.15 * layer
    prop[..., 3] = 0.0
    return prop


def _make_thermal_inputs(
    *,
    nwave: int,
    ncol: int,
    nlay: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    nlev = nlay + 1
    row = torch.arange(nwave * ncol, dtype=dtype, device=device).reshape(nwave, ncol, 1)
    level = torch.linspace(0.0, 1.0, nlev, dtype=dtype, device=device).reshape(1, 1, nlev)
    field = torch.arange(nwave * ncol, dtype=dtype, device=device).reshape(nwave, ncol)
    return {
        "planck": 1.0 + 0.25 * level + 0.0001 * row.remainder(17.0),
        "surfbb": 1.32 + 0.0002 * field.remainder(11.0),
        "emissivity": torch.full((nwave, ncol), 0.94, dtype=dtype, device=device),
        "albedo": torch.full((nwave, ncol), 0.04, dtype=dtype, device=device),
    }


def _make_solar_inputs(
    *,
    nwave: int,
    ncol: int,
    nlay: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    field = torch.arange(nwave * ncol, dtype=dtype, device=device).reshape(nwave, ncol)
    return {
        "albedo": 0.08 + 0.0001 * field.remainder(13.0),
        "flux_factor": torch.ones((nwave, ncol), dtype=dtype, device=device),
        "chapman": torch.tril(torch.ones((nlay, nlay), dtype=dtype, device=device)),
        "pxsq": torch.ones(2, dtype=dtype, device=device),
        "px0x": torch.ones(2, dtype=dtype, device=device),
    }


def _thermal_call(
    native: Any,
    *,
    nwave: int,
    ncol: int,
    nlay: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[], torch.Tensor]:
    module = native.TwoStreamEssNative(_make_options(native))
    prop = _make_prop(nwave=nwave, ncol=ncol, nlay=nlay, dtype=dtype, device=device)
    inputs = _make_thermal_inputs(nwave=nwave, ncol=ncol, nlay=nlay, dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return module.thermal_2s_flux(prop, **inputs)

    return run


def _solar_call(
    native: Any,
    *,
    nwave: int,
    ncol: int,
    nlay: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[], torch.Tensor]:
    module = native.TwoStreamEssNative(_make_options(native))
    prop = _make_prop(nwave=nwave, ncol=ncol, nlay=nlay, dtype=dtype, device=device)
    inputs = _make_solar_inputs(nwave=nwave, ncol=ncol, nlay=nlay, dtype=dtype, device=device)

    def run() -> torch.Tensor:
        return module.solar_2s_flux(prop, **inputs)

    return run


def _assert_close(
    label: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    rtol, atol = (1.0e-8, 1.0e-10) if dtype == torch.float64 else (2.0e-4, 2.0e-5)
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    scale = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    max_rel = float((diff / scale).max().item()) if diff.numel() else 0.0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(f"{label}: parity ok, max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")


def _run_parity(native: Any, args: argparse.Namespace, dtype: torch.dtype) -> None:
    rows = args.parity_rows
    ncol = args.ncol
    if rows % ncol:
        raise SystemExit("--parity-rows must be divisible by --ncol")
    nwave = rows // ncol
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    with torch.no_grad():
        thermal_cpu = _thermal_call(
            native, nwave=nwave, ncol=ncol, nlay=args.layers, dtype=dtype, device=cpu
        )()
        thermal_cuda = _thermal_call(
            native, nwave=nwave, ncol=ncol, nlay=args.layers, dtype=dtype, device=cuda
        )()
        _sync(cuda)
        _assert_close("thermal module CPU vs CUDA", thermal_cuda.cpu(), thermal_cpu, dtype)

        solar_cpu = _solar_call(
            native, nwave=nwave, ncol=ncol, nlay=args.layers, dtype=dtype, device=cpu
        )()
        solar_cuda = _solar_call(
            native, nwave=nwave, ncol=ncol, nlay=args.layers, dtype=dtype, device=cuda
        )()
        _sync(cuda)
        _assert_close("solar module CPU vs CUDA", solar_cuda.cpu(), solar_cpu, dtype)


def _time_call(
    *,
    label: str,
    run: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
    warmups: int,
) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last = None
    with torch.no_grad():
        for _ in range(warmups):
            last = run()
            _sync(device)
            del last
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            last = run()
            _sync(device)
            timings.append(time.perf_counter() - start)

    assert last is not None
    checksum = _sample_checksum(last)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else float("nan")
    )
    print(
        f"{label}: best={min(timings):.6f}s median={statistics.median(timings):.6f}s "
        f"mean={statistics.mean(timings):.6f}s shape={tuple(last.shape)} "
        f"checksum={checksum:.6e} peak_cuda_mem={peak_mb:.1f} MiB"
    )


def _run_speed(native: Any, args: argparse.Namespace, dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    cases = (
        ("TIR thermal module", args.tir_rows, _thermal_call),
        ("UV solar module", args.uv_rows, _solar_call),
    )
    for label, rows, factory in cases:
        if rows % args.ncol:
            raise SystemExit(f"{label}: row count must be divisible by --ncol")
        nwave = rows // args.ncol
        run = factory(
            native, nwave=nwave, ncol=args.ncol, nlay=args.layers, dtype=dtype, device=device
        )
        _time_call(
            label=label,
            run=run,
            device=device,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        del run
        gc.collect()
        torch.cuda.empty_cache()


def _run_cpu_only_smoke(native: Any, args: argparse.Namespace, dtype: torch.dtype) -> None:
    rows = min(args.parity_rows, 512)
    if rows % args.ncol:
        rows = args.ncol * max(1, rows // args.ncol)
    device = torch.device("cpu")
    nwave = rows // args.ncol
    with torch.no_grad():
        thermal = _thermal_call(
            native, nwave=nwave, ncol=args.ncol, nlay=args.layers, dtype=dtype, device=device
        )()
        solar = _solar_call(
            native, nwave=nwave, ncol=args.ncol, nlay=args.layers, dtype=dtype, device=device
        )()
    print(
        f"CPU-only thermal smoke shape={tuple(thermal.shape)} checksum={_sample_checksum(thermal):.6e}"
    )
    print(f"CPU-only solar smoke shape={tuple(solar.shape)} checksum={_sample_checksum(solar):.6e}")


def _load_native() -> tuple[Any, dict[str, Any], bool]:
    try:
        import torch as _torch

        _torch.cuda.is_available()  # force PyTorch shared-library initialization
        _native = importlib.import_module("py2sess._native")
        from py2sess import native_backend_info
        from py2sess.rtsolver.native_backend import native_backend_supports_device
    except Exception as exc:  # pragma: no cover - exercised on a broken install
        raise SystemExit(
            "Could not import py2sess native extension. Build first with "
            "PY2SESS_BUILD_CUDA=1 python setup.py build_ext --inplace"
        ) from exc
    return _native, native_backend_info(), native_backend_supports_device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--layers", type=int, default=114)
    parser.add_argument("--ncol", type=int, default=1)
    parser.add_argument("--parity-rows", type=int, default=256)
    parser.add_argument("--tir-rows", type=int, default=200_000)
    parser.add_argument("--uv-rows", type=int, default=280_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use small benchmark sizes.")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--allow-cpu-only", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.parity_rows = min(args.parity_rows, 64)
        args.tir_rows = min(args.tir_rows, 20_000)
        args.uv_rows = min(args.uv_rows, 20_000)
        args.repeats = min(args.repeats, 2)
        args.warmups = min(args.warmups, 1)

    dtype = _dtype(args.dtype)
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability(0)}")

    native, info, native_cuda = _load_native()
    print(f"native_backend_info={info}")

    if not torch.cuda.is_available() or not native_cuda:
        if args.allow_cpu_only:
            print("CUDA native backend is not available; running CPU-only smoke checks.")
            _run_cpu_only_smoke(native, args, dtype)
            return
        raise SystemExit(
            "Native CUDA is not available. In Colab, switch to a GPU runtime and rebuild with "
            "PY2SESS_BUILD_CUDA=1."
        )

    _run_parity(native, args, dtype)
    if not args.skip_speed:
        _run_speed(native, args, dtype)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
