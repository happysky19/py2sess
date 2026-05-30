#!/usr/bin/env python3
"""Run full-spectrum py2sess and 2S-ESS Fortran RT benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _full_spectrum_benchmark_common import (  # noqa: E402
    accuracy_summary,
    recommended_chunk_size,
    scalar_value,
    slice_spectral_rows,
)
from benchmark_scene_full_spectrum import (  # noqa: E402
    SOLAR_COMPONENT_KEYS,
    THERMAL_COMPONENT_KEYS,
    _run_numpy_component_timing,
)
from py2sess.rtsolver.backend import has_torch, to_numpy  # noqa: E402
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (  # noqa: E402
    fo_solar_obs_batch_precompute,
)
from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors  # noqa: E402
from py2sess.rtsolver.solar_obs_batch_torch import solve_solar_obs_batch_torch  # noqa: E402
from py2sess.rtsolver.fo_solar_obs_batch_torch import (  # noqa: E402
    solve_fo_solar_obs_eps_batch_torch,
)
from py2sess.rtsolver.native_backend import (  # noqa: E402
    native_backend_supports_device,
    solve_solar_2s,
    solve_solar_fo,
    solve_thermal_2s,
    solve_thermal_fo,
)
from py2sess.rtsolver.thermal_batch_numpy import (  # noqa: E402
    precompute_fo_thermal_geometry_numpy,
)
from py2sess.rtsolver.thermal_batch_torch import (  # noqa: E402
    _fo_thermal_toa_batch,
    _two_stream_thermal_toa_batch,
    fo_thermal_geometry_to_torch,
)
from py2sess.scene import load_scene  # noqa: E402


DEFAULT_EXTERNAL_ROOT = Path("/Users/thl/MyFolder/Research/2S-ESS")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "full_spectrum_benchmark"
DEFAULT_TORCH_DTYPES = ("float64",)
RAW_FIELDS = (
    "created_utc",
    "system",
    "case",
    "mode",
    "backend",
    "device",
    "dtype",
    "timing_kind",
    "repeat_index",
    "seconds",
    "wall_seconds",
    "fo_seconds",
    "two_stream_seconds",
    "setup_seconds",
    "fo2s_op_seconds",
    "create_prop_seconds",
    "write_seconds",
    "overall_seconds",
    "wavelengths",
    "layers",
    "chunk_size",
    "rows_per_second",
    "max_abs_diff",
    "max_rel_diff_pct",
    "status",
    "skip_reason",
    "source",
)
SUMMARY_FIELDS = (
    "created_utc",
    "system",
    "case",
    "mode",
    "backend",
    "device",
    "dtype",
    "timing_kind",
    "wavelengths",
    "layers",
    "chunk_size",
    "n_repeats",
    "best_s",
    "mean_s",
    "total_mean_s",
    "median_s",
    "std_s",
    "min_s",
    "max_s",
    "rows_per_second_best",
    "fo_mean_s",
    "two_stream_mean_s",
    "setup_mean_s",
    "fo2s_op_mean_s",
    "max_abs_diff",
    "max_rel_diff_pct",
    "status",
)
MANIFEST_FIELDS = (
    "created_utc",
    "kind",
    "key",
    "value",
    "status",
    "reason",
)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    case: str
    mode: str
    profile: Path
    scene: Path
    wavelengths: int
    layers: int


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    label: str
    device: str
    dtype: str
    compiled: bool = False
    compile_mode: str = "reduce-overhead"


@dataclass(frozen=True)
class FortranCase:
    key: str
    case: str
    mode: str
    relative_dir: Path
    executable: str
    args: tuple[str, ...]
    env: dict[str, str]
    timing_glob: str
    wavelengths: int
    layers: int


def _torch_module():
    if not has_torch():
        return None
    import torch

    return torch


def _split_csv(value: str, *, allowed: set[str], label: str) -> tuple[str, ...]:
    parsed = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{label} must not be empty")
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported values: {', '.join(unknown)}")
    return parsed


def _case_specs(input_root: Path | None) -> dict[str, CaseSpec]:
    if input_root is None:
        bundle_root = ROOT / "benchmark_bundles"
        profile_root = DEFAULT_EXTERNAL_ROOT / "geocape_data" / "Profile_Data"
    else:
        bundle_root = input_root / "benchmark_bundles"
        profile_root = input_root / "profiles"
    return {
        "tir": CaseSpec(
            key="tir",
            case="TIR",
            mode="thermal",
            profile=profile_root / "Profiles_1_2006726_0000.dat",
            scene=bundle_root / "tir_scene_python.yaml",
            wavelengths=200000,
            layers=114,
        ),
        "uv": CaseSpec(
            key="uv",
            case="UV",
            mode="solar",
            profile=profile_root / "Profiles_1_2006726_1500.dat",
            scene=bundle_root / "uv_scene_python.yaml",
            wavelengths=280000,
            layers=114,
        ),
    }


def _fortran_cases() -> dict[str, FortranCase]:
    return {
        "tir": FortranCase(
            key="tir",
            case="TIR",
            mode="thermal",
            relative_dir=Path("TIR") / "2S-ESS",
            executable="./test_Thermal_Rad_Exact_opt.exe",
            args=("1", "26", "0000"),
            env={"TIR_USE_DUMP": "T"},
            timing_glob="Results_Exact_Opt/Exact*_L1_D26_T0000.Tim",
            wavelengths=200000,
            layers=114,
        ),
        "uv": FortranCase(
            key="uv",
            case="UV",
            mode="solar",
            relative_dir=Path("UVVSWIR") / "2S-ESS",
            executable="./test_UVVSWIR_Rad_Exact_opt.exe",
            args=("1", "26", "1500"),
            env={"UVVSWIR_USE_DUMP": "T"},
            timing_glob="Results_Exact_Opt/Exact*L1D26T1500.Tim",
            wavelengths=280000,
            layers=114,
        ),
    }


def _backend_configs(
    backend_set: str,
    torch_dtypes: tuple[str, ...],
    *,
    torch_compile: bool,
    torch_compile_mode: str,
    created_utc: str,
) -> tuple[list[BackendConfig], list[dict[str, str]]]:
    configs: list[BackendConfig] = []
    manifest: list[dict[str, str]] = []
    if backend_set in {"all", "cpu", "numpy"}:
        configs.append(BackendConfig("numpy", "NumPy", "", "float64"))
    if backend_set == "numpy":
        return configs, manifest

    torch = _torch_module()
    if torch is None:
        manifest.append(
            _manifest_row(
                created_utc,
                "backend",
                "torch",
                "",
                status="skipped",
                reason="PyTorch is not installed",
            )
        )
        return configs, manifest
    if backend_set in {"all", "cpu"}:
        for dtype in torch_dtypes:
            label = "Torch CPU torch.compile" if torch_compile else "Torch CPU"
            configs.append(
                BackendConfig(
                    "torch",
                    label,
                    "cpu",
                    dtype,
                    compiled=torch_compile,
                    compile_mode=torch_compile_mode,
                )
            )
            if native_backend_supports_device("cpu"):
                configs.append(BackendConfig("native", "Native CPU", "cpu", dtype))
            else:
                manifest.append(
                    _manifest_row(
                        created_utc,
                        "backend",
                        f"native-cpu-{dtype}",
                        "unavailable",
                        status="skipped",
                        reason="native extension is not built for CPU",
                    )
                )
    if backend_set == "native":
        for dtype in torch_dtypes:
            if native_backend_supports_device("cpu"):
                configs.append(BackendConfig("native", "Native CPU", "cpu", dtype))
            else:
                manifest.append(
                    _manifest_row(
                        created_utc,
                        "backend",
                        f"native-cpu-{dtype}",
                        "unavailable",
                        status="skipped",
                        reason="native extension is not built for CPU",
                    )
                )
    if backend_set in {"all", "cuda"}:
        if torch.cuda.is_available():
            for dtype in torch_dtypes:
                label = "Torch CUDA torch.compile" if torch_compile else "Torch CUDA"
                configs.append(
                    BackendConfig(
                        "torch",
                        label,
                        "cuda",
                        dtype,
                        compiled=torch_compile,
                        compile_mode=torch_compile_mode,
                    )
                )
                if native_backend_supports_device("cuda"):
                    configs.append(BackendConfig("native", "Native CUDA", "cuda", dtype))
                else:
                    manifest.append(
                        _manifest_row(
                            created_utc,
                            "backend",
                            f"native-cuda-{dtype}",
                            "unavailable",
                            status="skipped",
                            reason="native extension is not built for CUDA",
                        )
                    )
        else:
            for dtype in torch_dtypes:
                manifest.append(
                    _manifest_row(
                        created_utc,
                        "backend",
                        f"torch-cuda-{dtype}",
                        "unavailable",
                        status="skipped",
                        reason="torch.cuda.is_available() is false",
                    )
                )
    return configs, manifest


def _manifest_row(
    created_utc: str,
    kind: str,
    key: str,
    value: str,
    *,
    status: str = "ok",
    reason: str = "",
) -> dict[str, str]:
    return {
        "created_utc": created_utc,
        "kind": kind,
        "key": key,
        "value": value,
        "status": status,
        "reason": reason,
    }


def _require_paths(specs: list[CaseSpec]) -> None:
    missing = []
    for spec in specs:
        for path in (spec.profile, spec.scene):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "missing full-spectrum inputs:\n"
            + "\n".join(missing)
            + "\n\nOn Colab, upload or mount the exported full-spectrum input bundle "
            "and pass --input-root /path/to/py2sess_full_spectrum_inputs."
        )


def _load_scene_and_inputs(spec: CaseSpec) -> tuple[Any, Any, float, dict[str, float]]:
    load_start = time.perf_counter()
    scene = load_scene(
        profile=spec.profile,
        config=spec.scene,
        strict_runtime_inputs=True,
    )
    load_seconds = time.perf_counter() - load_start
    inputs = scene.to_forward_inputs()
    return scene, inputs, load_seconds, dict(inputs.timings)


def _sync_if_cuda(config: BackendConfig) -> None:
    if config.backend in {"torch", "native"} and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.synchronize(torch.device(config.device))


def _cuda_peak(config: BackendConfig) -> str:
    if config.backend in {"torch", "native"} and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            return str(int(torch.cuda.max_memory_allocated(torch.device(config.device))))
    return ""


def _reset_cuda_peak(config: BackendConfig) -> None:
    if config.backend in {"torch", "native"} and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(torch.device(config.device))


def _run_repeats(
    *,
    created_utc: str,
    spec: CaseSpec,
    run_once: Callable[[], dict[str, Any]],
    repeats: int,
    warmups: int,
    base_row: dict[str, Any],
    config: BackendConfig | None = None,
) -> list[dict[str, Any]]:
    for _ in range(warmups):
        run_once()
        if config is not None:
            _sync_if_cuda(config)
    rows = []
    for repeat in range(repeats):
        try:
            if config is not None:
                _reset_cuda_peak(config)
                _sync_if_cuda(config)
            metrics = run_once()
            if config is not None:
                _sync_if_cuda(config)
            row = {
                "created_utc": created_utc,
                "repeat_index": repeat,
                "wavelengths": spec.wavelengths,
                "layers": spec.layers,
                "status": "ok",
                "skip_reason": "",
                **base_row,
                **metrics,
            }
            rows.append(row)
        except Exception as exc:  # pragma: no cover - exercised by real benchmark failures
            rows.append(
                {
                    "created_utc": created_utc,
                    "repeat_index": repeat,
                    "wavelengths": spec.wavelengths,
                    "layers": spec.layers,
                    "status": "failed",
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                    **base_row,
                }
            )
    return rows


def _run_scene_forward_once(
    scene: Any,
    inputs: Any,
    config: BackendConfig,
    *,
    torch_threads: int,
    torch_bvp_engine: str,
    numpy_bvp_engine: str,
    output_levels: bool,
    output_fluxes: bool,
    fo_flux_n_mu: int,
) -> dict[str, Any]:
    common_options = {
        "output_levels": output_levels,
        "output_fluxes": output_fluxes,
        "fo_flux_n_mu": fo_flux_n_mu,
    }
    native_flux_only = config.backend == "native" and output_fluxes
    if native_flux_only:
        common_options.pop("output_levels")
        common_options.pop("output_fluxes")
    if output_fluxes:
        common_options["plane_parallel"] = True
    if config.backend in {"torch", "native"}:
        torch = _torch_module()
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if config.backend == "native" and not native_backend_supports_device(config.device):
            raise RuntimeError(f"native extension is not built for {config.device!r}")
        torch.set_num_threads(torch_threads)
        options = {
            "backend": config.backend,
            "torch_device": config.device,
            "torch_dtype": config.dtype,
            "torch_enable_grad": False,
            "bvp_solver": "auto" if torch_bvp_engine == "auto" else torch_bvp_engine,
            **common_options,
        }
    else:
        options = {
            "backend": "numpy",
            "bvp_solver": "auto" if numpy_bvp_engine == "auto" else numpy_bvp_engine,
            **common_options,
        }
    _sync_if_cuda(config)
    start = time.perf_counter()
    if native_flux_only:
        result = scene.forward_flux(**options, include_fo=True, return_net=True)
    else:
        result = scene.forward(**options, include_fo=True)
    _sync_if_cuda(config)
    seconds = time.perf_counter() - start
    if native_flux_only:
        max_abs = ""
        max_rel = ""
    else:
        radiance = np.asarray(to_numpy(result.radiance_total), dtype=float)
        max_abs, max_rel = accuracy_summary(radiance, inputs.reference_total)
    return {
        "seconds": seconds,
        "wall_seconds": seconds,
        "chunk_size": _recommended_chunk(inputs, config.backend),
        "rows_per_second": inputs.wavelengths.shape[0] / seconds,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "source": _cuda_peak(config),
    }


def _normalize_timing_kinds(args: argparse.Namespace) -> tuple[str, ...]:
    allowed = {"components", "scene-forward", "level-fluxes"}
    if args.timing_kinds is None:
        parsed = ("level-fluxes",) if args.output_levels or args.output_fluxes else ("components",)
    else:
        parsed = _split_csv(args.timing_kinds, allowed=allowed, label="--timing-kinds")
    if args.components and "components" not in parsed:
        parsed = (*parsed, "components")
    return parsed


def _recommended_chunk(inputs: Any, backend: str) -> int:
    workload = "solar_obs" if inputs.mode == "solar" else "thermal"
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(inputs.kwargs["tau"]).shape[-1])
    return recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend=backend,
        workload=workload,
    )


def _run_numpy_components_once(
    scene: Any,
    inputs: Any,
    *,
    numpy_bvp_engine: str,
) -> dict[str, Any]:
    args = SimpleNamespace(chunk_size=None, numpy_bvp_engine=numpy_bvp_engine)
    row = _run_numpy_component_timing(scene, args=args)
    return {
        "seconds": row.rt_seconds,
        "wall_seconds": row.wall_seconds,
        "fo_seconds": row.fo_seconds,
        "two_stream_seconds": row.two_stream_seconds,
        "chunk_size": row.chunk_size,
        "rows_per_second": row.rows_per_second_rt,
        "max_abs_diff": row.max_abs_diff,
        "max_rel_diff_pct": row.max_rel_diff_pct,
        "source": "",
    }


def _torch_dtype(dtype: str):
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    return {"float64": torch.float64, "float32": torch.float32}[dtype]


def _tensor(value: Any, *, dtype: Any, device: Any):
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    return torch.as_tensor(value, dtype=dtype, device=device)


def _torch_fo_solar_precompute(precomputed: Any, *, dtype: Any, device: Any) -> Any:
    fields = {
        "inv_layer_thickness": _tensor(precomputed.inv_layer_thickness, dtype=dtype, device=device),
        "do_nadir": precomputed.do_nadir,
        "mu0": precomputed.mu0,
        "ntrav_nl": precomputed.ntrav_nl,
        "sunpathsnl": _tensor(precomputed.sunpathsnl, dtype=dtype, device=device),
        "cota": _tensor(precomputed.cota, dtype=dtype, device=device),
        "cotfine": _tensor(precomputed.cotfine, dtype=dtype, device=device),
        "csqfine": _tensor(precomputed.csqfine, dtype=dtype, device=device),
        "wfine": _tensor(precomputed.wfine, dtype=dtype, device=device),
        "nfinedivs": precomputed.nfinedivs,
        "rayconv": precomputed.rayconv,
        "xfine": _tensor(precomputed.xfine, dtype=dtype, device=device),
        "sunpathsfine": _tensor(precomputed.sunpathsfine, dtype=dtype, device=device),
        "ntraversefine": precomputed.ntraversefine,
        "fine_path_matrix": (
            None
            if precomputed.fine_path_matrix is None
            else _tensor(precomputed.fine_path_matrix, dtype=dtype, device=device)
        ),
        "fine_column_index": precomputed.fine_column_index,
    }
    return SimpleNamespace(**fields)


def _compile_torch_callable(func: Callable[..., Any], config: BackendConfig) -> Callable[..., Any]:
    if config.backend != "torch" or not config.compiled:
        return func
    torch = _torch_module()
    if torch is None or not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch installation")
    if config.compile_mode == "default":
        return torch.compile(func)
    return torch.compile(func, mode=config.compile_mode)


def _timed_torch_call(
    ctx_device: Any, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, float]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if ctx_device.type == "cuda":
        torch.cuda.synchronize(ctx_device)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if ctx_device.type == "cuda":
        torch.cuda.synchronize(ctx_device)
    return result, time.perf_counter() - start


def _run_torch_components_once(
    scene: Any,
    inputs: Any,
    config: BackendConfig,
    *,
    torch_bvp_engine: str,
) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    if inputs.mode == "solar":
        return _run_torch_solar_components_once(
            inputs,
            dtype=dtype,
            device=device,
            bvp_engine=torch_bvp_engine,
            config=config,
        )
    return _run_torch_thermal_components_once(
        inputs,
        dtype=dtype,
        device=device,
        bvp_engine=torch_bvp_engine,
        config=config,
    )


def _run_native_components_once(
    scene: Any,
    inputs: Any,
    config: BackendConfig,
) -> dict[str, Any]:
    del scene
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not native_backend_supports_device(config.device):
        raise RuntimeError(f"native extension is not built for {config.device!r}")
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    if inputs.mode == "solar":
        return _run_native_solar_components_once(inputs, dtype=dtype, device=device, config=config)
    return _run_native_thermal_components_once(inputs, dtype=dtype, device=device, config=config)


def _run_torch_solar_components_once(
    inputs: Any,
    *,
    dtype: Any,
    device: Any,
    bvp_engine: str,
    config: BackendConfig,
) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="torch",
        workload="solar_obs",
    )
    heights = np.asarray(kwargs["z"], dtype=float)
    angles = np.asarray(kwargs["angles"], dtype=float)
    geoms = angles.reshape(1, 3) if angles.ndim == 1 else angles
    if geoms.shape[0] != 1:
        raise ValueError("component timing supports one solar geometry")
    stream = float(kwargs.get("stream", 1.0 / np.sqrt(3.0)))
    sza, vza, raz = geoms[0]
    x0 = np.array([np.cos(np.deg2rad(sza))], dtype=float)
    user_stream = np.array([np.cos(np.deg2rad(vza))], dtype=float)
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
    fo_precomputed_t = _torch_fo_solar_precompute(fo_precomputed, dtype=dtype, device=device)
    chapman_t = _tensor(chapman, dtype=dtype, device=device)
    pxsq_values = tuple(float(item) for item in np.asarray(pxsq, dtype=float).reshape(-1))
    px0x_values = tuple(float(item) for item in np.asarray(px0x[0], dtype=float).reshape(-1))
    bundle = dict(kwargs)

    def fo_kernel(tau, omega, scaling, albedo, fbeam, exact_scatter):
        return solve_fo_solar_obs_eps_batch_torch(
            tau=tau,
            omega=omega,
            scaling=scaling,
            albedo=albedo,
            flux_factor=fbeam,
            exact_scatter=exact_scatter,
            precomputed=fo_precomputed_t,
            dtype=dtype,
            device=device,
        )

    def two_stream_kernel(tau, omega, asymm, scaling, albedo, fbeam):
        return solve_solar_obs_batch_torch(
            tau=tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            albedo=albedo,
            flux_factor=fbeam,
            stream_value=stream,
            chapman=chapman_t,
            x0=scalar_value(x0),
            user_stream=scalar_value(user_stream),
            user_secant=1.0 / scalar_value(user_stream),
            azmfac=scalar_value(azmfac),
            px11=px11,
            pxsq=pxsq_values,
            px0x=px0x_values,
            ulp=scalar_value(ulp),
            dtype=dtype,
            device=device,
            bvp_engine=bvp_engine,
        )

    fo_kernel = _compile_torch_callable(fo_kernel, config)
    two_stream_kernel = _compile_torch_callable(two_stream_kernel, config)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts = []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, wavelengths, chunk_size):
            stop = min(start + chunk_size, wavelengths)
            chunk = slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
            tau = _tensor(chunk["tau"], dtype=dtype, device=device)
            omega = _tensor(chunk["ssa"], dtype=dtype, device=device)
            scaling = _tensor(chunk["delta_m_truncation_factor"], dtype=dtype, device=device)
            albedo = _tensor(chunk["albedo"], dtype=dtype, device=device)
            fbeam = _tensor(chunk["fbeam"], dtype=dtype, device=device)
            fo, elapsed = _timed_torch_call(
                device,
                fo_kernel,
                tau,
                omega,
                scaling,
                albedo,
                fbeam,
                _tensor(chunk["fo_scatter_term"], dtype=dtype, device=device),
            )
            fo_seconds += elapsed
            two_stream, elapsed = _timed_torch_call(
                device,
                two_stream_kernel,
                tau,
                omega,
                _tensor(chunk["g"], dtype=dtype, device=device),
                scaling,
                albedo,
                fbeam,
            )
            two_stream_seconds += elapsed
            total_parts.append((fo + two_stream).detach().cpu().numpy())
    wall_seconds = time.perf_counter() - wall_start
    total = np.concatenate(total_parts)
    max_abs, max_rel = accuracy_summary(total, inputs.reference_total)
    seconds = fo_seconds + two_stream_seconds
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "fo_seconds": fo_seconds,
        "two_stream_seconds": two_stream_seconds,
        "chunk_size": chunk_size,
        "rows_per_second": wavelengths / seconds,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "source": _cuda_peak(config),
    }


def _run_native_solar_components_once(
    inputs: Any,
    *,
    dtype: Any,
    device: Any,
    config: BackendConfig,
) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="native",
        workload="solar_obs",
    )
    heights = np.asarray(kwargs["z"], dtype=float)
    angles = np.asarray(kwargs["angles"], dtype=float)
    geoms = angles.reshape(1, 3) if angles.ndim == 1 else angles
    if geoms.shape[0] != 1:
        raise ValueError("component timing supports one solar geometry")
    stream = float(kwargs.get("stream", 1.0 / np.sqrt(3.0)))
    sza, vza, raz = geoms[0]
    x0 = np.array([np.cos(np.deg2rad(sza))], dtype=float)
    user_stream = np.array([np.cos(np.deg2rad(vza))], dtype=float)
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
    fo_precomputed_t = _torch_fo_solar_precompute(fo_precomputed, dtype=dtype, device=device)
    chapman_t = _tensor(chapman, dtype=dtype, device=device)
    pxsq_t = _tensor(np.asarray(pxsq, dtype=float), dtype=dtype, device=device)
    px0x_t = _tensor(np.asarray(px0x[0], dtype=float), dtype=dtype, device=device)
    bundle = dict(kwargs)

    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts = []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, wavelengths, chunk_size):
            stop = min(start + chunk_size, wavelengths)
            chunk = slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
            tau = _tensor(chunk["tau"], dtype=dtype, device=device)
            omega = _tensor(chunk["ssa"], dtype=dtype, device=device)
            scaling = _tensor(chunk["delta_m_truncation_factor"], dtype=dtype, device=device)
            albedo = _tensor(chunk["albedo"], dtype=dtype, device=device)
            fbeam = _tensor(chunk["fbeam"], dtype=dtype, device=device)
            fo, elapsed = _timed_torch_call(
                device,
                solve_solar_fo,
                tau=tau,
                omega=omega,
                scaling=scaling,
                albedo=albedo,
                flux_factor=fbeam,
                exact_scatter=_tensor(chunk["fo_scatter_term"], dtype=dtype, device=device),
                precomputed=fo_precomputed_t,
            )
            fo_seconds += elapsed
            two_stream, elapsed = _timed_torch_call(
                device,
                solve_solar_2s,
                tau=tau,
                omega=omega,
                asymm=_tensor(chunk["g"], dtype=dtype, device=device),
                scaling=scaling,
                albedo=albedo,
                flux_factor=fbeam,
                chapman=chapman_t,
                pxsq=pxsq_t,
                px0x=px0x_t,
                stream_value=stream,
                x0=scalar_value(x0),
                user_stream=scalar_value(user_stream),
                user_secant=1.0 / scalar_value(user_stream),
                azmfac=scalar_value(azmfac),
                px11=px11,
                ulp=scalar_value(ulp),
                return_profile=False,
            )
            two_stream_seconds += elapsed
            total_parts.append((fo + two_stream).detach().cpu().numpy())
    wall_seconds = time.perf_counter() - wall_start
    total = np.concatenate(total_parts)
    max_abs, max_rel = accuracy_summary(total, inputs.reference_total)
    seconds = fo_seconds + two_stream_seconds
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "fo_seconds": fo_seconds,
        "two_stream_seconds": two_stream_seconds,
        "chunk_size": chunk_size,
        "rows_per_second": wavelengths / seconds,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "source": _cuda_peak(config),
    }


def _run_torch_thermal_components_once(
    inputs: Any,
    *,
    dtype: Any,
    device: Any,
    bvp_engine: str,
    config: BackendConfig,
) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="torch",
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
    fo_geometry = fo_thermal_geometry_to_torch(geometry, dtype=dtype, device=device)
    heights_t = _tensor(heights, dtype=dtype, device=device)
    bundle = dict(kwargs)

    def two_stream_kernel(tau, omega, asymm, scaling, planck, surfbb, emissivity, albedo):
        return _two_stream_thermal_toa_batch(
            tau=tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            thermal_bb_input=planck,
            surfbb=surfbb,
            emissivity=emissivity,
            albedo=albedo,
            stream_value=stream,
            user_stream=user_stream,
            pxsq=stream * stream,
            thermal_tcutoff=1.0e-8,
            bvp_engine=bvp_engine,
        )

    def fo_kernel(tau, omega, scaling, planck, surfbb, emissivity):
        return _fo_thermal_toa_batch(
            tau=tau,
            omega=omega,
            scaling=scaling,
            thermal_bb_input=planck,
            surfbb=surfbb,
            emissivity=emissivity,
            heights=heights_t,
            user_angle_degrees=user_angle,
            earth_radius=6371.0,
            nfine=3,
            fo_geometry=fo_geometry,
        )

    two_stream_kernel = _compile_torch_callable(two_stream_kernel, config)
    fo_kernel = _compile_torch_callable(fo_kernel, config)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts = []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, wavelengths, chunk_size):
            stop = min(start + chunk_size, wavelengths)
            chunk = slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
            tau = _tensor(chunk["tau"], dtype=dtype, device=device)
            omega = _tensor(chunk["ssa"], dtype=dtype, device=device)
            scaling = _tensor(chunk["delta_m_truncation_factor"], dtype=dtype, device=device)
            planck = _tensor(chunk["planck"], dtype=dtype, device=device)
            surfbb = _tensor(chunk["surface_planck"], dtype=dtype, device=device)
            emissivity = _tensor(chunk["emissivity"], dtype=dtype, device=device)
            albedo = _tensor(chunk["albedo"], dtype=dtype, device=device)
            two_stream, elapsed = _timed_torch_call(
                device,
                two_stream_kernel,
                tau,
                omega,
                _tensor(chunk["g"], dtype=dtype, device=device),
                scaling,
                planck,
                surfbb,
                emissivity,
                albedo,
            )
            two_stream_seconds += elapsed
            fo, elapsed = _timed_torch_call(
                device,
                fo_kernel,
                tau,
                omega,
                scaling,
                planck,
                surfbb,
                emissivity,
            )
            fo_seconds += elapsed
            total_parts.append((fo + two_stream).detach().cpu().numpy())
    wall_seconds = time.perf_counter() - wall_start
    total = np.concatenate(total_parts)
    max_abs, max_rel = accuracy_summary(total, inputs.reference_total)
    seconds = fo_seconds + two_stream_seconds
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "fo_seconds": fo_seconds,
        "two_stream_seconds": two_stream_seconds,
        "chunk_size": chunk_size,
        "rows_per_second": wavelengths / seconds,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "source": _cuda_peak(config),
    }


def _run_native_thermal_components_once(
    inputs: Any,
    *,
    dtype: Any,
    device: Any,
    config: BackendConfig,
) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = inputs.kwargs
    wavelengths = int(inputs.wavelengths.shape[0])
    layers = int(np.asarray(kwargs["tau"]).shape[-1])
    chunk_size = recommended_chunk_size(
        total_rows=wavelengths,
        nlayers=layers,
        backend="native",
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
    fo_geometry = fo_thermal_geometry_to_torch(geometry, dtype=dtype, device=device)
    heights_t = _tensor(heights, dtype=dtype, device=device)
    bundle = dict(kwargs)

    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts = []
    wall_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, wavelengths, chunk_size):
            stop = min(start + chunk_size, wavelengths)
            chunk = slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
            tau = _tensor(chunk["tau"], dtype=dtype, device=device)
            omega = _tensor(chunk["ssa"], dtype=dtype, device=device)
            scaling = _tensor(chunk["delta_m_truncation_factor"], dtype=dtype, device=device)
            planck = _tensor(chunk["planck"], dtype=dtype, device=device)
            surfbb = _tensor(chunk["surface_planck"], dtype=dtype, device=device)
            emissivity = _tensor(chunk["emissivity"], dtype=dtype, device=device)
            albedo = _tensor(chunk["albedo"], dtype=dtype, device=device)
            two_stream, elapsed = _timed_torch_call(
                device,
                solve_thermal_2s,
                tau=tau,
                omega=omega,
                asymm=_tensor(chunk["g"], dtype=dtype, device=device),
                scaling=scaling,
                planck=planck,
                surfbb=surfbb,
                emissivity=emissivity,
                albedo=albedo,
                stream_value=stream,
                user_stream=user_stream,
                thermal_tcutoff=1.0e-8,
                return_profile=False,
            )
            two_stream_seconds += elapsed
            fo, elapsed = _timed_torch_call(
                device,
                solve_thermal_fo,
                tau=tau,
                omega=omega,
                scaling=scaling,
                planck=planck,
                surfbb=surfbb,
                emissivity=emissivity,
                heights=heights_t,
                geometry=fo_geometry,
            )
            fo_seconds += elapsed
            total_parts.append((fo + two_stream).detach().cpu().numpy())
    wall_seconds = time.perf_counter() - wall_start
    total = np.concatenate(total_parts)
    max_abs, max_rel = accuracy_summary(total, inputs.reference_total)
    seconds = fo_seconds + two_stream_seconds
    return {
        "seconds": seconds,
        "wall_seconds": wall_seconds,
        "fo_seconds": fo_seconds,
        "two_stream_seconds": two_stream_seconds,
        "chunk_size": chunk_size,
        "rows_per_second": wavelengths / seconds,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "source": _cuda_peak(config),
    }


def _prepare_fortran_bundle(external_root: Path, bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _symlink_or_keep(external_root / "geocape_data", bundle_dir / "geocape_data")

    specs = (
        (
            external_root / "TIR" / "2S-ESS",
            bundle_dir / "TIR" / "2S-ESS",
            (
                "test_Thermal_Rad_Exact_opt.exe",
                "test_Thermal_Rad_Exact_optimized.f90",
                "makefile_opt",
                "Configfiles_Thermal",
                "Dump_1_26_0000.dat_25",
            ),
        ),
        (
            external_root / "UVVSWIR" / "2S-ESS",
            bundle_dir / "UVVSWIR" / "2S-ESS",
            (
                "test_UVVSWIR_Rad_Exact_opt.exe",
                "test_UVVSWIR_Rad_Exact_optimized.f90",
                "makefile_opt",
                "Configfiles_Rad",
                "Dump_9_26_1500.dat_11_114",
            ),
        ),
    )
    for source_dir, target_dir, names in specs:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "Results_Exact_Opt").mkdir(exist_ok=True)
        (target_dir / "Debug_stuff").mkdir(exist_ok=True)
        for name in names:
            source = source_dir / name
            if source.exists():
                _symlink_or_keep(source, target_dir / name)
    readme = bundle_dir / "README.md"
    readme.write_text(
        "Local 2S-ESS full-spectrum benchmark rerun bundle.\n\n"
        "This folder is generated by scripts/benchmark_full_spectrum_rt.py and is ignored by git.\n"
        "Large dump files and source assets are symlinked to the external 2S-ESS checkout.\n\n"
        "TIR command:\n"
        "  cd TIR/2S-ESS && TIR_USE_DUMP=T ./test_Thermal_Rad_Exact_opt.exe 1 26 0000\n\n"
        "UV command:\n"
        "  cd UVVSWIR/2S-ESS && UVVSWIR_USE_DUMP=T ./test_UVVSWIR_Rad_Exact_opt.exe 1 26 1500\n",
        encoding="utf-8",
    )
    return bundle_dir


def _symlink_or_keep(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def _run_fortran_repeats(
    *,
    created_utc: str,
    case: FortranCase,
    bundle_dir: Path,
    output_dir: Path,
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    workdir = bundle_dir / case.relative_dir
    if not (workdir / case.executable.removeprefix("./")).exists():
        raise FileNotFoundError(f"missing Fortran executable in bundle: {workdir}")
    env = os.environ.copy()
    env.update(case.env)
    rows = []
    timing_dir = output_dir / "fortran_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    for index in range(warmups + repeats):
        measured = index >= warmups
        repeat_index = index - warmups
        log_path = (
            timing_dir / f"{case.key}_repeat_{repeat_index if measured else 'warmup'}_stdout.txt"
        )
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [case.executable, *case.args],
                cwd=workdir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall_seconds = time.perf_counter() - start
        if not measured:
            continue
        row_base = {
            "created_utc": created_utc,
            "system": "Fortran",
            "case": case.case,
            "mode": case.mode,
            "backend": "2S-ESS optimized",
            "device": "cpu",
            "dtype": "float64",
            "timing_kind": "module",
            "repeat_index": repeat_index,
            "wall_seconds": wall_seconds,
            "wavelengths": case.wavelengths,
            "layers": case.layers,
            "chunk_size": case.wavelengths,
            "source": str(log_path),
        }
        if completed.returncode != 0:
            rows.append(
                {
                    **row_base,
                    "status": "failed",
                    "skip_reason": f"Fortran exited with {completed.returncode}",
                }
            )
            continue
        timing_files = sorted(workdir.glob(case.timing_glob), key=lambda path: path.stat().st_mtime)
        if not timing_files:
            rows.append(
                {
                    **row_base,
                    "status": "failed",
                    "skip_reason": f"no timing file matched {case.timing_glob}",
                }
            )
            continue
        timing_file = timing_files[-1]
        copied_timing = timing_dir / f"{case.key}_repeat_{repeat_index}.Tim"
        copied_timing.write_text(timing_file.read_text(encoding="utf-8"), encoding="utf-8")
        metrics = _parse_fortran_timing_text(copied_timing.read_text(encoding="utf-8"))
        seconds = metrics["module_seconds"]
        rows.append(
            {
                **row_base,
                "seconds": seconds,
                "fo_seconds": metrics.get("fo_seconds"),
                "two_stream_seconds": metrics.get("two_stream_seconds"),
                "setup_seconds": metrics.get("setup_seconds"),
                "fo2s_op_seconds": metrics.get("fo2s_op_seconds"),
                "create_prop_seconds": metrics.get("create_prop_seconds"),
                "write_seconds": metrics.get("write_seconds"),
                "overall_seconds": metrics.get("overall_seconds"),
                "rows_per_second": case.wavelengths / seconds,
                "status": "ok",
                "skip_reason": "",
                "source": str(copied_timing),
            }
        )
    return rows


def _parse_fortran_timing_text(text: str) -> dict[str, float]:
    values = {
        "create_prop_seconds": _first_float(text, r"CreatePropTime \(1\)\s*=\s*([-+0-9.Ee]+)"),
        "rtm_setup_seconds": _first_float(text, r"(?:Exact)?RTMSetUps\s*=\s*([-+0-9.Ee]+)"),
        "fo_geom_seconds": _first_float(text, r"FOGeomTime\s*=\s*([-+0-9.Ee]+)"),
        "fo_spher_seconds": _first_float(text, r"FOSpherFnTime\s*=\s*([-+0-9.Ee]+)"),
        "two_geom_seconds": _first_float(text, r"2SGeomTime\s*=\s*([-+0-9.Ee]+)"),
        "fo2s_op_seconds": _first_float(text, r"(?:Exact)?FO2SOpTime\s*=\s*([-+0-9.Ee]+)"),
        "fo_seconds": _first_float(text, r"(?:Exact)?FOCalcTime\s*=\s*([-+0-9.Ee]+)"),
        "two_stream_seconds": _first_float(text, r"(?:Exact)?2SCalcTime\s*=\s*([-+0-9.Ee]+)"),
        "module_seconds": _first_float(text, r"(?:Exact)?ModuleTime \(2\)\s*=\s*([-+0-9.Ee]+)"),
        "write_seconds": _first_float(text, r"WriteTime \(3\)\s*=\s*([-+0-9.Ee]+)"),
        "overall_seconds": _first_float(text, r"OverallRunTime\s*=\s*([-+0-9.Ee]+)"),
    }
    setup = sum(
        values[key] or 0.0
        for key in ("rtm_setup_seconds", "fo_geom_seconds", "fo_spher_seconds", "two_geom_seconds")
    )
    values["setup_seconds"] = setup
    if values["module_seconds"] is None:
        raise ValueError("could not parse Fortran module RT time")
    return {key: value for key, value in values.items() if value is not None}


def _first_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return None if match is None else float(match.group(1))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _summarize(rows: list[dict[str, Any]], created_utc: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("system"),
            row.get("case"),
            row.get("mode"),
            row.get("backend"),
            row.get("device"),
            row.get("dtype"),
            row.get("timing_kind"),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        seconds = [float(row["seconds"]) for row in group]
        wavelengths = int(group[0]["wavelengths"])
        best_s = min(seconds)
        summary = {
            "created_utc": created_utc,
            "system": key[0],
            "case": key[1],
            "mode": key[2],
            "backend": key[3],
            "device": key[4],
            "dtype": key[5],
            "timing_kind": key[6],
            "wavelengths": wavelengths,
            "layers": group[0].get("layers", ""),
            "chunk_size": group[0].get("chunk_size", ""),
            "n_repeats": len(seconds),
            "best_s": best_s,
            "mean_s": statistics.fmean(seconds),
            "total_mean_s": statistics.fmean(seconds),
            "median_s": statistics.median(seconds),
            "std_s": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
            "min_s": min(seconds),
            "max_s": max(seconds),
            "rows_per_second_best": wavelengths / best_s,
            "fo_mean_s": _mean_optional(group, "fo_seconds"),
            "two_stream_mean_s": _mean_optional(group, "two_stream_seconds"),
            "setup_mean_s": _mean_optional(group, "setup_seconds"),
            "fo2s_op_mean_s": _mean_optional(group, "fo2s_op_seconds"),
            "max_abs_diff": _last_optional(group, "max_abs_diff"),
            "max_rel_diff_pct": _last_optional(group, "max_rel_diff_pct"),
            "status": "ok",
        }
        summaries.append(summary)
    return summaries


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("")
    print("Mean full-spectrum RT timing summary")
    print(
        f"{'case':<4} {'system/backend':<28} {'kind':<10} "
        f"{'n':>3} {'mean total (s)':>15} {'mean 2S (s)':>12} {'mean FO (s)':>12} "
        f"{'max abs diff':>12} {'max rel diff (%)':>16}"
    )
    print("-" * 121)
    for row in rows:
        label = str(row["system"])
        if row["system"] == "py2sess":
            label = str(row["backend"])
            if row.get("device"):
                label += f" {row['device']}"
        print(
            f"{row['case']:<4} {label:<28} {row['timing_kind']:<10} "
            f"{int(row['n_repeats']):>3d} {_format_optional(row.get('total_mean_s')):>15} "
            f"{_format_optional(row.get('two_stream_mean_s')):>12} "
            f"{_format_optional(row.get('fo_mean_s')):>12} "
            f"{_format_optional(row.get('max_abs_diff')):>12} "
            f"{_format_optional(row.get('max_rel_diff_pct')):>16}"
        )


def _format_optional(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.6g}"


def _mean_optional(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.fmean(values) if values else ""


def _last_optional(rows: list[dict[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if row.get(key) not in (None, ""):
            return row[key]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", default="tir,uv")
    parser.add_argument("--systems", default="python")
    parser.add_argument(
        "--timing-kinds",
        default=None,
        help=(
            "Comma-separated timing modes: components, scene-forward, or level-fluxes. "
            "Defaults to components unless --output-levels or --output-fluxes is set."
        ),
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="Also run component timing. Kept as a shorthand for --timing-kinds components.",
    )
    parser.add_argument(
        "--backend-set",
        choices=["all", "cpu", "numpy", "native", "cuda"],
        default="all",
    )
    parser.add_argument("--torch-dtypes", default=",".join(DEFAULT_TORCH_DTYPES))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--torch-bvp-engine", choices=["auto", "block", "pentadiagonal"], default="auto"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Compile torch component kernels with torch.compile. The backend label is "
            "tagged with 'torch.compile'; eager mode remains the default."
        ),
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="Compilation mode passed to torch.compile when --torch-compile is set.",
    )
    parser.add_argument(
        "--numpy-bvp-engine", choices=["auto", "block", "pentadiagonal"], default="auto"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--output-levels", action="store_true")
    parser.add_argument("--output-fluxes", action="store_true")
    parser.add_argument("--fo-flux-n-mu", type=int, default=8)
    parser.add_argument("--fortran-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--fortran-bundle-dir", type=Path, default=None)
    args = parser.parse_args()

    cases = _split_csv(args.cases, allowed={"tir", "uv"}, label="--cases")
    systems = _split_csv(args.systems, allowed={"python", "fortran"}, label="--systems")
    timing_kinds = _normalize_timing_kinds(args)
    torch_dtypes = _split_csv(
        args.torch_dtypes,
        allowed={"float64", "float32"},
        label="--torch-dtypes",
    )
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.fo_flux_n_mu <= 0:
        raise ValueError("--fo-flux-n-mu must be positive")
    if "level-fluxes" in timing_kinds:
        args.output_levels = True
        args.output_fluxes = True

    if has_torch():
        torch = _torch_module()
        if torch is not None:
            torch.set_num_threads(args.torch_threads)

    created_utc = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        _manifest_row(created_utc, "platform", "python", sys.version.replace("\n", " ")),
        _manifest_row(created_utc, "platform", "machine", platform.platform()),
        _manifest_row(created_utc, "repo", "root", str(ROOT)),
        _manifest_row(created_utc, "metadata", "torch_threads", str(args.torch_threads)),
        _manifest_row(created_utc, "metadata", "torch_compile", str(args.torch_compile)),
        _manifest_row(created_utc, "metadata", "torch_compile_mode", args.torch_compile_mode),
        _manifest_row(created_utc, "metadata", "timing_kinds", ",".join(timing_kinds)),
        _manifest_row(created_utc, "metadata", "output_levels", str(args.output_levels)),
        _manifest_row(created_utc, "metadata", "output_fluxes", str(args.output_fluxes)),
        _manifest_row(created_utc, "metadata", "fo_flux_n_mu", str(args.fo_flux_n_mu)),
    ]
    raw_rows: list[dict[str, Any]] = []

    specs_by_key = _case_specs(args.input_root)
    selected_specs = [specs_by_key[key] for key in cases]
    if "python" in systems:
        _require_paths(selected_specs)
        configs, backend_manifest = _backend_configs(
            args.backend_set,
            torch_dtypes,
            torch_compile=args.torch_compile,
            torch_compile_mode=args.torch_compile_mode,
            created_utc=created_utc,
        )
        manifest.extend(backend_manifest)
        for spec in selected_specs:
            scene, inputs, load_seconds, prep_timings = _load_scene_and_inputs(spec)
            prep_total = sum(prep_timings.values())
            manifest.append(
                _manifest_row(
                    created_utc,
                    "python-case",
                    spec.key,
                    f"load_s={load_seconds:.6g};preprocess_s={prep_total:.6g};scene={spec.scene}",
                )
            )
            for config in configs:
                if "components" in timing_kinds:
                    base = {
                        "system": "py2sess",
                        "case": spec.case,
                        "mode": spec.mode,
                        "backend": config.label,
                        "device": config.device,
                        "dtype": config.dtype,
                        "timing_kind": "components",
                    }
                    if config.backend == "numpy":

                        def run_once(scene=scene, inputs=inputs):
                            return _run_numpy_components_once(
                                scene,
                                inputs,
                                numpy_bvp_engine=args.numpy_bvp_engine,
                            )
                    elif config.backend == "native":

                        def run_once(scene=scene, inputs=inputs, config=config):
                            return _run_native_components_once(scene, inputs, config)
                    else:

                        def run_once(scene=scene, inputs=inputs, config=config):
                            return _run_torch_components_once(
                                scene,
                                inputs,
                                config,
                                torch_bvp_engine=args.torch_bvp_engine,
                            )

                    raw_rows.extend(
                        _run_repeats(
                            created_utc=created_utc,
                            spec=spec,
                            run_once=run_once,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            base_row=base,
                            config=config,
                        )
                    )
                if "scene-forward" in timing_kinds or "level-fluxes" in timing_kinds:
                    timing_kind = (
                        "level_fluxes" if "level-fluxes" in timing_kinds else "scene_forward"
                    )
                    base = {
                        "system": "py2sess",
                        "case": spec.case,
                        "mode": spec.mode,
                        "backend": config.label,
                        "device": config.device,
                        "dtype": config.dtype,
                        "timing_kind": timing_kind,
                    }

                    def run_once(scene=scene, inputs=inputs, config=config):
                        return _run_scene_forward_once(
                            scene,
                            inputs,
                            config,
                            torch_threads=args.torch_threads,
                            torch_bvp_engine=args.torch_bvp_engine,
                            numpy_bvp_engine=args.numpy_bvp_engine,
                            output_levels=args.output_levels,
                            output_fluxes=args.output_fluxes,
                            fo_flux_n_mu=args.fo_flux_n_mu,
                        )

                    raw_rows.extend(
                        _run_repeats(
                            created_utc=created_utc,
                            spec=spec,
                            run_once=run_once,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            base_row=base,
                            config=config,
                        )
                    )

    if "fortran" in systems:
        bundle_dir = args.fortran_bundle_dir or (args.output_dir / "fortran_bundle")
        bundle_dir = _prepare_fortran_bundle(args.fortran_root, bundle_dir)
        manifest.append(_manifest_row(created_utc, "fortran", "bundle_dir", str(bundle_dir)))
        for key in cases:
            fortran_case = _fortran_cases()[key]
            raw_rows.extend(
                _run_fortran_repeats(
                    created_utc=created_utc,
                    case=fortran_case,
                    bundle_dir=bundle_dir,
                    output_dir=args.output_dir,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
            )

    summary_rows = _summarize(raw_rows, created_utc)
    raw_path = args.output_dir / "raw_full_spectrum_timings.csv"
    summary_path = args.output_dir / "summary_full_spectrum.csv"
    manifest_path = args.output_dir / "manifest_full_spectrum.csv"
    _write_csv(raw_path, RAW_FIELDS, raw_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(manifest_path, MANIFEST_FIELDS, manifest)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    _print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
