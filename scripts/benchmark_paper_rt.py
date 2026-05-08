#!/usr/bin/env python3
"""Run reproducible py2sess RT benchmarks for paper figures."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from py2sess import TwoStreamEss, TwoStreamEssOptions  # noqa: E402
from py2sess.optical.planck import thermal_source_from_temperature_profile  # noqa: E402
from py2sess.rtsolver.backend import has_torch, to_numpy  # noqa: E402
from py2sess.rtsolver.fo_solar_obs import fo_scatter_term_henyey_greenstein  # noqa: E402
from py2sess.scene import load_scene  # noqa: E402


DEFAULT_LAYER_COUNTS = (5, 10, 20, 50, 100, 114, 200)
DEFAULT_WAVELENGTH_COUNTS = (300, 1000, 3000, 10000, 30000, 100000, 300000)
DEFAULT_JACOBIAN_WAVELENGTH_COUNTS = (300, 1000, 3000, 10000)
DEFAULT_GRAD_LAYER_COUNTS = (1, 2, 5, 10, 20, 50, 114)
DEFAULT_BASE_LAYERS = 114
DEFAULT_BASE_WAVELENGTHS = 50000
SMOKE_LAYER_COUNTS = (2, 3)
SMOKE_WAVELENGTH_COUNTS = (2, 3)
SMOKE_GRAD_LAYER_COUNTS = (1, 2)
UV_TAU_PER_LAYER = 0.01
UV_OMEGA = 0.2
UV_G = 0.1
TIR_TAU_PER_LAYER = 0.01
TIR_OMEGA = 0.05
TIR_G = 0.1
RAW_FIELDS = (
    "experiment",
    "case",
    "mode",
    "backend",
    "device",
    "dtype",
    "sweep_axis",
    "gradient_target",
    "wavelengths",
    "layers",
    "levels",
    "active_tau_layers",
    "n_grad_vars",
    "repeat_index",
    "seconds",
    "forward_seconds",
    "backward_seconds",
    "cuda_peak_bytes",
    "checksum",
    "grad_checksum",
    "grad_l2",
    "max_abs_diff",
    "max_rel_diff_pct",
    "status",
    "skip_reason",
)
SUMMARY_FIELDS = (
    "experiment",
    "case",
    "mode",
    "backend",
    "device",
    "dtype",
    "sweep_axis",
    "gradient_target",
    "wavelengths",
    "layers",
    "levels",
    "active_tau_layers",
    "n_grad_vars",
    "n_repeats",
    "best_s",
    "mean_s",
    "median_s",
    "std_s",
    "min_s",
    "max_s",
    "rows_per_second",
    "best_speedup_vs_numpy",
    "best_speedup_vs_torch_cpu",
    "forward_mean_s",
    "backward_mean_s",
    "backward_fraction",
    "cuda_peak_bytes_max",
    "checksum",
    "grad_checksum",
    "grad_l2",
    "max_abs_diff",
    "max_rel_diff_pct",
    "status",
)
MANIFEST_FIELDS = (
    "created_utc",
    "kind",
    "experiment",
    "backend",
    "device",
    "dtype",
    "status",
    "reason",
    "value",
)


@dataclass(frozen=True)
class BackendConfig:
    """Execution backend selected for one benchmark row."""

    backend: str
    label: str
    device: str
    dtype: str


@dataclass(frozen=True)
class RtCase:
    """Prepared direct RT inputs for one benchmark case."""

    case: str
    mode: str
    kwargs: dict[str, Any]
    wavelengths: int
    layers: int
    reference_total: np.ndarray | None = None


def _torch_module():
    if not has_torch():
        return None
    import torch

    return torch


def _parse_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("counts must be a comma-separated list of positive integers")
    return parsed


def _parse_jacobian_targets(value: str) -> tuple[str, ...]:
    aliases = {
        "ssa": "omega",
        "albedo": "surface_albedo",
    }
    allowed = {"tau", "omega", "surface_albedo"}
    parsed = tuple(
        aliases.get(part.strip(), part.strip()) for part in value.split(",") if part.strip()
    )
    if not parsed or any(target not in allowed for target in parsed):
        raise ValueError("--jacobian-targets must contain tau, omega, and/or surface_albedo")
    return tuple(dict.fromkeys(parsed))


def _parse_dtypes(value: str) -> tuple[str, ...]:
    dtypes = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"float64"}
    if not dtypes or any(dtype not in allowed for dtype in dtypes):
        raise ValueError("--torch-dtypes must be float64 for this benchmark workflow")
    return dtypes


def _active_layer_indices(layers: int, active_tau_layers: int) -> np.ndarray:
    if layers <= 0:
        raise ValueError("layers must be positive")
    if active_tau_layers <= 0:
        raise ValueError("active_tau_layers must be positive")
    if active_tau_layers > layers:
        raise ValueError("active_tau_layers cannot exceed layers")
    if active_tau_layers == layers:
        return np.arange(layers, dtype=int)
    indices = np.floor((np.arange(active_tau_layers) + 0.5) * layers / active_tau_layers).astype(
        int
    )
    return np.clip(indices, 0, layers - 1)


def _standard_heights_and_temperature(layers: int) -> tuple[np.ndarray, np.ndarray]:
    bottom_to_top = np.linspace(0.0, 50.0, layers + 1)
    temp = np.where(
        bottom_to_top <= 11.0,
        288.15 - 6.5 * bottom_to_top,
        np.where(bottom_to_top <= 20.0, 216.65, 216.65 + 1.0 * (bottom_to_top - 20.0)),
    )
    temp = np.minimum(temp, 270.0)
    return bottom_to_top[::-1].copy(), temp[::-1].copy()


def build_synthetic_uv_case(wavelengths: int, layers: int) -> RtCase:
    """Build deterministic solar direct RT inputs without opacity files."""
    if wavelengths <= 0 or layers <= 0:
        raise ValueError("wavelengths and layers must be positive")
    z, _ = _standard_heights_and_temperature(layers)
    tau = np.full((wavelengths, layers), UV_TAU_PER_LAYER, dtype=float)
    ssa = np.full_like(tau, UV_OMEGA)
    g = np.full_like(tau, UV_G)
    scaling = np.zeros_like(tau)
    angles = np.array([47.70200090144217, 49.514425392048906, 275.7465175402976])
    fo_scatter = fo_scatter_term_henyey_greenstein(
        ssa=ssa,
        g=g,
        angles=angles,
        delta_m_truncation_factor=scaling,
        n_moments=5000,
    )
    kwargs = {
        "tau": tau,
        "ssa": ssa,
        "g": g,
        "z": z,
        "angles": angles,
        "albedo": np.full(wavelengths, 0.05, dtype=float),
        "fbeam": np.ones(wavelengths, dtype=float),
        "delta_m_truncation_factor": scaling,
        "fo_scatter_term": fo_scatter,
    }
    return RtCase("UV", "solar", kwargs, wavelengths, layers)


def build_synthetic_tir_case(wavelengths: int, layers: int) -> RtCase:
    """Build deterministic thermal direct RT inputs without opacity files."""
    if wavelengths <= 0 or layers <= 0:
        raise ValueError("wavelengths and layers must be positive")
    z, temperature = _standard_heights_and_temperature(layers)
    tau = np.full((wavelengths, layers), TIR_TAU_PER_LAYER, dtype=float)
    ssa = np.full_like(tau, TIR_OMEGA)
    g = np.full_like(tau, TIR_G)
    scaling = np.zeros_like(tau)
    wavenumber = np.linspace(700.0, 1300.0, wavelengths)
    thermal = thermal_source_from_temperature_profile(
        temperature,
        np.array([288.15]),
        wavenumber_cm_inv=wavenumber,
    )
    kwargs = {
        "tau": tau,
        "ssa": ssa,
        "g": g,
        "z": z,
        "angles": 49.514425392048906,
        "albedo": np.full(wavelengths, 0.02, dtype=float),
        "emissivity": np.full(wavelengths, 0.98, dtype=float),
        "delta_m_truncation_factor": scaling,
        "planck": np.asarray(thermal.planck, dtype=float),
        "surface_planck": np.asarray(thermal.surface_planck, dtype=float).reshape(wavelengths),
        "stream": 0.5,
    }
    return RtCase("TIR", "thermal", kwargs, wavelengths, layers)


def _load_fortran_forward_cases(limit: int | None) -> list[RtCase]:
    specs = (
        (
            "UV",
            "solar",
            ROOT / "benchmarks" / "uv_profile1" / "profile.csv",
            ROOT / "benchmarks" / "uv_profile1" / "scene.yaml",
        ),
        (
            "TIR",
            "thermal",
            ROOT / "benchmarks" / "tir_profile1" / "profile.csv",
            ROOT / "benchmarks" / "tir_profile1" / "scene.yaml",
        ),
    )
    cases: list[RtCase] = []
    for case_name, mode, profile, scene_path in specs:
        scene = load_scene(
            profile=profile,
            config=scene_path,
            spectral_limit=limit,
            strict_runtime_inputs=True,
        )
        inputs = scene.to_forward_inputs()
        tau = np.asarray(inputs.kwargs["tau"])
        cases.append(
            RtCase(
                case=case_name,
                mode=mode,
                kwargs=dict(inputs.kwargs),
                wavelengths=int(tau.shape[0]),
                layers=int(tau.shape[-1]),
                reference_total=inputs.reference_total,
            )
        )
    return cases


def _float_fields_to_torch(
    kwargs: dict[str, Any],
    *,
    dtype: str,
    device: str,
    requires_grad: tuple[str, ...] = (),
    omit: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch_module()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    dtype_obj = {"float64": torch.float64, "float32": torch.float32}[dtype]
    device_obj = torch.device(device)
    tensor_keys = {
        "tau",
        "ssa",
        "g",
        "z",
        "albedo",
        "fbeam",
        "delta_m_truncation_factor",
        "fo_scatter_term",
        "planck",
        "surface_planck",
        "emissivity",
    }
    out: dict[str, Any] = {}
    tracked: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in omit:
            continue
        if key in tensor_keys:
            tensor = torch.as_tensor(value, dtype=dtype_obj, device=device_obj)
            if key in requires_grad:
                tensor = tensor.detach().clone().requires_grad_(True)
                tracked[key] = tensor
            out[key] = tensor
        else:
            out[key] = value
    return out, tracked


def _backend_configs(
    *,
    backend_set: str,
    torch_dtypes: tuple[str, ...],
    include_numpy: bool,
    include_torch: bool,
    created_utc: str,
) -> tuple[list[BackendConfig], list[dict[str, str]]]:
    configs: list[BackendConfig] = []
    manifest: list[dict[str, str]] = []
    if include_numpy and backend_set in {"all", "cpu", "numpy"}:
        configs.append(BackendConfig("numpy", "NumPy", "", "float64"))
    if not include_torch or backend_set == "numpy":
        return configs, manifest

    torch = _torch_module()
    if torch is None:
        for device in ("cpu", "cuda"):
            for dtype in torch_dtypes:
                manifest.append(
                    _manifest_row(
                        created_utc,
                        kind="backend",
                        backend="Torch",
                        device=device,
                        dtype=dtype,
                        status="skipped",
                        reason="PyTorch is not installed",
                    )
                )
        return configs, manifest

    if backend_set in {"all", "cpu"}:
        for dtype in torch_dtypes:
            configs.append(BackendConfig("torch", "Torch CPU", "cpu", dtype))

    if backend_set in {"all", "cuda"}:
        if torch.cuda.is_available():
            for dtype in torch_dtypes:
                configs.append(BackendConfig("torch", "Torch CUDA", "cuda", dtype))
        else:
            for dtype in torch_dtypes:
                manifest.append(
                    _manifest_row(
                        created_utc,
                        kind="backend",
                        backend="Torch CUDA",
                        device="cuda",
                        dtype=dtype,
                        status="skipped",
                        reason="torch.cuda.is_available() is false",
                    )
                )
    return configs, manifest


def _manifest_row(
    created_utc: str,
    *,
    kind: str,
    experiment: str = "",
    backend: str = "",
    device: str = "",
    dtype: str = "",
    status: str = "ok",
    reason: str = "",
    value: str = "",
) -> dict[str, str]:
    return {
        "created_utc": created_utc,
        "kind": kind,
        "experiment": experiment,
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "status": status,
        "reason": reason,
        "value": value,
    }


def _accuracy_summary(
    value: np.ndarray, reference: np.ndarray | None
) -> tuple[float | None, float | None]:
    if reference is None:
        return None, None
    diff = value - reference
    max_abs = float(np.max(np.abs(diff)))
    scale = np.maximum(np.abs(reference), 1.0e-15)
    max_rel_pct = float(np.max(np.abs(diff) / scale) * 100.0)
    return max_abs, max_rel_pct


def _extract_radiance(result: Any) -> np.ndarray:
    return np.asarray(to_numpy(result.radiance_total), dtype=float)


def _checksum(array: np.ndarray) -> float:
    total = float(np.sum(array))
    if not math.isfinite(total):
        raise RuntimeError("benchmark checksum is not finite")
    return total


def _sync_if_cuda(config: BackendConfig) -> None:
    if config.backend == "torch" and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.synchronize()


def _reset_peak_memory(config: BackendConfig) -> None:
    if config.backend == "torch" and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(torch.device(config.device))


def _peak_memory(config: BackendConfig) -> int | None:
    if config.backend == "torch" and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            return int(torch.cuda.max_memory_allocated(torch.device(config.device)))
    return None


def _time_repeats(
    run: Callable[[], Any],
    metrics: Callable[[Any], dict[str, float | None]],
    *,
    config: BackendConfig,
    warmups: int,
    repeats: int,
) -> list[dict[str, float | int | None]]:
    for _ in range(warmups):
        run()
        _sync_if_cuda(config)

    rows: list[dict[str, float | int | None]] = []
    for repeat in range(repeats):
        _reset_peak_memory(config)
        _sync_if_cuda(config)
        start = time.perf_counter()
        result = run()
        _sync_if_cuda(config)
        seconds = time.perf_counter() - start
        row: dict[str, float | int | None] = {
            "repeat_index": repeat,
            "seconds": seconds,
            "cuda_peak_bytes": _peak_memory(config),
        }
        row.update(metrics(result))
        rows.append(row)
    return rows


def _make_solver(case: RtCase, config: BackendConfig, *, enable_grad: bool) -> TwoStreamEss:
    return TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=case.layers,
            mode=case.mode,
            backend=config.backend,
            torch_device=config.device or None,
            torch_dtype=config.dtype if config.backend == "torch" else None,
            torch_enable_grad=enable_grad,
        )
    )


def _forward_runtime_rows(
    *,
    experiment: str,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    if config.backend == "torch":
        kwargs, _ = _float_fields_to_torch(case.kwargs, dtype=config.dtype, device=config.device)
    else:
        kwargs = case.kwargs
    solver = _make_solver(case, config, enable_grad=False)

    def run():
        return solver.forward(**kwargs, include_fo=True)

    def metrics(result):
        radiance = _extract_radiance(result)
        max_abs, max_rel = _accuracy_summary(radiance, case.reference_total)
        return {
            "checksum": _checksum(radiance),
            "grad_checksum": None,
            "grad_l2": None,
            "max_abs_diff": max_abs,
            "max_rel_diff_pct": max_rel,
        }

    try:
        timings = _time_repeats(
            run,
            metrics,
            config=config,
            warmups=warmups,
            repeats=repeats,
        )
    except Exception as exc:
        return [
            _failure_row(
                experiment=experiment,
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                active_tau_layers=0,
                n_grad_vars=0,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]
    return [
        _raw_row(
            experiment=experiment,
            case=case,
            config=config,
            sweep_axis=sweep_axis,
            active_tau_layers=0,
            n_grad_vars=0,
            timing=timing,
        )
        for timing in timings
    ]


def _profile_jacobian_runtime_rows(
    *,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    gradient_field: str,
    gradient_target: str,
    active_tau_layers: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    if config.backend != "torch":
        return []
    active_indices = _active_layer_indices(case.layers, active_tau_layers)
    kwargs, _ = _float_fields_to_torch(
        case.kwargs,
        dtype=config.dtype,
        device=config.device,
    )
    torch = _torch_module()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if gradient_field not in kwargs:
        raise RuntimeError(f"{gradient_field!r} is not available for Jacobian timing")
    field_base = kwargs[gradient_field].detach()
    active_index_t = torch.as_tensor(active_indices, dtype=torch.long, device=config.device)
    active_field = field_base[:, active_index_t].detach().clone().requires_grad_(True)
    solver = _make_solver(case, config, enable_grad=True)

    def make_field():
        if active_tau_layers == case.layers:
            return active_field
        field = field_base.clone()
        field[:, active_index_t] = active_field
        return field

    def run_once(*, measure: bool) -> dict[str, float | int | None]:
        active_field.grad = None
        kwargs[gradient_field] = make_field()
        if measure:
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            total_start = time.perf_counter()
            forward_start = total_start
        result = solver.forward(**kwargs, include_fo=True)
        if measure:
            _sync_if_cuda(config)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
        loss = result.radiance_total.sum()
        loss.backward()
        if measure:
            _sync_if_cuda(config)
            backward_seconds = time.perf_counter() - backward_start
            seconds = time.perf_counter() - total_start
        else:
            _sync_if_cuda(config)
            forward_seconds = None
            backward_seconds = None
            seconds = None
        radiance = _extract_radiance(result)
        if active_field.grad is None:
            raise RuntimeError(f"expected {gradient_target} gradient was not populated")
        grad = active_field.grad.detach()
        if not bool(grad.isfinite().all().item()):
            raise RuntimeError(f"{gradient_target} Jacobian gradient contains non-finite values")
        grad_checksum = float(grad.sum().detach().cpu())
        grad_l2 = math.sqrt(float((grad * grad).sum().detach().cpu()))
        return {
            "seconds": seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "cuda_peak_bytes": _peak_memory(config) if measure else None,
            "checksum": _checksum(radiance),
            "grad_checksum": grad_checksum,
            "grad_l2": grad_l2,
            "max_abs_diff": None,
            "max_rel_diff_pct": None,
        }

    n_grad_vars = case.wavelengths * active_tau_layers
    try:
        for _ in range(warmups):
            run_once(measure=False)
        timings = []
        for repeat in range(repeats):
            timing = run_once(measure=True)
            timing["repeat_index"] = repeat
            timings.append(timing)
    except Exception as exc:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target=gradient_target,
                active_tau_layers=active_tau_layers,
                n_grad_vars=n_grad_vars,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]
    return [
        _raw_row(
            experiment="synthetic-jacobian",
            case=case,
            config=config,
            sweep_axis=sweep_axis,
            gradient_target=gradient_target,
            active_tau_layers=active_tau_layers,
            n_grad_vars=n_grad_vars,
            timing=timing,
        )
        for timing in timings
    ]


def _jacobian_runtime_rows(
    *,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    active_tau_layers: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    return _profile_jacobian_runtime_rows(
        case=case,
        config=config,
        sweep_axis=sweep_axis,
        gradient_field="tau",
        gradient_target="tau",
        active_tau_layers=active_tau_layers,
        warmups=warmups,
        repeats=repeats,
    )


def _omega_jacobian_runtime_rows(
    *,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    active_tau_layers: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    return _profile_jacobian_runtime_rows(
        case=case,
        config=config,
        sweep_axis=sweep_axis,
        gradient_field="ssa",
        gradient_target="omega",
        active_tau_layers=active_tau_layers,
        warmups=warmups,
        repeats=repeats,
    )


def _surface_albedo_jacobian_runtime_rows(
    *,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    if config.backend != "torch":
        return []
    kwargs, tracked = _float_fields_to_torch(
        case.kwargs,
        dtype=config.dtype,
        device=config.device,
        requires_grad=("albedo",),
    )
    torch = _torch_module()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    albedo = tracked["albedo"]
    solver = _make_solver(case, config, enable_grad=True)

    def run_once(*, measure: bool) -> dict[str, float | int | None]:
        albedo.grad = None
        if measure:
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            total_start = time.perf_counter()
            forward_start = total_start
        result = solver.forward(**kwargs, include_fo=True)
        if measure:
            _sync_if_cuda(config)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
        loss = result.radiance_total.sum()
        loss.backward()
        if measure:
            _sync_if_cuda(config)
            backward_seconds = time.perf_counter() - backward_start
            seconds = time.perf_counter() - total_start
        else:
            _sync_if_cuda(config)
            forward_seconds = None
            backward_seconds = None
            seconds = None
        radiance = _extract_radiance(result)
        if albedo.grad is None:
            raise RuntimeError("expected surface albedo gradient was not populated")
        grad = albedo.grad.detach()
        if not bool(grad.isfinite().all().item()):
            raise RuntimeError("surface albedo gradient contains non-finite values")
        grad_checksum = float(grad.sum().detach().cpu())
        grad_l2 = math.sqrt(float((grad * grad).sum().detach().cpu()))
        return {
            "seconds": seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "cuda_peak_bytes": _peak_memory(config) if measure else None,
            "checksum": _checksum(radiance),
            "grad_checksum": grad_checksum,
            "grad_l2": grad_l2,
            "max_abs_diff": None,
            "max_rel_diff_pct": None,
        }

    n_grad_vars = case.wavelengths
    try:
        for _ in range(warmups):
            run_once(measure=False)
        timings = []
        for repeat in range(repeats):
            timing = run_once(measure=True)
            timing["repeat_index"] = repeat
            timings.append(timing)
    except Exception as exc:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target="surface_albedo",
                active_tau_layers=0,
                n_grad_vars=n_grad_vars,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]
    return [
        _raw_row(
            experiment="synthetic-jacobian",
            case=case,
            config=config,
            sweep_axis=sweep_axis,
            gradient_target="surface_albedo",
            active_tau_layers=0,
            n_grad_vars=n_grad_vars,
            timing=timing,
        )
        for timing in timings
    ]


def _raw_row(
    *,
    experiment: str,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    active_tau_layers: int,
    n_grad_vars: int,
    timing: dict[str, float | int | None],
    gradient_target: str = "",
) -> dict[str, str]:
    row = {
        "experiment": experiment,
        "case": case.case,
        "mode": case.mode,
        "backend": config.label,
        "device": config.device,
        "dtype": config.dtype,
        "sweep_axis": sweep_axis,
        "gradient_target": gradient_target,
        "wavelengths": case.wavelengths,
        "layers": case.layers,
        "levels": case.layers + 1,
        "active_tau_layers": active_tau_layers,
        "n_grad_vars": n_grad_vars,
        "status": "ok",
        "skip_reason": "",
    }
    row.update(timing)
    return {field: _csv_value(row.get(field)) for field in RAW_FIELDS}


def _failure_row(
    *,
    experiment: str,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    active_tau_layers: int,
    n_grad_vars: int,
    reason: str,
    gradient_target: str = "",
) -> dict[str, str]:
    row = {
        "experiment": experiment,
        "case": case.case,
        "mode": case.mode,
        "backend": config.label,
        "device": config.device,
        "dtype": config.dtype,
        "sweep_axis": sweep_axis,
        "gradient_target": gradient_target,
        "wavelengths": case.wavelengths,
        "layers": case.layers,
        "levels": case.layers + 1,
        "active_tau_layers": active_tau_layers,
        "n_grad_vars": n_grad_vars,
        "repeat_index": "",
        "seconds": "",
        "forward_seconds": "",
        "backward_seconds": "",
        "cuda_peak_bytes": "",
        "checksum": "",
        "grad_checksum": "",
        "grad_l2": "",
        "max_abs_diff": "",
        "max_rel_diff_pct": "",
        "status": "failed",
        "skip_reason": reason,
    }
    return {field: _csv_value(row.get(field)) for field in RAW_FIELDS}


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _benchmark_specs(
    *,
    layer_counts: tuple[int, ...],
    wavelength_counts: tuple[int, ...],
    base_layers: int,
    base_wavelengths: int,
    full_grid: bool,
) -> list[tuple[int, int, str]]:
    if full_grid:
        return [
            (wavelengths, layers, "full_grid")
            for layers in layer_counts
            for wavelengths in wavelength_counts
        ]
    specs = [(wavelengths, base_layers, "wavelengths") for wavelengths in wavelength_counts]
    specs.extend((base_wavelengths, layers, "layers") for layers in layer_counts)
    return specs


def _jacobian_specs(
    *,
    layer_counts: tuple[int, ...],
    wavelength_counts: tuple[int, ...],
    grad_layer_counts: tuple[int, ...],
    base_layers: int,
    base_wavelengths: int,
    full_grid: bool,
) -> list[tuple[int, int, int, str, str]]:
    specs: list[tuple[int, int, int, str, str]] = []
    if full_grid:
        specs.extend(
            (wavelengths, layers, layers, "full_grid", "tau")
            for layers in layer_counts
            for wavelengths in wavelength_counts
        )
    else:
        specs.extend(
            (wavelengths, base_layers, base_layers, "wavelengths", "tau")
            for wavelengths in wavelength_counts
        )
        specs.extend((base_wavelengths, layers, layers, "layers", "tau") for layers in layer_counts)
    specs.extend(
        (base_wavelengths, base_layers, count, "grad_vars", "tau")
        for count in grad_layer_counts
        if count <= base_layers
    )
    omega_counts = tuple(dict.fromkeys(count for count in (1, base_layers) if count <= base_layers))
    specs.extend(
        (base_wavelengths, base_layers, count, "omega_grad_vars", "omega") for count in omega_counts
    )
    specs.append((base_wavelengths, base_layers, 0, "surface_albedo", "surface_albedo"))
    return specs


def _synthetic_case_builders() -> tuple[Callable[[int, int], RtCase], ...]:
    return (build_synthetic_uv_case, build_synthetic_tir_case)


def _summarize(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    key_fields = (
        "experiment",
        "case",
        "mode",
        "backend",
        "device",
        "dtype",
        "sweep_axis",
        "gradient_target",
        "wavelengths",
        "layers",
        "levels",
        "active_tau_layers",
        "n_grad_vars",
    )
    for row in raw_rows:
        if row["status"] != "ok":
            continue
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, str]] = []
    for key, rows in groups.items():
        seconds = [float(row["seconds"]) for row in rows]
        best = min(seconds)
        n_repeats = len(seconds)
        summary = dict(zip(key_fields, key, strict=True))
        summary.update(
            {
                "n_repeats": n_repeats,
                "best_s": best,
                "mean_s": statistics.fmean(seconds),
                "median_s": statistics.median(seconds),
                "std_s": statistics.stdev(seconds) if n_repeats > 1 else 0.0,
                "min_s": min(seconds),
                "max_s": max(seconds),
                "rows_per_second": float(summary["wavelengths"]) / best if best > 0.0 else 0.0,
                "best_speedup_vs_numpy": "",
                "cuda_peak_bytes_max": _max_optional_int(row["cuda_peak_bytes"] for row in rows),
                "checksum": rows[-1]["checksum"],
                "grad_checksum": rows[-1]["grad_checksum"],
                "grad_l2": rows[-1]["grad_l2"],
                "max_abs_diff": rows[-1]["max_abs_diff"],
                "max_rel_diff_pct": rows[-1]["max_rel_diff_pct"],
                "status": "ok",
            }
        )
        forward_seconds = _optional_floats(row["forward_seconds"] for row in rows)
        backward_seconds = _optional_floats(row["backward_seconds"] for row in rows)
        if forward_seconds:
            summary["forward_mean_s"] = statistics.fmean(forward_seconds)
        if backward_seconds:
            backward_mean = statistics.fmean(backward_seconds)
            summary["backward_mean_s"] = backward_mean
            if float(summary["mean_s"]) > 0.0:
                summary["backward_fraction"] = backward_mean / float(summary["mean_s"])
        summaries.append({field: _csv_value(summary.get(field)) for field in SUMMARY_FIELDS})

    numpy_best = {
        (
            row["experiment"],
            row["case"],
            row["mode"],
            row["sweep_axis"],
            row["gradient_target"],
            row["wavelengths"],
            row["layers"],
            row["levels"],
            row["active_tau_layers"],
        ): float(row["best_s"])
        for row in summaries
        if row["backend"] == "NumPy"
    }
    torch_cpu_best = {
        (
            row["experiment"],
            row["case"],
            row["mode"],
            row["sweep_axis"],
            row["gradient_target"],
            row["wavelengths"],
            row["layers"],
            row["levels"],
            row["active_tau_layers"],
        ): float(row["best_s"])
        for row in summaries
        if row["backend"] == "Torch CPU"
    }
    for row in summaries:
        key = (
            row["experiment"],
            row["case"],
            row["mode"],
            row["sweep_axis"],
            row["gradient_target"],
            row["wavelengths"],
            row["layers"],
            row["levels"],
            row["active_tau_layers"],
        )
        baseline = numpy_best.get(key)
        if baseline is not None and float(row["best_s"]) > 0.0:
            row["best_speedup_vs_numpy"] = _csv_value(baseline / float(row["best_s"]))
        torch_cpu_baseline = torch_cpu_best.get(key)
        if torch_cpu_baseline is not None and float(row["best_s"]) > 0.0:
            row["best_speedup_vs_torch_cpu"] = _csv_value(torch_cpu_baseline / float(row["best_s"]))

    return sorted(
        summaries,
        key=lambda row: (
            row["experiment"],
            row["case"],
            row["sweep_axis"],
            row["gradient_target"],
            int(row["layers"]),
            int(row["wavelengths"]),
            int(row["active_tau_layers"]),
            row["backend"],
            row["dtype"],
        ),
    )


def _optional_floats(values) -> list[float]:
    return [float(value) for value in values if str(value) != ""]


def _max_optional_int(values) -> int | None:
    parsed = [int(value) for value in values if str(value) != ""]
    return max(parsed) if parsed else None


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_rows(created_utc: str, args: argparse.Namespace) -> list[dict[str, str]]:
    torch = _torch_module()
    rows = [
        _manifest_row(created_utc, kind="metadata", value=f"python={platform.python_version()}"),
        _manifest_row(created_utc, kind="metadata", value=f"platform={platform.platform()}"),
        _manifest_row(created_utc, kind="metadata", value=f"preset={args.preset}"),
        _manifest_row(created_utc, kind="metadata", value=f"warmups={args.warmups}"),
        _manifest_row(created_utc, kind="metadata", value=f"repeats={args.repeats}"),
    ]
    if torch is None:
        rows.append(_manifest_row(created_utc, kind="metadata", value="torch=unavailable"))
    else:
        rows.append(_manifest_row(created_utc, kind="metadata", value=f"torch={torch.__version__}"))
        rows.append(
            _manifest_row(
                created_utc,
                kind="metadata",
                value=f"torch_cuda_available={torch.cuda.is_available()}",
            )
        )
        rows.append(
            _manifest_row(
                created_utc,
                kind="metadata",
                value=f"torch_cuda_version={getattr(torch.version, 'cuda', None)}",
            )
        )
    return rows


def _progress(args: argparse.Namespace, message: str) -> None:
    if not args.no_progress:
        print(message, file=sys.stderr, flush=True)


def _completed_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no rows"
    ok = sum(row["status"] == "ok" for row in rows)
    return f"{ok}/{len(rows)} ok"


def run_benchmarks(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir = args.output_dir
    torch_dtypes = _parse_dtypes(args.torch_dtypes)
    jacobian_targets = _parse_jacobian_targets(args.jacobian_targets)
    layer_counts = _parse_ints(args.layer_counts, DEFAULT_LAYER_COUNTS)
    wavelength_counts = _parse_ints(args.wavelength_counts, DEFAULT_WAVELENGTH_COUNTS)
    jacobian_wavelength_counts = _parse_ints(
        args.jacobian_wavelength_counts,
        DEFAULT_JACOBIAN_WAVELENGTH_COUNTS,
    )
    grad_layer_counts = _parse_ints(args.grad_layer_counts, DEFAULT_GRAD_LAYER_COUNTS)
    base_layers = args.base_layers
    base_wavelengths = args.base_wavelengths
    fortran_limit = args.fortran_limit
    if args.preset == "smoke":
        layer_counts = _parse_ints(args.layer_counts, SMOKE_LAYER_COUNTS)
        wavelength_counts = _parse_ints(args.wavelength_counts, SMOKE_WAVELENGTH_COUNTS)
        jacobian_wavelength_counts = _parse_ints(
            args.jacobian_wavelength_counts,
            SMOKE_WAVELENGTH_COUNTS,
        )
        grad_layer_counts = _parse_ints(args.grad_layer_counts, SMOKE_GRAD_LAYER_COUNTS)
        base_layers = args.base_layers or SMOKE_LAYER_COUNTS[0]
        base_wavelengths = args.base_wavelengths or SMOKE_WAVELENGTH_COUNTS[-1]
        fortran_limit = args.fortran_limit if args.fortran_limit is not None else 2
    else:
        base_layers = base_layers or DEFAULT_BASE_LAYERS
        base_wavelengths = base_wavelengths or DEFAULT_BASE_WAVELENGTHS

    if has_torch():
        torch = _torch_module()
        torch.set_num_threads(args.torch_threads)

    groups = set(args.groups)
    raw_rows: list[dict[str, str]] = []
    manifest_rows = _metadata_rows(created_utc, args)

    forward_configs, manifest = _backend_configs(
        backend_set=args.backend_set,
        torch_dtypes=torch_dtypes,
        include_numpy=True,
        include_torch=True,
        created_utc=created_utc,
    )
    manifest_rows.extend(manifest)

    if "fortran-forward" in groups:
        for case in _load_fortran_forward_cases(fortran_limit):
            for config in forward_configs:
                label = (
                    "fortran-forward "
                    f"case={case.case} wavelengths={case.wavelengths} layers={case.layers} "
                    f"backend={config.label} {config.device}"
                )
                _progress(args, f"start {label}")
                step_start = time.perf_counter()
                rows = _forward_runtime_rows(
                    experiment="fortran-forward",
                    case=case,
                    config=config,
                    sweep_axis="reference",
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                raw_rows.extend(rows)
                _progress(
                    args,
                    f"done {label} {_completed_status(rows)} "
                    f"wall={time.perf_counter() - step_start:.1f}s",
                )

    specs = _benchmark_specs(
        layer_counts=layer_counts,
        wavelength_counts=wavelength_counts,
        base_layers=base_layers,
        base_wavelengths=base_wavelengths,
        full_grid=args.full_grid,
    )

    if "synthetic-forward" in groups:
        for wavelengths, layers, sweep_axis in specs:
            for builder in _synthetic_case_builders():
                case = builder(wavelengths, layers)
                for config in forward_configs:
                    label = (
                        "synthetic-forward "
                        f"case={case.case} sweep={sweep_axis} wavelengths={wavelengths} "
                        f"layers={layers} backend={config.label} {config.device}"
                    )
                    _progress(args, f"start {label}")
                    step_start = time.perf_counter()
                    rows = _forward_runtime_rows(
                        experiment="synthetic-forward",
                        case=case,
                        config=config,
                        sweep_axis=sweep_axis,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                    raw_rows.extend(rows)
                    _progress(
                        args,
                        f"done {label} {_completed_status(rows)} "
                        f"wall={time.perf_counter() - step_start:.1f}s",
                    )

    if "synthetic-jacobian" in groups:
        jac_specs = _jacobian_specs(
            layer_counts=layer_counts,
            wavelength_counts=jacobian_wavelength_counts,
            grad_layer_counts=grad_layer_counts,
            base_layers=base_layers,
            base_wavelengths=base_wavelengths,
            full_grid=args.full_grid,
        )
        jacobian_configs, jac_manifest = _backend_configs(
            backend_set=args.backend_set,
            torch_dtypes=torch_dtypes,
            include_numpy=False,
            include_torch=True,
            created_utc=created_utc,
        )
        manifest_rows.extend(jac_manifest)
        for wavelengths, layers, active_tau_layers, sweep_axis, gradient_target in jac_specs:
            if gradient_target not in jacobian_targets:
                continue
            for builder in _synthetic_case_builders():
                case = builder(wavelengths, layers)
                for config in jacobian_configs:
                    label = (
                        "synthetic-jacobian "
                        f"case={case.case} target={gradient_target} sweep={sweep_axis} "
                        f"wavelengths={wavelengths} layers={layers} "
                        f"active_layers={active_tau_layers} backend={config.label} {config.device}"
                    )
                    _progress(args, f"start {label}")
                    step_start = time.perf_counter()
                    if gradient_target == "surface_albedo":
                        rows = _surface_albedo_jacobian_runtime_rows(
                            case=case,
                            config=config,
                            sweep_axis=sweep_axis,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                    elif gradient_target == "omega":
                        rows = _omega_jacobian_runtime_rows(
                            case=case,
                            config=config,
                            sweep_axis=sweep_axis,
                            active_tau_layers=active_tau_layers,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                    else:
                        rows = _jacobian_runtime_rows(
                            case=case,
                            config=config,
                            sweep_axis=sweep_axis,
                            active_tau_layers=active_tau_layers,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                    raw_rows.extend(rows)
                    _progress(
                        args,
                        f"done {label} {_completed_status(rows)} "
                        f"wall={time.perf_counter() - step_start:.1f}s",
                    )

    summary_rows = _summarize(raw_rows)
    raw_path = output_dir / "raw_timings.csv"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.csv"
    _write_csv(raw_path, raw_rows, RAW_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    return raw_path, summary_path, manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("fortran-forward", "synthetic-forward", "synthetic-jacobian"),
        default=("synthetic-forward",),
    )
    parser.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "forward_scaling_benchmark",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend-set", choices=("all", "cpu", "numpy", "cuda"), default="all")
    parser.add_argument("--torch-dtypes", default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--layer-counts", default=None)
    parser.add_argument("--wavelength-counts", default=None)
    parser.add_argument(
        "--jacobian-wavelength-counts",
        default=None,
        help=(
            "Comma-separated wavelength counts for synthetic-jacobian spectral sweeps. "
            "Defaults to a bounded Jacobian grid independent of --wavelength-counts."
        ),
    )
    parser.add_argument(
        "--grad-layer-counts",
        default=None,
        help="Comma-separated active tau-layer counts for the synthetic Jacobian gradient sweep.",
    )
    parser.add_argument(
        "--jacobian-targets",
        default="tau,omega,surface_albedo",
        help="Comma-separated synthetic Jacobian targets: tau, omega/ssa, surface_albedo.",
    )
    parser.add_argument("--base-layers", type=int, default=None)
    parser.add_argument("--base-wavelengths", type=int, default=None)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument(
        "--fortran-limit",
        type=int,
        default=None,
        help="Optional spectral-row limit for the checked-in Fortran-reference scenes.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-subcase progress messages written outside timed RT regions.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("--warmups must be non-negative and --repeats must be positive")
    paths = run_benchmarks(args)
    print("paper RT benchmark complete")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
