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
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from py2sess import TwoStreamEss, TwoStreamEssOptions  # noqa: E402
from py2sess.optical.planck import thermal_source_from_temperature_profile  # noqa: E402
from py2sess.rtsolver.backend import has_torch, to_numpy  # noqa: E402
from py2sess.rtsolver.native_backend import (  # noqa: E402
    native_backend_supports_device,
    solve_solar_2s,
    solve_solar_fo,
    solve_thermal_2s,
    solve_thermal_fo,
)
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (  # noqa: E402
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.rtsolver.fo_solar_obs_batch_torch import (  # noqa: E402
    solve_fo_solar_obs_eps_batch_torch,
)
from py2sess.rtsolver.fo_solar_obs import fo_scatter_term_henyey_greenstein  # noqa: E402
from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors  # noqa: E402
from py2sess.rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy  # noqa: E402
from py2sess.rtsolver.solar_obs_batch_torch import solve_solar_obs_batch_torch  # noqa: E402
from py2sess.rtsolver.thermal_batch_numpy import (  # noqa: E402
    _fo_thermal_toa,
    _two_stream_thermal_toa,
    precompute_fo_thermal_geometry_numpy,
)
from py2sess.rtsolver.thermal_batch_torch import (  # noqa: E402
    _fo_thermal_toa_batch,
    _two_stream_thermal_toa_batch,
    fo_thermal_geometry_to_torch,
)
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
RAW_FIELDS = (
    "experiment",
    "case",
    "mode",
    "backend",
    "device",
    "dtype",
    "timing_kind",
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
    "timing_kind",
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
    compiled: bool = False
    compile_mode: str = "reduce-overhead"


@dataclass(frozen=True)
class RtCase:
    """Prepared direct RT inputs for one benchmark case."""

    case: str
    mode: str
    kwargs: dict[str, Any]
    wavelengths: int
    layers: int
    reference_total: np.ndarray | None = None


@dataclass(frozen=True)
class _RadianceOnly:
    """Small result wrapper used by compiled timing callables."""

    radiance_total: Any


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
        "emissivity": "surface_emissivity",
        "asymmetry": "g",
        "phase": "g",
    }
    allowed = {"tau", "omega", "g", "surface_albedo", "surface_emissivity"}
    parsed = tuple(
        aliases.get(part.strip(), part.strip()) for part in value.split(",") if part.strip()
    )
    if not parsed or any(target not in allowed for target in parsed):
        raise ValueError(
            "--jacobian-targets must contain tau, omega, g, surface_albedo, "
            "and/or surface_emissivity"
        )
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


def _layer_centers_km(z_top_to_bottom: np.ndarray) -> np.ndarray:
    return 0.5 * (z_top_to_bottom[:-1] + z_top_to_bottom[1:])


def _normalized_profile(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def build_synthetic_uv_case(wavelengths: int, layers: int) -> RtCase:
    """Build deterministic solar direct RT inputs without opacity files.

    The profile is intentionally inhomogeneous: a gas-like absorbing background,
    a low aerosol layer, and an elevated cloud-like scattering layer. This keeps
    the benchmark synthetic and prepared-input-only while making the local VJP
    targets closer to retrieval Jacobian tests than a constant slab.
    """
    if wavelengths <= 0 or layers <= 0:
        raise ValueError("wavelengths and layers must be positive")
    z, _ = _standard_heights_and_temperature(layers)
    height = _layer_centers_km(z)
    wavelength_nm = np.linspace(645.0, 665.0, wavelengths, dtype=float)
    spectral = wavelength_nm[:, None]
    gas_line = 1.0 + 3.0 * np.exp(-0.5 * ((spectral - 656.0) / 2.2) ** 2)
    gas_profile = _normalized_profile(np.exp(-height / 8.0))[None, :]
    gas_tau = 0.018 * gas_line * gas_profile

    rayleigh_profile = _normalized_profile(np.exp(-height / 7.5))[None, :]
    rayleigh_tau = 0.010 * (spectral / 650.0) ** -4.0 * rayleigh_profile

    aerosol_profile = _normalized_profile(np.exp(-0.5 * ((height - 2.0) / 1.3) ** 2))[None, :]
    aerosol_tau = 0.050 * (spectral / 650.0) ** -1.2 * aerosol_profile

    cloud_profile = _normalized_profile(np.exp(-0.5 * ((height - 10.5) / 2.0) ** 2))[None, :]
    cloud_tau = 0.020 * (1.0 + 0.15 * np.cos((spectral - 645.0) / 20.0 * np.pi)) * cloud_profile

    scattering_tau = rayleigh_tau + aerosol_tau + cloud_tau
    tau = gas_tau + scattering_tau
    ssa = scattering_tau / np.maximum(tau, 1.0e-15)
    g = (0.70 * aerosol_tau + 0.85 * cloud_tau) / np.maximum(scattering_tau, 1.0e-15)
    scaling = np.zeros_like(tau)
    angles = np.array([40.0, 50.0, 90.0])
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
    height = _layer_centers_km(z)
    wavenumber = np.linspace(700.0, 1300.0, wavelengths)
    spectral = wavenumber[:, None]
    absorber_profile = _normalized_profile(
        np.exp(-height / 6.0) + 0.15 * np.exp(-0.5 * ((height - 15.0) / 5.0) ** 2)
    )[None, :]
    gas_lines = (
        1.0
        + 2.0 * np.exp(-0.5 * ((spectral - 820.0) / 35.0) ** 2)
        + 1.4 * np.exp(-0.5 * ((spectral - 1120.0) / 55.0) ** 2)
    )
    absorption_tau = 0.030 * gas_lines * absorber_profile
    haze_profile = _normalized_profile(np.exp(-0.5 * ((height - 3.0) / 2.0) ** 2))[None, :]
    scattering_tau = 0.004 * (1.0 + 0.1 * np.sin((spectral - 700.0) / 600.0 * np.pi)) * haze_profile
    tau = absorption_tau + scattering_tau
    ssa = scattering_tau / np.maximum(tau, 1.0e-15)
    g = np.full_like(tau, 0.45)
    scaling = np.zeros_like(tau)
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
        "angles",
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
    torch_compile: bool,
    torch_compile_mode: str,
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
                        kind="backend",
                        backend=f"Native CPU {dtype}",
                        device="cpu",
                        dtype=dtype,
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
                        kind="backend",
                        backend=f"Native CPU {dtype}",
                        device="cpu",
                        dtype=dtype,
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
                            kind="backend",
                            backend=f"Native CUDA {dtype}",
                            device="cuda",
                            dtype=dtype,
                            status="skipped",
                            reason="native extension is not built for CUDA",
                        )
                    )
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


def _compile_forward_callable(run: Callable[..., Any], config: BackendConfig) -> Callable[..., Any]:
    if config.backend != "torch" or not config.compiled:
        return run
    torch = _torch_module()
    if torch is None or not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch installation")
    if config.compile_mode == "default":
        return torch.compile(run)
    return torch.compile(run, mode=config.compile_mode)


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


def _solar_scattering_cosine(angles: np.ndarray) -> float:
    geoms = angles.reshape(1, 3) if angles.ndim == 1 else angles
    if geoms.shape != (1, 3):
        raise ValueError("compiled direct solar timing supports one observation geometry")
    sza, vza, raz = geoms[0]
    sza_rad = np.deg2rad(sza)
    vza_rad = np.deg2rad(vza)
    raz_rad = np.deg2rad(raz)
    user_mu = float(np.cos(vza_rad))
    cosscat = -user_mu * float(np.cos(sza_rad)) + float(
        np.sin(vza_rad) * np.sin(sza_rad) * np.cos(raz_rad)
    )
    if np.isclose(sza, 0.0):
        return 0.0 if np.isclose(user_mu, 0.0) else -user_mu
    return cosscat


def _solar_hg_scatter_term_torch(ssa: Any, g: Any, scaling: Any, cosscat: Any) -> Any:
    denominator = 1.0 - scaling * ssa
    phase_denominator = 1.0 + g * g - 2.0 * g * cosscat
    phase = (1.0 - g * g) / phase_denominator.pow(1.5)
    return phase * (ssa / denominator)


def _direct_torch_evaluator(
    case: RtCase,
    kwargs: dict[str, Any],
    config: BackendConfig,
    *,
    bvp_engine: str = "auto",
) -> Callable[[dict[str, Any]], Any]:
    if config.backend != "torch":
        raise RuntimeError("direct torch evaluator requires a torch backend")
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)

    if case.mode == "solar":
        heights = np.asarray(case.kwargs["z"], dtype=float)
        angles = np.asarray(case.kwargs["angles"], dtype=float)
        geoms = angles.reshape(1, 3) if angles.ndim == 1 else angles
        if geoms.shape != (1, 3):
            raise ValueError("compiled direct solar timing supports one observation geometry")
        stream = float(case.kwargs.get("stream", 1.0 / np.sqrt(3.0)))
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
        fo_precomputed = _torch_fo_solar_precompute(
            fo_solar_obs_batch_precompute(
                user_obsgeom=geoms,
                heights=heights,
                earth_radius=6371.0,
                nfine=3,
            ),
            dtype=dtype,
            device=device,
        )
        chapman_t = _tensor(chapman, dtype=dtype, device=device)
        pxsq_values = tuple(float(item) for item in np.asarray(pxsq, dtype=float).reshape(-1))
        px0x_values = tuple(float(item) for item in np.asarray(px0x[0], dtype=float).reshape(-1))
        cosscat_t = _tensor(_solar_scattering_cosine(angles), dtype=dtype, device=device)

        def evaluate(local_kwargs: dict[str, Any]):
            tau = local_kwargs["tau"]
            omega = local_kwargs["ssa"]
            asymm = local_kwargs["g"]
            scaling = local_kwargs["delta_m_truncation_factor"]
            albedo = local_kwargs["albedo"]
            fbeam = local_kwargs.get("fbeam")
            if fbeam is None:
                fbeam = torch.ones(tau.shape[0], dtype=tau.dtype, device=tau.device)
            exact_scatter = local_kwargs.get("fo_scatter_term")
            if exact_scatter is None:
                exact_scatter = _solar_hg_scatter_term_torch(
                    omega,
                    asymm,
                    scaling,
                    cosscat_t,
                )
            fo = solve_fo_solar_obs_eps_batch_torch(
                tau=tau,
                omega=omega,
                scaling=scaling,
                albedo=albedo,
                flux_factor=fbeam,
                exact_scatter=exact_scatter,
                precomputed=fo_precomputed,
                dtype=dtype,
                device=device,
            )
            two_stream = solve_solar_obs_batch_torch(
                tau=tau,
                omega=omega,
                asymm=asymm,
                scaling=scaling,
                albedo=albedo,
                flux_factor=fbeam,
                stream_value=stream,
                chapman=chapman_t,
                x0=float(x0[0]),
                user_stream=float(user_stream[0]),
                user_secant=1.0 / float(user_stream[0]),
                azmfac=float(azmfac[0]),
                px11=px11,
                pxsq=pxsq_values,
                px0x=px0x_values,
                ulp=float(np.asarray(ulp).reshape(-1)[0]),
                dtype=dtype,
                device=device,
                bvp_engine=bvp_engine,
            )
            return fo + two_stream

        return evaluate

    if case.mode != "thermal":
        raise ValueError(f"unsupported RT mode for direct torch evaluator: {case.mode!r}")

    heights = np.asarray(case.kwargs["z"], dtype=float)
    user_angle = float(np.asarray(case.kwargs["angles"], dtype=float).reshape(-1)[0])
    user_stream = float(np.cos(np.deg2rad(user_angle)))
    stream = float(case.kwargs.get("stream", 0.5))
    geometry = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    fo_geometry = fo_thermal_geometry_to_torch(geometry, dtype=dtype, device=device)
    heights_t = _tensor(heights, dtype=dtype, device=device)

    def evaluate(local_kwargs: dict[str, Any]):
        tau = local_kwargs["tau"]
        omega = local_kwargs["ssa"]
        asymm = local_kwargs["g"]
        scaling = local_kwargs["delta_m_truncation_factor"]
        albedo = local_kwargs["albedo"]
        planck = local_kwargs["planck"]
        surfbb = local_kwargs["surface_planck"]
        emissivity = local_kwargs.get("emissivity")
        if emissivity is None:
            emissivity = 1.0 - albedo
        two_stream = _two_stream_thermal_toa_batch(
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
        fo = _fo_thermal_toa_batch(
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
        return fo + two_stream

    return evaluate


def _torch_compile_skip_reason(case: RtCase, config: BackendConfig) -> str:
    if not config.compiled:
        return ""
    return ""


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
    if config.backend in {"torch", "native"} and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.synchronize()


def _reset_peak_memory(config: BackendConfig) -> None:
    if config.backend in {"torch", "native"} and config.device == "cuda":
        torch = _torch_module()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(torch.device(config.device))


def _peak_memory(config: BackendConfig) -> int | None:
    if config.backend in {"torch", "native"} and config.device == "cuda":
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


def _slice_spectral_rows(
    bundle: dict[str, Any],
    keys: tuple[str, ...],
    start: int,
    stop: int,
) -> dict[str, Any]:
    return {key: bundle[key][start:stop] for key in keys}


def _component_chunk_size(case: RtCase, backend: str) -> int:
    total_rows = int(case.wavelengths)
    if total_rows <= 0:
        return 0
    n_layers = max(int(case.layers), 1)
    if case.mode == "solar":
        row_floats = (48 if backend == "torch" else 40) * n_layers + 64
        target_mib = 1024 if backend == "torch" else 1400
    elif case.mode == "thermal":
        row_floats = (6 if backend == "torch" else 4) * n_layers + 32
        target_mib = 560 if backend == "torch" else 384
    else:  # pragma: no cover
        raise ValueError(f"unsupported RT mode: {case.mode!r}")
    target_bytes = target_mib * 1024 * 1024
    chunk = target_bytes // (8 * row_floats)
    granularity = 2000 if backend == "torch" else 1000
    chunk = max(granularity, int(chunk))
    chunk = min(total_rows, ((chunk + granularity - 1) // granularity) * granularity)
    return max(1, chunk)


def _scalar_value(value: Any) -> float:
    arr = np.asarray(to_numpy(value), dtype=float).reshape(-1)
    return float(arr[0])


def _timed_torch_call(
    ctx_device: Any,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
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


def _component_timing(
    *,
    case: RtCase,
    total: np.ndarray,
    seconds: float,
    fo_seconds: float,
    two_stream_seconds: float,
    chunk_size: int,
) -> dict[str, float | int | None]:
    max_abs, max_rel = _accuracy_summary(total, case.reference_total)
    return {
        "seconds": seconds,
        "timing_kind": "components",
        "forward_seconds": None,
        "backward_seconds": None,
        "checksum": _checksum(total),
        "grad_checksum": None,
        "grad_l2": None,
        "max_abs_diff": max_abs,
        "max_rel_diff_pct": max_rel,
        "component_fo_seconds": fo_seconds,
        "component_two_stream_seconds": two_stream_seconds,
        "component_chunk_size": chunk_size,
    }


def _run_numpy_solar_components_once(case: RtCase) -> dict[str, float | int | None]:
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "numpy")
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
    bundle = dict(kwargs)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    for start in range(0, case.wavelengths, chunk_size):
        stop = min(start + chunk_size, case.wavelengths)
        chunk = _slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
        timer = time.perf_counter()
        fo = solve_fo_solar_obs_eps_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            scaling=chunk["delta_m_truncation_factor"],
            albedo=chunk["albedo"],
            flux_factor=chunk["fbeam"],
            exact_scatter=chunk["fo_scatter_term"],
            precomputed=fo_precomputed,
        )
        fo_seconds += time.perf_counter() - timer
        timer = time.perf_counter()
        two_stream = solve_solar_obs_batch_numpy(
            tau=chunk["tau"],
            omega=chunk["ssa"],
            asymm=chunk["g"],
            scaling=chunk["delta_m_truncation_factor"],
            albedo=chunk["albedo"],
            flux_factor=chunk["fbeam"],
            stream_value=stream,
            chapman=chapman,
            x0=_scalar_value(x0),
            user_stream=_scalar_value(user_stream),
            user_secant=1.0 / _scalar_value(user_stream),
            azmfac=_scalar_value(azmfac),
            px11=px11,
            pxsq=pxsq,
            px0x=px0x[0],
            ulp=_scalar_value(ulp),
            bvp_engine="auto",
        )
        two_stream_seconds += time.perf_counter() - timer
        total_parts.append(fo + two_stream)
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_numpy_thermal_components_once(case: RtCase) -> dict[str, float | int | None]:
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "numpy")
    heights = np.asarray(kwargs["z"], dtype=float)
    user_angle = _scalar_value(kwargs["angles"])
    user_stream = float(np.cos(np.deg2rad(user_angle)))
    stream = float(kwargs.get("stream", 0.5))
    geometry = precompute_fo_thermal_geometry_numpy(
        heights=heights,
        user_angle_degrees=user_angle,
        earth_radius=6371.0,
        nfine=3,
    )
    bundle = dict(kwargs)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    for start in range(0, case.wavelengths, chunk_size):
        stop = min(start + chunk_size, case.wavelengths)
        chunk = _slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
        timer = time.perf_counter()
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
            bvp_engine="auto",
        )
        two_stream_seconds += time.perf_counter() - timer
        timer = time.perf_counter()
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
        fo_seconds += time.perf_counter() - timer
        total_parts.append(fo + two_stream)
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_numpy_components_once(case: RtCase) -> dict[str, float | int | None]:
    if case.mode == "solar":
        return _run_numpy_solar_components_once(case)
    return _run_numpy_thermal_components_once(case)


def _run_torch_solar_components_once(
    case: RtCase,
    config: BackendConfig,
    *,
    dtype: Any,
    device: Any,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "torch")
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
            x0=_scalar_value(x0),
            user_stream=_scalar_value(user_stream),
            user_secant=1.0 / _scalar_value(user_stream),
            azmfac=_scalar_value(azmfac),
            px11=px11,
            pxsq=pxsq_values,
            px0x=px0x_values,
            ulp=_scalar_value(ulp),
            dtype=dtype,
            device=device,
            bvp_engine="auto",
        )

    fo_kernel = _compile_forward_callable(fo_kernel, config)
    two_stream_kernel = _compile_forward_callable(two_stream_kernel, config)
    bundle = dict(kwargs)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, case.wavelengths, chunk_size):
            stop = min(start + chunk_size, case.wavelengths)
            chunk = _slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
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
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_torch_thermal_components_once(
    case: RtCase,
    config: BackendConfig,
    *,
    dtype: Any,
    device: Any,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "torch")
    heights = np.asarray(kwargs["z"], dtype=float)
    user_angle = _scalar_value(kwargs["angles"])
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
            bvp_engine="auto",
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

    two_stream_kernel = _compile_forward_callable(two_stream_kernel, config)
    fo_kernel = _compile_forward_callable(fo_kernel, config)
    bundle = dict(kwargs)
    fo_seconds = 0.0
    two_stream_seconds = 0.0
    total_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, case.wavelengths, chunk_size):
            stop = min(start + chunk_size, case.wavelengths)
            chunk = _slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
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
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_torch_components_once(
    case: RtCase,
    config: BackendConfig,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    if case.mode == "solar":
        return _run_torch_solar_components_once(case, config, dtype=dtype, device=device)
    return _run_torch_thermal_components_once(case, config, dtype=dtype, device=device)


def _run_native_solar_components_once(
    case: RtCase,
    *,
    dtype: Any,
    device: Any,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "native")
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
    total_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, case.wavelengths, chunk_size):
            stop = min(start + chunk_size, case.wavelengths)
            chunk = _slice_spectral_rows(bundle, SOLAR_COMPONENT_KEYS, start, stop)
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
                x0=_scalar_value(x0),
                user_stream=_scalar_value(user_stream),
                user_secant=1.0 / _scalar_value(user_stream),
                azmfac=_scalar_value(azmfac),
                px11=px11,
                ulp=_scalar_value(ulp),
                return_profile=False,
            )
            two_stream_seconds += elapsed
            total_parts.append((fo + two_stream).detach().cpu().numpy())
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_native_thermal_components_once(
    case: RtCase,
    *,
    dtype: Any,
    device: Any,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    kwargs = case.kwargs
    chunk_size = _component_chunk_size(case, "native")
    heights = np.asarray(kwargs["z"], dtype=float)
    user_angle = _scalar_value(kwargs["angles"])
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
    total_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, case.wavelengths, chunk_size):
            stop = min(start + chunk_size, case.wavelengths)
            chunk = _slice_spectral_rows(bundle, THERMAL_COMPONENT_KEYS, start, stop)
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
    total = np.concatenate(total_parts)
    return _component_timing(
        case=case,
        total=total,
        seconds=fo_seconds + two_stream_seconds,
        fo_seconds=fo_seconds,
        two_stream_seconds=two_stream_seconds,
        chunk_size=chunk_size,
    )


def _run_native_components_once(
    case: RtCase,
    config: BackendConfig,
) -> dict[str, float | int | None]:
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not native_backend_supports_device(config.device):
        raise RuntimeError(f"native extension is not built for {config.device!r}")
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    if case.mode == "solar":
        return _run_native_solar_components_once(case, dtype=dtype, device=device)
    return _run_native_thermal_components_once(case, dtype=dtype, device=device)


def _component_runtime_rows(
    *,
    experiment: str,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    skip_reason = _torch_compile_skip_reason(case, config)
    if skip_reason:
        return [
            _failure_row(
                experiment=experiment,
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                active_tau_layers=0,
                n_grad_vars=0,
                reason=skip_reason,
                status="skipped",
            )
        ]

    if config.backend == "numpy":

        def run_once() -> dict[str, float | int | None]:
            return _run_numpy_components_once(case)

    elif config.backend == "native":

        def run_once() -> dict[str, float | int | None]:
            return _run_native_components_once(case, config)

    else:

        def run_once() -> dict[str, float | int | None]:
            return _run_torch_components_once(case, config)

    try:
        for _ in range(warmups):
            run_once()
            _sync_if_cuda(config)
        timings = []
        for repeat in range(repeats):
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            timing = run_once()
            _sync_if_cuda(config)
            timing["repeat_index"] = repeat
            timing["cuda_peak_bytes"] = _peak_memory(config)
            timings.append(timing)
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


def _make_solver(case: RtCase, config: BackendConfig, *, enable_grad: bool) -> TwoStreamEss:
    return TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=case.layers,
            mode=case.mode,
            backend=config.backend,
            torch_device=config.device or None,
            torch_dtype=config.dtype if config.backend in {"torch", "native"} else None,
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
    skip_reason = _torch_compile_skip_reason(case, config)
    if skip_reason:
        return [
            _failure_row(
                experiment=experiment,
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                active_tau_layers=0,
                n_grad_vars=0,
                reason=skip_reason,
                status="skipped",
            )
        ]
    if config.backend in {"torch", "native"}:
        kwargs, _ = _float_fields_to_torch(case.kwargs, dtype=config.dtype, device=config.device)
    else:
        kwargs = case.kwargs

    try:
        if config.backend == "torch" and config.compiled:
            evaluator = _direct_torch_evaluator(case, kwargs, config)

            def forward_radiance():
                return evaluator(kwargs)

        else:
            solver = _make_solver(case, config, enable_grad=False)

            def forward_radiance():
                return solver.forward(**kwargs, include_fo=True).radiance_total

        timed_forward = _compile_forward_callable(forward_radiance, config)
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

    def run():
        return _RadianceOnly(timed_forward())

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
    skip_reason = _torch_compile_skip_reason(case, config)
    if skip_reason:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target=gradient_target,
                active_tau_layers=active_tau_layers,
                n_grad_vars=case.wavelengths * active_tau_layers,
                reason=skip_reason,
                status="skipped",
            )
        ]
    active_indices = _active_layer_indices(case.layers, active_tau_layers)
    omit = ("fo_scatter_term",) if case.mode == "solar" and gradient_field in {"ssa", "g"} else ()
    kwargs, _ = _float_fields_to_torch(
        case.kwargs,
        dtype=config.dtype,
        device=config.device,
        omit=omit,
    )
    torch = _torch_module()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    if gradient_field not in kwargs:
        raise RuntimeError(f"{gradient_field!r} is not available for Jacobian timing")
    field_base = kwargs[gradient_field].detach()
    active_index_t = torch.as_tensor(active_indices, dtype=torch.long, device=config.device)
    active_field = field_base[:, active_index_t].detach().clone().requires_grad_(True)

    def make_field(field):
        if active_tau_layers == case.layers:
            return field
        expanded = field_base.clone()
        expanded[:, active_index_t] = field
        return expanded

    try:
        if config.compiled:
            evaluator = _direct_torch_evaluator(case, kwargs, config)

            def forward_radiance(field):
                local_kwargs = dict(kwargs)
                local_kwargs[gradient_field] = make_field(field)
                return evaluator(local_kwargs)

        else:
            solver = _make_solver(case, config, enable_grad=True)

            def forward_radiance(field):
                local_kwargs = dict(kwargs)
                local_kwargs[gradient_field] = make_field(field)
                return solver.forward(**local_kwargs, include_fo=True).radiance_total

        timed_forward = _compile_forward_callable(forward_radiance, config)
    except Exception as exc:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target=gradient_target,
                active_tau_layers=active_tau_layers,
                n_grad_vars=case.wavelengths * active_tau_layers,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]

    def run_once(*, measure: bool) -> dict[str, float | int | None]:
        active_field.grad = None
        if measure:
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            total_start = time.perf_counter()
            forward_start = total_start
        radiance_total = timed_forward(active_field)
        if measure:
            _sync_if_cuda(config)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
        loss = radiance_total.sum()
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
        radiance = np.asarray(to_numpy(radiance_total), dtype=float)
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


def _g_jacobian_runtime_rows(
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
        gradient_field="g",
        gradient_target="g",
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
    skip_reason = _torch_compile_skip_reason(case, config)
    if skip_reason:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target="surface_albedo",
                active_tau_layers=0,
                n_grad_vars=case.wavelengths,
                reason=skip_reason,
                status="skipped",
            )
        ]
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

    try:
        if config.compiled:
            evaluator = _direct_torch_evaluator(case, kwargs, config)

            def forward_radiance(albedo_value):
                local_kwargs = dict(kwargs)
                local_kwargs["albedo"] = albedo_value
                return evaluator(local_kwargs)

        else:
            solver = _make_solver(case, config, enable_grad=True)

            def forward_radiance(albedo_value):
                local_kwargs = dict(kwargs)
                local_kwargs["albedo"] = albedo_value
                return solver.forward(**local_kwargs, include_fo=True).radiance_total

        timed_forward = _compile_forward_callable(forward_radiance, config)
    except Exception as exc:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target="surface_albedo",
                active_tau_layers=0,
                n_grad_vars=case.wavelengths,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]

    def run_once(*, measure: bool) -> dict[str, float | int | None]:
        albedo.grad = None
        if measure:
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            total_start = time.perf_counter()
            forward_start = total_start
        radiance_total = timed_forward(albedo)
        if measure:
            _sync_if_cuda(config)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
        loss = radiance_total.sum()
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
        radiance = np.asarray(to_numpy(radiance_total), dtype=float)
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


def _surface_emissivity_jacobian_runtime_rows(
    *,
    case: RtCase,
    config: BackendConfig,
    sweep_axis: str,
    warmups: int,
    repeats: int,
) -> list[dict[str, str]]:
    if config.backend != "torch" or "emissivity" not in case.kwargs:
        return []
    skip_reason = _torch_compile_skip_reason(case, config)
    if skip_reason:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target="surface_emissivity",
                active_tau_layers=0,
                n_grad_vars=case.wavelengths,
                reason=skip_reason,
                status="skipped",
            )
        ]
    kwargs, tracked = _float_fields_to_torch(
        case.kwargs,
        dtype=config.dtype,
        device=config.device,
        requires_grad=("emissivity",),
    )
    torch = _torch_module()
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    emissivity = tracked["emissivity"]

    try:
        if config.compiled:
            evaluator = _direct_torch_evaluator(case, kwargs, config)

            def forward_radiance(emissivity_value):
                local_kwargs = dict(kwargs)
                local_kwargs["emissivity"] = emissivity_value
                return evaluator(local_kwargs)

        else:
            solver = _make_solver(case, config, enable_grad=True)

            def forward_radiance(emissivity_value):
                local_kwargs = dict(kwargs)
                local_kwargs["emissivity"] = emissivity_value
                return solver.forward(**local_kwargs, include_fo=True).radiance_total

        timed_forward = _compile_forward_callable(forward_radiance, config)
    except Exception as exc:
        return [
            _failure_row(
                experiment="synthetic-jacobian",
                case=case,
                config=config,
                sweep_axis=sweep_axis,
                gradient_target="surface_emissivity",
                active_tau_layers=0,
                n_grad_vars=case.wavelengths,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]

    def run_once(*, measure: bool) -> dict[str, float | int | None]:
        emissivity.grad = None
        if measure:
            _reset_peak_memory(config)
            _sync_if_cuda(config)
            total_start = time.perf_counter()
            forward_start = total_start
        radiance_total = timed_forward(emissivity)
        if measure:
            _sync_if_cuda(config)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
        loss = radiance_total.sum()
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
        radiance = np.asarray(to_numpy(radiance_total), dtype=float)
        if emissivity.grad is None:
            raise RuntimeError("expected surface emissivity gradient was not populated")
        grad = emissivity.grad.detach()
        if not bool(grad.isfinite().all().item()):
            raise RuntimeError("surface emissivity gradient contains non-finite values")
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
                gradient_target="surface_emissivity",
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
            gradient_target="surface_emissivity",
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
        "timing_kind": timing.get("timing_kind", ""),
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
    status: str = "failed",
) -> dict[str, str]:
    row = {
        "experiment": experiment,
        "case": case.case,
        "mode": case.mode,
        "backend": config.label,
        "device": config.device,
        "dtype": config.dtype,
        "timing_kind": "",
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
        "status": status,
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
    specs.extend(
        (base_wavelengths, base_layers, count, "g_grad_vars", "g") for count in omega_counts
    )
    specs.append((base_wavelengths, base_layers, 0, "surface_albedo", "surface_albedo"))
    specs.append((base_wavelengths, base_layers, 0, "surface_emissivity", "surface_emissivity"))
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
        "timing_kind",
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
            row["timing_kind"],
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
            row["timing_kind"],
            row["sweep_axis"],
            row["gradient_target"],
            row["wavelengths"],
            row["layers"],
            row["levels"],
            row["active_tau_layers"],
        ): float(row["best_s"])
        for row in summaries
        if row["backend"].startswith("Torch CPU")
    }
    for row in summaries:
        key = (
            row["experiment"],
            row["case"],
            row["mode"],
            row["timing_kind"],
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
        _manifest_row(created_utc, kind="metadata", value=f"torch_compile={args.torch_compile}"),
        _manifest_row(
            created_utc,
            kind="metadata",
            value=f"torch_compile_mode={args.torch_compile_mode}",
        ),
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
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
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
                    rows = _component_runtime_rows(
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
            torch_compile=args.torch_compile,
            torch_compile_mode=args.torch_compile_mode,
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
                    elif gradient_target == "surface_emissivity":
                        rows = _surface_emissivity_jacobian_runtime_rows(
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
                    elif gradient_target == "g":
                        rows = _g_jacobian_runtime_rows(
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
    parser.add_argument(
        "--backend-set",
        choices=("all", "cpu", "numpy", "native", "cuda"),
        default="all",
    )
    parser.add_argument("--torch-dtypes", default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Run torch benchmark rows through torch.compile. The output backend label is "
            "tagged with 'torch.compile'; eager mode remains the default."
        ),
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="Compilation mode passed to torch.compile when --torch-compile is set.",
    )
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
        default="tau,omega,g,surface_albedo,surface_emissivity",
        help=(
            "Comma-separated synthetic Jacobian targets: tau, omega/ssa, g/phase-shape, "
            "surface_albedo, surface_emissivity."
        ),
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
