#!/usr/bin/env python3
"""Compare py2sess torch gradients with a compact Fortran Jacobian fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.optical.planck import (
    _FORTRAN_C2,
    _FORTRAN_PLANCK_CONC,
    _FORTRAN_PLANCK_CRITERION,
    _FORTRAN_PLANCK_EPSIL,
    _FORTRAN_PLANCK_NSIMPSON,
    _FORTRAN_PLANCK_VMAX,
    _FORTRAN_SIGMA_OVER_PI,
)
from py2sess.rtsolver.backend import has_torch
from py2sess.scene import load_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("thermal", "solar"), default="thermal")
    parser.add_argument("--profile")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--reference")
    parser.add_argument(
        "--unit-scale",
        default="auto",
        help="Scale py2sess radiance/Jacobian before comparison, or 'auto'.",
    )
    parser.add_argument(
        "--plot", type=Path, default=None, help="Optional PNG path for comparison plot."
    )
    args = parser.parse_args()
    if not has_torch():
        raise RuntimeError("PyTorch is required for gradient comparison")

    if args.case == "solar":
        result, reference = solar_toa_jacobians(args.scene, args.reference)
        compare_solar(result, reference, args.plot)
        return

    if args.profile is None:
        raise ValueError("thermal comparison requires --profile")
    scene = load_scene(profile=args.profile, config=args.scene, strict_runtime_inputs=True)
    reference = load_scene_reference(args.scene, args.reference)
    result = thermal_toa_jacobians(scene)
    indices = matching_indices(result["wavelength_nm"], reference["wavelength_nm"])
    radiance = result["radiance_total"][indices]
    scale = unit_scale(args.unit_scale, radiance, reference["radiance_total"])

    print("Fortran Jacobian comparison")
    print(f"scene wavelengths: {result['wavelength_nm'].size}")
    print(f"reference wavelengths: {reference['wavelength_nm'].size}")
    print(f"unit scale: {scale:.12g}")
    print_summary("radiance_total", scale * radiance, reference["radiance_total"])
    print_summary(
        "surface_emissivity_jacobian_total",
        scale * result["surface_emissivity_jacobian_total"][indices],
        reference["surface_emissivity_jacobian_total"],
    )
    if "surface_temperature_jacobian_total" in reference:
        print_summary(
            "surface_temperature_jacobian_total_normalized",
            scale * result["surface_temperature_jacobian_total_normalized"][indices],
            reference["surface_temperature_jacobian_total"],
        )
    if args.plot is not None:
        plot_thermal_comparison(args.plot, result, reference, indices, scale)
        print(f"plot: {args.plot}")


def solar_toa_jacobians(
    scene_path: str | Path, reference_path: str | Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch

    scene_path = Path(scene_path)
    config = read_scene_config(scene_path)
    if config.get("mode") != "solar":
        raise ValueError("solar Jacobian comparison requires mode: solar")
    inputs_path = scene_path.parent / config["rt_inputs"]["path"]
    inputs = dict(np.load(inputs_path))
    reference = load_scene_reference(scene_path, reference_path)

    def tensor(name: str, *, requires_grad: bool = False):
        return torch.tensor(
            np.asarray(inputs[name]),
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    tau = tensor("tau")
    ssa = tensor("ssa")
    asymm = tensor("g")
    trunc = tensor("delta_m_truncation_factor")
    albedo = tensor("albedo", requires_grad=True)
    fbeam = tensor("fbeam")
    fo_scatter = tensor("fo_scatter_term")

    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tau.shape[-1],
            mode="solar",
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=ssa,
        g=asymm,
        z=inputs["z"],
        angles=inputs["angles"],
        albedo=albedo,
        fbeam=fbeam,
        delta_m_truncation_factor=trunc,
        fo_scatter_term=fo_scatter,
        include_fo=True,
    )
    radiance_2s = result.radiance_2s.reshape(-1)
    radiance_fo = result.radiance_fo.reshape(-1)
    radiance_total = result.radiance_total.reshape(-1)
    jac_2s = torch.autograd.grad(radiance_2s.sum(), albedo, retain_graph=True)[0]
    jac_fo = torch.autograd.grad(radiance_fo.sum(), albedo, retain_graph=True)[0]
    jac_total = torch.autograd.grad(radiance_total.sum(), albedo)[0]
    return (
        {
            "wavelength_nm": np.asarray(inputs["wavelength_nm"], dtype=float),
            "radiance_2s": radiance_2s.detach().cpu().numpy(),
            "radiance_fo": radiance_fo.detach().cpu().numpy(),
            "radiance_total": radiance_total.detach().cpu().numpy(),
            "surface_albedo_jacobian_2s": jac_2s.detach().cpu().numpy(),
            "surface_albedo_jacobian_fo": jac_fo.detach().cpu().numpy(),
            "surface_albedo_jacobian_total": jac_total.detach().cpu().numpy(),
        },
        reference,
    )


def compare_solar(
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    plot_path: Path | None,
) -> None:
    indices = matching_indices(result["wavelength_nm"], reference["wavelength_nm"])
    print("Fortran solar Jacobian comparison")
    print(f"scene wavelengths: {result['wavelength_nm'].size}")
    print(f"reference wavelengths: {reference['wavelength_nm'].size}")
    for component in ("2s", "fo", "total"):
        print_summary(
            f"radiance_{component}",
            result[f"radiance_{component}"][indices],
            reference[f"radiance_{component}"],
        )
        print_summary(
            f"surface_albedo_jacobian_{component}",
            result[f"surface_albedo_jacobian_{component}"][indices],
            reference[f"surface_albedo_jacobian_{component}"],
        )
    if plot_path is not None:
        plot_solar_comparison(plot_path, result, reference, indices)
        print(f"plot: {plot_path}")


def read_scene_config(scene_path: str | Path) -> dict:
    scene_path = Path(scene_path)
    config = yaml.safe_load(scene_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{scene_path} is not a scene mapping")
    return config


def load_scene_reference(
    scene_path: str | Path, reference_path: str | Path | None = None
) -> dict[str, np.ndarray]:
    scene_path = Path(scene_path)
    if reference_path is None:
        config = read_scene_config(scene_path)
        reference = config.get("jacobian_reference")
        if not isinstance(reference, dict) or "path" not in reference:
            raise ValueError("scene YAML must define jacobian_reference.path or pass --reference")
        reference_path = scene_path.parent / reference["path"]
    return dict(np.load(reference_path))


def thermal_toa_jacobians(scene) -> dict[str, np.ndarray]:
    import torch

    inputs = scene.to_forward_inputs()
    kwargs = inputs.kwargs
    context = {"dtype": torch.float64}

    def tensor(name: str, *, requires_grad: bool = False):
        return torch.tensor(np.asarray(kwargs[name]), **context, requires_grad=requires_grad)

    tau = tensor("tau")
    ssa = tensor("ssa")
    asymm = tensor("g")
    trunc = tensor("delta_m_truncation_factor")
    planck = tensor("planck")
    surface_planck = tensor("surface_planck")
    emissivity = tensor("emissivity", requires_grad=True)
    surface_temperature = torch.full(
        surface_planck.shape,
        _surface_temperature_k(scene),
        dtype=torch.float64,
        requires_grad=True,
    )
    surface_planck_from_temperature = _surface_planck_from_temperature(scene, surface_temperature)

    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tau.shape[-1],
            mode="thermal",
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=ssa,
        g=asymm,
        z=kwargs["z"],
        angles=kwargs["angles"],
        planck=planck,
        surface_planck=surface_planck_from_temperature,
        emissivity=emissivity,
        albedo=1.0 - emissivity,
        delta_m_truncation_factor=trunc,
        stream=kwargs.get("stream"),
        include_fo=True,
    )
    radiance_2s = result.radiance_2s.reshape(-1)
    radiance_fo = result.radiance_fo.reshape(-1)
    radiance_total = result.radiance_total.reshape(-1)
    jac_2s = torch.autograd.grad(radiance_2s.sum(), emissivity, retain_graph=True)[0]
    jac_fo = torch.autograd.grad(radiance_fo.sum(), emissivity, retain_graph=True)[0]
    jac_total = torch.autograd.grad(radiance_total.sum(), emissivity, retain_graph=True)[0]
    t_jac_total = torch.autograd.grad(radiance_total.sum(), surface_temperature)[0]
    t_jac_total_normalized = surface_temperature * t_jac_total
    return {
        "wavelength_nm": np.asarray(inputs.wavelengths, dtype=float),
        "radiance_2s": radiance_2s.detach().cpu().numpy(),
        "radiance_fo": radiance_fo.detach().cpu().numpy(),
        "radiance_total": radiance_total.detach().cpu().numpy(),
        "surface_emissivity_jacobian_2s": jac_2s.detach().cpu().numpy(),
        "surface_emissivity_jacobian_fo": jac_fo.detach().cpu().numpy(),
        "surface_emissivity_jacobian_total": jac_total.detach().cpu().numpy(),
        "surface_temperature_jacobian_total": t_jac_total.detach().cpu().numpy(),
        "surface_temperature_jacobian_total_normalized": (
            t_jac_total_normalized.detach().cpu().numpy()
        ),
    }


def _surface_temperature_k(scene) -> float:
    bundle = getattr(scene, "_bundle", None)
    if not bundle or "surface_temperature_k" not in bundle:
        raise ValueError("scene does not expose surface_temperature_k")
    return float(np.asarray(bundle["surface_temperature_k"], dtype=float).reshape(-1)[0])


def _surface_planck_from_temperature(scene, surface_temperature):
    import torch

    bundle = getattr(scene, "_bundle", None)
    if not bundle or "wavenumber_band_cm_inv" not in bundle:
        raise ValueError("surface-temperature Jacobian requires wavenumber_band_cm_inv")
    bands = torch.as_tensor(
        np.asarray(bundle["wavenumber_band_cm_inv"], dtype=np.float64),
        dtype=surface_temperature.dtype,
        device=surface_temperature.device,
    )
    return _planck_wavenumber_band_torch(surface_temperature, bands[:, 0], bands[:, 1])


def _planck_wavenumber_band_torch(temperature, low, high):
    import torch

    gamma = _FORTRAN_C2 / temperature
    x_low = gamma * low
    x_high = gamma * high
    if not bool(
        (
            (x_low > _FORTRAN_PLANCK_EPSIL)
            & (x_high < _FORTRAN_PLANCK_VMAX)
            & ((high - low) / high < 1.0e-2)
        ).all()
    ):
        raise ValueError("torch Jacobian benchmark currently supports narrow bands only")

    interval = x_high - x_low
    f_low = x_low**3 / torch.expm1(x_low)
    f_high = x_high**3 / torch.expm1(x_high)
    endpoints = f_low + f_high
    previous = endpoints * 0.5 * interval
    value = previous
    for n in range(1, _FORTRAN_PLANCK_NSIMPSON + 1):
        step = 0.5 * interval / float(n)
        current = endpoints
        for k in range(1, 2 * n):
            x_current = x_low + float(k) * step
            current = current + float(2 * (1 + (k % 2))) * x_current**3 / torch.expm1(x_current)
        value = current * step / 3.0
        if bool(
            (torch.abs(value - previous) / torch.abs(value)).le(_FORTRAN_PLANCK_CRITERION).all()
        ):
            break
        previous = value
    else:
        raise RuntimeError("Fortran-compatible torch Planck integration did not converge")

    return _FORTRAN_SIGMA_OVER_PI * temperature**4 * value * _FORTRAN_PLANCK_CONC


def matching_indices(scene_wavelength: np.ndarray, reference_wavelength: np.ndarray) -> np.ndarray:
    if (
        reference_wavelength.size <= scene_wavelength.size
        and np.all(np.diff(reference_wavelength) < 0.0)
        and np.all(np.diff(scene_wavelength) > 0.0)
    ):
        return np.arange(reference_wavelength.size)
    indices = np.searchsorted(scene_wavelength, reference_wavelength)
    indices = np.clip(indices, 1, scene_wavelength.size - 1)
    left = indices - 1
    choose_left = np.abs(scene_wavelength[left] - reference_wavelength) <= np.abs(
        scene_wavelength[indices] - reference_wavelength
    )
    out = np.where(choose_left, left, indices)
    if not np.allclose(scene_wavelength[out], reference_wavelength, rtol=0.0, atol=1.0e-4):
        raise ValueError("reference wavelengths are not on the scene grid")
    return out


def unit_scale(value: str, py_value: np.ndarray, reference: np.ndarray) -> float:
    if value == "auto":
        denom = float(np.dot(py_value, py_value))
        if denom == 0.0:
            raise ValueError("cannot infer unit scale from zero py2sess radiance")
        return float(np.dot(py_value, reference) / denom)
    return float(value)


def print_summary(name: str, value: np.ndarray, reference: np.ndarray) -> None:
    diff = value - reference
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / np.maximum(np.abs(reference), 1.0e-30)))
    print(f"{name}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}")


def plot_thermal_comparison(
    path: Path,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
    scale: float,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if "wavenumber_cm_inv" in reference:
        x_axis = reference["wavenumber_cm_inv"]
        xlabel = "Wavenumber (cm$^{-1}$)"
    else:
        x_axis = reference["wavelength_nm"] / 1000.0
        xlabel = "Wavelength (um)"
    order = np.argsort(x_axis)
    x_axis = x_axis[order]

    series = [
        (
            "TOA radiance",
            scale * result["radiance_total"][indices],
            reference["radiance_total"],
            "Radiance",
            "log",
        ),
        (
            "Surface-emissivity Jacobian",
            scale * result["surface_emissivity_jacobian_total"][indices],
            reference["surface_emissivity_jacobian_total"],
            "dI / d emissivity",
            "linear",
        ),
        (
            "Normalized surface-temperature Jacobian",
            scale * result["surface_temperature_jacobian_total_normalized"][indices],
            reference["surface_temperature_jacobian_total"],
            "Tsurf dI / dTsurf",
            "linear",
        ),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for row, (title, py_value, ref_value, ylabel, yscale) in enumerate(series):
        py_value = py_value[order]
        ref_value = ref_value[order]
        axes[row, 0].plot(x_axis, ref_value, color="black", linewidth=1.8, label="Fortran")
        axes[row, 0].plot(
            x_axis, py_value, color="#0072B2", linewidth=1.2, linestyle="--", label="py2sess"
        )
        axes[row, 0].set(title=title, ylabel=ylabel, yscale=yscale)
        axes[row, 0].grid(True, alpha=0.25)
        if row == 0:
            axes[row, 0].legend(frameon=False)

        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].plot(x_axis, py_value - ref_value, color="#D55E00", linewidth=1.2)
        axes[row, 1].set(title="py2sess - Fortran", ylabel="Difference")
        axes[row, 1].grid(True, alpha=0.25)
    for ax in axes.ravel():
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[-1, 0].set(xlabel=xlabel)
    axes[-1, 1].set(xlabel=xlabel)
    fig.suptitle("Thermal Fortran Jacobian Validation", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_solar_comparison(
    path: Path,
    result: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    indices: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_axis = reference["wavelength_nm"] / 1000.0
    series = [
        (
            "2S TOA radiance",
            result["radiance_2s"][indices],
            reference["radiance_2s"],
            "Radiance",
        ),
        (
            "FO TOA radiance",
            result["radiance_fo"][indices],
            reference["radiance_fo"],
            "Radiance",
        ),
        (
            "Total TOA radiance",
            result["radiance_total"][indices],
            reference["radiance_total"],
            "Radiance",
        ),
        (
            "Total surface-albedo Jacobian",
            result["surface_albedo_jacobian_total"][indices],
            reference["surface_albedo_jacobian_total"],
            "dI / d albedo",
        ),
        (
            "2S surface-albedo Jacobian",
            result["surface_albedo_jacobian_2s"][indices],
            reference["surface_albedo_jacobian_2s"],
            "dI / d albedo",
        ),
    ]
    fig, axes = plt.subplots(len(series), 2, figsize=(11, 10), sharex=True)
    for row, (title, py_value, ref_value, ylabel) in enumerate(series):
        axes[row, 0].plot(x_axis, ref_value, color="black", linewidth=1.8, label="Fortran")
        axes[row, 0].plot(
            x_axis, py_value, color="#0072B2", linewidth=1.2, linestyle="--", label="py2sess"
        )
        axes[row, 0].set(title=title, ylabel=ylabel)
        axes[row, 0].grid(True, alpha=0.25)
        if row == 0:
            axes[row, 0].legend(frameon=False)
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].plot(x_axis, py_value - ref_value, color="#D55E00", linewidth=1.2)
        axes[row, 1].set(title="py2sess - Fortran", ylabel="Difference")
        axes[row, 1].grid(True, alpha=0.25)
    axes[-1, 0].set(xlabel="Wavelength (um)")
    axes[-1, 1].set(xlabel="Wavelength (um)")
    fig.suptitle("Solar Fortran Jacobian Validation", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
