#!/usr/bin/env python3
"""Compare py2sess torch gradients with a compact Fortran Jacobian fixture."""

from __future__ import annotations

import argparse

import numpy as np

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
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--unit-scale",
        default="auto",
        help="Scale py2sess radiance/Jacobian before comparison, or 'auto'.",
    )
    args = parser.parse_args()
    if not has_torch():
        raise RuntimeError("PyTorch is required for gradient comparison")

    scene = load_scene(profile=args.profile, config=args.scene, strict_runtime_inputs=True)
    reference = dict(np.load(args.reference))
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
    if "radiance_2s" in reference and "surface_emissivity_jacobian_2s" in reference:
        print("component radiance/Jacobian columns: diagnostic only; convention differs")
    for key in sorted(reference):
        if key.endswith("_profile_jacobian_total"):
            print(f"{key}: diagnostic only; profile chain-rule state is not pass/fail")


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


if __name__ == "__main__":
    main()
