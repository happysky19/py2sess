#!/usr/bin/env python3
"""Generate the small finite-difference Jacobian validation table for the paper."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from py2sess import TwoStreamEss, TwoStreamEssOptions  # noqa: E402
from py2sess.rtsolver.fo_solar_obs import fo_scatter_term_henyey_greenstein  # noqa: E402
from py2sess.rtsolver.backend import has_torch  # noqa: E402

FIELDS = (
    "regime",
    "case_size",
    "checked_variables",
    "n_columns_checked",
    "step",
    "max_abs_fd",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "n_anomalous_columns",
)


@dataclass(frozen=True)
class ValidationGroup:
    regime: str
    mode: str
    variables: tuple[str, ...]
    label: str


GROUPS = (
    ValidationGroup("UV solar", "solar", ("tau", "ssa", "g"), "tau, omega, g"),
    ValidationGroup("UV solar", "solar", ("albedo", "fbeam"), "surface albedo, incident flux"),
    ValidationGroup("TIR thermal", "thermal", ("tau", "ssa", "g"), "tau, omega, g"),
    ValidationGroup(
        "TIR thermal",
        "thermal",
        ("planck", "surface_planck"),
        "level Planck, surface Planck",
    ),
    ValidationGroup(
        "TIR thermal",
        "thermal",
        ("emissivity", "albedo"),
        "emissivity, surface albedo",
    ),
)


def _base_solar_inputs() -> dict[str, np.ndarray | float | list[float]]:
    shape = (2, 3)
    inputs = {
        "tau": np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]], dtype=float),
        "ssa": np.full(shape, 0.2, dtype=float),
        "g": np.full(shape, 0.1, dtype=float),
        "delta_m_truncation_factor": np.zeros(shape, dtype=float),
        "z": np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        "angles": [30.0, 20.0, 0.0],
        "albedo": np.array([0.05, 0.08], dtype=float),
        "fbeam": np.array([1.0, 0.9], dtype=float),
    }
    inputs["fo_scatter_term"] = fo_scatter_term_henyey_greenstein(
        ssa=inputs["ssa"],
        g=inputs["g"],
        angles=inputs["angles"],
        delta_m_truncation_factor=inputs["delta_m_truncation_factor"],
        n_moments=5,
    )
    return inputs


def _base_thermal_inputs() -> dict[str, np.ndarray | float]:
    shape = (2, 3)
    return {
        "tau": np.array([[0.10, 0.20, 0.30], [0.20, 0.30, 0.40]], dtype=float),
        "ssa": np.full(shape, 0.1, dtype=float),
        "g": np.full(shape, 0.05, dtype=float),
        "delta_m_truncation_factor": np.zeros(shape, dtype=float),
        "z": np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        "angles": 30.0,
        "stream": 0.5,
        "planck": np.array([[1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.1, 1.2]], dtype=float),
        "surface_planck": np.array([1.4, 1.3], dtype=float),
        "emissivity": np.array([0.98, 0.97], dtype=float),
        "albedo": np.array([0.02, 0.03], dtype=float),
    }


def _copy_inputs(inputs: dict[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _torch_inputs(inputs: dict[str, object], grad_variable: str | None = None) -> dict[str, object]:
    import torch

    converted: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray) and key != "z":
            tensor = torch.tensor(value, dtype=torch.float64)
            if key == grad_variable:
                tensor.requires_grad_(True)
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def _solver(mode: str) -> TwoStreamEss:
    return TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=3,
            mode=mode,
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )


def _radiance_sum(mode: str, inputs: dict[str, object]):
    solver = _solver(mode)
    if mode == "solar":
        result = solver.forward(
            tau=inputs["tau"],
            ssa=inputs["ssa"],
            g=inputs["g"],
            z=inputs["z"],
            angles=inputs["angles"],
            albedo=inputs["albedo"],
            fbeam=inputs["fbeam"],
            delta_m_truncation_factor=inputs["delta_m_truncation_factor"],
            fo_scatter_term=inputs["fo_scatter_term"],
            include_fo=True,
        )
    else:
        result = solver.forward(
            tau=inputs["tau"],
            ssa=inputs["ssa"],
            g=inputs["g"],
            z=inputs["z"],
            angles=inputs["angles"],
            stream=inputs["stream"],
            albedo=inputs["albedo"],
            emissivity=inputs["emissivity"],
            delta_m_truncation_factor=inputs["delta_m_truncation_factor"],
            planck=inputs["planck"],
            surface_planck=inputs["surface_planck"],
            include_fo=True,
        )
    return result.radiance_total.sum()


def _objective(mode: str, inputs: dict[str, object]) -> float:
    value = _radiance_sum(mode, _torch_inputs(inputs))
    return float(value.detach().cpu())


def _autograd(mode: str, inputs: dict[str, object], variable: str) -> np.ndarray:
    converted = _torch_inputs(inputs, variable)
    value = _radiance_sum(mode, converted)
    value.backward()
    grad = converted[variable].grad
    if grad is None:
        return np.zeros_like(np.asarray(inputs[variable], dtype=float))
    return grad.detach().cpu().numpy()


def _indices(value: np.ndarray) -> list[tuple[int, ...]]:
    if value.ndim == 0:
        return [()]
    return list(np.ndindex(value.shape))


def _finite_difference_step(_variable: str, _value: float) -> float:
    return 1.0e-6


def _group_summary(group: ValidationGroup) -> dict[str, str]:
    inputs = _base_solar_inputs() if group.mode == "solar" else _base_thermal_inputs()
    max_abs_fd = 0.0
    max_abs_error = 0.0
    max_rel_error = 0.0
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    step_used = 0.0

    for variable in group.variables:
        base_value = np.asarray(inputs[variable], dtype=float)
        autograd_values = _autograd(group.mode, inputs, variable)
        for index in _indices(base_value):
            step = _finite_difference_step(variable, float(base_value[index]))
            step_used = step
            plus = _copy_inputs(inputs)
            minus = _copy_inputs(inputs)
            np.asarray(plus[variable])[index] += step
            np.asarray(minus[variable])[index] -= step
            finite_difference = (_objective(group.mode, plus) - _objective(group.mode, minus)) / (
                2.0 * step
            )
            autograd_value = float(autograd_values[index])
            abs_error = abs(autograd_value - finite_difference)
            rel_error = abs_error / max(abs(finite_difference), 1.0e-12)
            abs_errors.append(abs_error)
            rel_errors.append(rel_error)
            max_abs_fd = max(max_abs_fd, abs(finite_difference))
            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)

    if not all(math.isfinite(value) for value in (max_abs_fd, max_abs_error, max_rel_error)):
        raise RuntimeError(f"non-finite validation error for {group}")

    return {
        "regime": group.regime,
        "case_size": "2 wavelengths x 3 layers",
        "checked_variables": group.label,
        "n_columns_checked": str(
            sum(np.asarray(inputs[variable], dtype=float).size for variable in group.variables)
        ),
        "step": f"{step_used:.1e}",
        "max_abs_fd": f"{max_abs_fd:.12e}",
        "max_abs_error": f"{max_abs_error:.12e}",
        "mean_abs_error": f"{float(np.mean(abs_errors)):.12e}",
        "max_rel_error": f"{max_rel_error:.12e}",
        "mean_rel_error": f"{float(np.mean(rel_errors)):.12e}",
        "n_anomalous_columns": str(sum(error > 1.0e-2 for error in rel_errors)),
    }


def _standard_heights_and_temperature(layers: int) -> tuple[np.ndarray, np.ndarray]:
    bottom_to_top = np.linspace(0.0, 50.0, layers + 1)
    temp = np.where(
        bottom_to_top <= 11.0,
        288.15 - 6.5 * bottom_to_top,
        np.where(bottom_to_top <= 20.0, 216.65, 216.65 + bottom_to_top - 20.0),
    )
    temp = np.minimum(temp, 270.0)
    return bottom_to_top[::-1].copy(), temp[::-1].copy()


def _medium_solar_inputs(wavelengths: int = 1000, layers: int = 50) -> dict[str, object]:
    z, _temperature = _standard_heights_and_temperature(layers)
    shape = (wavelengths, layers)
    tau = np.full(shape, 0.01, dtype=float)
    ssa = np.full(shape, 0.2, dtype=float)
    g = np.full(shape, 0.1, dtype=float)
    scaling = np.zeros(shape, dtype=float)
    angles = [30.0, 20.0, 0.0]
    return {
        "tau": tau,
        "ssa": ssa,
        "g": g,
        "delta_m_truncation_factor": scaling,
        "z": z,
        "angles": angles,
        "albedo": np.full(wavelengths, 0.05, dtype=float),
        "fbeam": np.ones(wavelengths, dtype=float),
        "fo_scatter_term": fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=g,
            angles=angles,
            delta_m_truncation_factor=scaling,
            n_moments=5,
        ),
    }


def _medium_thermal_inputs(wavelengths: int = 1000, layers: int = 50) -> dict[str, object]:
    from py2sess.optical.planck import thermal_source_from_temperature_profile

    z, temperature = _standard_heights_and_temperature(layers)
    shape = (wavelengths, layers)
    thermal = thermal_source_from_temperature_profile(
        temperature,
        np.array([288.15]),
        wavenumber_cm_inv=np.linspace(700.0, 1300.0, wavelengths),
    )
    return {
        "tau": np.full(shape, 0.10, dtype=float),
        "ssa": np.full(shape, 0.10, dtype=float),
        "g": np.full(shape, 0.05, dtype=float),
        "delta_m_truncation_factor": np.zeros(shape, dtype=float),
        "z": z,
        "angles": 30.0,
        "stream": 0.5,
        "planck": thermal.planck,
        "surface_planck": thermal.surface_planck,
        "emissivity": np.full(wavelengths, 0.98, dtype=float),
        "albedo": np.full(wavelengths, 0.02, dtype=float),
    }


def _medium_objective(mode: str, inputs: dict[str, object]) -> float:
    value = _medium_radiance_sum(mode, _torch_inputs(inputs))
    return float(value.detach().cpu())


def _medium_radiance_sum(mode: str, inputs: dict[str, object]):
    n_layers = int(inputs["tau"].shape[-1])
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=n_layers,
            mode=mode,
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    if mode == "solar":
        result = solver.forward(
            tau=inputs["tau"],
            ssa=inputs["ssa"],
            g=inputs["g"],
            z=inputs["z"],
            angles=inputs["angles"],
            albedo=inputs["albedo"],
            fbeam=inputs["fbeam"],
            delta_m_truncation_factor=inputs["delta_m_truncation_factor"],
            fo_scatter_term=inputs["fo_scatter_term"],
            include_fo=True,
        )
    else:
        result = solver.forward(
            tau=inputs["tau"],
            ssa=inputs["ssa"],
            g=inputs["g"],
            z=inputs["z"],
            angles=inputs["angles"],
            stream=inputs["stream"],
            albedo=inputs["albedo"],
            emissivity=inputs["emissivity"],
            delta_m_truncation_factor=inputs["delta_m_truncation_factor"],
            planck=inputs["planck"],
            surface_planck=inputs["surface_planck"],
            include_fo=True,
        )
    return result.radiance_total.sum()


def _medium_autograd(mode: str, inputs: dict[str, object], variable: str) -> np.ndarray:
    converted = _torch_inputs(inputs, variable)
    value = _medium_radiance_sum(mode, converted)
    value.backward()
    grad = converted[variable].grad
    if grad is None:
        return np.zeros_like(np.asarray(inputs[variable], dtype=float))
    return grad.detach().cpu().numpy()


def _medium_sample_indices() -> tuple[tuple[int, int], ...]:
    return ((0, 0), (99, 7), (499, 25), (999, 49))


def _medium_group_summary(regime: str, mode: str) -> dict[str, str]:
    inputs = _medium_solar_inputs() if mode == "solar" else _medium_thermal_inputs()
    variables = ("tau", "ssa", "g")
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    max_abs_fd = 0.0
    step_used = ""
    for variable in variables:
        base_value = np.asarray(inputs[variable], dtype=float)
        autograd_values = _medium_autograd(mode, inputs, variable)
        for index in _medium_sample_indices():
            step = _finite_difference_step(variable, float(base_value[index]))
            step_used = f"{step:.1e}" if not step_used else "variable"
            plus = _copy_inputs(inputs)
            minus = _copy_inputs(inputs)
            np.asarray(plus[variable])[index] += step
            np.asarray(minus[variable])[index] -= step
            finite_difference = (_medium_objective(mode, plus) - _medium_objective(mode, minus)) / (
                2.0 * step
            )
            autograd_value = float(autograd_values[index])
            abs_error = abs(autograd_value - finite_difference)
            rel_error = abs_error / max(abs(finite_difference), 1.0e-12)
            abs_errors.append(abs_error)
            rel_errors.append(rel_error)
            max_abs_fd = max(max_abs_fd, abs(finite_difference))

    if not abs_errors or not all(math.isfinite(value) for value in abs_errors + rel_errors):
        raise RuntimeError(f"non-finite medium-scale validation error for {regime}")
    n_anomalous = sum(error > 1.0e-2 for error in rel_errors)
    return {
        "regime": regime,
        "case_size": "1000 wavelengths x 50 layers",
        "checked_variables": "sampled tau, omega, g",
        "n_columns_checked": str(len(abs_errors)),
        "step": step_used,
        "max_abs_fd": f"{max_abs_fd:.12e}",
        "max_abs_error": f"{max(abs_errors):.12e}",
        "mean_abs_error": f"{float(np.mean(abs_errors)):.12e}",
        "max_rel_error": f"{max(rel_errors):.12e}",
        "mean_rel_error": f"{float(np.mean(rel_errors)):.12e}",
        "n_anomalous_columns": str(n_anomalous),
    }


def generate_rows() -> list[dict[str, str]]:
    if not has_torch():
        raise RuntimeError("PyTorch is required to generate Jacobian validation summary")
    rows = [_group_summary(group) for group in GROUPS]
    rows.append(_medium_group_summary("Solar/VNIR", "solar"))
    rows.append(_medium_group_summary("Thermal", "thermal"))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "jacobian_gradient_validation_summary.csv",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rows = generate_rows()
    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
