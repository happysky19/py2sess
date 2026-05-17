#!/usr/bin/env python3
"""Quantify prepared-input delta-M convention effects for the paper."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from py2sess import TwoStreamEss, TwoStreamEssOptions  # noqa: E402
from py2sess.optical.planck import thermal_source_from_temperature_profile  # noqa: E402
from py2sess.rtsolver.fo_solar_obs_torch import (  # noqa: E402
    fo_scatter_term_henyey_greenstein_torch,
)

FIELDS = (
    "diagnostic",
    "regime",
    "wavelengths",
    "layers",
    "metric",
    "value",
    "notes",
)


def _standard_heights_and_temperature(layers: int) -> tuple[np.ndarray, np.ndarray]:
    bottom_to_top = np.linspace(0.0, 50.0, layers + 1)
    temp = np.where(
        bottom_to_top <= 11.0,
        288.15 - 6.5 * bottom_to_top,
        np.where(bottom_to_top <= 20.0, 216.65, 216.65 + bottom_to_top - 20.0),
    )
    temp = np.minimum(temp, 270.0)
    return bottom_to_top[::-1].copy(), temp[::-1].copy()


def _solar_g_gradient(*, differentiable_delta_m: bool, wavelengths: int, layers: int) -> np.ndarray:
    import torch

    z, _temperature = _standard_heights_and_temperature(layers)
    angles = [47.7, 49.5, 275.7]
    tau = torch.full((wavelengths, layers), 0.01, dtype=torch.float64)
    ssa = torch.full((wavelengths, layers), 0.2, dtype=torch.float64)
    g = torch.full((wavelengths, layers), 0.3, dtype=torch.float64, requires_grad=True)
    trunc = g * g if differentiable_delta_m else (g.detach() * g.detach())
    fo_scatter = fo_scatter_term_henyey_greenstein_torch(
        ssa=ssa,
        g=g,
        angles=angles,
        delta_m_truncation_factor=trunc,
        n_moments=5,
        dtype=torch.float64,
    )
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=layers,
            mode="solar",
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=ssa,
        g=g,
        z=z,
        angles=angles,
        albedo=torch.full((wavelengths,), 0.05, dtype=torch.float64),
        fbeam=torch.ones(wavelengths, dtype=torch.float64),
        delta_m_truncation_factor=trunc,
        fo_scatter_term=fo_scatter,
        include_fo=True,
    )
    result.radiance_total.sum().backward()
    return g.grad.detach().cpu().numpy()


def _thermal_g_gradient(
    *, differentiable_delta_m: bool, wavelengths: int, layers: int
) -> np.ndarray:
    import torch

    z, temperature = _standard_heights_and_temperature(layers)
    thermal = thermal_source_from_temperature_profile(
        temperature,
        np.array([288.15]),
        wavenumber_cm_inv=np.linspace(700.0, 1300.0, wavelengths),
    )
    tau = torch.full((wavelengths, layers), 0.01, dtype=torch.float64)
    ssa = torch.full((wavelengths, layers), 0.05, dtype=torch.float64)
    g = torch.full((wavelengths, layers), 0.3, dtype=torch.float64, requires_grad=True)
    trunc = g * g if differentiable_delta_m else (g.detach() * g.detach())
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=layers,
            mode="thermal",
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=ssa,
        g=g,
        z=z,
        angles=49.5,
        stream=0.5,
        albedo=torch.full((wavelengths,), 0.02, dtype=torch.float64),
        emissivity=torch.full((wavelengths,), 0.98, dtype=torch.float64),
        delta_m_truncation_factor=trunc,
        planck=torch.tensor(thermal.planck, dtype=torch.float64),
        surface_planck=torch.tensor(thermal.surface_planck, dtype=torch.float64),
        include_fo=True,
    )
    result.radiance_total.sum().backward()
    return g.grad.detach().cpu().numpy()


def _thermal_source_convention(*, wavelengths: int, layers: int) -> tuple[float, float]:
    z, temperature = _standard_heights_and_temperature(layers)
    thermal = thermal_source_from_temperature_profile(
        temperature,
        np.array([288.15]),
        wavenumber_cm_inv=np.linspace(700.0, 1300.0, wavelengths),
    )
    tau = np.full((wavelengths, layers), 0.01, dtype=float)
    ssa = np.full_like(tau, 0.05)
    g = np.full_like(tau, 0.1)
    trunc = g * g
    kwargs: dict[str, Any] = {
        "tau": tau,
        "ssa": ssa,
        "g": g,
        "z": z,
        "angles": 49.5,
        "stream": 0.5,
        "albedo": np.full(wavelengths, 0.02, dtype=float),
        "emissivity": np.full(wavelengths, 0.98, dtype=float),
        "delta_m_truncation_factor": trunc,
        "planck": thermal.planck,
        "surface_planck": thermal.surface_planck,
        "include_fo": True,
    }
    base = (
        TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=layers,
                mode="thermal",
                backend="numpy",
                fo_thermal_source_delta_m_scaling=False,
            )
        )
        .forward(**kwargs)
        .radiance_total
    )
    alternative = (
        TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=layers,
                mode="thermal",
                backend="numpy",
                fo_thermal_source_delta_m_scaling=True,
            )
        )
        .forward(**kwargs)
        .radiance_total
    )
    rel = np.abs(alternative - base) / np.maximum(np.abs(base), 1.0e-30)
    return float(np.max(rel) * 100.0), float(np.mean(rel) * 100.0)


def _chain_rule_rows(
    regime: str, fixed: np.ndarray, differentiable: np.ndarray
) -> list[dict[str, str]]:
    contribution = differentiable - fixed
    fixed_norm = float(np.linalg.norm(fixed))
    contribution_norm = float(np.linalg.norm(contribution))
    rows = [
        ("fixed_gradient_l2", fixed_norm),
        ("omitted_chain_rule_l2", contribution_norm),
        (
            "omitted_chain_rule_relative_l2",
            contribution_norm / max(fixed_norm, 1.0e-30),
        ),
        ("max_abs_fixed_gradient", float(np.max(np.abs(fixed)))),
        ("max_abs_omitted_chain_rule", float(np.max(np.abs(contribution)))),
        (
            "max_pointwise_relative_omitted_chain_rule",
            float(np.max(np.abs(contribution) / np.maximum(np.abs(fixed), 1.0e-20))),
        ),
    ]
    return [
        {
            "diagnostic": "delta_m_chain_rule",
            "regime": regime,
            "wavelengths": "200",
            "layers": "50",
            "metric": metric,
            "value": f"{value:.12e}",
            "notes": "fixed f versus differentiable f=g^2 for g sensitivity",
        }
        for metric, value in rows
    ]


def generate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        _chain_rule_rows(
            "Solar/VNIR",
            _solar_g_gradient(differentiable_delta_m=False, wavelengths=200, layers=50),
            _solar_g_gradient(differentiable_delta_m=True, wavelengths=200, layers=50),
        )
    )
    rows.extend(
        _chain_rule_rows(
            "Thermal",
            _thermal_g_gradient(differentiable_delta_m=False, wavelengths=200, layers=50),
            _thermal_g_gradient(differentiable_delta_m=True, wavelengths=200, layers=50),
        )
    )
    max_rel, mean_rel = _thermal_source_convention(wavelengths=1000, layers=50)
    for metric, value in (
        ("max_relative_radiance_difference_percent", max_rel),
        ("mean_relative_radiance_difference_percent", mean_rel),
    ):
        rows.append(
            {
                "diagnostic": "thermal_fo_source_convention",
                "regime": "Thermal",
                "wavelengths": "1000",
                "layers": "50",
                "metric": metric,
                "value": f"{value:.12e}",
                "notes": "source-side delta-M multiplier on/off",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "delta_m_chain_rule_sensitivity.csv",
    )
    args = parser.parse_args()
    rows = generate_rows()
    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
