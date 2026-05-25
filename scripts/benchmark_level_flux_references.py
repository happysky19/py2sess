#!/usr/bin/env python3
"""Compare py2sess level fluxes with analytic, pydisort, and KINETICS references."""

from __future__ import annotations

import argparse
import csv
import math
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expn

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.benchmarks.flux_references import (
    rayleigh_phase_moments,
    run_pydisort_absorbing_solar_flux,
    run_pydisort_solar_flux,
)
from py2sess.optical.phase import build_solar_fo_scatter_term
from py2sess.benchmarks.kinetics_flux import kinetics_flux_to_py2sess, read_kinetics_flux_table


def _to_numpy(value: Any) -> np.ndarray:
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flux_field(actual: Any, field: str) -> Any:
    if isinstance(actual, dict):
        return actual[field]
    return getattr(actual, field)


def _solar_direct_surface_reference(
    tau: np.ndarray,
    *,
    sza: float,
    fbeam: float,
    albedo: float,
) -> dict[str, np.ndarray]:
    mu0 = math.cos(math.radians(sza))
    levels = np.concatenate(([0.0], np.cumsum(tau)))
    distance_to_surface = levels[-1] - levels
    reflected_flux_boa = fbeam * mu0 * albedo * math.exp(-levels[-1] / mu0)
    flux_up = 2.0 * reflected_flux_boa * expn(3, distance_to_surface)
    flux_down = np.zeros_like(flux_up)
    flux_mean = 0.5 * reflected_flux_boa / math.pi * expn(2, distance_to_surface)
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up,
        "flux_mean": flux_mean,
    }


def _solar_clear_surface_reference(
    tau: np.ndarray,
    *,
    sza: float,
    fbeam: float,
    albedo: float,
) -> dict[str, np.ndarray]:
    mu0 = math.cos(math.radians(sza))
    levels = np.concatenate(([0.0], np.cumsum(tau)))
    direct_down = fbeam * mu0 * np.exp(-levels / mu0)
    direct_mean = fbeam * np.exp(-levels / mu0) / (4.0 * math.pi)
    reflected = _solar_direct_surface_reference(
        tau,
        sza=sza,
        fbeam=fbeam,
        albedo=albedo,
    )
    flux_up = reflected["flux_up"]
    flux_down = direct_down
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": direct_mean + reflected["flux_mean"],
    }


def _thermal_surface_reference(
    tau: np.ndarray,
    *,
    surface_radiance: float,
) -> dict[str, np.ndarray]:
    levels = np.concatenate(([0.0], np.cumsum(tau)))
    distance_to_surface = levels[-1] - levels
    flux_up = 2.0 * math.pi * surface_radiance * expn(3, distance_to_surface)
    flux_down = np.zeros_like(flux_up)
    flux_mean = 0.5 * surface_radiance * expn(2, distance_to_surface)
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up,
        "flux_mean": flux_mean,
    }


def _case_absorbing_solar() -> tuple[Any, dict[str, np.ndarray]]:
    sza = 30.0
    mu0 = math.cos(math.radians(sza))
    tau = np.array([0.1, 0.2], dtype=float)
    result = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_fluxes=True,
        )
    ).forward(
        tau=tau,
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=[sza, 20.0, 0.0],
        fbeam=np.pi,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    cumulative_tau = np.array([0.0, 0.1, 0.3], dtype=float)
    direct_down = np.pi * mu0 * np.exp(-cumulative_tau / mu0)
    mean = 0.25 * np.exp(-cumulative_tau / mu0)
    reference = {
        "flux_up": np.zeros(3, dtype=float),
        "flux_down": direct_down,
        "flux_net": -direct_down,
        "flux_mean": mean,
    }
    return result, reference


def _case_solar_surface_total() -> tuple[Any, dict[str, np.ndarray]]:
    sza = 30.0
    tau = np.array([0.2, 0.3], dtype=float)
    result = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_fluxes=True,
        )
    ).forward(
        tau=tau,
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=[sza, 20.0, 0.0],
        fbeam=2.0,
        albedo=0.3,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
        include_fo=True,
    )
    return result, _solar_clear_surface_reference(tau, sza=sza, fbeam=2.0, albedo=0.3)


def _case_solar_fo_surface_correction() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sza = 30.0
    tau = np.array([0.2, 0.3], dtype=float)
    kwargs = dict(
        tau=tau,
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=[sza, 20.0, 0.0],
        fbeam=2.0,
        albedo=0.3,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_fluxes=True,
        )
    )
    base = solver.forward(**kwargs)
    total = solver.forward(**kwargs, include_fo=True)
    correction = {
        field: getattr(total, field) - getattr(base, field)
        for field in ("flux_up", "flux_down", "flux_net", "flux_mean")
    }
    total_reference = _solar_clear_surface_reference(tau, sza=sza, fbeam=2.0, albedo=0.3)
    correction_reference = {
        field: total_reference[field][np.newaxis, :] - getattr(base, field)
        for field in ("flux_up", "flux_down", "flux_net", "flux_mean")
    }
    return correction, correction_reference


def _case_solar_isotropic_scattering_runs() -> tuple[Any, Any]:
    sza = 30.0
    tau = np.array([0.12, 0.18], dtype=float)
    ssa = np.array([0.05, 0.08], dtype=float)
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_fluxes=True,
        )
    )
    kwargs = dict(
        tau=tau,
        ssa=ssa,
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=[sza, 20.0, 0.0],
        fbeam=1.7,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    base = solver.forward(**kwargs)
    total = solver.forward(**kwargs, include_fo=True, fo_n_moments=1)
    return base, total


def _rayleigh_scatter_term(
    *,
    ssa: np.ndarray,
    angles: np.ndarray,
    rayleigh_delta: float = 1.0,
) -> np.ndarray:
    depol = (2.0 * (1.0 - rayleigh_delta)) / (rayleigh_delta + 2.0)
    aerosol_moments = np.zeros((2, 3, 1), dtype=float)
    aerosol_moments[:, 0, :] = 1.0
    return build_solar_fo_scatter_term(
        ssa=ssa,
        depol=depol,
        rayleigh_fraction=np.ones_like(ssa),
        aerosol_fraction=np.zeros(ssa.shape + (1,), dtype=float),
        aerosol_moments=aerosol_moments,
        aerosol_interp_fraction=0.0,
        angles=angles,
        delta_m_truncation_factor=np.zeros_like(ssa),
    )


def _case_solar_rayleigh_scattering_runs() -> tuple[Any, Any]:
    sza = 30.0
    tau = np.array([0.12, 0.18], dtype=float)
    ssa = np.array([0.05, 0.08], dtype=float)
    angles = np.array([[sza, 20.0, 0.0], [sza, 50.0, 120.0]], dtype=float)
    scatter = _rayleigh_scatter_term(ssa=ssa, angles=angles)
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_fluxes=True,
        )
    )
    kwargs = dict(
        tau=tau,
        ssa=ssa,
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=angles,
        fbeam=1.7,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    base = solver.forward(**kwargs)
    total = solver.forward(**kwargs, include_fo=True, fo_scatter_term=scatter)
    return base, total


def _case_zero_thermal() -> tuple[Any, dict[str, np.ndarray]]:
    result = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="thermal",
            downwelling=True,
            output_fluxes=True,
        )
    ).forward(
        tau=np.array([0.1, 0.2], dtype=float),
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=30.0,
        planck=np.zeros(3, dtype=float),
        surface_planck=0.0,
        emissivity=1.0,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    zero = np.zeros(3, dtype=float)
    return result, {"flux_up": zero, "flux_down": zero, "flux_net": zero, "flux_mean": zero}


def _case_thermal_surface_total() -> tuple[Any, dict[str, np.ndarray]]:
    tau = np.array([0.2, 0.3], dtype=float)
    surface_planck = 1.2
    emissivity = 0.8
    result = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="thermal",
            plane_parallel=True,
            downwelling=True,
            output_fluxes=True,
        )
    ).forward(
        tau=tau,
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=30.0,
        planck=np.zeros(3, dtype=float),
        surface_planck=surface_planck,
        emissivity=emissivity,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
        include_fo=True,
    )
    return result, _thermal_surface_reference(
        tau,
        surface_radiance=surface_planck * emissivity,
    )


def _case_thermal_fo_surface_correction() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tau = np.array([0.2, 0.3], dtype=float)
    surface_planck = 1.2
    emissivity = 0.8
    kwargs = dict(
        tau=tau,
        ssa=np.zeros(2, dtype=float),
        g=np.zeros(2, dtype=float),
        z=np.array([2.0, 1.0, 0.0], dtype=float),
        angles=30.0,
        planck=np.zeros(3, dtype=float),
        surface_planck=surface_planck,
        emissivity=emissivity,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros(2, dtype=float),
    )
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=2,
            mode="thermal",
            plane_parallel=True,
            downwelling=True,
            output_fluxes=True,
        )
    )
    base = solver.forward(**kwargs)
    total = solver.forward(**kwargs, include_fo=True)
    correction = {
        field: getattr(total, field) - getattr(base, field)
        for field in ("flux_up", "flux_down", "flux_net", "flux_mean")
    }
    total_reference = _thermal_surface_reference(
        tau,
        surface_radiance=surface_planck * emissivity,
    )
    correction_reference = {
        field: total_reference[field][np.newaxis, :] - getattr(base, field)
        for field in ("flux_up", "flux_down", "flux_net", "flux_mean")
    }
    return correction, correction_reference


def _compare(
    *,
    source: str,
    case: str,
    actual: Any,
    reference: dict[str, Any],
    fields: tuple[str, ...] = ("flux_up", "flux_down", "flux_net", "flux_mean"),
    weak: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        actual_field = _to_numpy(_flux_field(actual, field))[0]
        reference_field = _to_numpy(reference[field])
        if reference_field.ndim > 1:
            reference_field = reference_field.reshape(-1, reference_field.shape[-1])[0]
        if actual_field.shape != reference_field.shape:
            raise ValueError(
                f"{source} {case} {field} shape mismatch: "
                f"py2sess has {actual_field.shape}, reference has {reference_field.shape}"
            )
        diff = np.abs(actual_field - reference_field)
        scale = np.maximum(np.abs(reference_field), 1.0)
        rows.append(
            {
                "source": source,
                "case": case,
                "field": field,
                "max_abs_diff": float(np.max(diff)),
                "max_rel_diff": float(np.max(diff / scale)),
                "weak_diagnostic": weak,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinetics-flxout", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--components",
        action="store_true",
        help="Also report FO increment checks used to validate the added FO flux term.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    solar, solar_reference = _case_absorbing_solar()
    rows.extend(
        _compare(source="analytic", case="absorbing_solar", actual=solar, reference=solar_reference)
    )
    solar_surface_total, solar_surface_reference = _case_solar_surface_total()
    rows.extend(
        _compare(
            source="analytic",
            case="solar_surface_total",
            actual=solar_surface_total,
            reference=solar_surface_reference,
        )
    )
    thermal, thermal_reference = _case_zero_thermal()
    rows.extend(
        _compare(
            source="analytic", case="zero_thermal", actual=thermal, reference=thermal_reference
        )
    )
    thermal_surface_total, thermal_surface_reference = _case_thermal_surface_total()
    rows.extend(
        _compare(
            source="analytic",
            case="thermal_surface_total",
            actual=thermal_surface_total,
            reference=thermal_surface_reference,
        )
    )
    solar_scatter_base, solar_scatter_total = _case_solar_isotropic_scattering_runs()
    solar_rayleigh_base, solar_rayleigh_total = _case_solar_rayleigh_scattering_runs()
    if args.components:
        solar_fo, solar_fo_reference = _case_solar_fo_surface_correction()
        rows.extend(
            _compare(
                source="analytic_component",
                case="solar_fo_surface_correction",
                actual=solar_fo,
                reference=solar_fo_reference,
            )
        )
        thermal_fo, thermal_fo_reference = _case_thermal_fo_surface_correction()
        rows.extend(
            _compare(
                source="analytic_component",
                case="thermal_fo_surface_correction",
                actual=thermal_fo,
                reference=thermal_fo_reference,
            )
        )

    if find_spec("pydisort") is not None:
        pydisort_reference = run_pydisort_absorbing_solar_flux(
            np.array([0.1, 0.2], dtype=float),
            mu0=math.cos(math.radians(30.0)),
            fbeam=float(np.pi),
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="absorbing_solar",
                actual=solar,
                reference=pydisort_reference,
            )
        )
        pydisort_surface_reference = run_pydisort_absorbing_solar_flux(
            np.array([0.2, 0.3], dtype=float),
            mu0=math.cos(math.radians(30.0)),
            fbeam=2.0,
            albedo=0.3,
            nstr=32,
            nmom=32,
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="solar_surface_total",
                actual=solar_surface_total,
                reference=pydisort_surface_reference,
            )
        )
        pydisort_scatter_reference = run_pydisort_solar_flux(
            np.array([0.12, 0.18], dtype=float),
            ssa=np.array([0.05, 0.08], dtype=float),
            phase_moments=np.zeros(16, dtype=float),
            mu0=math.cos(math.radians(30.0)),
            fbeam=1.7,
            albedo=0.0,
            nstr=16,
            nmom=16,
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="solar_isotropic_scattering_2s",
                actual=solar_scatter_base,
                reference=pydisort_scatter_reference,
                weak=True,
            )
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="solar_isotropic_scattering_total",
                actual=solar_scatter_total,
                reference=pydisort_scatter_reference,
                weak=True,
            )
        )
        pydisort_rayleigh_reference = run_pydisort_solar_flux(
            np.array([0.12, 0.18], dtype=float),
            ssa=np.array([0.05, 0.08], dtype=float),
            phase_moments=rayleigh_phase_moments(16),
            mu0=math.cos(math.radians(30.0)),
            fbeam=1.7,
            albedo=0.0,
            nstr=16,
            nmom=16,
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="solar_rayleigh_scattering_2s",
                actual=solar_rayleigh_base,
                reference=pydisort_rayleigh_reference,
                weak=True,
            )
        )
        rows.extend(
            _compare(
                source="pydisort",
                case="solar_rayleigh_scattering_total",
                actual=solar_rayleigh_total,
                reference=pydisort_rayleigh_reference,
                weak=True,
            )
        )

    if args.kinetics_flxout is not None:
        kinetics_reference = kinetics_flux_to_py2sess(
            read_kinetics_flux_table(args.kinetics_flxout)
        )
        rows.extend(
            _compare(
                source="KINETICS",
                case=args.kinetics_flxout.stem,
                actual=solar,
                reference=kinetics_reference,
                weak=True,
            )
        )

    for row in rows:
        print(
            f"{row['source']:9s} {row['case']:18s} {row['field']:10s} "
            f"max_abs={row['max_abs_diff']:.6e} max_rel={row['max_rel_diff']:.6e} "
            f"weak={row['weak_diagnostic']}"
        )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
