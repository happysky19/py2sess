#!/usr/bin/env python3
"""Compare py2sess level fluxes against DISOTEST flux benchmark cases."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.optical.brdf_solar_obs import DISORT_HAPKE_IDX
from py2sess.benchmarks.flux_references import (
    rayleigh_phase_moments,
    run_pydisort_absorbing_solar_flux,
    run_pydisort_solar_flux,
)
from py2sess.optical.planck import planck_radiance_wavenumber_band
from py2sess.optical.phase import build_solar_fo_scatter_term

FIELDS = ("flux_up", "flux_down", "flux_net")
SECTION6_STREAM = 0.5
PUBLIC_DEFAULT_STREAM = 1.0 / math.sqrt(3.0)
DEFAULT_DISORT_STREAM = PUBLIC_DEFAULT_STREAM
PAPER_ROUNDOFF_ATOL = 1.0e-9

# DISORT c_getmom tabulated moments before division by (2l + 1).
HAZE_L_MOMENTS = (
    2.41260,
    3.23047,
    3.37296,
    3.23150,
    2.89350,
    2.49594,
    2.11361,
    1.74812,
    1.44692,
    1.17714,
    0.96643,
    0.78237,
    0.64114,
    0.51966,
    0.42563,
    0.34688,
    0.28351,
    0.23317,
    0.18963,
    0.15788,
    0.12739,
    0.10762,
    0.08597,
    0.07381,
    0.05828,
    0.05089,
    0.03971,
    0.03524,
    0.02720,
    0.02451,
    0.01874,
    0.01711,
)

CLOUD_C1_MOMENTS = (
    2.544,
    3.883,
    4.568,
    5.235,
    5.887,
    6.457,
    7.177,
    7.859,
    8.494,
    9.286,
    9.856,
    10.615,
    11.229,
    11.851,
    12.503,
    13.058,
    13.626,
    14.209,
    14.660,
    15.231,
    15.641,
    16.126,
    16.539,
    16.934,
    17.325,
    17.673,
    17.999,
    18.329,
    18.588,
    18.885,
    19.103,
    19.345,
    19.537,
    19.721,
    19.884,
    20.024,
    20.145,
    20.251,
    20.330,
    20.401,
    20.444,
    20.477,
    20.489,
    20.483,
    20.467,
    20.427,
    20.382,
    20.310,
)


@dataclass(frozen=True)
class DisotestFluxCase:
    name: str
    tau: tuple[float, ...]
    ssa: tuple[float, ...]
    g: tuple[float, ...]
    mu0: float
    fbeam: float
    albedo: float
    phase: str
    nstr: int
    nmom: int
    fo_n_moments: int
    fisot: float = 0.0
    benchmark_direct_down: tuple[float, ...] | None = None
    benchmark_diffuse_down: tuple[float, ...] | None = None
    benchmark_flux_up: tuple[float, ...] | None = None
    pydisort_tau: tuple[float, ...] | None = None
    py2sess_tau: tuple[float, ...] | None = None
    mode: str = "solar"
    thermal_temperature: tuple[float, ...] | None = None
    thermal_wavenumber_band: tuple[float, float] | None = None
    surface_planck: float = 0.0
    emissivity: float = 1.0
    delta_m_truncation_factor: tuple[float, ...] | None = None
    compare_level_indices: tuple[int, ...] | None = None
    level_names: tuple[str, ...] | None = None
    surface_model: str = "lambertian"
    unsupported_category: str | None = None
    unsupported_reason: str | None = None


def _case_brdf(case: DisotestFluxCase) -> dict[str, object] | None:
    if case.surface_model != "hapke":
        return None
    return {
        "kernel_specs": [
            {
                "which_brdf": DISORT_HAPKE_IDX,
                "factor": 1.0,
                "nstreams_brdf": case.nstr,
            }
        ]
    }


def _layer_values(name: str, values: tuple[float, ...], nlyr: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return np.full(nlyr, float(array[0]), dtype=float)
    if array.shape == (nlyr,):
        return array
    raise ValueError(f"{name} must be scalar or have one value per layer")


def _hg_phase_moments(g: float, nmom: int) -> np.ndarray:
    return np.array([g**order for order in range(1, nmom + 1)], dtype=float)


def _disort_tabulated_phase_moments(raw_moments: tuple[float, ...], nmom: int) -> np.ndarray:
    moments = np.zeros(int(nmom), dtype=float)
    for order, value in enumerate(raw_moments[:nmom], start=1):
        moments[order - 1] = value / float(2 * order + 1)
    return moments


def _tabulated_moments(phase: str, nmom: int) -> np.ndarray:
    if phase == "haze_l":
        return _disort_tabulated_phase_moments(HAZE_L_MOMENTS, nmom)
    if phase == "cloud_c1":
        return _disort_tabulated_phase_moments(CLOUD_C1_MOMENTS, nmom)
    raise ValueError(f"unsupported tabulated phase {phase!r}")


def _tabulated_two_stream_inputs(
    phase: str,
    nstr: int,
) -> tuple[float, float]:
    moments = _tabulated_moments(phase, nstr)
    return float(moments[0]), float(moments[1])


def _pydisort_nmom(case: DisotestFluxCase) -> int:
    return max(case.nmom, case.nstr)


def _phase_moments(case: DisotestFluxCase) -> np.ndarray:
    nlyr = len(case.pydisort_tau or case.tau)
    nmom = _pydisort_nmom(case)
    if case.phase == "isotropic":
        return np.zeros((nlyr, nmom), dtype=float)
    if case.phase == "rayleigh":
        return np.broadcast_to(rayleigh_phase_moments(nmom), (nlyr, nmom)).copy()
    if case.phase == "hg":
        g = _layer_values("g", case.g, nlyr)
        return np.vstack([_hg_phase_moments(float(value), nmom) for value in g])
    if case.phase in {"haze_l", "cloud_c1"}:
        return np.broadcast_to(_tabulated_moments(case.phase, nmom), (nlyr, nmom)).copy()
    raise ValueError(f"unsupported phase {case.phase!r}")


def _rayleigh_scatter_term(ssa: np.ndarray, angles: np.ndarray) -> np.ndarray:
    aerosol_moments = np.zeros((2, 3, 1), dtype=float)
    aerosol_moments[:, 0, :] = 1.0
    return build_solar_fo_scatter_term(
        ssa=ssa,
        depol=0.0,
        rayleigh_fraction=np.ones_like(ssa),
        aerosol_fraction=np.zeros(ssa.shape + (1,), dtype=float),
        aerosol_moments=aerosol_moments,
        aerosol_interp_fraction=0.0,
        angles=angles,
        delta_m_truncation_factor=np.zeros_like(ssa),
    )


def _benchmark_flux(case: DisotestFluxCase) -> dict[str, np.ndarray]:
    if (
        case.benchmark_direct_down is None
        or case.benchmark_diffuse_down is None
        or case.benchmark_flux_up is None
    ):
        if case.unsupported_reason is not None:
            values = np.array([math.nan], dtype=float)
            return {"flux_up": values, "flux_down": values, "flux_net": values}
        return _run_pydisort(case)
    direct_down = np.asarray(case.benchmark_direct_down, dtype=float)
    diffuse_down = np.asarray(case.benchmark_diffuse_down, dtype=float)
    flux_up = np.asarray(case.benchmark_flux_up, dtype=float)
    flux_down = direct_down + diffuse_down
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
    }


def _select_case_levels(
    case: DisotestFluxCase,
    flux: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if case.compare_level_indices is None:
        return flux
    indices = np.asarray(case.compare_level_indices, dtype=int)
    return {field: np.asarray(values, dtype=float)[indices] for field, values in flux.items()}


def _level_names(case: DisotestFluxCase, nlevels: int) -> tuple[str, ...]:
    if case.level_names is not None:
        if len(case.level_names) != nlevels:
            raise ValueError(f"{case.name} level_names length does not match benchmark levels")
        return case.level_names
    return ("TOA", "BOA") if nlevels == 2 else tuple(f"L{index}" for index in range(nlevels))


def _section6_flux(
    direct_down: tuple[float, ...],
    diffuse_down: tuple[float, ...],
    flux_up: tuple[float, ...],
    *,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    direct = scale * np.asarray(direct_down, dtype=float)
    diffuse = scale * np.asarray(diffuse_down, dtype=float)
    up = scale * np.asarray(flux_up, dtype=float)
    down = direct + diffuse
    return {
        "flux_up": up,
        "flux_down": down,
        "flux_net": up - down,
    }


VIJAY_SECTION6_2SESS_FLUXES = {
    "DISOTEST 1a isotropic beam": _section6_flux(
        (1.0, 0.731616),
        (0.0, 2.61349e-2),
        (2.62195e-2, 0.0),
        scale=math.pi,
    ),
    "DISOTEST 1b isotropic beam": _section6_flux(
        (1.0, 0.731616),
        (3.72944e-9, 1.33981e-1),
        (1.34404e-1, 3.72980e-9),
        scale=math.pi,
    ),
    "DISOTEST 1d isotropic beam": _section6_flux(
        (1.0, 0.0),
        (0.0, 1.70407e-26),
        (8.95531e-2, -1.40874e-43),
        scale=math.pi,
    ),
    "DISOTEST 1e isotropic beam": _section6_flux(
        (1.0, 0.0),
        (-9.57183e-11, 1.81818e-2),
        (9.81818e-1, -9.57183e-11),
        scale=math.pi,
    ),
    "DISOTEST 2a Rayleigh beam": _section6_flux(
        (2.52716e-1, 2.10311e-2),
        (0.0, 4.80132e-2),
        (5.28552e-2, 0.0),
    ),
    "DISOTEST 2b Rayleigh beam": _section6_flux(
        (2.52716e-1, 2.10311e-2),
        (-3.56301e-9, 1.11188e-1),
        (1.20497e-1, -3.56301e-9),
    ),
    "DISOTEST 2c Rayleigh beam": _section6_flux(
        (2.52716e-1, 2.56077e-28),
        (0.0, 6.03691e-5),
        (6.30988e-2, 0.0),
    ),
    "DISOTEST 2d Rayleigh beam": _section6_flux(
        (2.52716e-1, 2.56077e-28),
        (-8.03962e-10, 2.49680e-2),
        (2.27748e-1, -8.04319e-10),
    ),
    "DISOTEST 3a HG beam": _section6_flux(
        (1.0, 3.67879e-1),
        (-3.25192e-10, 5.39822e-1),
        (9.22982e-2, -3.24835e-10),
        scale=math.pi,
    ),
    "DISOTEST 3b HG beam": _section6_flux(
        (1.0, 3.35463e-4),
        (-1.53578e-10, 4.39635e-1),
        (5.60029e-1, -1.52864e-10),
        scale=math.pi,
    ),
    "DISOTEST 6a clear beam": _section6_flux(
        (100.0, 100.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ),
    "DISOTEST 6b absorbing beam": _section6_flux(
        (100.0, 3.67879e1, 1.35335e1),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    "DISOTEST 6c absorbing surface": _section6_flux(
        (100.0, 3.67879e1, 1.35335e1),
        (0.0, 0.0, 0.0),
        (9.15782e-1, 2.48935, 6.76676),
    ),
    "DISOTEST 7a thermal internal": _section6_flux(
        (0.0, 0.0),
        (0.0, 1.3476e2),
        (9.4134e1, 6.9757e-16),
    ),
    "DISOTEST 7b thermal thick": _section6_flux(
        (0.0, 0.0),
        (0.0, 1.8623e-5),
        (7.8032e-7, 1.3305e-21),
    ),
}


def _vijay_section6_flux(case: DisotestFluxCase) -> dict[str, np.ndarray]:
    flux = VIJAY_SECTION6_2SESS_FLUXES.get(case.name)
    if flux is None:
        benchmark = _benchmark_flux(case)
        return {field: np.full_like(benchmark[field], np.nan) for field in FIELDS}
    return {field: values.copy() for field, values in flux.items()}


def _as_level_flux(result: object) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(result, field), dtype=float)[0] for field in FIELDS}


def _thermal_planck(case: DisotestFluxCase) -> np.ndarray:
    if case.thermal_temperature is None or case.thermal_wavenumber_band is None:
        raise ValueError(f"{case.name} is missing thermal source metadata")
    low, high = case.thermal_wavenumber_band
    return np.asarray(
        planck_radiance_wavenumber_band(
            np.asarray(case.thermal_temperature, dtype=float),
            float(low),
            float(high),
        ),
        dtype=float,
    )


def _run_py2sess(
    case: DisotestFluxCase,
    *,
    stream: float = DEFAULT_DISORT_STREAM,
    fo_flux_n_mu: int = 8,
    backend: str = "numpy",
) -> dict[str, np.ndarray]:
    if case.unsupported_reason is not None:
        benchmark = _benchmark_flux(case)
        return {field: np.full_like(benchmark[field], np.nan) for field in FIELDS}

    tau = np.asarray(case.py2sess_tau or case.tau, dtype=float)
    ssa = _layer_values("ssa", case.ssa, tau.size)
    g = _layer_values("g", case.g, tau.size)
    if case.mode == "thermal":
        result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=tau.size,
                mode="thermal",
                backend=backend,
                plane_parallel=True,
                delta_scaling=True,
                downwelling=True,
                output_fluxes=True,
                fo_flux_n_mu=fo_flux_n_mu,
            )
        ).forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.arange(tau.size, -1, -1, dtype=float),
            angles=0.0,
            planck=_thermal_planck(case),
            surface_planck=case.surface_planck,
            emissivity=case.emissivity,
            albedo=case.albedo,
            stream=stream,
            include_fo=True,
        )
        return _select_case_levels(case, _as_level_flux(result))
    if case.mode != "solar":
        raise ValueError(f"unsupported mode {case.mode!r}")

    sza = math.degrees(math.acos(case.mu0))
    angles = np.array([[sza, 20.0, 0.0], [sza, 50.0, 120.0]], dtype=float)
    kwargs = {
        "tau": tau,
        "ssa": ssa,
        "g": g,
        "z": np.arange(tau.size, -1, -1, dtype=float),
        "angles": angles,
        "fbeam": case.fbeam,
        "fisot": case.fisot,
        "albedo": case.albedo,
        "stream": stream,
        "include_fo": case.fbeam != 0.0,
        "fo_n_moments": case.fo_n_moments,
    }
    brdf = _case_brdf(case)
    if brdf is not None:
        kwargs["brdf"] = brdf
    if case.delta_m_truncation_factor is not None:
        kwargs["delta_m_truncation_factor"] = _layer_values(
            "delta_m_truncation_factor",
            case.delta_m_truncation_factor,
            tau.size,
        )
    # Keep py2sess default delta-M for forward-peaked HG cases.
    elif case.phase != "hg":
        kwargs["delta_m_truncation_factor"] = np.zeros_like(tau)
    if case.phase == "rayleigh":
        kwargs["fo_scatter_term"] = _rayleigh_scatter_term(ssa, angles)

    result = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tau.size,
            mode="solar",
            backend=backend,
            plane_parallel=True,
            delta_scaling=True,
            downwelling=True,
            output_fluxes=True,
            brdf_surface=brdf is not None,
            fo_flux_n_mu=fo_flux_n_mu,
        )
    ).forward(**kwargs)
    return _select_case_levels(case, _as_level_flux(result))


def _run_pydisort(case: DisotestFluxCase) -> dict[str, np.ndarray]:
    if case.surface_model != "lambertian" or case.fisot != 0.0:
        benchmark = _benchmark_flux(case)
        return {field: values.copy() for field, values in benchmark.items()}
    if case.unsupported_reason is not None:
        benchmark = _benchmark_flux(case)
        return {field: np.full_like(benchmark[field], np.nan) for field in FIELDS}

    if case.mode == "thermal":
        benchmark = _benchmark_flux(case)
        return {field: np.full_like(benchmark[field], np.nan) for field in FIELDS}
    if case.mode != "solar":
        raise ValueError(f"unsupported mode {case.mode!r}")

    tau = np.asarray(case.pydisort_tau or case.tau, dtype=float)
    if case.phase == "absorbing":
        result = run_pydisort_absorbing_solar_flux(
            tau,
            mu0=case.mu0,
            fbeam=case.fbeam,
            albedo=case.albedo,
            nstr=case.nstr,
            nmom=max(case.nmom, case.nstr),
        )
    else:
        result = run_pydisort_solar_flux(
            tau,
            ssa=_layer_values("ssa", case.ssa, tau.size),
            phase_moments=_phase_moments(case),
            mu0=case.mu0,
            fbeam=case.fbeam,
            albedo=case.albedo,
            nstr=case.nstr,
            nmom=_pydisort_nmom(case),
        )
    return _select_case_levels(
        case,
        {
            field: np.asarray(result[field], dtype=float).reshape(-1, result[field].shape[-1])[0]
            for field in FIELDS
        },
    )


def _roundoff_zero(value: float, *, atol: float = PAPER_ROUNDOFF_ATOL) -> float:
    return 0.0 if abs(value) <= atol else value


def _rel_percent(actual: float, benchmark: float) -> float:
    actual = _roundoff_zero(actual)
    benchmark = _roundoff_zero(benchmark)
    if benchmark == 0.0:
        return 0.0 if actual == 0.0 else math.nan
    return 100.0 * (actual - benchmark) / benchmark


def _stream_mode(stream: float) -> str:
    if math.isclose(stream, SECTION6_STREAM, rel_tol=0.0, abs_tol=1.0e-12):
        return "section6"
    if math.isclose(stream, PUBLIC_DEFAULT_STREAM, rel_tol=0.0, abs_tol=1.0e-12):
        return "public-default"
    return "custom"


def _rows(
    case: DisotestFluxCase,
    *,
    stream: float = DEFAULT_DISORT_STREAM,
    fo_flux_n_mu: int = 8,
    backend: str = "numpy",
) -> list[dict[str, object]]:
    benchmark = _benchmark_flux(case)
    pydisort = _run_pydisort(case)
    py2sess = _run_py2sess(
        case,
        stream=stream,
        fo_flux_n_mu=fo_flux_n_mu,
        backend=backend,
    )
    stream_mode = _stream_mode(stream)
    rows: list[dict[str, object]] = []
    levels = _level_names(case, len(benchmark["flux_up"]))
    for field in FIELDS:
        for level_index, level_name in enumerate(levels):
            bench = float(benchmark[field][level_index])
            dis = float(pydisort[field][level_index])
            py = float(py2sess[field][level_index])
            bench = _roundoff_zero(bench)
            dis = _roundoff_zero(dis)
            py = _roundoff_zero(py)
            rows.append(
                {
                    "case": case.name,
                    "status": "unsupported" if case.unsupported_reason else "run",
                    "unsupported_category": case.unsupported_category or "",
                    "surface_model": case.surface_model,
                    "reason": "" if case.unsupported_reason is None else case.unsupported_reason,
                    "stream": stream,
                    "stream_mode": stream_mode,
                    "fo_flux_n_mu": fo_flux_n_mu,
                    "py2sess_backend": backend,
                    "field": field,
                    "level": level_name,
                    "benchmark": bench,
                    "pydisort": dis,
                    "py2sess": py,
                    "pydisort_abs_err": abs(dis - bench),
                    "py2sess_abs_err": abs(py - bench),
                    "pydisort_rel_percent": _rel_percent(dis, bench),
                    "py2sess_rel_percent": _rel_percent(py, bench),
                }
            )
    return rows


def _case_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    finite_py = [
        row
        for row in rows
        if isinstance(row["py2sess_rel_percent"], float)
        and math.isfinite(row["py2sess_rel_percent"])
    ]
    pydisort_max = max(float(row["pydisort_abs_err"]) for row in rows)
    py2sess_max = max(float(row["py2sess_abs_err"]) for row in rows)
    if not finite_py:
        return {
            "case": rows[0]["case"],
            "status": rows[0].get("status", "run"),
            "reason": rows[0].get("reason", ""),
            "stream": rows[0].get("stream", math.nan),
            "stream_mode": rows[0].get("stream_mode", ""),
            "fo_flux_n_mu": rows[0].get("fo_flux_n_mu", math.nan),
            "py2sess_backend": rows[0].get("py2sess_backend", ""),
            "pydisort_max_abs": pydisort_max,
            "py2sess_max_abs": py2sess_max,
            "worst_field": rows[0]["field"],
            "worst_level": rows[0]["level"],
            "benchmark": rows[0]["benchmark"],
            "pydisort": rows[0]["pydisort"],
            "py2sess": rows[0]["py2sess"],
            "py2sess_rel_percent": math.nan,
        }
    worst = max(finite_py, key=lambda row: abs(float(row["py2sess_rel_percent"])))
    return {
        "case": rows[0]["case"],
        "status": rows[0].get("status", "run"),
        "reason": rows[0].get("reason", ""),
        "stream": rows[0].get("stream", math.nan),
        "stream_mode": rows[0].get("stream_mode", ""),
        "fo_flux_n_mu": rows[0].get("fo_flux_n_mu", math.nan),
        "py2sess_backend": rows[0].get("py2sess_backend", ""),
        "pydisort_max_abs": pydisort_max,
        "py2sess_max_abs": py2sess_max,
        "worst_field": worst["field"],
        "worst_level": worst["level"],
        "benchmark": worst["benchmark"],
        "pydisort": worst["pydisort"],
        "py2sess": worst["py2sess"],
        "py2sess_rel_percent": worst["py2sess_rel_percent"],
    }


def _comparable_cases() -> list[DisotestFluxCase]:
    isotropic = [
        ("1a", 0.03125, 0.2, (3.14159, 2.29844), (0.0, 7.94108e-2), (7.99451e-2, 0.0)),
        ("1b", 0.03125, 1.0, (3.14159, 2.29844), (0.0, 4.20233e-1), (4.22922e-1, 0.0)),
        ("1d", 32.0, 0.2, (3.14159, 0.0), (0.0, 0.0), (2.59686e-1, 0.0)),
        ("1e", 32.0, 1.0, (3.14159, 0.0), (0.0, 6.76954e-2), (3.07390, 0.0)),
    ]
    rayleigh = [
        ("2a", 0.2, 0.5, (2.52716e-1, 2.10311e-2), (0.0, 4.41791e-2), (5.35063e-2, 0.0)),
        ("2b", 0.2, 1.0, (2.52716e-1, 2.10311e-2), (0.0, 1.06123e-1), (1.25561e-1, 0.0)),
        ("2c", 5.0, 0.5, (2.52716e-1, 2.56077e-28), (0.0, 2.51683e-4), (6.24730e-2, 0.0)),
        ("2d", 5.0, 1.0, (2.52716e-1, 0.0), (0.0, 2.68008e-2), (2.25915e-1, 0.0)),
    ]
    hg = [
        ("3a", 1.0, (3.14159, 1.15573), (0.0, 1.73849), (2.47374e-1, 0.0)),
        ("3b", 8.0, (3.14159, 1.05389e-3), (0.0, 1.54958), (1.59096, 0.0)),
    ]
    cases: list[DisotestFluxCase] = []
    for suffix, tau, ssa, direct_down, diffuse_down, flux_up in isotropic:
        cases.append(
            DisotestFluxCase(
                name=f"DISOTEST {suffix} isotropic beam",
                tau=(tau,),
                ssa=(ssa,),
                g=(0.0,),
                mu0=0.1,
                fbeam=math.pi / 0.1,
                albedo=0.0,
                phase="isotropic",
                nstr=16,
                nmom=16,
                fo_n_moments=1,
                benchmark_direct_down=direct_down,
                benchmark_diffuse_down=diffuse_down,
                benchmark_flux_up=flux_up,
            )
        )
    for suffix, tau, ssa, direct_down, diffuse_down, flux_up in rayleigh:
        cases.append(
            DisotestFluxCase(
                name=f"DISOTEST {suffix} Rayleigh beam",
                tau=(tau,),
                ssa=(ssa,),
                g=(0.0,),
                mu0=0.080442,
                fbeam=math.pi,
                albedo=0.0,
                phase="rayleigh",
                nstr=16,
                nmom=16,
                fo_n_moments=64,
                benchmark_direct_down=direct_down,
                benchmark_diffuse_down=diffuse_down,
                benchmark_flux_up=flux_up,
            )
        )
    for suffix, tau, direct_down, diffuse_down, flux_up in hg:
        cases.append(
            DisotestFluxCase(
                name=f"DISOTEST {suffix} HG beam",
                tau=(tau,),
                ssa=(1.0,),
                g=(0.75,),
                mu0=1.0,
                fbeam=math.pi,
                albedo=0.0,
                phase="hg",
                nstr=16,
                nmom=32,
                fo_n_moments=64,
                benchmark_direct_down=direct_down,
                benchmark_diffuse_down=diffuse_down,
                benchmark_flux_up=flux_up,
            )
        )
    cases.extend(
        [
            DisotestFluxCase(
                name="DISOTEST 6a clear beam",
                tau=(0.0,),
                ssa=(0.0,),
                g=(0.0,),
                mu0=0.5,
                fbeam=200.0,
                albedo=0.0,
                phase="absorbing",
                nstr=16,
                nmom=2,
                fo_n_moments=1,
                benchmark_direct_down=(100.0, 100.0),
                benchmark_diffuse_down=(0.0, 0.0),
                benchmark_flux_up=(0.0, 0.0),
            ),
            DisotestFluxCase(
                name="DISOTEST 6b absorbing beam",
                tau=(1.0,),
                ssa=(0.0,),
                g=(0.0,),
                mu0=0.5,
                fbeam=200.0,
                albedo=0.0,
                phase="absorbing",
                nstr=16,
                nmom=2,
                fo_n_moments=1,
                benchmark_direct_down=(100.0, 3.67879e1, 1.35335e1),
                benchmark_diffuse_down=(0.0, 0.0, 0.0),
                benchmark_flux_up=(0.0, 0.0, 0.0),
                pydisort_tau=(0.5, 0.5),
                py2sess_tau=(0.5, 0.5),
            ),
            DisotestFluxCase(
                name="DISOTEST 6c absorbing surface",
                tau=(1.0,),
                ssa=(0.0,),
                g=(0.0,),
                mu0=0.5,
                fbeam=200.0,
                albedo=0.5,
                phase="absorbing",
                nstr=16,
                nmom=2,
                fo_n_moments=1,
                benchmark_direct_down=(100.0, 3.67879e1, 1.35335e1),
                benchmark_diffuse_down=(0.0, 0.0, 0.0),
                benchmark_flux_up=(1.48450, 2.99914, 6.76676),
                pydisort_tau=(0.5, 0.5),
                py2sess_tau=(0.5, 0.5),
            ),
        ]
    )
    return cases


def _top_isotropic_disotest_cases() -> list[DisotestFluxCase]:
    return [
        DisotestFluxCase(
            name="DISOTEST 1c isotropic top illumination",
            tau=(0.03125,),
            ssa=(0.99,),
            g=(0.0,),
            mu0=1.0,
            fbeam=0.0,
            fisot=1.0,
            albedo=0.0,
            phase="isotropic",
            nstr=16,
            nmom=16,
            fo_n_moments=1,
            benchmark_direct_down=(0.0, 0.0),
            benchmark_diffuse_down=(3.14159, 3.04897),
            benchmark_flux_up=(9.06556e-2, 0.0),
        ),
        DisotestFluxCase(
            name="DISOTEST 1f isotropic top illumination",
            tau=(32.0,),
            ssa=(0.99,),
            g=(0.0,),
            mu0=1.0,
            fbeam=0.0,
            fisot=1.0,
            albedo=0.0,
            phase="isotropic",
            nstr=16,
            nmom=16,
            fo_n_moments=1,
            benchmark_direct_down=(0.0, 0.0),
            benchmark_diffuse_down=(3.14159, 4.60048e-3),
            benchmark_flux_up=(2.49618, 0.0),
        ),
        DisotestFluxCase(
            name="DISOTEST 8a two-layer isotropic source",
            tau=(0.25, 0.25),
            ssa=(0.5, 0.3),
            g=(0.0, 0.0),
            mu0=1.0,
            fbeam=0.0,
            fisot=1.0 / math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=8,
            nmom=8,
            fo_n_moments=1,
            benchmark_direct_down=(0.0, 0.0, 0.0),
            benchmark_diffuse_down=(1.0, 7.22235e-1, 5.13132e-1),
            benchmark_flux_up=(9.29633e-2, 2.78952e-2, 0.0),
        ),
        DisotestFluxCase(
            name="DISOTEST 8b two-layer conservative source",
            tau=(0.25, 0.25),
            ssa=(0.8, 0.95),
            g=(0.0, 0.0),
            mu0=1.0,
            fbeam=0.0,
            fisot=1.0 / math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=8,
            nmom=8,
            fo_n_moments=1,
            benchmark_direct_down=(0.0, 0.0, 0.0),
            benchmark_diffuse_down=(1.0, 7.95332e-1, 6.50417e-1),
            benchmark_flux_up=(2.25136e-1, 1.26349e-1, 0.0),
        ),
        DisotestFluxCase(
            name="DISOTEST 8c two-layer thick source",
            tau=(1.0, 2.0),
            ssa=(0.8, 0.95),
            g=(0.0, 0.0),
            mu0=1.0,
            fbeam=0.0,
            fisot=1.0 / math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=8,
            nmom=8,
            fo_n_moments=1,
            benchmark_direct_down=(0.0, 0.0, 0.0),
            benchmark_diffuse_down=(1.0, 4.86157e-1, 1.59984e-1),
            benchmark_flux_up=(3.78578e-1, 2.43397e-1, 0.0),
        ),
    ]


def _tabulated_phase_disotest_cases() -> list[DisotestFluxCase]:
    haze_g, haze_delta = _tabulated_two_stream_inputs("haze_l", 32)
    cloud_g, cloud_delta = _tabulated_two_stream_inputs("cloud_c1", 48)
    return [
        DisotestFluxCase(
            name="DISOTEST 4a Haze-L beam",
            tau=(0.5, 0.5),
            ssa=(1.0,),
            g=(haze_g,),
            mu0=1.0,
            fbeam=math.pi,
            albedo=0.0,
            phase="haze_l",
            nstr=32,
            nmom=32,
            fo_n_moments=64,
            benchmark_direct_down=(3.14159, 1.90547, 1.15573),
            benchmark_diffuse_down=(0.0, 1.17401, 1.81264),
            benchmark_flux_up=(1.73223e-1, 1.11113e-1, 0.0),
            delta_m_truncation_factor=(haze_delta,),
        ),
        DisotestFluxCase(
            name="DISOTEST 4b Haze-L absorbing beam",
            tau=(0.5, 0.5),
            ssa=(0.9,),
            g=(haze_g,),
            mu0=1.0,
            fbeam=math.pi,
            albedo=0.0,
            phase="haze_l",
            nstr=32,
            nmom=32,
            fo_n_moments=64,
            benchmark_direct_down=(3.14159, 1.90547, 1.15573),
            benchmark_diffuse_down=(0.0, 1.01517, 1.51554),
            benchmark_flux_up=(1.23665e-1, 7.88690e-2, 0.0),
            delta_m_truncation_factor=(haze_delta,),
        ),
        DisotestFluxCase(
            name="DISOTEST 4c Haze-L oblique beam",
            tau=(0.5, 0.5),
            ssa=(0.9,),
            g=(haze_g,),
            mu0=0.5,
            fbeam=math.pi,
            albedo=0.0,
            phase="haze_l",
            nstr=32,
            nmom=32,
            fo_n_moments=64,
            benchmark_direct_down=(1.57080, 5.77864e-1, 2.12584e-1),
            benchmark_diffuse_down=(0.0, 7.02764e-1, 8.03294e-1),
            benchmark_flux_up=(2.25487e-1, 1.23848e-1, 0.0),
            delta_m_truncation_factor=(haze_delta,),
        ),
        DisotestFluxCase(
            name="DISOTEST 5a Cloud C.1 conservative",
            tau=(32.0, 32.0),
            ssa=(1.0,),
            g=(cloud_g,),
            mu0=1.0,
            fbeam=math.pi,
            albedo=0.0,
            phase="cloud_c1",
            nstr=48,
            nmom=299,
            fo_n_moments=64,
            benchmark_direct_down=(3.14159, 3.97856e-14, 5.03852e-28),
            benchmark_diffuse_down=(0.0, 2.24768, 4.79851e-1),
            benchmark_flux_up=(2.66174, 1.76783, 0.0),
            delta_m_truncation_factor=(cloud_delta,),
        ),
        DisotestFluxCase(
            name="DISOTEST 5b Cloud C.1 absorbing",
            tau=(3.2, 9.6, 35.2, 16.0),
            ssa=(0.9,),
            g=(cloud_g,),
            mu0=1.0,
            fbeam=math.pi,
            albedo=0.0,
            phase="cloud_c1",
            nstr=48,
            nmom=299,
            fo_n_moments=64,
            benchmark_direct_down=(1.28058e-1, 8.67322e-6, 4.47729e-21),
            benchmark_diffuse_down=(1.74767, 2.33975e-1, 6.38345e-5),
            benchmark_flux_up=(2.70485e-1, 3.74252e-2, 1.02904e-5),
            delta_m_truncation_factor=(cloud_delta,),
            compare_level_indices=(1, 2, 3),
            level_names=("3.2", "12.8", "48"),
        ),
    ]


def _thermal_disotest_cases() -> list[DisotestFluxCase]:
    return [
        DisotestFluxCase(
            name="DISOTEST 7a thermal internal",
            tau=(1.0,),
            ssa=(0.1,),
            g=(0.05,),
            mu0=0.5,
            fbeam=0.0,
            albedo=0.0,
            phase="hg",
            nstr=16,
            nmom=16,
            fo_n_moments=64,
            benchmark_direct_down=(0.0, 0.0),
            benchmark_diffuse_down=(0.0, 1.21204e2),
            benchmark_flux_up=(8.62936e1, 0.0),
            mode="thermal",
            thermal_temperature=(200.0, 300.0),
            thermal_wavenumber_band=(300.0, 800.0),
            surface_planck=0.0,
            emissivity=1.0,
        ),
        DisotestFluxCase(
            name="DISOTEST 7b thermal thick",
            tau=(100.0,),
            ssa=(0.95,),
            g=(0.75,),
            mu0=0.5,
            fbeam=0.0,
            albedo=0.0,
            phase="hg",
            nstr=16,
            nmom=16,
            fo_n_moments=64,
            benchmark_direct_down=(0.0, 0.0),
            benchmark_diffuse_down=(0.0, 2.07786e-5),
            benchmark_flux_up=(1.10949e-6, 0.0),
            mode="thermal",
            thermal_temperature=(200.0, 300.0),
            thermal_wavenumber_band=(2702.99, 2703.01),
            surface_planck=0.0,
            emissivity=1.0,
        ),
    ]


def _unsupported_flux_case(
    name: str,
    reason: str,
    *,
    category: str,
    surface_model: str = "lambertian",
    direct: tuple[float, ...] | None = None,
    diffuse_down: tuple[float, ...] | None = None,
    flux_up: tuple[float, ...] | None = None,
) -> DisotestFluxCase:
    return DisotestFluxCase(
        name=name,
        tau=(1.0,),
        ssa=(0.0,),
        g=(0.0,),
        mu0=0.5,
        fbeam=0.0,
        albedo=0.0,
        phase="isotropic",
        nstr=16,
        nmom=16,
        fo_n_moments=1,
        benchmark_direct_down=direct,
        benchmark_diffuse_down=diffuse_down,
        benchmark_flux_up=flux_up,
        surface_model=surface_model,
        unsupported_category=category,
        unsupported_reason=reason,
    )


def _unsupported_official_disotest_cases() -> list[DisotestFluxCase]:
    top_isotropic = "requires top isotropic illumination on non-boundary output levels"
    surface_not_equivalent = (
        "DISORT Hapke/BDR surface is not equivalent to py2sess' current "
        "Lambertian/Ross/RPV/Cox-Munk BRDF kernels"
    )
    mixed_sources = (
        "mixed solar, isotropic-boundary, and thermal source terms require separate py2sess modes"
    )
    internal_regression = "DISORT internal consistency regression without a fixed level-flux table"
    return [
        DisotestFluxCase(
            name="DISOTEST 6d absorbing non-Lambert surface",
            tau=(1.0,),
            ssa=(0.0,),
            g=(0.0,),
            mu0=0.5,
            fbeam=200.0,
            albedo=0.0,
            phase="absorbing",
            nstr=16,
            nmom=16,
            fo_n_moments=1,
            surface_model="hapke",
            benchmark_direct_down=(100.0, 3.67879e1, 1.35335e1),
            benchmark_diffuse_down=(0.0, 0.0, 0.0),
            benchmark_flux_up=(6.70783e-1, 1.39084, 3.31655),
            pydisort_tau=(0.5, 0.5),
            py2sess_tau=(0.5, 0.5),
        ),
        _unsupported_flux_case(
            "DISOTEST 6e absorbing bottom emission",
            surface_not_equivalent,
            category="surface_model_not_equivalent",
            surface_model="disort-bdr-function",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(0.0, 0.0, 0.0),
            flux_up=(7.95458e1, 1.59902e2, 3.56410e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 6f absorbing top/bottom emission",
            mixed_sources,
            category="mixed_sources",
            surface_model="disort-bdr-function",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(3.21497e2, 1.42493e2, 7.05305e1),
            flux_up=(8.27917e1, 1.66532e2, 3.71743e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 6g absorbing internal emission",
            mixed_sources,
            category="mixed_sources",
            surface_model="disort-bdr-function",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(3.21497e2, 3.04775e2, 3.63632e2),
            flux_up=(3.35292e2, 4.12540e2, 4.41125e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 6h absorbing thick emission",
            mixed_sources,
            category="mixed_sources",
            surface_model="disort-bdr-function",
            direct=(100.0, 1.35335e1, 2.06115e-7),
            diffuse_down=(3.21497e2, 2.55455e2, 4.43444e2),
            flux_up=(2.37350e2, 2.61130e2, 4.55861e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 7c all sources Lambertian",
            mixed_sources,
            category="mixed_sources",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(3.19830e2, 3.54099e2, 3.01334e2),
            flux_up=(4.29572e2, 4.47018e2, 5.94576e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 7d all sources Hapke",
            mixed_sources,
            category="mixed_sources",
            surface_model="hapke",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(3.19830e2, 3.50555e2, 2.92063e2),
            flux_up=(3.12563e2, 2.68126e2, 3.05596e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 7e all sources BDR",
            mixed_sources,
            category="mixed_sources",
            surface_model="disort-bdr-function",
            direct=(100.0, 3.67879e1, 1.35335e1),
            diffuse_down=(3.19830e2, 3.53275e2, 2.99002e2),
            flux_up=(4.04300e2, 4.07843e2, 5.29248e2),
        ),
        _unsupported_flux_case(
            "DISOTEST 9a multilayer isotropic source",
            top_isotropic,
            category="top_isotropic",
            direct=(0.0, 0.0, 0.0, 0.0, 0.0),
            diffuse_down=(1.0, 3.55151e-1, 1.44265e-1, 6.71445e-3, 6.16968e-7),
            flux_up=(2.27973e-1, 8.75098e-2, 3.61819e-2, 2.19291e-3, 0.0),
        ),
        _unsupported_flux_case(
            "DISOTEST 9b multilayer anisotropic source",
            "top isotropic illumination plus arbitrary layer phase moments",
            category="top_isotropic",
            direct=(0.0, 0.0, 0.0, 0.0, 0.0),
            diffuse_down=(1.0, 4.52357e-1, 2.36473e-1, 2.76475e-2, 7.41853e-5),
            flux_up=(1.00079e-1, 4.52014e-2, 2.41941e-2, 4.16016e-3, 0.0),
        ),
        _unsupported_flux_case(
            "DISOTEST 9c multilayer all sources",
            mixed_sources,
            category="mixed_sources",
            direct=(1.57080, 1.92354e-1, 2.35550e-2, 9.65131e-6, 9.03133e-19),
            diffuse_down=(6.09217, 4.97279, 4.46616, 4.22731, 4.73767),
            flux_up=(4.68414, 4.24381, 4.16941, 4.30667, 5.11524),
        ),
        *[
            _unsupported_flux_case(name, internal_regression, category="internal_regression")
            for name in (
                "DISOTEST 10a usrang true internal regression",
                "DISOTEST 10b usrang false internal regression",
                "DISOTEST 11a one-layer internal regression",
                "DISOTEST 11b multi-layer internal regression",
                "DISOTEST 12a absorption shortcut reference",
                "DISOTEST 12b absorption shortcut variant",
                "DISOTEST 13a albedo/transmissivity shortcut single",
                "DISOTEST 13b albedo/transmissivity regular single",
                "DISOTEST 13c albedo/transmissivity shortcut multi",
                "DISOTEST 13d albedo/transmissivity regular multi",
                "DISOTEST 14a disort/twostr reference",
                "DISOTEST 14b twostr comparison",
            )
        ],
    ]


def _direct_disotest_cases() -> list[DisotestFluxCase]:
    return [
        *_comparable_cases(),
        *_top_isotropic_disotest_cases(),
        *_tabulated_phase_disotest_cases(),
        *_thermal_disotest_cases(),
    ]


def _official_disotest_cases() -> list[DisotestFluxCase]:
    return [*_direct_disotest_cases(), *_unsupported_official_disotest_cases()]


def _surface_disotest_cases() -> list[DisotestFluxCase]:
    names = {
        "DISOTEST 6d absorbing non-Lambert surface",
        "DISOTEST 7d all sources Hapke",
        "DISOTEST 7e all sources BDR",
    }
    return [case for case in _official_disotest_cases() if case.name in names]


def _pydisort_grid_cases() -> list[DisotestFluxCase]:
    cases = [
        DisotestFluxCase(
            name="GRID absorbing thin black",
            tau=(0.05, 0.15),
            ssa=(0.0,),
            g=(0.0,),
            mu0=0.85,
            fbeam=math.pi,
            albedo=0.0,
            phase="absorbing",
            nstr=16,
            nmom=2,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID absorbing oblique black",
            tau=(0.2, 0.4, 0.6),
            ssa=(0.0,),
            g=(0.0,),
            mu0=0.25,
            fbeam=math.pi,
            albedo=0.0,
            phase="absorbing",
            nstr=16,
            nmom=2,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID absorbing surface low",
            tau=(0.15, 0.25),
            ssa=(0.0,),
            g=(0.0,),
            mu0=0.6,
            fbeam=2.0,
            albedo=0.15,
            phase="absorbing",
            nstr=24,
            nmom=2,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID absorbing surface high",
            tau=(0.3, 0.7),
            ssa=(0.0,),
            g=(0.0,),
            mu0=0.4,
            fbeam=2.0,
            albedo=0.55,
            phase="absorbing",
            nstr=24,
            nmom=2,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID iso weak thin",
            tau=(0.08, 0.12),
            ssa=(0.2, 0.25),
            g=(0.0,),
            mu0=0.75,
            fbeam=math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=24,
            nmom=16,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID iso weak surface",
            tau=(0.1, 0.3),
            ssa=(0.25, 0.35),
            g=(0.0,),
            mu0=0.35,
            fbeam=math.pi,
            albedo=0.2,
            phase="isotropic",
            nstr=24,
            nmom=16,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID iso conservative thin",
            tau=(0.2, 0.2),
            ssa=(1.0,),
            g=(0.0,),
            mu0=0.7,
            fbeam=math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=32,
            nmom=16,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID iso conservative thick",
            tau=(0.5, 0.5, 0.5),
            ssa=(1.0,),
            g=(0.0,),
            mu0=0.5,
            fbeam=math.pi,
            albedo=0.0,
            phase="isotropic",
            nstr=32,
            nmom=16,
            fo_n_moments=1,
        ),
        DisotestFluxCase(
            name="GRID rayleigh weak thin",
            tau=(0.05, 0.15),
            ssa=(0.2, 0.25),
            g=(0.0,),
            mu0=0.8,
            fbeam=math.pi,
            albedo=0.0,
            phase="rayleigh",
            nstr=24,
            nmom=16,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID rayleigh weak oblique",
            tau=(0.2, 0.3),
            ssa=(0.3, 0.4),
            g=(0.0,),
            mu0=0.3,
            fbeam=math.pi,
            albedo=0.0,
            phase="rayleigh",
            nstr=24,
            nmom=16,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID rayleigh conservative",
            tau=(0.25, 0.25),
            ssa=(1.0,),
            g=(0.0,),
            mu0=0.65,
            fbeam=math.pi,
            albedo=0.0,
            phase="rayleigh",
            nstr=32,
            nmom=16,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID rayleigh surface",
            tau=(0.15, 0.35),
            ssa=(0.7, 0.8),
            g=(0.0,),
            mu0=0.45,
            fbeam=math.pi,
            albedo=0.25,
            phase="rayleigh",
            nstr=32,
            nmom=16,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID HG moderate thin",
            tau=(0.1, 0.2),
            ssa=(0.6, 0.7),
            g=(0.45, 0.55),
            mu0=0.8,
            fbeam=math.pi,
            albedo=0.0,
            phase="hg",
            nstr=32,
            nmom=32,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID HG forward thin",
            tau=(0.1, 0.2),
            ssa=(0.85, 0.9),
            g=(0.75, 0.82),
            mu0=0.6,
            fbeam=math.pi,
            albedo=0.0,
            phase="hg",
            nstr=48,
            nmom=48,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID HG forward surface",
            tau=(0.25, 0.35),
            ssa=(0.8, 0.9),
            g=(0.7, 0.85),
            mu0=0.45,
            fbeam=math.pi,
            albedo=0.2,
            phase="hg",
            nstr=48,
            nmom=48,
            fo_n_moments=64,
        ),
        DisotestFluxCase(
            name="GRID HG conservative",
            tau=(0.3, 0.4, 0.3),
            ssa=(1.0,),
            g=(0.6,),
            mu0=0.55,
            fbeam=math.pi,
            albedo=0.0,
            phase="hg",
            nstr=48,
            nmom=48,
            fo_n_moments=64,
        ),
    ]
    cases.extend(
        [
            DisotestFluxCase(
                name="GRID absorbing multilayer surface",
                tau=(0.05, 0.15, 0.35, 0.45),
                ssa=(0.0,),
                g=(0.0,),
                mu0=0.65,
                fbeam=2.5,
                albedo=0.35,
                phase="absorbing",
                nstr=24,
                nmom=2,
                fo_n_moments=1,
            ),
            DisotestFluxCase(
                name="GRID absorbing grazing surface",
                tau=(0.02, 0.03),
                ssa=(0.0,),
                g=(0.0,),
                mu0=0.15,
                fbeam=math.pi,
                albedo=0.3,
                phase="absorbing",
                nstr=24,
                nmom=2,
                fo_n_moments=1,
            ),
            DisotestFluxCase(
                name="GRID iso moderate multilayer",
                tau=(0.05, 0.15, 0.3),
                ssa=(0.4, 0.6, 0.8),
                g=(0.0,),
                mu0=0.55,
                fbeam=math.pi,
                albedo=0.1,
                phase="isotropic",
                nstr=32,
                nmom=16,
                fo_n_moments=1,
            ),
            DisotestFluxCase(
                name="GRID iso thick surface",
                tau=(0.4, 0.6, 0.8),
                ssa=(0.8,),
                g=(0.0,),
                mu0=0.4,
                fbeam=math.pi,
                albedo=0.3,
                phase="isotropic",
                nstr=32,
                nmom=16,
                fo_n_moments=1,
            ),
            DisotestFluxCase(
                name="GRID rayleigh moderate multilayer",
                tau=(0.05, 0.2, 0.35),
                ssa=(0.5, 0.6, 0.7),
                g=(0.0,),
                mu0=0.55,
                fbeam=math.pi,
                albedo=0.0,
                phase="rayleigh",
                nstr=32,
                nmom=16,
                fo_n_moments=64,
            ),
            DisotestFluxCase(
                name="GRID rayleigh thick oblique",
                tau=(0.3, 0.7, 1.0),
                ssa=(0.9,),
                g=(0.0,),
                mu0=0.25,
                fbeam=math.pi,
                albedo=0.1,
                phase="rayleigh",
                nstr=32,
                nmom=16,
                fo_n_moments=64,
            ),
            DisotestFluxCase(
                name="GRID HG weakly forward",
                tau=(0.15, 0.25),
                ssa=(0.55, 0.65),
                g=(0.25, 0.35),
                mu0=0.7,
                fbeam=math.pi,
                albedo=0.0,
                phase="hg",
                nstr=32,
                nmom=32,
                fo_n_moments=64,
            ),
            DisotestFluxCase(
                name="GRID HG high surface",
                tau=(0.15, 0.25, 0.4),
                ssa=(0.75, 0.85, 0.9),
                g=(0.65, 0.75, 0.85),
                mu0=0.5,
                fbeam=math.pi,
                albedo=0.35,
                phase="hg",
                nstr=48,
                nmom=48,
                fo_n_moments=64,
            ),
            DisotestFluxCase(
                name="GRID HG layered",
                tau=(0.05, 0.2, 0.5, 0.25),
                ssa=(0.4, 0.7, 0.9, 0.6),
                g=(0.2, 0.5, 0.85, 0.65),
                mu0=0.6,
                fbeam=math.pi,
                albedo=0.05,
                phase="hg",
                nstr=48,
                nmom=48,
                fo_n_moments=64,
            ),
            DisotestFluxCase(
                name="GRID HG forward oblique",
                tau=(0.2, 0.4, 0.6),
                ssa=(0.95,),
                g=(0.85,),
                mu0=0.3,
                fbeam=math.pi,
                albedo=0.0,
                phase="hg",
                nstr=48,
                nmom=48,
                fo_n_moments=64,
            ),
        ]
    )
    return cases


def _selected_cases(suite: str) -> list[DisotestFluxCase]:
    if suite == "disotest":
        return _official_disotest_cases()
    if suite in {"disort-test", "disotest-runnable"}:
        return _direct_disotest_cases()
    if suite == "disotest-solar":
        return _comparable_cases()
    if suite == "disotest-thermal":
        return _thermal_disotest_cases()
    if suite == "disotest-surface":
        return _surface_disotest_cases()
    if suite == "disotest-top-isotropic":
        return _top_isotropic_disotest_cases()
    if suite == "pydisort-grid":
        return _pydisort_grid_cases()
    if suite == "all":
        return [*_official_disotest_cases(), *_pydisort_grid_cases()]
    raise ValueError(f"unsupported suite {suite!r}")


def _format_float(value: object, *, precision: int = 8) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{precision}g}"


def _print_run_context(
    *,
    suite: str,
    stream: float,
    fo_flux_n_mu: int,
    backend: str = "numpy",
) -> None:
    mode = _stream_mode(stream)
    print(
        f"suite={suite} stream={stream:.8g} stream_mode={mode} "
        f"fo_flux_n_mu={fo_flux_n_mu} py2sess_backend={backend}"
    )
    if suite in {
        "all",
        "disotest",
        "disort-test",
        "disotest-runnable",
        "disotest-solar",
        "disotest-thermal",
        "disotest-surface",
        "disotest-top-isotropic",
    }:
        print(
            "DISOTEST benchmark/pydisort columns are DISORT/LIDORT reference fluxes; "
            "stream_mode=public-default is the py2sess default and "
            "stream_mode=section6 is the 2S-ESS paper Section 6 reproduction setting."
        )
    if suite == "disotest-surface":
        print(
            "Surface suite: DISOTEST 6d uses the cDISORT Hapke surface kernel; "
            "7d/7e remain diagnostic until their mixed-source terms are represented."
        )
    if suite == "disotest-top-isotropic":
        print(
            "Top-isotropic suite: pydisort is populated from official DISOTEST benchmark "
            "rows because this adapter path exercises py2sess' fisot boundary directly."
        )
    if mode == "public-default":
        print("stream_mode=public-default uses py2sess' public two-stream closure, 1/sqrt(3).")
    elif mode == "custom":
        print("stream_mode=custom is a sensitivity run, not a public default.")
    print()


def _print_summary(summaries: list[dict[str, object]]) -> None:
    print(
        f"{'case':34s} {'status':11s} {'worst field':10s} {'level':5s} "
        f"{'benchmark':>12s} {'pydisort':>12s} {'py2sess':>12s} {'py2sess err %':>14s}"
    )
    print("-" * 117)
    for summary in summaries:
        rel = float(summary["py2sess_rel_percent"])
        rel_text = "nan" if math.isnan(rel) else f"{rel:14.4f}"
        print(
            f"{str(summary['case']):34s} "
            f"{str(summary['status']):11s} "
            f"{str(summary['worst_field']):10s} "
            f"{str(summary['worst_level']):5s} "
            f"{_format_float(summary['benchmark']):>12s} "
            f"{_format_float(summary['pydisort']):>12s} "
            f"{_format_float(summary['py2sess']):>12s} "
            f"{rel_text:>14s}"
        )


def _print_details(rows: list[dict[str, object]]) -> None:
    print(
        f"{'case':34s} {'status':11s} {'field':10s} {'level':5s} "
        f"{'benchmark':>12s} {'pydisort':>12s} {'py2sess':>12s} {'py2sess err %':>14s}"
    )
    print("-" * 117)
    for row in rows:
        rel = row["py2sess_rel_percent"]
        rel_text = "nan" if not isinstance(rel, float) or math.isnan(rel) else f"{rel:.4f}"
        print(
            f"{str(row['case']):34s} "
            f"{str(row['status']):11s} "
            f"{str(row['field']):10s} "
            f"{str(row['level']):5s} "
            f"{_format_float(row['benchmark']):>12s} "
            f"{_format_float(row['pydisort']):>12s} "
            f"{_format_float(row['py2sess']):>12s} "
            f"{rel_text:>14s}"
        )


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paper_table_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "case": row["case"],
            "quantity": row["field"],
            "level": row["level"],
            "DISORT": row["benchmark"],
            "py2sess": row["py2sess"],
            "percent_error": row["py2sess_rel_percent"],
        }
        for row in rows
        if row.get("status") == "run"
    ]


def _format_percent(value: object) -> str:
    if not isinstance(value, float) or math.isnan(value):
        return "nan"
    rounded = round(value, 2)
    if rounded == 0.0:
        return "0"
    return f"{rounded:.2f}"


def _latex_escape(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _print_paper_table(rows: list[dict[str, object]], *, table_format: str) -> None:
    paper_rows = _paper_table_rows(rows)
    if table_format == "markdown":
        print("| Case | Quantity | Level | DISORT | py2sess | Error (%) |")
        print("|---|---:|---:|---:|---:|---:|")
        for row in paper_rows:
            print(
                f"| {row['case']} | {row['quantity']} | {row['level']} | "
                f"{_format_float(row['DISORT'], precision=6)} | "
                f"{_format_float(row['py2sess'], precision=6)} | "
                f"{_format_percent(row['percent_error'])} |"
            )
        return

    if table_format == "latex":
        print(r"\begin{tabular}{lllrrr}")
        print(r"\hline")
        print(r"Case & Quantity & Level & DISORT & py2sess & Error (\%) \\")
        print(r"\hline")
        for row in paper_rows:
            print(
                f"{_latex_escape(row['case'])} & "
                f"{_latex_escape(row['quantity'])} & "
                f"{_latex_escape(row['level'])} & "
                f"{_format_float(row['DISORT'], precision=6)} & "
                f"{_format_float(row['py2sess'], precision=6)} & "
                f"{_format_percent(row['percent_error'])} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        return

    print(
        f"{'case':34s} {'quantity':10s} {'level':5s} "
        f"{'DISORT':>12s} {'py2sess':>12s} {'err %':>10s}"
    )
    print("-" * 91)
    for row in paper_rows:
        print(
            f"{str(row['case']):34s} "
            f"{str(row['quantity']):10s} "
            f"{str(row['level']):5s} "
            f"{_format_float(row['DISORT'], precision=6):>12s} "
            f"{_format_float(row['py2sess'], precision=6):>12s} "
            f"{_format_percent(row['percent_error']):>10s}"
        )


def _parse_stream_list(value: str) -> list[float]:
    streams: list[float] = []
    aliases = {
        "sqrt3": 1.0 / math.sqrt(3.0),
        "1/sqrt3": 1.0 / math.sqrt(3.0),
        "gauss": 1.0 / math.sqrt(3.0),
        "hemispheric": 0.5,
    }
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        stream = aliases.get(token, float(token) if token not in aliases else aliases[token])
        if not 0.0 < stream <= 1.0:
            raise argparse.ArgumentTypeError("stream values must satisfy 0 < stream <= 1")
        streams.append(float(stream))
    if not streams:
        raise argparse.ArgumentTypeError("stream sweep cannot be empty")
    return streams


def _parse_positive_int_list(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            number = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("n_mu values must be positive integers") from exc
        if number <= 0:
            raise argparse.ArgumentTypeError("n_mu values must be positive integers")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("n_mu sweep cannot be empty")
    return values


def _case_worst_rel_percent(rows: list[dict[str, object]]) -> float:
    values = [
        abs(float(row["py2sess_rel_percent"]))
        for row in rows
        if isinstance(row["py2sess_rel_percent"], float)
        and math.isfinite(row["py2sess_rel_percent"])
    ]
    return max(values) if values else math.nan


def _print_stream_sweep(
    cases: list[DisotestFluxCase],
    streams: list[float],
    *,
    fo_flux_n_mu: int,
    backend: str = "numpy",
) -> list[dict[str, object]]:
    per_case: dict[str, dict[float, float]] = {case.name: {} for case in cases}
    rows: list[dict[str, object]] = []
    for stream in streams:
        for case in cases:
            case_rows = _rows(
                case,
                stream=stream,
                fo_flux_n_mu=fo_flux_n_mu,
                backend=backend,
            )
            rows.extend(case_rows)
            per_case[case.name][stream] = _case_worst_rel_percent(case_rows)

    stream_stats: list[dict[str, object]] = []
    for stream in streams:
        values = np.array(
            [
                per_stream[stream]
                for per_stream in per_case.values()
                if math.isfinite(per_stream[stream])
            ],
            dtype=float,
        )
        wins = 0
        for per_stream in per_case.values():
            finite = {s: v for s, v in per_stream.items() if math.isfinite(v)}
            if finite and min(finite, key=finite.get) == stream:
                wins += 1
        stream_stats.append(
            {
                "stream": stream,
                "cases": int(values.size),
                "worst": float(np.max(values)) if values.size else math.nan,
                "p95": float(np.percentile(values, 95.0)) if values.size else math.nan,
                "median": float(np.median(values)) if values.size else math.nan,
                "mean": float(np.mean(values)) if values.size else math.nan,
                "wins": wins,
            }
        )

    print(
        "Two-stream mu sweep; metric is each case's worst finite abs(relative error %) vs benchmark."
    )
    print(
        "stream modes: " + ", ".join(f"{stream:.8g}={_stream_mode(stream)}" for stream in streams)
    )
    print(
        f"{'stream':>12s} {'cases':>6s} {'worst %':>12s} {'p95 %':>12s} "
        f"{'median %':>12s} {'mean %':>12s} {'case wins':>10s}"
    )
    print("-" * 86)
    for stat in stream_stats:
        print(
            f"{float(stat['stream']):12.8f} "
            f"{int(stat['cases']):6d} "
            f"{float(stat['worst']):12.4f} "
            f"{float(stat['p95']):12.4f} "
            f"{float(stat['median']):12.4f} "
            f"{float(stat['mean']):12.4f} "
            f"{int(stat['wins']):10d}"
        )

    print()
    stream_headers = [f"{stream:.6g}" for stream in streams]
    case_col = 34
    print(
        f"{'case':{case_col}s} {'best stream':>12s} "
        + " ".join(f"{h:>12s}" for h in stream_headers)
    )
    print("-" * (case_col + 13 + 13 * len(stream_headers)))
    unsupported: list[DisotestFluxCase] = []
    for case in cases:
        metrics = per_case[case.name]
        finite = {stream: value for stream, value in metrics.items() if math.isfinite(value)}
        if not finite:
            unsupported.append(case)
            continue
        best = min(finite, key=finite.get)
        metric_text = [
            "nan" if not math.isfinite(metrics[stream]) else f"{metrics[stream]:12.4f}"
            for stream in streams
        ]
        best_text = f"{best:.8f}"
        print(f"{case.name:{case_col}s} {best_text:>12s} " + " ".join(metric_text))
    if unsupported:
        print()
        print("Registered official DISOTEST cases not included in the mu metric:")
        for case in unsupported:
            reason = case.unsupported_reason or "no finite py2sess comparison metric"
            print(f"- {case.name}: {reason}")
    return rows


def _print_n_mu_sweep(
    cases: list[DisotestFluxCase],
    n_mu_values: list[int],
    *,
    stream: float,
    backend: str = "numpy",
) -> list[dict[str, object]]:
    per_case: dict[str, dict[int, float]] = {case.name: {} for case in cases}
    rows: list[dict[str, object]] = []
    for n_mu in n_mu_values:
        for case in cases:
            case_rows = _rows(case, stream=stream, fo_flux_n_mu=n_mu, backend=backend)
            rows.extend(case_rows)
            per_case[case.name][n_mu] = _case_worst_rel_percent(case_rows)

    stats: list[dict[str, object]] = []
    for n_mu in n_mu_values:
        values = np.array(
            [per_n_mu[n_mu] for per_n_mu in per_case.values() if math.isfinite(per_n_mu[n_mu])],
            dtype=float,
        )
        wins = 0
        for per_n_mu in per_case.values():
            finite = {n: v for n, v in per_n_mu.items() if math.isfinite(v)}
            if finite and min(finite, key=finite.get) == n_mu:
                wins += 1
        stats.append(
            {
                "n_mu": n_mu,
                "cases": int(values.size),
                "worst": float(np.max(values)) if values.size else math.nan,
                "p95": float(np.percentile(values, 95.0)) if values.size else math.nan,
                "median": float(np.median(values)) if values.size else math.nan,
                "mean": float(np.mean(values)) if values.size else math.nan,
                "wins": wins,
            }
        )

    print(
        "FO flux n_mu sweep; metric is each case's worst finite abs(relative error %) vs benchmark."
    )
    print(f"stream fixed at {stream:.8g} ({_stream_mode(stream)})")
    print(
        f"{'n_mu':>8s} {'cases':>6s} {'worst %':>12s} {'p95 %':>12s} "
        f"{'median %':>12s} {'mean %':>12s} {'case wins':>10s}"
    )
    print("-" * 82)
    for stat in stats:
        print(
            f"{int(stat['n_mu']):8d} "
            f"{int(stat['cases']):6d} "
            f"{float(stat['worst']):12.4f} "
            f"{float(stat['p95']):12.4f} "
            f"{float(stat['median']):12.4f} "
            f"{float(stat['mean']):12.4f} "
            f"{int(stat['wins']):10d}"
        )

    print()
    headers = [str(n_mu) for n_mu in n_mu_values]
    case_col = 34
    print(f"{'case':{case_col}s} {'best n_mu':>10s} " + " ".join(f"{h:>12s}" for h in headers))
    print("-" * (case_col + 11 + 13 * len(headers)))
    unsupported: list[DisotestFluxCase] = []
    for case in cases:
        metrics = per_case[case.name]
        finite = {n_mu: value for n_mu, value in metrics.items() if math.isfinite(value)}
        if not finite:
            unsupported.append(case)
            continue
        best = min(finite, key=finite.get)
        metric_text = [
            "nan" if not math.isfinite(metrics[n_mu]) else f"{metrics[n_mu]:12.4f}"
            for n_mu in n_mu_values
        ]
        print(f"{case.name:{case_col}s} {best:10d} " + " ".join(metric_text))
    if unsupported:
        print()
        print("Registered official DISOTEST cases not included in the n_mu metric:")
        for case in unsupported:
            reason = case.unsupported_reason or "no finite py2sess comparison metric"
            print(f"- {case.name}: {reason}")
    return rows


def _vijay_comparison_rows(
    cases: list[DisotestFluxCase],
    *,
    n_mu_values: list[int],
    stream: float,
    backend: str = "numpy",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.unsupported_reason is not None or case.name not in VIJAY_SECTION6_2SESS_FLUXES:
            continue
        benchmark = _benchmark_flux(case)
        pydisort = _run_pydisort(case)
        vijay = _vijay_section6_flux(case)
        py2sess_by_n_mu = {
            n_mu: _run_py2sess(case, stream=stream, fo_flux_n_mu=n_mu, backend=backend)
            for n_mu in n_mu_values
        }
        levels = _level_names(case, len(benchmark["flux_up"]))
        for field in FIELDS:
            for level_index, level_name in enumerate(levels):
                row: dict[str, object] = {
                    "case": case.name,
                    "field": field,
                    "level": level_name,
                    "benchmark": float(benchmark[field][level_index]),
                    "pydisort": float(pydisort[field][level_index]),
                    "vijay_section6": float(vijay[field][level_index]),
                    "stream": stream,
                    "stream_mode": _stream_mode(stream),
                    "py2sess_backend": backend,
                }
                for n_mu in n_mu_values:
                    py = float(py2sess_by_n_mu[n_mu][field][level_index])
                    row[f"py2sess_n{n_mu}"] = py
                    row[f"err_vs_benchmark_n{n_mu}_percent"] = _rel_percent(
                        py,
                        float(benchmark[field][level_index]),
                    )
                    row[f"err_vs_vijay_n{n_mu}_percent"] = _rel_percent(
                        py,
                        float(vijay[field][level_index]),
                    )
                rows.append(row)
    return rows


def _print_vijay_section6_comparison(rows: list[dict[str, object]], n_mu_values: list[int]) -> None:
    print(
        "Vijay Section 6 comparison. benchmark/pydisort are DISORT/LIDORT references; "
        "vijay_section6 is the paper's 2S-ESS output transcribed from Zenodo "
        "DISORT_Comparisons/2S-ESS outputs."
    )
    if rows:
        print(f"py2sess stream={float(rows[0]['stream']):.8g} stream_mode={rows[0]['stream_mode']}")
    print()
    headers = [
        "case",
        "field",
        "level",
        "benchmark",
        "pydisort",
        "vijay_section6",
        *[f"py2sess_n{n_mu}" for n_mu in n_mu_values],
        f"err_vijay_n{n_mu_values[-1]}%",
    ]
    widths = [31, 10, 5, 13, 13, 15, *([13] * len(n_mu_values)), 15]
    print(" ".join(f"{header:{width}s}" for header, width in zip(headers, widths)))
    print("-" * (sum(widths) + len(widths) - 1))
    for row in rows:
        values = [
            str(row["case"])[:31],
            str(row["field"]),
            str(row["level"]),
            row["benchmark"],
            row["pydisort"],
            row["vijay_section6"],
            *[row[f"py2sess_n{n_mu}"] for n_mu in n_mu_values],
            row[f"err_vs_vijay_n{n_mu_values[-1]}_percent"],
        ]
        parts: list[str] = []
        for idx, (value, width) in enumerate(zip(values, widths)):
            if idx < 3:
                parts.append(f"{str(value):{width}s}")
            else:
                parts.append(f"{_format_float(value, precision=6):>{width}s}")
        print(" ".join(parts))


def _unsupported_official_cases() -> list[tuple[str, str, str, str | None]]:
    return [
        (
            case.name,
            case.unsupported_category or "unsupported",
            case.surface_model,
            case.unsupported_reason,
        )
        for case in _unsupported_official_disotest_cases()
        if case.unsupported_reason is not None
    ]


def _print_unsupported_official_cases() -> None:
    print("Official DISOTEST cases registered but not run through py2sess:")
    for case, category, surface_model, reason in _unsupported_official_cases():
        surface = "" if surface_model == "lambertian" else f"; surface={surface_model}"
        print(f"- {case}: {category}{surface}; {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="print every field/level row")
    parser.add_argument(
        "--paper-table",
        action="store_true",
        help=(
            "print paper-ready rows with columns Case, Quantity, Level, DISORT, "
            "py2sess, and percent error"
        ),
    )
    parser.add_argument(
        "--paper-table-format",
        choices=("plain", "markdown", "latex"),
        default="markdown",
        help="paper table output format",
    )
    parser.add_argument(
        "--fail-rel-percent",
        type=float,
        help="print only rows where abs(py2sess relative error percent) exceeds this threshold",
    )
    parser.add_argument(
        "--suite",
        choices=(
            "all",
            "disotest",
            "disort-test",
            "disotest-runnable",
            "disotest-solar",
            "disotest-thermal",
            "disotest-surface",
            "disotest-top-isotropic",
            "pydisort-grid",
        ),
        default="all",
        help="case suite to run; pydisort-grid uses pydisort itself as the benchmark column",
    )
    parser.add_argument(
        "--stream",
        type=float,
        default=None,
        help=(
            "two-stream quadrature cosine used by py2sess; defaults to 1/sqrt(3). "
            "Pass 0.5 to reproduce the 2S-ESS Section 6 setting"
        ),
    )
    parser.add_argument(
        "--fo-flux-n-mu",
        type=int,
        default=None,
        help=(
            "positive-hemisphere polar quadrature count used by FO flux source replacement; "
            "defaults to 8"
        ),
    )
    parser.add_argument(
        "--py2sess-backend",
        choices=("numpy", "torch", "native"),
        default="numpy",
        help="py2sess backend used for the comparison table",
    )
    parser.add_argument(
        "--stream-sweep",
        type=_parse_stream_list,
        help="comma-separated two-stream mu values to compare, e.g. 0.5,1/sqrt3,0.6",
    )
    parser.add_argument(
        "--n-mu-sweep",
        type=_parse_positive_int_list,
        help="comma-separated FO flux n_mu values to compare, e.g. 4,8,16,32,48",
    )
    parser.add_argument(
        "--compare-vijay-section6",
        action="store_true",
        help=(
            "print benchmark/pydisort/Vijay Section 6/py2sess n_mu comparison rows "
            "for official DISOTEST cases with transcribed Section 6 rows"
        ),
    )
    parser.add_argument(
        "--list-unsupported",
        action="store_true",
        help="also print official DISOTEST cases not directly represented by this benchmark",
    )
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    args = parser.parse_args()

    cases = _selected_cases(args.suite)
    fo_flux_n_mu = 8 if args.fo_flux_n_mu is None else args.fo_flux_n_mu
    stream = (
        PUBLIC_DEFAULT_STREAM
        if args.compare_vijay_section6 and args.stream is None
        else DEFAULT_DISORT_STREAM
        if args.stream is None
        else float(args.stream)
    )
    if fo_flux_n_mu <= 0:
        parser.error("--fo-flux-n-mu must be a positive integer")
    if args.stream_sweep is not None and args.n_mu_sweep is not None:
        parser.error("--stream-sweep and --n-mu-sweep are separate sweeps; run one at a time")
    if args.compare_vijay_section6 and args.stream_sweep is not None:
        parser.error(
            "--compare-vijay-section6 uses one stream; use --stream instead of --stream-sweep"
        )
    if args.compare_vijay_section6:
        n_mu_values = args.n_mu_sweep or [4, 8, 16, 32]
        rows = _vijay_comparison_rows(
            cases,
            n_mu_values=n_mu_values,
            stream=stream,
            backend=args.py2sess_backend,
        )
        _print_vijay_section6_comparison(rows, n_mu_values)
        if args.csv is not None:
            _write_csv(rows, args.csv)
            print(f"wrote {args.csv}")
        return
    if args.stream_sweep is not None:
        rows = _print_stream_sweep(
            cases,
            args.stream_sweep,
            fo_flux_n_mu=fo_flux_n_mu,
            backend=args.py2sess_backend,
        )
        if args.list_unsupported:
            print()
            _print_unsupported_official_cases()
        if args.csv is not None:
            _write_csv(rows, args.csv)
            print(f"wrote {args.csv}")
        return
    if args.n_mu_sweep is not None:
        rows = _print_n_mu_sweep(
            cases,
            args.n_mu_sweep,
            stream=stream,
            backend=args.py2sess_backend,
        )
        if args.list_unsupported:
            print()
            _print_unsupported_official_cases()
        if args.csv is not None:
            _write_csv(rows, args.csv)
            print(f"wrote {args.csv}")
        return

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case in cases:
        case_rows = _rows(
            case,
            stream=stream,
            fo_flux_n_mu=fo_flux_n_mu,
            backend=args.py2sess_backend,
        )
        rows.extend(case_rows)
        summaries.append(_case_summary(case_rows))

    if args.paper_table:
        _print_paper_table(rows, table_format=args.paper_table_format)
    elif args.fail_rel_percent is not None:
        _print_run_context(
            suite=args.suite,
            stream=stream,
            fo_flux_n_mu=fo_flux_n_mu,
            backend=args.py2sess_backend,
        )
        threshold = float(args.fail_rel_percent)
        filtered = [
            row
            for row in rows
            if isinstance(row["py2sess_rel_percent"], float)
            and math.isfinite(row["py2sess_rel_percent"])
            and abs(row["py2sess_rel_percent"]) > threshold
        ]
        _print_details(filtered)
    elif args.details:
        _print_run_context(
            suite=args.suite,
            stream=stream,
            fo_flux_n_mu=fo_flux_n_mu,
            backend=args.py2sess_backend,
        )
        _print_details(rows)
    else:
        _print_run_context(
            suite=args.suite,
            stream=stream,
            fo_flux_n_mu=fo_flux_n_mu,
            backend=args.py2sess_backend,
        )
        _print_summary(summaries)
    if args.list_unsupported:
        print()
        _print_unsupported_official_cases()
    if args.csv is not None:
        _write_csv(_paper_table_rows(rows) if args.paper_table else rows, args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
