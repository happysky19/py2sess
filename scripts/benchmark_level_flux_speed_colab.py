#!/usr/bin/env python3
"""Benchmark batched level-flux output backends on Colab/local machines."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions, native_backend_info
from py2sess.rtsolver.backend import has_torch, to_numpy


def _cuda_available() -> bool:
    if not has_torch():
        return False
    import torch

    return bool(torch.cuda.is_available())


def _sync(device: str) -> None:
    if device != "cuda" or not has_torch():
        return
    import torch

    torch.cuda.synchronize()


def _thermal_case(rows: int, nlay: int) -> dict[str, Any]:
    row = np.arange(rows, dtype=np.float64)[:, None]
    layer = np.arange(nlay, dtype=np.float64)[None, :]
    level = np.arange(nlay + 1, dtype=np.float64)[None, :]
    return {
        "tau": 0.01 + 0.02 * (1.0 + np.sin(0.00007 * row + 0.31 * layer)),
        "ssa": 0.06 + 0.03 * (1.0 + np.cos(0.00005 * row + 0.23 * layer)),
        "g": 0.15 + 0.04 * np.sin(0.00003 * row + 0.17 * layer),
        "z": np.linspace(float(nlay), 0.0, nlay + 1, dtype=np.float64),
        "angles": 30.0,
        "stream": 0.5,
        "planck": 0.8 + 0.015 * level + 0.02 * np.sin(0.00004 * row + 0.19 * level),
        "surface_planck": 1.2 + 0.03 * np.sin(np.arange(rows, dtype=np.float64) * 0.00009),
        "emissivity": 0.88 + 0.02 * np.cos(np.arange(rows, dtype=np.float64) * 0.00011),
        "albedo": 0.04 + 0.01 * np.sin(np.arange(rows, dtype=np.float64) * 0.00013),
        "delta_m_truncation_factor": np.zeros((rows, nlay), dtype=np.float64),
    }


def _solar_case(rows: int, nlay: int) -> dict[str, Any]:
    row = np.arange(rows, dtype=np.float64)[:, None]
    layer = np.arange(nlay, dtype=np.float64)[None, :]
    return {
        "tau": 0.005 + 0.015 * (1.0 + np.sin(0.00008 * row + 0.29 * layer)),
        "ssa": 0.08 + 0.04 * (1.0 + np.cos(0.00006 * row + 0.21 * layer)),
        "g": 0.1 + 0.05 * np.sin(0.00004 * row + 0.13 * layer),
        "z": np.linspace(float(nlay), 0.0, nlay + 1, dtype=np.float64),
        "angles": [30.0, 20.0, 0.0],
        "stream": 0.5,
        "fbeam": 1.0 + 0.1 * np.sin(np.arange(rows, dtype=np.float64) * 0.0001),
        "albedo": 0.12 + 0.03 * np.cos(np.arange(rows, dtype=np.float64) * 0.00012),
        "delta_m_truncation_factor": np.zeros((rows, nlay), dtype=np.float64),
    }


def _torch_inputs(inputs: dict[str, Any], *, device: str, dtype: str) -> dict[str, Any]:
    import torch

    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray) and key != "z":
            out[key] = torch.as_tensor(value, dtype=torch_dtype, device=device)
        else:
            out[key] = value
    return out


def _checksum(result: Any) -> float:
    total = 0.0
    for field in ("flux_up", "flux_down", "flux_net", "flux_mean"):
        value = getattr(result, field)
        if value is not None:
            total += float(np.asarray(to_numpy(value)).sum())
    return total


def _run_level_flux_solver(solver: TwoStreamEss, run_inputs: dict[str, Any], *, include_fo: bool):
    if solver.options.backend in {"native", "torch"}:
        return solver.forward_flux(**run_inputs, include_fo=include_fo, return_net=True)
    return solver.forward(**run_inputs, include_fo=include_fo, fo_n_moments=3)


def _sample_indices(rows: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=int)
    if count >= rows:
        return np.arange(rows, dtype=int)
    return np.unique(np.linspace(0, rows - 1, count, dtype=int))


def _row_subset(inputs: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    rows = int(inputs["tau"].shape[0])
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if key != "z" and isinstance(value, np.ndarray) and value.shape[:1] == (rows,):
            out[key] = value[indices]
        else:
            out[key] = value
    return out


def _hg_phase_moments(g: np.ndarray, nmom: int) -> np.ndarray:
    orders = np.arange(1, int(nmom) + 1, dtype=float)
    return np.asarray(g, dtype=float)[:, None] ** orders[None, :]


def _level_values(values: Any, row: int | None = None) -> np.ndarray:
    array = np.asarray(to_numpy(values), dtype=float)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array
    if row is None:
        raise ValueError("row is required for batched level values")
    return np.asarray(array[row], dtype=float)


def _max_rel_percent(actual: np.ndarray, reference: np.ndarray) -> float:
    diff = np.abs(actual - reference)
    denom = np.maximum(np.abs(reference), 1.0e-12)
    return float(100.0 * np.max(diff / denom))


def _time_backend(
    *,
    case_name: str,
    inputs: dict[str, Any],
    backend: str,
    device: str,
    dtype: str,
    repeats: int,
    warmup: int,
    include_fo: bool,
    fo_flux_n_mu: int,
) -> list[dict[str, Any]]:
    if backend in {"torch", "native"} and not has_torch():
        return []
    if device == "cuda" and not _cuda_available():
        return []
    if backend == "native":
        info = native_backend_info()
        if not info.get("available", False):
            return []
        if device == "cuda" and not info.get("cuda", False):
            return []

    mode = "thermal" if case_name == "thermal" else "solar"
    run_inputs = (
        _torch_inputs(inputs, device=device, dtype=dtype)
        if backend in {"torch", "native"}
        else inputs
    )
    options = TwoStreamEssOptions(
        nlyr=int(inputs["tau"].shape[-1]),
        mode=mode,
        backend=backend,
        torch_device=device if backend in {"torch", "native"} else None,
        torch_dtype=dtype,
        plane_parallel=True,
        output_levels=backend not in {"native", "torch"},
        output_fluxes=backend not in {"native", "torch"},
        fo_flux_n_mu=fo_flux_n_mu,
    )
    solver = TwoStreamEss(options)
    rows = []
    for _ in range(warmup):
        result = _run_level_flux_solver(solver, run_inputs, include_fo=include_fo)
        _sync(device)
        _checksum(result)
    for rep in range(repeats):
        _sync(device)
        start = time.perf_counter()
        result = _run_level_flux_solver(solver, run_inputs, include_fo=include_fo)
        _sync(device)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "case": case_name,
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "rows": int(inputs["tau"].shape[0]),
                "nlay": int(inputs["tau"].shape[1]),
                "include_fo": include_fo,
                "fo_flux_n_mu": fo_flux_n_mu,
                "repeat": rep,
                "seconds": elapsed,
                "checksum": _checksum(result),
            }
        )
    return rows


def _run_backend_once(
    *,
    case_name: str,
    inputs: dict[str, Any],
    backend: str,
    device: str,
    dtype: str,
    include_fo: bool,
    fo_flux_n_mu: int,
) -> tuple[Any, float] | None:
    if backend in {"torch", "native"} and not has_torch():
        return None
    if device == "cuda" and not _cuda_available():
        return None
    if backend == "native":
        info = native_backend_info()
        if not info.get("available", False):
            return None
        if device == "cuda" and not info.get("cuda", False):
            return None

    mode = "thermal" if case_name == "thermal" else "solar"
    run_inputs = (
        _torch_inputs(inputs, device=device, dtype=dtype)
        if backend in {"torch", "native"}
        else inputs
    )
    options = TwoStreamEssOptions(
        nlyr=int(inputs["tau"].shape[-1]),
        mode=mode,
        backend=backend,
        torch_device=device if backend in {"torch", "native"} else None,
        torch_dtype=dtype,
        plane_parallel=True,
        output_levels=backend not in {"native", "torch"},
        output_fluxes=backend not in {"native", "torch"},
        fo_flux_n_mu=fo_flux_n_mu,
    )
    solver = TwoStreamEss(options)
    _sync(device)
    start = time.perf_counter()
    result = _run_level_flux_solver(solver, run_inputs, include_fo=include_fo)
    _sync(device)
    return result, time.perf_counter() - start


def _pydisort_same_input_comparison(
    *,
    solar_inputs: dict[str, Any],
    sample_rows: int,
    backends: list[str],
    devices: list[str],
    dtype: str,
    include_fo: bool,
    fo_flux_n_mu: int,
    nstr: int,
    nmom: int,
    warmup: int,
) -> list[dict[str, Any]]:
    try:
        from py2sess.benchmarks.flux_references import run_pydisort_solar_flux
    except Exception:
        return []

    indices = _sample_indices(int(solar_inputs["tau"].shape[0]), sample_rows)
    if indices.size == 0:
        return []

    sample_inputs = _row_subset(solar_inputs, indices)
    mu0 = math.cos(math.radians(float(sample_inputs["angles"][0])))
    for _ in range(max(0, int(warmup))):
        run_pydisort_solar_flux(
            sample_inputs["tau"][0],
            ssa=sample_inputs["ssa"][0],
            phase_moments=_hg_phase_moments(sample_inputs["g"][0], nmom),
            mu0=mu0,
            fbeam=float(sample_inputs["fbeam"][0]),
            albedo=float(sample_inputs["albedo"][0]),
            nstr=nstr,
            nmom=nmom,
            dtype=dtype,
        )

    references: list[tuple[int, dict[str, Any], float]] = []
    for local_row, source_row in enumerate(indices):
        start = time.perf_counter()
        reference = run_pydisort_solar_flux(
            sample_inputs["tau"][local_row],
            ssa=sample_inputs["ssa"][local_row],
            phase_moments=_hg_phase_moments(sample_inputs["g"][local_row], nmom),
            mu0=mu0,
            fbeam=float(sample_inputs["fbeam"][local_row]),
            albedo=float(sample_inputs["albedo"][local_row]),
            nstr=nstr,
            nmom=nmom,
            dtype=dtype,
        )
        references.append((int(source_row), reference, time.perf_counter() - start))

    rows: list[dict[str, Any]] = []
    for backend in backends:
        run_devices = ["cpu"] if backend == "numpy" else devices
        for device in run_devices:
            run = _run_backend_once(
                case_name="solar",
                inputs=sample_inputs,
                backend=backend,
                device=device,
                dtype=dtype,
                include_fo=include_fo,
                fo_flux_n_mu=fo_flux_n_mu,
            )
            if run is None:
                continue
            result, py2sess_seconds = run
            for local_row, (source_row, reference, pydisort_seconds) in enumerate(references):
                for field in ("flux_up", "flux_down", "flux_net", "flux_mean"):
                    field_value = getattr(result, field)
                    if field_value is None:
                        continue
                    actual = _level_values(field_value, row=local_row)
                    expected = _level_values(reference[field])
                    rows.append(
                        {
                            "case": "solar_same_input",
                            "row_index": source_row,
                            "field": field,
                            "backend": backend,
                            "device": device,
                            "dtype": dtype,
                            "nlay": int(sample_inputs["tau"].shape[1]),
                            "include_fo": include_fo,
                            "fo_flux_n_mu": fo_flux_n_mu,
                            "pydisort_nstr": int(nstr),
                            "pydisort_nmom": int(nmom),
                            "pydisort_seconds": pydisort_seconds,
                            "py2sess_sample_seconds": py2sess_seconds,
                            "max_abs_diff": float(np.max(np.abs(actual - expected))),
                            "max_rel_diff_percent": _max_rel_percent(actual, expected),
                        }
                    )
    return rows


def _pydisort_smoke(repeats: int) -> list[dict[str, Any]]:
    try:
        from py2sess.benchmarks.flux_references import run_pydisort_absorbing_solar_flux
    except Exception:
        return []
    tau = np.array([0.02, 0.03, 0.04], dtype=np.float64)
    mu0 = math.cos(math.radians(30.0))
    rows = []
    for rep in range(repeats):
        start = time.perf_counter()
        result = run_pydisort_absorbing_solar_flux(tau, mu0=mu0, fbeam=1.0, albedo=0.1)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "case": "pydisort_absorbing_solar_scalar",
                "backend": "pydisort",
                "device": "cpu",
                "dtype": "float64",
                "rows": 1,
                "nlay": int(tau.size),
                "include_fo": False,
                "fo_flux_n_mu": "",
                "repeat": rep,
                "seconds": elapsed,
                "checksum": float(np.asarray(result["flux_up"]).sum()),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = (
            row["case"],
            row["backend"],
            row["device"],
            row["dtype"],
            row["rows"],
            row["nlay"],
            row["include_fo"],
            row["fo_flux_n_mu"],
        )
        groups.setdefault(key, []).append(float(row["seconds"]))
    summary = []
    for key, values in sorted(groups.items()):
        summary.append(
            {
                "case": key[0],
                "backend": key[1],
                "device": key[2],
                "dtype": key[3],
                "rows": key[4],
                "nlay": key[5],
                "include_fo": key[6],
                "fo_flux_n_mu": key[7],
                "n": len(values),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values)),
                "min_seconds": float(np.min(values)),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--nlay", type=int, default=26)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--fo-flux-n-mu", type=int, default=8)
    parser.add_argument("--case", action="append", choices=("thermal", "solar"))
    parser.add_argument("--backend", action="append", choices=("numpy", "torch", "native"))
    parser.add_argument("--device", action="append", choices=("cpu", "cuda"))
    parser.add_argument("--no-fo", action="store_true")
    parser.add_argument("--pydisort-smoke", action="store_true")
    parser.add_argument("--pydisort-same-inputs", action="store_true")
    parser.add_argument("--pydisort-sample-rows", type=int, default=3)
    parser.add_argument("--pydisort-nstr", type=int, default=16)
    parser.add_argument("--pydisort-nmom", type=int, default=16)
    parser.add_argument("--pydisort-warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/level_flux_speed_colab"))
    args = parser.parse_args()

    cases = args.case or ["thermal", "solar"]
    backends = args.backend or ["numpy", "torch", "native"]
    devices = args.device or ["cpu", "cuda"]
    include_fo = not args.no_fo

    inputs_by_case = {
        "thermal": _thermal_case(args.rows, args.nlay),
        "solar": _solar_case(args.rows, args.nlay),
    }
    rows: list[dict[str, Any]] = []
    for case_name in cases:
        for backend in backends:
            run_devices = ["cpu"] if backend == "numpy" else devices
            for device in run_devices:
                rows.extend(
                    _time_backend(
                        case_name=case_name,
                        inputs=inputs_by_case[case_name],
                        backend=backend,
                        device=device,
                        dtype=args.dtype,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        include_fo=include_fo,
                        fo_flux_n_mu=args.fo_flux_n_mu,
                    )
                )
    if args.pydisort_smoke:
        rows.extend(_pydisort_smoke(args.repeats))

    pydisort_comparison: list[dict[str, Any]] = []
    if args.pydisort_same_inputs:
        pydisort_comparison = _pydisort_same_input_comparison(
            solar_inputs=inputs_by_case["solar"],
            sample_rows=args.pydisort_sample_rows,
            backends=backends,
            devices=devices,
            dtype=args.dtype,
            include_fo=include_fo,
            fo_flux_n_mu=args.fo_flux_n_mu,
            nstr=args.pydisort_nstr,
            nmom=args.pydisort_nmom,
            warmup=args.pydisort_warmup,
        )

    summary = _summarize(rows)
    _write_csv(args.output_dir / "raw_level_flux_speed.csv", rows)
    _write_csv(args.output_dir / "summary_level_flux_speed.csv", summary)
    _write_csv(args.output_dir / "pydisort_same_input_comparison.csv", pydisort_comparison)
    print(f"wrote {args.output_dir / 'raw_level_flux_speed.csv'}")
    print(f"wrote {args.output_dir / 'summary_level_flux_speed.csv'}")
    if pydisort_comparison:
        print(f"wrote {args.output_dir / 'pydisort_same_input_comparison.csv'}")
    print(
        f"{'case':<36s} {'backend':<8s} {'device':<6s} {'rows':>8s} "
        f"{'nlay':>4s} {'mean s':>10s} {'std s':>10s}"
    )
    for row in summary:
        print(
            f"{row['case']:<36s} {row['backend']:<8s} {row['device']:<6s} "
            f"{row['rows']:8d} {row['nlay']:4d} "
            f"{row['mean_seconds']:10.6g} {row['std_seconds']:10.3g}"
        )
    if pydisort_comparison:
        print("\npydisort same-input solar comparison; max over levels")
        print(
            f"{'row':>8s} {'field':<10s} {'backend':<8s} {'device':<6s} "
            f"{'abs diff':>12s} {'rel diff %':>12s} {'pydisort s':>12s}"
        )
        for row in pydisort_comparison:
            print(
                f"{row['row_index']:8d} {row['field']:<10s} {row['backend']:<8s} "
                f"{row['device']:<6s} {row['max_abs_diff']:12.6g} "
                f"{row['max_rel_diff_percent']:12.6g} {row['pydisort_seconds']:12.6g}"
            )


if __name__ == "__main__":
    main()
