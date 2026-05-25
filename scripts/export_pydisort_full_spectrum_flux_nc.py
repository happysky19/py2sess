#!/usr/bin/env python3
"""Export pydisort full-spectrum level flux references to NetCDF."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from py2sess.benchmarks.flux_references import (
    PYDISORT_FLUX_CHANNELS,
    pydisort_flux_to_py2sess,
)
from py2sess.optical.delta_m import delta_m_scale_optical_properties
from py2sess.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT_ENV = "PY2SESS_EXTERNAL_ROOT"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pydisort_full_spectrum_flux"
CHANNELS = PYDISORT_FLUX_CHANNELS

try:
    import torch
except ImportError:  # pragma: no cover - exercised in minimal CI environments
    torch = None


@dataclass(frozen=True)
class CaseSpec:
    key: str
    mode: str
    profile: Path
    scene: Path
    default_rows: int


def _case_specs(input_root: Path | None) -> dict[str, CaseSpec]:
    if input_root is None:
        env_root = os.environ.get(EXTERNAL_ROOT_ENV)
        if not env_root:
            raise ValueError(
                f"full-spectrum pydisort export requires --input-root or {EXTERNAL_ROOT_ENV}"
            )
        bundle_root = ROOT / "benchmark_bundles"
        profile_root = Path(env_root).expanduser() / "geocape_data" / "Profile_Data"
    else:
        bundle_root = input_root / "benchmark_bundles"
        profile_root = input_root / "profiles"
    return {
        "tir": CaseSpec(
            key="tir",
            mode="thermal",
            profile=profile_root / "Profiles_1_2006726_0000.dat",
            scene=bundle_root / "tir_scene_python.yaml",
            default_rows=200_000,
        ),
        "uv": CaseSpec(
            key="uv",
            mode="solar",
            profile=profile_root / "Profiles_1_2006726_1500.dat",
            scene=bundle_root / "uv_scene_python.yaml",
            default_rows=280_000,
        ),
    }


def _split_cases(value: str) -> tuple[str, ...]:
    cases = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(set(cases) - {"tir", "uv"})
    if unknown:
        raise ValueError(f"unsupported case(s): {', '.join(unknown)}")
    return cases


def _require_netcdf4():
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required: pip install netCDF4") from exc
    return netCDF4


def _require_pydisort():
    try:
        import pydisort
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pydisort is required: pip install pydisort") from exc
    return pydisort


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required: pip install torch") from None
    return torch


def _hg_moments(g: np.ndarray, nmom: int) -> np.ndarray:
    orders = np.arange(1, int(nmom) + 1, dtype=np.float64)
    return np.asarray(g, dtype=np.float64)[..., None] ** orders


def _prop_tensor(
    *,
    tau: np.ndarray,
    ssa: np.ndarray,
    g: np.ndarray,
    nmom: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    nrow, nlay = tau.shape
    prop = np.empty((nrow, 1, nlay, 2 + int(nmom)), dtype=np.float64)
    prop[..., 0] = tau[:, None, :]
    prop[..., 1] = ssa[:, None, :]
    prop[..., 2:] = _hg_moments(g, nmom)[:, None, :, :]
    return torch.as_tensor(prop, dtype=dtype)


def _maybe_delta_m(
    kwargs: dict[str, Any],
    start: int,
    stop: int,
    *,
    apply_delta_m: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(kwargs["tau"][start:stop], dtype=np.float64)
    ssa = np.asarray(kwargs["ssa"][start:stop], dtype=np.float64)
    g = np.asarray(kwargs["g"][start:stop], dtype=np.float64)
    if not apply_delta_m:
        return tau, ssa, g
    scaling = np.asarray(kwargs["delta_m_truncation_factor"][start:stop], dtype=np.float64)
    return delta_m_scale_optical_properties(tau, ssa, g, scaling)


def _make_solver(
    *,
    mode: str,
    nrow: int,
    nlay: int,
    nstr: int,
    nmom: int,
    wave_lower: np.ndarray | None = None,
    wave_upper: np.ndarray | None = None,
):
    pydisort = _require_pydisort()
    flags = "onlyfl,lamber,quiet"
    options = pydisort.DisortOptions().flags(flags).nwave(int(nrow)).ncol(1)
    if mode == "thermal":
        if wave_lower is None or wave_upper is None:
            raise ValueError("thermal pydisort export requires wavenumber band edges")
        flags = "onlyfl,lamber,planck,quiet"
        options = (
            pydisort.DisortOptions()
            .flags(flags)
            .nwave(int(nrow))
            .ncol(1)
            .wave_lower([float(v) for v in wave_lower])
            .wave_upper([float(v) for v in wave_upper])
        )
    options.ds().nlyr = int(nlay)
    options.ds().nstr = int(nstr)
    options.ds().nmom = int(nmom)
    options.ds().nphase = int(nmom)
    return pydisort.Disort(options)


def _run_solar_chunk(
    *,
    kwargs: dict[str, Any],
    start: int,
    stop: int,
    nstr: int,
    nmom: int,
    dtype: torch.dtype,
    apply_delta_m: bool,
) -> np.ndarray:
    tau, ssa, g = _maybe_delta_m(kwargs, start, stop, apply_delta_m=apply_delta_m)
    prop = _prop_tensor(tau=tau, ssa=ssa, g=g, nmom=nmom, dtype=dtype)
    angles = np.asarray(kwargs["angles"], dtype=float).reshape(-1)
    mu0 = math.cos(math.radians(float(angles[0])))
    solver = _make_solver(mode="solar", nrow=stop - start, nlay=tau.shape[1], nstr=nstr, nmom=nmom)
    bc = {
        "umu0": torch.tensor([mu0], dtype=dtype),
        "fbeam": torch.as_tensor(kwargs["fbeam"][start:stop, None], dtype=dtype),
        "albedo": torch.as_tensor(kwargs["albedo"][start:stop, None], dtype=dtype),
    }
    solver.forward(prop, bc, "", None)
    return solver.gather_flx().detach().cpu().numpy()[:, 0, :, :]


def _thermal_bands(bundle: dict[str, Any]) -> np.ndarray:
    if "wavenumber_band_cm_inv" in bundle:
        bands = np.asarray(bundle["wavenumber_band_cm_inv"], dtype=np.float64)
        if bands.ndim == 2 and bands.shape[1] == 2:
            return bands
    if "wavenumber_cm_inv" in bundle:
        centers = np.asarray(bundle["wavenumber_cm_inv"], dtype=np.float64)
        if centers.size < 2:
            raise ValueError("cannot infer thermal band edges from one wavenumber")
        edges = np.empty((centers.size, 2), dtype=np.float64)
        mid = 0.5 * (centers[1:] + centers[:-1])
        edges[1:, 0] = mid
        edges[:-1, 1] = mid
        edges[0, 0] = centers[0] - (mid[0] - centers[0])
        edges[-1, 1] = centers[-1] + (centers[-1] - mid[-1])
        return edges
    raise ValueError("thermal scene must provide wavenumber_band_cm_inv or wavenumber_cm_inv")


def _temperature_profile(bundle: dict[str, Any]) -> np.ndarray:
    if "level_temperature_k" not in bundle:
        raise ValueError("thermal scene is missing level_temperature_k")
    temperature = np.asarray(bundle["level_temperature_k"], dtype=np.float64)
    if temperature.ndim != 1:
        raise ValueError("pydisort thermal export requires one shared temperature profile")
    return temperature


def _surface_temperature(bundle: dict[str, Any]) -> float:
    if "surface_temperature_k" in bundle:
        return float(np.asarray(bundle["surface_temperature_k"], dtype=np.float64).reshape(-1)[0])
    temperature = _temperature_profile(bundle)
    return float(temperature[-1])


def _run_thermal_chunk(
    *,
    kwargs: dict[str, Any],
    bundle: dict[str, Any],
    start: int,
    stop: int,
    nstr: int,
    nmom: int,
    dtype: torch.dtype,
    apply_delta_m: bool,
) -> np.ndarray:
    tau, ssa, g = _maybe_delta_m(kwargs, start, stop, apply_delta_m=apply_delta_m)
    bands = _thermal_bands(bundle)[start:stop]
    prop = _prop_tensor(tau=tau, ssa=ssa, g=g, nmom=nmom, dtype=dtype)
    solver = _make_solver(
        mode="thermal",
        nrow=stop - start,
        nlay=tau.shape[1],
        nstr=nstr,
        nmom=nmom,
        wave_lower=bands[:, 0],
        wave_upper=bands[:, 1],
    )
    level_temperature = _temperature_profile(bundle)
    surface_temperature = _surface_temperature(bundle)
    emissivity = np.asarray(kwargs["emissivity"][start:stop], dtype=np.float64)
    bc = {
        "albedo": torch.as_tensor(kwargs["albedo"][start:stop, None], dtype=dtype),
        "temis": torch.as_tensor(emissivity[:, None], dtype=dtype),
        "btemp": torch.tensor([surface_temperature], dtype=dtype),
        "ttemp": torch.tensor([0.0], dtype=dtype),
    }
    temf = torch.as_tensor(np.ascontiguousarray(level_temperature.reshape(1, -1)), dtype=dtype)
    solver.forward(prop, bc, "", temf)
    return solver.gather_flx().detach().cpu().numpy()[:, 0, :, :]


def _mapped_flux(raw: np.ndarray) -> dict[str, np.ndarray]:
    mapped = pydisort_flux_to_py2sess(raw)
    return {
        "flux_up": mapped["flux_up"],
        "flux_down": mapped["flux_down"],
        "flux_net": mapped["flux_net"],
        "flux_mean": mapped["flux_mean"],
    }


def _create_output(
    path: Path,
    *,
    case: CaseSpec,
    nrow: int,
    nlev: int,
    chunk_size: int,
    output_dtype: str,
    compression: int,
    store: str,
    metadata: dict[str, Any],
):
    netCDF4 = _require_netcdf4()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = netCDF4.Dataset(path, "w")
    data.createDimension("spectral_row", nrow)
    data.createDimension("level", nlev)
    data.createDimension("channel", len(CHANNELS))
    data.createDimension("band_edge", 2)
    data.createDimension("chunk", None)
    data.setncattr("case", case.key)
    data.setncattr("mode", case.mode)
    data.setncattr("level_axis", "toa_to_boa")
    data.setncattr("pydisort_flux_channels", ",".join(CHANNELS))
    for key, value in metadata.items():
        data.setncattr(key, value)
    data.createVariable("completed", "i1", ("spectral_row",))[:] = 0
    data.createVariable("wavelength_nm", "f8", ("spectral_row",))
    data.createVariable("chunk_start", "i8", ("chunk",))
    data.createVariable("chunk_stop", "i8", ("chunk",))
    data.createVariable("chunk_seconds", "f8", ("chunk",))
    data.createVariable("chunk_rows_per_second", "f8", ("chunk",))
    kwargs = {
        "zlib": compression > 0,
        "complevel": int(compression),
        "shuffle": True,
        "fill_value": np.nan,
        "chunksizes": (min(int(chunk_size), nrow), nlev),
    }
    if store in {"mapped", "both"}:
        for name in ("flux_up", "flux_down", "flux_net", "flux_mean"):
            data.createVariable(name, output_dtype, ("spectral_row", "level"), **kwargs)
    if store in {"raw", "both"}:
        raw_kwargs = dict(kwargs)
        raw_kwargs["chunksizes"] = (min(int(chunk_size), nrow), nlev, len(CHANNELS))
        data.createVariable(
            "pydisort_flux_raw",
            output_dtype,
            ("spectral_row", "level", "channel"),
            **raw_kwargs,
        )
    return data


def _open_or_create_output(
    path: Path,
    *,
    args: argparse.Namespace,
    case: CaseSpec,
    nrow: int,
    nlev: int,
    metadata: dict[str, Any],
):
    netCDF4 = _require_netcdf4()
    if path.exists():
        if args.overwrite:
            path.unlink()
        elif args.resume:
            return netCDF4.Dataset(path, "r+")
        else:
            raise FileExistsError(f"{path} exists; use --overwrite or --resume")
    return _create_output(
        path,
        case=case,
        nrow=nrow,
        nlev=nlev,
        chunk_size=args.chunk_size,
        output_dtype=args.output_dtype,
        compression=args.compression,
        store=args.store,
        metadata=metadata,
    )


def _write_coordinates(data: Any, *, inputs: Any, bundle: dict[str, Any]) -> None:
    nrow = len(data.dimensions["spectral_row"])
    data.variables["wavelength_nm"][:] = np.asarray(inputs.wavelengths, dtype=np.float64)[:nrow]
    if "wavenumber_cm_inv" in bundle:
        if "wavenumber_cm_inv" not in data.variables:
            data.createVariable("wavenumber_cm_inv", "f8", ("spectral_row",))
        data.variables["wavenumber_cm_inv"][:] = np.asarray(
            bundle["wavenumber_cm_inv"], dtype=np.float64
        )[:nrow]
    if "wavenumber_band_cm_inv" in bundle:
        if "wavenumber_band_cm_inv" not in data.variables:
            data.createVariable("wavenumber_band_cm_inv", "f8", ("spectral_row", "band_edge"))
        data.variables["wavenumber_band_cm_inv"][:] = np.asarray(
            bundle["wavenumber_band_cm_inv"], dtype=np.float64
        )[:nrow, :]


def _write_flux(data: Any, *, start: int, stop: int, raw: np.ndarray, store: str) -> None:
    if store in {"raw", "both"}:
        data.variables["pydisort_flux_raw"][start:stop, :, :] = raw
    if store in {"mapped", "both"}:
        mapped = _mapped_flux(raw)
        for name, values in mapped.items():
            data.variables[name][start:stop, :] = values
    data.variables["completed"][start:stop] = 1


def _runtime_csv_path(path: Path) -> Path:
    return path.with_suffix(".runtime.csv")


def _write_runtime_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with _runtime_csv_path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_case(case: CaseSpec, *, args: argparse.Namespace) -> dict[str, Any]:
    load_start = time.perf_counter()
    scene = load_scene(profile=case.profile, config=case.scene, strict_runtime_inputs=True)
    inputs = scene.to_forward_inputs()
    load_seconds = time.perf_counter() - load_start
    kwargs = inputs.kwargs
    bundle = getattr(scene, "_bundle", {})
    nrows_total = int(np.asarray(kwargs["tau"]).shape[0])
    nrows = min(args.limit or nrows_total, nrows_total)
    nlay = int(np.asarray(kwargs["tau"]).shape[1])
    nlev = nlay + 1
    if nrows != case.default_rows and args.limit is None:
        print(f"warning: {case.key} loaded {nrows} rows; expected {case.default_rows}")

    output = args.output_dir / f"pydisort_{case.key}_flux.nc"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(case.profile),
        "scene": str(case.scene),
        "pydisort_nstr": int(args.nstr),
        "pydisort_nmom": int(args.nmom),
        "compute_dtype": args.compute_dtype,
        "output_dtype": args.output_dtype,
        "apply_delta_m": int(args.apply_delta_m),
        "store": args.store,
    }
    if case.mode == "thermal":
        metadata["thermal_top_boundary_temperature_k"] = 0.0
    data = _open_or_create_output(
        output, args=args, case=case, nrow=nrows, nlev=nlev, metadata=metadata
    )
    _write_coordinates(data, inputs=inputs, bundle=bundle)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.compute_dtype]
    completed = np.asarray(data.variables["completed"][:], dtype=np.int8)
    chunk_index = len(data.dimensions["chunk"])
    rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    recorded_seconds = 0.0
    try:
        for start in range(0, nrows, args.chunk_size):
            stop = min(start + args.chunk_size, nrows)
            if args.resume and np.all(completed[start:stop] == 1):
                continue
            chunk_start = time.perf_counter()
            if case.mode == "solar":
                raw = _run_solar_chunk(
                    kwargs=kwargs,
                    start=start,
                    stop=stop,
                    nstr=args.nstr,
                    nmom=args.nmom,
                    dtype=dtype,
                    apply_delta_m=args.apply_delta_m,
                )
            else:
                raw = _run_thermal_chunk(
                    kwargs=kwargs,
                    bundle=bundle,
                    start=start,
                    stop=stop,
                    nstr=args.nstr,
                    nmom=args.nmom,
                    dtype=dtype,
                    apply_delta_m=args.apply_delta_m,
                )
            seconds = time.perf_counter() - chunk_start
            _write_flux(data, start=start, stop=stop, raw=raw, store=args.store)
            data.variables["chunk_start"][chunk_index] = start
            data.variables["chunk_stop"][chunk_index] = stop
            data.variables["chunk_seconds"][chunk_index] = seconds
            data.variables["chunk_rows_per_second"][chunk_index] = (stop - start) / seconds
            data.sync()
            row = {
                "case": case.key,
                "mode": case.mode,
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "seconds": seconds,
                "rows_per_second": (stop - start) / seconds,
            }
            rows.append(row)
            print(
                f"{case.key} {start:>8d}:{stop:<8d} "
                f"{seconds:9.3f} s {row['rows_per_second']:10.1f} row/s",
                flush=True,
            )
            chunk_index += 1
    finally:
        current_seconds = time.perf_counter() - run_start
        done = np.asarray(data.variables["completed"][:], dtype=np.int8)
        chunk_seconds = np.ma.filled(data.variables["chunk_seconds"][:], np.nan)
        chunk_seconds = np.asarray(chunk_seconds, dtype=np.float64)
        recorded_seconds = float(np.nansum(chunk_seconds))
        data.setncattr("load_seconds", float(load_seconds))
        data.setncattr("current_run_seconds", float(current_seconds))
        data.setncattr("run_seconds", float(recorded_seconds))
        data.setncattr("completed_rows", int(np.sum(done)))
        data.setncattr(
            "rows_per_second",
            float(np.sum(done) / recorded_seconds) if recorded_seconds > 0.0 else 0.0,
        )
        data.close()
    _write_runtime_csv(output, rows)
    return {
        "case": case.key,
        "mode": case.mode,
        "rows": nrows,
        "layers": nlay,
        "output": str(output),
        "runtime_csv": str(_runtime_csv_path(output)),
        "seconds": recorded_seconds,
        "rows_per_second": nrows / recorded_seconds if recorded_seconds > 0.0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Portable full-spectrum input bundle root. When omitted, profile files "
            f"are read from ${EXTERNAL_ROOT_ENV}/geocape_data/Profile_Data and scenes "
            "from the repo benchmark_bundles directory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", default="tir,uv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--nstr", type=int, default=16)
    parser.add_argument("--nmom", type=int, default=16)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dtype", choices=("f4", "f8"), default="f8")
    parser.add_argument("--compression", type=int, default=4)
    parser.add_argument("--store", choices=("mapped", "raw", "both"), default="mapped")
    parser.add_argument("--apply-delta-m", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not 0 <= args.compression <= 9:
        raise ValueError("--compression must be in [0, 9]")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    torch_module = _require_torch()
    torch_module.set_num_threads(args.torch_threads)

    specs = _case_specs(args.input_root)
    summaries = [_run_case(specs[key], args=args) for key in _split_cases(args.cases)]
    summary_path = args.output_dir / "pydisort_full_spectrum_flux_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {summary_path}", flush=True)
    for row in summaries:
        print(
            f"{row['case']:<4s} rows={row['rows']} layers={row['layers']} "
            f"seconds={row['seconds']:.3f} rows/s={row['rows_per_second']:.1f} "
            f"output={row['output']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
