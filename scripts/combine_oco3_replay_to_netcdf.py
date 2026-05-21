#!/usr/bin/env python3
"""Combine raw OCO-3 replay CSV outputs into one NetCDF product."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


BAND_ORDER = ("o2", "wco2", "sco2")
BAND_LABELS = {
    "o2": "O2 A",
    "wco2": "weak CO2",
    "sco2": "strong CO2",
}
GROUP_LABELS = {
    "land_nd": "Land-Nadir",
    "land_am": "Land-SAM",
    "ocean_gl": "Ocean-Glint",
}
SUMMARY_FLOATS = {
    "continuum_referenced_bias_percent": "bias_percent",
    "continuum_referenced_rmse_percent": "rmse_percent",
    "continuum_referenced_max_abs_percent": "max_abs_percent",
    "surface_adjustment_scale": "surface_adjustment_scale",
    "surface_adjustment_tilt": "surface_adjustment_tilt",
    "surface_adjustment_fit_rmse_percent": "surface_adjustment_fit_rmse_percent",
    "aerosol_total_aod_used": "aerosol_total_aod_used",
    "polarization_median_continuum_percent": "polarization_median_continuum_percent",
    "polarization_max_abs_continuum_percent": "polarization_max_abs_continuum_percent",
}
SPECTRUM_FLOATS = {
    "wavelength_um": "wavelength_um",
    "measured_radiance_w_m2_sr_um": "measured_radiance_w_m2_sr_um",
    "py2sess_radiance_w_m2_sr_um": "py2sess_radiance_w_m2_sr_um",
    "py2sess_scalar_radiance_w_m2_sr_um": "py2sess_scalar_radiance_w_m2_sr_um",
    "py2sess_polarization_correction_w_m2_sr_um": ("py2sess_polarization_correction_w_m2_sr_um"),
    "py2sess_minus_measured_continuum_percent": "residual_percent_continuum",
    "oco_continuum_signal_radiance_w_m2_sr_um": "oco_continuum_signal_w_m2_sr_um",
}
SELECTED_FLOATS = {
    "latitude": "latitude",
    "longitude": "longitude",
    "land_fraction": "land_fraction",
    "solar_zenith_deg": "solar_zenith_deg",
    "view_zenith_deg": "view_zenith_deg",
    "aerosol_total_aod": "l2_aerosol_total_aod",
    "st_aod": "l2_st_aod",
    "surface_pressure_hpa": "l2_surface_pressure_hpa",
    "xco2_ppm": "l2_xco2_ppm",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _replay_dirs(raw_root: Path) -> list[Path]:
    return sorted(path.parent for path in raw_root.glob("*/*/py2sess_replay_summary.csv"))


def _case_key(case_dir: Path, row: dict[str, str]) -> tuple[str, str, int]:
    granule = case_dir.parent.name
    group_key = case_dir.name
    return granule, group_key, int(row["sounding_id"])


def _collect_cases(raw_root: Path) -> tuple[list[tuple[str, str, int]], dict, dict, dict]:
    selected_rows: dict[tuple[str, str, int], dict[str, str]] = {}
    summary_rows: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    spectrum_rows: dict[tuple[str, str, int], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for case_dir in _replay_dirs(raw_root):
        selected_path = case_dir / "selected_soundings.csv"
        if selected_path.exists():
            for row in _read_csv(selected_path):
                key = (case_dir.parent.name, case_dir.name, int(row["sounding_id"]))
                selected_rows[key] = row
        for row in _read_csv(case_dir / "py2sess_replay_summary.csv"):
            key = _case_key(case_dir, row)
            summary_rows[key][row["band"]] = row
        spectrum_path = case_dir / "py2sess_replay_spectrum.csv"
        for row in _read_csv(spectrum_path):
            key = _case_key(case_dir, row)
            spectrum_rows[key][row["band"]].append(row)

    keys = sorted(summary_rows, key=lambda item: (item[1], item[0], item[2]))
    return keys, selected_rows, summary_rows, spectrum_rows


def _numeric_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, data in dataset.data_vars.items():
        if np.issubdtype(data.dtype, np.number):
            encoding[name] = {"zlib": True, "complevel": 4}
    return encoding


def combine(raw_root: Path, output: Path) -> xr.Dataset:
    keys, selected_rows, summary_rows, spectrum_rows = _collect_cases(raw_root)
    if not keys:
        raise FileNotFoundError(f"no replay CSV outputs found under {raw_root}")

    max_colors = 0
    for key in keys:
        for band in BAND_ORDER:
            max_colors = max(max_colors, len(spectrum_rows[key].get(band, ())))
    if max_colors == 0:
        raise FileNotFoundError(f"no replay spectrum rows found under {raw_root}")

    n_soundings = len(keys)
    n_bands = len(BAND_ORDER)
    shape_sb = (n_soundings, n_bands)
    shape_sbc = (n_soundings, n_bands, max_colors)

    summary_data = {
        out_name: np.full(shape_sb, np.nan, dtype=np.float64)
        for out_name in SUMMARY_FLOATS.values()
    }
    spectrum_data = {
        out_name: np.full(shape_sbc, np.nan, dtype=np.float64)
        for out_name in SPECTRUM_FLOATS.values()
    }
    selected_data = {
        out_name: np.full(n_soundings, np.nan, dtype=np.float64)
        for out_name in SELECTED_FLOATS.values()
    }
    sample_index = np.full(shape_sbc, -1, dtype=np.int32)
    packed_color_index = np.full(shape_sbc, -1, dtype=np.int16)
    retrieval_index = np.full(n_soundings, -1, dtype=np.int32)
    sounding_id = np.empty(n_soundings, dtype=np.int64)
    granule = np.empty(n_soundings, dtype=object)
    group_key = np.empty(n_soundings, dtype=object)
    group = np.empty(n_soundings, dtype=object)
    operation_mode = np.empty(n_soundings, dtype=object)
    surface_type = np.empty(n_soundings, dtype=object)

    for sounding_idx, key in enumerate(keys):
        granule_name, group_name, sid = key
        granule[sounding_idx] = granule_name
        group_key[sounding_idx] = group_name
        group[sounding_idx] = GROUP_LABELS.get(group_name, group_name)
        sounding_id[sounding_idx] = sid

        selected = selected_rows.get(key, {})
        operation_mode[sounding_idx] = selected.get("operation_mode", "")
        surface_type[sounding_idx] = selected.get("surface_type", "")
        retrieval_index[sounding_idx] = int(float(selected.get("retrieval_index", -1)))
        for csv_name, out_name in SELECTED_FLOATS.items():
            selected_data[out_name][sounding_idx] = _as_float(selected.get(csv_name, "nan"))

        for band_idx, band in enumerate(BAND_ORDER):
            summary = summary_rows[key].get(band)
            if summary is not None:
                for csv_name, out_name in SUMMARY_FLOATS.items():
                    summary_data[out_name][sounding_idx, band_idx] = _as_float(
                        summary.get(csv_name, "nan")
                    )
            rows = sorted(
                spectrum_rows[key].get(band, ()),
                key=lambda row: int(float(row["packed_color_index"])),
            )
            for color_idx, row in enumerate(rows):
                packed_color_index[sounding_idx, band_idx, color_idx] = int(
                    float(row["packed_color_index"])
                )
                sample_index[sounding_idx, band_idx, color_idx] = int(float(row["sample_index"]))
                for csv_name, out_name in SPECTRUM_FLOATS.items():
                    spectrum_data[out_name][sounding_idx, band_idx, color_idx] = _as_float(
                        row.get(csv_name, "nan")
                    )

    dataset = xr.Dataset(
        coords={
            "sounding": np.arange(n_soundings, dtype=np.int32),
            "band": np.asarray(BAND_ORDER, dtype=object),
            "color": np.arange(max_colors, dtype=np.int16),
        },
        data_vars={
            "sounding_id": ("sounding", sounding_id),
            "retrieval_index": ("sounding", retrieval_index),
            "granule": ("sounding", granule),
            "group_key": ("sounding", group_key),
            "group": ("sounding", group),
            "operation_mode": ("sounding", operation_mode),
            "surface_type": ("sounding", surface_type),
            "band_label": ("band", np.asarray([BAND_LABELS[band] for band in BAND_ORDER])),
            "sample_index": (("sounding", "band", "color"), sample_index),
            "packed_color_index": (("sounding", "band", "color"), packed_color_index),
            **{name: ("sounding", values) for name, values in selected_data.items()},
            **{name: (("sounding", "band"), values) for name, values in summary_data.items()},
            **{
                name: (("sounding", "band", "color"), values)
                for name, values in spectrum_data.items()
            },
        },
        attrs={
            "title": "OCO-3 py2sess three-band replay outputs",
            "raw_root": str(raw_root),
            "n_soundings": int(n_soundings),
            "bands": ",".join(BAND_ORDER),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output, engine="netcdf4", encoding=_numeric_encoding(dataset))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = combine(args.raw_root, args.output)
    print(args.output)
    print(f"n_soundings={dataset.sizes['sounding']}")
    print(f"n_colors={dataset.sizes['color']}")


if __name__ == "__main__":
    main()
