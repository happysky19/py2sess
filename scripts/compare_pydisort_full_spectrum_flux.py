#!/usr/bin/env python3
"""Compare full-spectrum py2sess level fluxes with cached pydisort NetCDF files."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.rtsolver.backend import to_numpy
from py2sess.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path(
    "/Users/thl/MyFolder/Research/py2sess/outputs/full_spectrum_benchmark/input_bundle"
)
DEFAULT_PYDISORT_DIR = ROOT / "outputs" / "pydisort_full_spectrum_flux"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pydisort_py2sess_flux_comparison"
FIELDS = ("flux_up", "flux_down", "flux_net", "flux_mean")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    mode: str
    profile: Path
    scene: Path
    pydisort_nc: Path


def _case_specs(input_root: Path, pydisort_dir: Path) -> dict[str, CaseSpec]:
    return {
        "tir": CaseSpec(
            key="tir",
            label="TIR",
            mode="thermal",
            profile=input_root / "profiles" / "Profiles_1_2006726_0000.dat",
            scene=input_root / "benchmark_bundles" / "tir_scene_python.yaml",
            pydisort_nc=pydisort_dir / "pydisort_tir_flux.nc",
        ),
        "uv": CaseSpec(
            key="uv",
            label="UV/Solar",
            mode="solar",
            profile=input_root / "profiles" / "Profiles_1_2006726_1500.dat",
            scene=input_root / "benchmark_bundles" / "uv_scene_python.yaml",
            pydisort_nc=pydisort_dir / "pydisort_uv_flux.nc",
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


def _row_subset(kwargs: dict[str, Any], start: int, stop: int, nrows: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) == nrows:
            out[key] = value[start:stop]
        else:
            out[key] = value
    return out


def _field_array(result: Any, field: str) -> np.ndarray:
    values = np.asarray(to_numpy(getattr(result, field)), dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{field} must have shape (rows, levels), got {values.shape}")
    return values


def _safe_div_percent(numerator: np.ndarray, reference: np.ndarray, floor: float) -> np.ndarray:
    return 100.0 * numerator / np.maximum(np.abs(reference), floor)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e5:
        return f"{value:.4e}"
    return f"{value:.6g}"


class FieldStats:
    def __init__(self, nlevels: int, nbins: int, top_k: int, rel_floor: float):
        self.count = 0
        self.invalid_count = 0
        self.sum_ref = 0.0
        self.sum_py2sess = 0.0
        self.sum_ref_abs = 0.0
        self.sum_abs = 0.0
        self.sum_signed = 0.0
        self.sum_sq = 0.0
        self.max_abs = -1.0
        self.max_abs_location: tuple[int, int, float, float, float, float] | None = None
        self.max_rel = -1.0
        self.max_rel_location: tuple[int, int, float, float, float, float] | None = None
        self.level_count = np.zeros(nlevels, dtype=np.int64)
        self.level_sum_ref_abs = np.zeros(nlevels, dtype=np.float64)
        self.level_sum_abs = np.zeros(nlevels, dtype=np.float64)
        self.level_sum_signed = np.zeros(nlevels, dtype=np.float64)
        self.level_sum_sq = np.zeros(nlevels, dtype=np.float64)
        self.level_max_abs = np.zeros(nlevels, dtype=np.float64)
        self.level_max_rel = np.zeros(nlevels, dtype=np.float64)
        self.bin_count = np.zeros(nbins, dtype=np.int64)
        self.bin_sum_ref_abs = np.zeros(nbins, dtype=np.float64)
        self.bin_sum_abs = np.zeros(nbins, dtype=np.float64)
        self.bin_sum_sq = np.zeros(nbins, dtype=np.float64)
        self.bin_max_abs = np.zeros(nbins, dtype=np.float64)
        self.bin_max_rel = np.zeros(nbins, dtype=np.float64)
        self.heatmap_max_abs = np.zeros((nbins, nlevels), dtype=np.float64)
        self.heatmap_max_rel = np.zeros((nbins, nlevels), dtype=np.float64)
        self._top_k = int(top_k)
        self._rel_floor = float(rel_floor)
        self._top_abs: list[tuple[float, dict[str, Any]]] = []
        self._top_rel: list[tuple[float, dict[str, Any]]] = []

    def update(
        self,
        *,
        py2sess: np.ndarray,
        reference: np.ndarray,
        row_start: int,
        wavelengths: np.ndarray,
        bin_index: np.ndarray,
    ) -> None:
        diff = py2sess - reference
        finite = np.isfinite(py2sess) & np.isfinite(reference)
        count = int(np.sum(finite))
        self.invalid_count += int(finite.size - count)
        if count == 0:
            return
        diff = np.where(finite, diff, 0.0)
        reference = np.where(finite, reference, 0.0)
        py2sess = np.where(finite, py2sess, 0.0)
        abs_diff = np.abs(diff)
        rel_percent = np.where(
            finite,
            _safe_div_percent(abs_diff, reference, self._rel_floor),
            0.0,
        )
        self.count += count
        self.sum_ref += float(np.sum(reference))
        self.sum_py2sess += float(np.sum(py2sess))
        self.sum_ref_abs += float(np.sum(np.abs(reference)))
        self.sum_abs += float(np.sum(abs_diff))
        self.sum_signed += float(np.sum(diff))
        self.sum_sq += float(np.sum(diff * diff))

        self._update_global_max(
            abs_diff,
            rel_percent,
            finite,
            diff,
            py2sess,
            reference,
            row_start,
            wavelengths,
        )
        self._update_level_stats(abs_diff, rel_percent, finite, diff, reference)
        self._update_bin_stats(abs_diff, rel_percent, finite, reference, bin_index)
        self._update_top(
            abs_diff,
            rel_percent,
            finite,
            diff,
            py2sess,
            reference,
            row_start,
            wavelengths,
        )

    def _update_global_max(
        self,
        abs_diff: np.ndarray,
        rel_percent: np.ndarray,
        finite: np.ndarray,
        diff: np.ndarray,
        py2sess: np.ndarray,
        reference: np.ndarray,
        row_start: int,
        wavelengths: np.ndarray,
    ) -> None:
        ranked_abs = np.where(finite, abs_diff, -1.0)
        ranked_rel = np.where(finite, rel_percent, -1.0)
        abs_index = np.unravel_index(int(np.argmax(ranked_abs)), abs_diff.shape)
        abs_value = float(abs_diff[abs_index])
        if abs_value > self.max_abs:
            row, level = int(abs_index[0]), int(abs_index[1])
            self.max_abs = abs_value
            self.max_abs_location = (
                row_start + row,
                level,
                float(wavelengths[row]),
                float(py2sess[abs_index]),
                float(reference[abs_index]),
                float(diff[abs_index]),
            )
        rel_index = np.unravel_index(int(np.argmax(ranked_rel)), rel_percent.shape)
        rel_value = float(rel_percent[rel_index])
        if rel_value > self.max_rel:
            row, level = int(rel_index[0]), int(rel_index[1])
            self.max_rel = rel_value
            self.max_rel_location = (
                row_start + row,
                level,
                float(wavelengths[row]),
                float(py2sess[rel_index]),
                float(reference[rel_index]),
                float(diff[rel_index]),
            )

    def _update_level_stats(
        self,
        abs_diff: np.ndarray,
        rel_percent: np.ndarray,
        finite: np.ndarray,
        diff: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        self.level_count += np.sum(finite, axis=0)
        self.level_sum_ref_abs += np.sum(np.abs(reference), axis=0)
        self.level_sum_abs += np.sum(abs_diff, axis=0)
        self.level_sum_signed += np.sum(diff, axis=0)
        self.level_sum_sq += np.sum(diff * diff, axis=0)
        self.level_max_abs = np.maximum(self.level_max_abs, np.max(abs_diff, axis=0))
        self.level_max_rel = np.maximum(self.level_max_rel, np.max(rel_percent, axis=0))

    def _update_bin_stats(
        self,
        abs_diff: np.ndarray,
        rel_percent: np.ndarray,
        finite: np.ndarray,
        reference: np.ndarray,
        bin_index: np.ndarray,
    ) -> None:
        for bin_id in np.unique(bin_index):
            mask = bin_index == bin_id
            idx = int(bin_id)
            if not bool(np.any(finite[mask])):
                continue
            local_abs = abs_diff[mask]
            local_rel = rel_percent[mask]
            local_ref = reference[mask]
            local_finite = finite[mask]
            self.bin_count[idx] += int(np.sum(local_finite))
            self.bin_sum_ref_abs[idx] += float(np.sum(np.abs(local_ref)))
            self.bin_sum_abs[idx] += float(np.sum(local_abs))
            self.bin_sum_sq[idx] += float(np.sum(local_abs * local_abs))
            self.bin_max_abs[idx] = max(self.bin_max_abs[idx], float(np.max(local_abs)))
            self.bin_max_rel[idx] = max(self.bin_max_rel[idx], float(np.max(local_rel)))
            self.heatmap_max_abs[idx] = np.maximum(
                self.heatmap_max_abs[idx], np.max(local_abs, axis=0)
            )
            self.heatmap_max_rel[idx] = np.maximum(
                self.heatmap_max_rel[idx], np.max(local_rel, axis=0)
            )

    def _update_top(
        self,
        abs_diff: np.ndarray,
        rel_percent: np.ndarray,
        finite: np.ndarray,
        diff: np.ndarray,
        py2sess: np.ndarray,
        reference: np.ndarray,
        row_start: int,
        wavelengths: np.ndarray,
    ) -> None:
        if self._top_k <= 0:
            return
        self._push_top(
            heap=self._top_abs,
            values=abs_diff,
            finite=finite,
            abs_diff=abs_diff,
            rel_percent=rel_percent,
            diff=diff,
            py2sess=py2sess,
            reference=reference,
            row_start=row_start,
            wavelengths=wavelengths,
            metric="abs_diff",
        )
        self._push_top(
            heap=self._top_rel,
            values=rel_percent,
            finite=finite,
            abs_diff=abs_diff,
            rel_percent=rel_percent,
            diff=diff,
            py2sess=py2sess,
            reference=reference,
            row_start=row_start,
            wavelengths=wavelengths,
            metric="rel_percent",
        )

    def _push_top(
        self,
        *,
        heap: list[tuple[float, dict[str, Any]]],
        values: np.ndarray,
        finite: np.ndarray,
        abs_diff: np.ndarray,
        rel_percent: np.ndarray,
        diff: np.ndarray,
        py2sess: np.ndarray,
        reference: np.ndarray,
        row_start: int,
        wavelengths: np.ndarray,
        metric: str,
    ) -> None:
        flat = np.where(finite, values, -math.inf).reshape(-1)
        keep = min(self._top_k, flat.size)
        if keep == 0:
            return
        indices = np.argpartition(flat, -keep)[-keep:]
        nlevels = values.shape[1]
        for flat_index in indices:
            value = float(flat[flat_index])
            if not math.isfinite(value):
                continue
            row = int(flat_index // nlevels)
            level = int(flat_index % nlevels)
            item = {
                "metric": metric,
                "metric_value": value,
                "spectral_row": row_start + row,
                "level": level,
                "wavelength_nm": float(wavelengths[row]),
                "pydisort": float(reference[row, level]),
                "py2sess": float(py2sess[row, level]),
                "diff": float(diff[row, level]),
                "abs_diff": float(abs_diff[row, level]),
                "rel_percent": float(rel_percent[row, level]),
            }
            if len(heap) < self._top_k:
                heapq.heappush(heap, (value, item))
            elif value > heap[0][0]:
                heapq.heapreplace(heap, (value, item))

    def summary_row(self, *, case: CaseSpec, field: str) -> dict[str, Any]:
        mean_ref_abs = self.sum_ref_abs / self.count if self.count else math.nan
        mean_abs = self.sum_abs / self.count if self.count else math.nan
        bias = self.sum_signed / self.count if self.count else math.nan
        rmse = math.sqrt(self.sum_sq / self.count) if self.count else math.nan
        max_abs_location = self.max_abs_location or (-1, -1, math.nan, math.nan, math.nan, math.nan)
        max_rel_location = self.max_rel_location or (-1, -1, math.nan, math.nan, math.nan, math.nan)
        scale = max(mean_ref_abs, self._rel_floor)
        return {
            "case": case.label,
            "field": field,
            "valid_points": self.count,
            "invalid_points": self.invalid_count,
            "mean_pydisort": self.sum_ref / self.count if self.count else math.nan,
            "mean_py2sess": self.sum_py2sess / self.count if self.count else math.nan,
            "mean_abs_pydisort": mean_ref_abs,
            "bias": bias,
            "mean_abs_diff": mean_abs,
            "rmse": rmse,
            "nmae_percent": 100.0 * mean_abs / scale,
            "nrmse_percent": 100.0 * rmse / scale,
            "max_abs_diff": self.max_abs,
            "max_abs_row": max_abs_location[0],
            "max_abs_level": max_abs_location[1],
            "max_abs_wavelength_nm": max_abs_location[2],
            "max_abs_py2sess": max_abs_location[3],
            "max_abs_pydisort": max_abs_location[4],
            "max_rel_percent": self.max_rel,
            "max_rel_row": max_rel_location[0],
            "max_rel_level": max_rel_location[1],
            "max_rel_wavelength_nm": max_rel_location[2],
            "max_rel_py2sess": max_rel_location[3],
            "max_rel_pydisort": max_rel_location[4],
        }

    def level_rows(self, *, case: CaseSpec, field: str) -> list[dict[str, Any]]:
        rows = []
        for level in range(self.level_count.size):
            count = int(self.level_count[level])
            mean_ref_abs = self.level_sum_ref_abs[level] / count if count else math.nan
            mean_abs = self.level_sum_abs[level] / count if count else math.nan
            rmse = math.sqrt(self.level_sum_sq[level] / count) if count else math.nan
            scale = max(mean_ref_abs, self._rel_floor)
            rows.append(
                {
                    "case": case.label,
                    "field": field,
                    "level": level,
                    "points": count,
                    "mean_abs_pydisort": mean_ref_abs,
                    "bias": self.level_sum_signed[level] / count if count else math.nan,
                    "mean_abs_diff": mean_abs,
                    "rmse": rmse,
                    "nmae_percent": 100.0 * mean_abs / scale,
                    "nrmse_percent": 100.0 * rmse / scale,
                    "max_abs_diff": self.level_max_abs[level],
                    "max_rel_percent": self.level_max_rel[level],
                }
            )
        return rows

    def bin_rows(
        self,
        *,
        case: CaseSpec,
        field: str,
        bin_edges: np.ndarray,
        wavelength_min: np.ndarray,
        wavelength_max: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows = []
        for bin_id in range(self.bin_count.size):
            count = int(self.bin_count[bin_id])
            mean_ref_abs = self.bin_sum_ref_abs[bin_id] / count if count else math.nan
            mean_abs = self.bin_sum_abs[bin_id] / count if count else math.nan
            rmse = math.sqrt(self.bin_sum_sq[bin_id] / count) if count else math.nan
            scale = max(mean_ref_abs, self._rel_floor)
            rows.append(
                {
                    "case": case.label,
                    "field": field,
                    "bin": bin_id,
                    "row_start": int(bin_edges[bin_id]),
                    "row_stop": int(bin_edges[bin_id + 1]),
                    "wavelength_min_nm": float(wavelength_min[bin_id]),
                    "wavelength_max_nm": float(wavelength_max[bin_id]),
                    "points": count,
                    "mean_abs_pydisort": mean_ref_abs,
                    "mean_abs_diff": mean_abs,
                    "rmse": rmse,
                    "nmae_percent": 100.0 * mean_abs / scale,
                    "nrmse_percent": 100.0 * rmse / scale,
                    "max_abs_diff": self.bin_max_abs[bin_id],
                    "max_rel_percent": self.bin_max_rel[bin_id],
                }
            )
        return rows

    def top_rows(self, *, case: CaseSpec, field: str) -> list[dict[str, Any]]:
        rows = []
        for _, item in sorted(self._top_abs, key=lambda pair: pair[0], reverse=True):
            rows.append({"case": case.label, "field": field, **item})
        for _, item in sorted(self._top_rel, key=lambda pair: pair[0], reverse=True):
            rows.append({"case": case.label, "field": field, **item})
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pretty_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case",
        "field",
        "valid_points",
        "invalid_points",
        "mean_abs_pydisort",
        "mean_abs_diff",
        "rmse",
        "nmae_percent",
        "nrmse_percent",
        "max_abs_diff",
        "max_rel_percent",
        "max_abs_wavelength_nm",
        "max_abs_level",
    )
    table = [
        {
            key: _format_float(float(row[key])) if key not in {"case", "field"} else row[key]
            for key in fields
        }
        for row in rows
    ]
    _write_csv(path, table)


def _plot_case(
    *,
    case: CaseSpec,
    stats: dict[str, FieldStats],
    output_dir: Path,
    bin_edges: np.ndarray,
    wavelength_min: np.ndarray,
    wavelength_max: np.ndarray,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as exc:  # pragma: no cover
        print(f"skipping plots for {case.key}: {exc}", flush=True)
        return

    x = 0.5 * (wavelength_min + wavelength_max)
    extent = [float(x[0]), float(x[-1]), stats[FIELDS[0]].heatmap_max_rel.shape[1] - 1, 0]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.5), constrained_layout=True)
    rel_values = np.concatenate([stats[field].heatmap_max_rel.ravel() for field in FIELDS])
    positive = rel_values[rel_values > 0.0]
    vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1.0e-3)
    for ax, field in zip(axes.ravel(), FIELDS, strict=True):
        image = np.maximum(stats[field].heatmap_max_rel.T, 1.0e-8)
        im = ax.imshow(
            image,
            aspect="auto",
            origin="upper",
            extent=extent,
            norm=LogNorm(vmin=1.0e-6, vmax=vmax),
            cmap="magma",
        )
        ax.set_title(field)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("level index (TOA=0)")
        fig.colorbar(im, ax=ax, label="max rel. error in bin (%)")
    fig.suptitle(f"{case.label}: py2sess vs pydisort level-flux relative error")
    fig.savefig(output_dir / f"{case.key}_relative_error_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.5), constrained_layout=True)
    abs_values = np.concatenate([stats[field].heatmap_max_abs.ravel() for field in FIELDS])
    positive = abs_values[abs_values > 0.0]
    vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1.0e-16)
    for ax, field in zip(axes.ravel(), FIELDS, strict=True):
        image = np.maximum(stats[field].heatmap_max_abs.T, 1.0e-30)
        im = ax.imshow(
            image,
            aspect="auto",
            origin="upper",
            extent=extent,
            norm=LogNorm(vmin=max(vmax * 1.0e-8, 1.0e-30), vmax=vmax),
            cmap="viridis",
        )
        ax.set_title(field)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("level index (TOA=0)")
        fig.colorbar(im, ax=ax, label="max abs. error in bin")
    fig.suptitle(f"{case.label}: py2sess vs pydisort level-flux absolute error")
    fig.savefig(output_dir / f"{case.key}_absolute_error_heatmap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for field in FIELDS:
        field_stats = stats[field]
        level_nrmse = (
            100.0
            * np.sqrt(field_stats.level_sum_sq / np.maximum(field_stats.level_count, 1))
            / np.maximum(
                field_stats.level_sum_ref_abs / np.maximum(field_stats.level_count, 1),
                field_stats._rel_floor,
            )
        )
        ax.plot(np.arange(level_nrmse.size), level_nrmse, label=field)
    ax.set_yscale("log")
    ax.set_xlabel("level index (TOA=0, BOA=nlyr)")
    ax.set_ylabel("NRMSE / mean |pydisort| (%)")
    ax.set_title(f"{case.label}: level-wise normalized RMSE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / f"{case.key}_level_nrmse_profile.png", dpi=220)
    plt.close(fig)


def _plot_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"skipping summary plot: {exc}", flush=True)
        return
    labels = [f"{row['case']}\n{row['field']}" for row in summary_rows]
    nmae = [float(row["nmae_percent"]) for row in summary_rows]
    nrmse = [float(row["nrmse_percent"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(labels)), 5.5), constrained_layout=True)
    ax.bar(x - width / 2, nmae, width, label="NMAE")
    ax.bar(x + width / 2, nrmse, width, label="NRMSE")
    ax.set_yscale("log")
    ax.set_ylabel("error / mean |pydisort| (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend()
    ax.set_title("Full-spectrum level-flux comparison summary")
    fig.savefig(output_dir / "summary_normalized_errors.png", dpi=220)
    plt.close(fig)


def _bin_metadata(wavelengths: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = np.empty(bin_edges.size - 1, dtype=np.float64)
    maxs = np.empty(bin_edges.size - 1, dtype=np.float64)
    for idx, (start, stop) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=True)):
        local = wavelengths[int(start) : int(stop)]
        mins[idx] = float(np.min(local))
        maxs[idx] = float(np.max(local))
    return mins, maxs


def _run_case(case: CaseSpec, *, args: argparse.Namespace) -> dict[str, Any]:
    netCDF4 = _require_netcdf4()
    load_start = time.perf_counter()
    scene = load_scene(profile=case.profile, config=case.scene, strict_runtime_inputs=True)
    inputs = scene.to_forward_inputs()
    load_seconds = time.perf_counter() - load_start
    kwargs = inputs.kwargs
    nrows_scene = int(np.asarray(kwargs["tau"]).shape[0])
    nlay = int(np.asarray(kwargs["tau"]).shape[1])

    pydisort = netCDF4.Dataset(case.pydisort_nc)
    try:
        nrows_ref = len(pydisort.dimensions["spectral_row"])
        nlevels = len(pydisort.dimensions["level"])
        nrows = min(nrows_scene, nrows_ref)
        if args.limit is not None:
            nrows = min(nrows, int(args.limit))
        if nlevels != nlay + 1:
            raise ValueError(f"{case.key}: pydisort levels={nlevels}, py2sess levels={nlay + 1}")
        completed = int(np.asarray(pydisort.variables["completed"][:nrows]).sum())
        if completed != nrows:
            raise ValueError(f"{case.key}: pydisort NetCDF has only {completed}/{nrows} rows done")
        apply_delta_m = bool(int(getattr(pydisort, "apply_delta_m", 0)))
        wavelengths = np.asarray(pydisort.variables["wavelength_nm"][:nrows], dtype=np.float64)
        bin_size = min(int(args.bin_size), nrows)
        bin_edges = np.arange(0, nrows + bin_size, bin_size, dtype=int)
        bin_edges[-1] = nrows
        bin_edges = np.unique(bin_edges)
        nbins = bin_edges.size - 1
        wavelength_min, wavelength_max = _bin_metadata(wavelengths, bin_edges)
        stats = {
            field: FieldStats(
                nlevels=nlevels,
                nbins=nbins,
                top_k=args.top_k,
                rel_floor=args.rel_floor,
            )
            for field in FIELDS
        }

        options = TwoStreamEssOptions(
            nlyr=nlay,
            mode=case.mode,
            backend=args.backend,
            plane_parallel=True,
            delta_scaling=apply_delta_m,
            output_levels=True,
            output_fluxes=True,
            torch_device=args.torch_device,
            torch_dtype=args.torch_dtype,
            torch_enable_grad=False,
            fo_flux_n_mu=args.fo_flux_n_mu,
        )
        solver = TwoStreamEss(options)
        run_start = time.perf_counter()
        for start in range(0, nrows, args.chunk_size):
            stop = min(start + args.chunk_size, nrows)
            chunk_kwargs = _row_subset(kwargs, start, stop, nrows_scene)
            result = solver.forward(**chunk_kwargs, include_fo=not args.no_fo)
            rows = stop - start
            bin_index = np.searchsorted(bin_edges, np.arange(start, stop), side="right") - 1
            bin_index = np.clip(bin_index, 0, nbins - 1)
            local_wavelengths = wavelengths[start:stop]
            for field in FIELDS:
                py2sess = _field_array(result, field)
                reference = np.asarray(pydisort.variables[field][start:stop, :], dtype=np.float64)
                if py2sess.shape != reference.shape:
                    raise ValueError(
                        f"{case.key} {field}: py2sess shape {py2sess.shape}, "
                        f"pydisort shape {reference.shape}"
                    )
                stats[field].update(
                    py2sess=py2sess,
                    reference=reference,
                    row_start=start,
                    wavelengths=local_wavelengths,
                    bin_index=bin_index,
                )
            print(
                f"{case.label:<8s} {start:>8d}:{stop:<8d} {rows:>6d} rows compared",
                flush=True,
            )
        run_seconds = time.perf_counter() - run_start
    finally:
        pydisort.close()

    case_dir = args.output_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [stats[field].summary_row(case=case, field=field) for field in FIELDS]
    level_rows = [
        row for field in FIELDS for row in stats[field].level_rows(case=case, field=field)
    ]
    bin_rows = [
        row
        for field in FIELDS
        for row in stats[field].bin_rows(
            case=case,
            field=field,
            bin_edges=bin_edges,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
    ]
    worst_rows = [row for field in FIELDS for row in stats[field].top_rows(case=case, field=field)]
    _write_csv(case_dir / f"{case.key}_summary.csv", summary_rows)
    _write_pretty_summary(case_dir / f"{case.key}_summary_pretty.csv", summary_rows)
    _write_csv(case_dir / f"{case.key}_by_level.csv", level_rows)
    _write_csv(case_dir / f"{case.key}_by_wavelength_bin.csv", bin_rows)
    _write_csv(case_dir / f"{case.key}_worst_points.csv", worst_rows)
    _plot_case(
        case=case,
        stats=stats,
        output_dir=case_dir,
        bin_edges=bin_edges,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )
    return {
        "case": case.label,
        "mode": case.mode,
        "backend": args.backend,
        "rows": nrows,
        "layers": nlay,
        "levels": nlevels,
        "chunk_size": args.chunk_size,
        "bin_size": args.bin_size,
        "include_fo": not args.no_fo,
        "fo_flux_n_mu": args.fo_flux_n_mu,
        "delta_scaling": apply_delta_m,
        "load_seconds": load_seconds,
        "compare_seconds": run_seconds,
        "rows_per_second": nrows / run_seconds if run_seconds > 0.0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--pydisort-dir", type=Path, default=DEFAULT_PYDISORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", default="tir,uv")
    parser.add_argument("--backend", choices=("numpy", "torch", "native"), default="numpy")
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--torch-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rel-floor", type=float, default=1.0e-12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fo-flux-n-mu", type=int, default=8)
    parser.add_argument("--no-fo", action="store_true")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.rel_floor <= 0.0:
        raise ValueError("--rel-floor must be positive")
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")
    mpl_config = args.output_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    specs = _case_specs(args.input_root, args.pydisort_dir)
    output_rows = [_run_case(specs[key], args=args) for key in _split_cases(args.cases)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "comparison_runtime.csv", output_rows)

    summary_rows: list[dict[str, Any]] = []
    for key in _split_cases(args.cases):
        path = args.output_dir / key / f"{key}_summary.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            summary_rows.extend(csv.DictReader(handle))
    _write_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    _write_pretty_summary(args.output_dir / "comparison_summary_pretty.csv", summary_rows)
    _plot_summary(summary_rows, args.output_dir)

    print(f"wrote {args.output_dir / 'comparison_runtime.csv'}", flush=True)
    print(f"wrote {args.output_dir / 'comparison_summary.csv'}", flush=True)
    print(
        f"{'case':<10s} {'field':<10s} {'NMAE %':>10s} {'NRMSE %':>10s} "
        f"{'max abs':>12s} {'max rel %':>12s}"
    )
    for row in summary_rows:
        print(
            f"{row['case']:<10s} {row['field']:<10s} "
            f"{float(row['nmae_percent']):10.4g} {float(row['nrmse_percent']):10.4g} "
            f"{float(row['max_abs_diff']):12.4g} {float(row['max_rel_percent']):12.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
