#!/usr/bin/env python3
"""Export matched Jacobian/forward timing ratios used by the paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_plot_module():
    path = ROOT / "scripts" / "plot_paper_rt_benchmarks.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row_id(row: dict[str, str]) -> str:
    keys = (
        "experiment",
        "case",
        "mode",
        "backend_group",
        "backend_label",
        "hardware",
        "device",
        "dtype",
        "timing_kind",
        "sweep_axis",
        "gradient_target",
        "wavelengths",
        "layers",
        "active_tau_layers",
        "n_grad_vars",
        "source_csv",
        "source_run",
    )
    payload = "\0".join(row.get(key, "") for key in keys)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _ratio_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["case"],
        row["mode"],
        row["series_key"],
        row["dtype"],
        row["sweep_axis"],
        row["gradient_target"],
        row["wavelengths"],
        row["layers"],
        row["active_tau_layers"],
    )


def _category_for_selected(row: dict[str, str]) -> str:
    target = row["gradient_target"]
    if target == "surface_albedo":
        return "surface_albedo"
    active = int(float(row["active_tau_layers"]))
    layers = int(float(row["layers"]))
    if target == "tau" and active == 1:
        return "tau_1_layer"
    if target == "tau" and active == layers:
        return "tau_all_layers"
    if target == "omega" and active == 1:
        return "omega_1_layer"
    if target == "omega" and active == layers:
        return "omega_all_layers"
    if target == "g" and active == 1:
        return "g_1_layer"
    if target == "g" and active == layers:
        return "g_all_layers"
    if target == "surface_emissivity":
        return "surface_emissivity"
    return ""


def _summary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(summary_csv: Path) -> list[dict[str, str]]:
    plot = _load_plot_module()
    input_rows = plot._load_rows(summary_csv)
    ratio_rows = plot._jacobian_forward_ratio_rows(input_rows)

    forward_lookup: dict[tuple[str, ...], dict[str, str]] = {}
    jacobian_lookup: dict[tuple[str, ...], dict[str, str]] = {}
    for row in input_rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("case", ""),
            row.get("mode", ""),
            plot._series_key(row),
            row.get("dtype", ""),
            row.get("wavelengths", ""),
            row.get("layers", ""),
        )
        if row.get("experiment") == "synthetic-forward" and plot._backend_group(row).startswith(
            "torch"
        ):
            forward_lookup[key] = row
        elif row.get("experiment") == "synthetic-jacobian":
            jacobian_lookup[
                key
                + (
                    row.get("sweep_axis", ""),
                    plot._gradient_target(row),
                    row.get("active_tau_layers", ""),
                )
            ] = row

    selected: dict[tuple[str, ...], str] = {}
    categories = (
        ("surface_albedo", "surface_albedo", 0),
        ("grad_vars", "tau", 1),
        ("grad_vars", "tau", None),
        ("omega_grad_vars", "omega", 1),
        ("omega_grad_vars", "omega", None),
        ("g_grad_vars", "g", 1),
        ("g_grad_vars", "g", None),
        ("surface_emissivity", "surface_emissivity", 0),
    )
    for case in ("TIR", "UV"):
        reference_dims = plot._representative_overhead_dims(ratio_rows, case=case)
        for series_key in plot._present_series_keys(ratio_rows, plot.JACOBIAN_SERIES_ORDER):
            for sweep_axis, target, active_layers in categories:
                match = plot._representative_overhead_row(
                    ratio_rows,
                    case=case,
                    series_key=series_key,
                    sweep_axis=sweep_axis,
                    gradient_target=target,
                    active_layers=active_layers,
                    reference_dims=reference_dims,
                )
                if match is not None:
                    selected[_ratio_key(match)] = _category_for_selected(match)

    summary_hash = _summary_sha256(summary_csv)
    output_rows = []
    for row in sorted(
        ratio_rows,
        key=lambda item: (
            item["case"],
            item["gradient_target"],
            item["sweep_axis"],
            item["series_key"],
            int(float(item["wavelengths"])),
            int(float(item["layers"])),
            int(float(item["active_tau_layers"])),
        ),
    ):
        forward_key = (
            row["case"],
            row["mode"],
            row["series_key"],
            row["dtype"],
            row["wavelengths"],
            row["layers"],
        )
        jacobian_key = forward_key + (
            row["sweep_axis"],
            row["gradient_target"],
            row["active_tau_layers"],
        )
        forward = forward_lookup[forward_key]
        jacobian = jacobian_lookup[jacobian_key]
        ratio_key = _ratio_key(row)
        output = {
            **row,
            "selected_for_figure": "true" if ratio_key in selected else "false",
            "figure_category": selected.get(ratio_key, ""),
            "forward_std_s": forward.get("std_s", ""),
            "jacobian_std_s": jacobian.get("std_s", ""),
            "forward_n_repeats": forward.get("n_repeats", ""),
            "jacobian_n_repeats": jacobian.get("n_repeats", ""),
            "forward_row_id": _row_id(forward),
            "jacobian_row_id": _row_id(jacobian),
            "forward_source_csv": forward.get("source_csv", ""),
            "jacobian_source_csv": jacobian.get("source_csv", ""),
            "forward_source_run": forward.get("source_run", ""),
            "jacobian_source_run": jacobian.get("source_run", ""),
            "source_summary_csv": str(summary_csv.relative_to(ROOT)),
            "source_summary_sha256": summary_hash,
        }
        output_rows.append(output)
    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "mode",
        "backend",
        "backend_group",
        "series_key",
        "backend_label",
        "device",
        "dtype",
        "sweep_axis",
        "gradient_target",
        "wavelengths",
        "layers",
        "levels",
        "active_tau_layers",
        "n_grad_vars",
        "forward_mean_s",
        "jacobian_mean_s",
        "jacobian_forward_ratio",
        "backward_mean_s",
        "backward_forward_ratio",
        "jacobian_forward_ratio_std_approx",
        "selected_for_figure",
        "figure_category",
        "forward_std_s",
        "jacobian_std_s",
        "forward_n_repeats",
        "jacobian_n_repeats",
        "forward_row_id",
        "jacobian_row_id",
        "forward_source_csv",
        "jacobian_source_csv",
        "forward_source_run",
        "jacobian_source_run",
        "source_summary_csv",
        "source_summary_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_synthetic_jacobian_overhead_summary.csv",
    )
    args = parser.parse_args()
    rows = build_rows(args.summary)
    write_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
