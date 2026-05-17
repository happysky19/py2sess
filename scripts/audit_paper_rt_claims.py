#!/usr/bin/env python3
"""Check that selected manuscript numbers match paper benchmark CSV assets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_export_module():
    path = ROOT / "scripts" / "export_jacobian_overhead_summary.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row for row in rows if all(row.get(key, "") == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _timing_row(
    rows: list[dict[str, str]],
    *,
    experiment: str,
    case: str,
    backend_label: str,
    sweep_axis: str,
    wavelengths: int,
    layers: int,
    gradient_target: str = "",
) -> dict[str, str]:
    criteria = {
        "experiment": experiment,
        "case": case,
        "backend_label": backend_label,
        "sweep_axis": sweep_axis,
        "wavelengths": str(wavelengths),
        "layers": str(layers),
    }
    if gradient_target:
        criteria["gradient_target"] = gradient_target
    return _row(rows, **criteria)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _fixed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _sci(value: float, coeff_digits: int = 2, suffix: str = "") -> str:
    if value == 0.0:
        return f"0{suffix}"
    exponent = math.floor(math.log10(abs(value)))
    coefficient = value / 10.0**exponent
    return f"{coefficient:.{coeff_digits}f}\\times10^{{{exponent}}}{suffix}"


def _range(
    rows: list[dict[str, str]], *, case: str, categories: tuple[str, ...]
) -> tuple[float, float]:
    values = [
        float(row["jacobian_forward_ratio"])
        for row in rows
        if row.get("selected_for_figure") == "true"
        and row.get("case") == case
        and row.get("figure_category") in categories
    ]
    if not values:
        raise AssertionError(f"no selected overhead rows for {case} {categories}")
    return min(values), max(values)


def _range_text(values: tuple[float, float]) -> str:
    return f"{values[0]:.1f}--{values[1]:.1f}"


def _endpoint_slope(rows: list[dict[str, str]], x_key: str) -> float:
    sorted_rows = sorted(rows, key=lambda row: float(row[x_key]))
    first = sorted_rows[0]
    last = sorted_rows[-1]
    return (math.log(float(last["mean_total_s"])) - math.log(float(first["mean_total_s"]))) / (
        math.log(float(last[x_key])) - math.log(float(first[x_key]))
    )


def _overhead_rows(path: Path, summary_csv: Path) -> list[dict[str, str]]:
    if path.exists():
        return _load_rows(path)
    exporter = _load_export_module()
    return exporter.build_rows(summary_csv)


def expected_strings(
    *,
    full_summary_csv: Path,
    spectrum_csv: Path,
    validation_csv: Path,
    overhead_csv: Path,
    combined_summary_csv: Path,
) -> list[str]:
    combined = _load_rows(combined_summary_csv)
    spectrum = _load_rows(spectrum_csv)
    validation = _load_rows(validation_csv)
    overhead = _overhead_rows(overhead_csv, combined_summary_csv)

    expected: list[str] = []

    for case in ("TIR", "UV"):
        case_rows = [row for row in spectrum if row["case"] == case]
        max_rel = max(float(row["case_max_rel_diff_pct"]) for row in case_rows)
        expected.append(_sci(max_rel, coeff_digits=1, suffix="\\%"))

    tir_numpy = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="TIR",
        backend="NumPy",
        source_run="local",
    )
    uv_numpy = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="UV",
        backend="NumPy",
        source_run="local",
    )
    tir_fortran = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="TIR",
        backend="2S-ESS optimized",
        source_run="local",
    )
    uv_fortran = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="UV",
        backend="2S-ESS optimized",
        source_run="local",
    )
    tir_torch = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="TIR",
        backend="Torch CPU",
        source_run="local",
    )
    uv_torch = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="UV",
        backend="Torch CPU",
        source_run="local",
    )
    tir_t4 = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="TIR",
        backend="Torch CUDA",
        source_run="colab_t4",
    )
    uv_t4 = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="UV",
        backend="Torch CUDA",
        source_run="colab_t4",
    )
    tir_a100 = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="TIR",
        backend="Torch CUDA",
        source_run="colab_a100",
    )
    uv_a100 = _row(
        combined,
        experiment="full-spectrum-benchmark",
        case="UV",
        backend="Torch CUDA",
        source_run="colab_a100",
    )

    expected.extend(
        [
            f"{_fixed(_float(tir_numpy, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(tir_numpy, 'median_s'), 2)} s",
            f"{_fixed(_float(tir_numpy, 'min_s'), 2)}--{_fixed(_float(tir_numpy, 'max_s'), 2)} s",
            f"{_fixed(_float(uv_numpy, 'median_s'), 2)} s",
            f"{_fixed(_float(uv_numpy, 'min_s'), 2)}--{_fixed(_float(uv_numpy, 'max_s'), 2)} s",
            f"{_fixed(_float(uv_numpy, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(tir_fortran, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(uv_fortran, 'mean_total_s'), 1)} s",
            f"{_fixed(_float(tir_torch, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(uv_torch, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(tir_a100, 'mean_total_s'), 3)} s",
            f"{_fixed(_float(uv_a100, 'mean_total_s'), 2)} s",
            f"{_fixed(_float(tir_t4, 'mean_total_s'), 3)} s",
            f"{_fixed(_float(uv_t4, 'mean_total_s'), 2)} s",
        ]
    )

    labels = {
        "numpy": "NumPy (Apple M2 Pro)",
        "torch_cpu": "Torch (Apple M2 Pro)",
        "t4": "Torch (Tesla T4)",
        "a100": "Torch (A100)",
    }

    # Synthetic forward scaling claims.
    forward_300k = [
        ("TIR", labels["numpy"], 2),
        ("TIR", labels["torch_cpu"], 2),
        ("TIR", labels["t4"], 2),
        ("TIR", labels["a100"], 3),
        ("UV", labels["numpy"], 2),
        ("UV", labels["torch_cpu"], 2),
        ("UV", labels["t4"], 2),
        ("UV", labels["a100"], 2),
    ]
    for case, backend_label, digits in forward_300k:
        row = _timing_row(
            combined,
            experiment="synthetic-forward",
            case=case,
            backend_label=backend_label,
            sweep_axis="wavelengths",
            wavelengths=300000,
            layers=114,
        )
        expected.append(f"{_fixed(_float(row, 'mean_total_s'), digits)} s")

    forward_layer_claims = [
        ("TIR", labels["numpy"], 5, 3),
        ("TIR", labels["numpy"], 200, 3),
        ("TIR", labels["torch_cpu"], 5, 3),
        ("TIR", labels["torch_cpu"], 200, 2),
        ("TIR", labels["t4"], 5, 3),
        ("TIR", labels["t4"], 200, 3),
        ("TIR", labels["a100"], 5, 4),
        ("TIR", labels["a100"], 200, 3),
        ("UV", labels["numpy"], 200, 2),
        ("UV", labels["torch_cpu"], 200, 2),
        ("UV", labels["t4"], 200, 2),
        ("UV", labels["a100"], 200, 3),
    ]
    for case, backend_label, layers, digits in forward_layer_claims:
        row = _timing_row(
            combined,
            experiment="synthetic-forward",
            case=case,
            backend_label=backend_label,
            sweep_axis="layers",
            wavelengths=50000,
            layers=layers,
        )
        expected.append(f"{_fixed(_float(row, 'mean_total_s'), digits)} s")

    # Synthetic reverse-mode sensitivity scaling claims.
    jac_wave_claims = [
        ("TIR", labels["torch_cpu"], 300, 3),
        ("TIR", labels["torch_cpu"], 10000, 2),
        ("UV", labels["torch_cpu"], 300, 3),
        ("UV", labels["torch_cpu"], 10000, 2),
        ("TIR", labels["a100"], 300, 3),
        ("TIR", labels["a100"], 10000, 3),
        ("UV", labels["a100"], 300, 3),
        ("UV", labels["a100"], 10000, 3),
    ]
    for case, backend_label, wavelengths, digits in jac_wave_claims:
        row = _timing_row(
            combined,
            experiment="synthetic-jacobian",
            case=case,
            backend_label=backend_label,
            sweep_axis="wavelengths",
            wavelengths=wavelengths,
            layers=114,
            gradient_target="tau",
        )
        expected.append(f"{_fixed(_float(row, 'mean_total_s'), digits)} s")

    jac_layer_claims = [
        ("TIR", labels["torch_cpu"], 5, 3),
        ("TIR", labels["torch_cpu"], 200, 1),
        ("TIR", labels["t4"], 5, 3),
        ("TIR", labels["t4"], 200, 2),
        ("TIR", labels["a100"], 5, 3),
        ("TIR", labels["a100"], 200, 2),
        ("UV", labels["torch_cpu"], 5, 3),
        ("UV", labels["torch_cpu"], 200, 1),
        ("UV", labels["t4"], 5, 3),
        ("UV", labels["t4"], 200, 1),
        ("UV", labels["a100"], 5, 3),
        ("UV", labels["a100"], 200, 2),
    ]
    for case, backend_label, layers, digits in jac_layer_claims:
        row = _timing_row(
            combined,
            experiment="synthetic-jacobian",
            case=case,
            backend_label=backend_label,
            sweep_axis="layers",
            wavelengths=50000,
            layers=layers,
            gradient_target="tau",
        )
        expected.append(f"{_fixed(_float(row, 'mean_total_s'), digits)} s")

    jac_rows = [
        row
        for row in combined
        if row["experiment"] == "synthetic-jacobian"
        and row["gradient_target"] == "tau"
        and row["status"] == "ok"
    ]
    slope_rows = {(row["sweep_axis"], row["backend_label"], row["case"]): [] for row in jac_rows}
    for row in jac_rows:
        slope_rows[(row["sweep_axis"], row["backend_label"], row["case"])].append(row)

    m2_tir_lambda = _endpoint_slope(
        slope_rows[("wavelengths", labels["torch_cpu"], "TIR")], "wavelengths"
    )
    m2_uv_lambda = _endpoint_slope(
        slope_rows[("wavelengths", labels["torch_cpu"], "UV")], "wavelengths"
    )
    gpu_lambda = [
        _endpoint_slope(slope_rows[("wavelengths", label, case)], "wavelengths")
        for label in (labels["t4"], labels["a100"])
        for case in ("TIR", "UV")
    ]
    cpu_t4_layers = [
        _endpoint_slope(slope_rows[("layers", label, case)], "layers")
        for label in (labels["torch_cpu"], labels["t4"])
        for case in ("TIR", "UV")
    ]
    a100_layers = [
        _endpoint_slope(slope_rows[("layers", labels["a100"], case)], "layers")
        for case in ("TIR", "UV")
    ]
    grad_vars = [
        _endpoint_slope(slope_rows[("grad_vars", label, case)], "n_grad_vars")
        for label in (labels["torch_cpu"], labels["t4"], labels["a100"])
        for case in ("TIR", "UV")
    ]
    expected.extend(
        [
            f"$a_{{N_\\lambda}}={m2_tir_lambda:.2f}$",
            f"${m2_uv_lambda:.2f}$",
            f"$a_{{N_\\lambda}}={min(gpu_lambda):.2f}$--{max(gpu_lambda):.2f}",
            f"$a_L={min(cpu_t4_layers):.2f}$--{max(cpu_t4_layers):.2f}",
            f"$a_L={min(a100_layers):.2f}$--{max(a100_layers):.2f}",
            f"${min(grad_vars):.3f} \\le a_{{N_g}} \\le {max(grad_vars):.3f}$",
        ]
    )

    for case, values in {
        "TIR": ((labels["torch_cpu"], 1), (labels["t4"], 2), (labels["a100"], 2)),
        "UV": ((labels["torch_cpu"], 0), (labels["t4"], 2), (labels["a100"], 2)),
    }.items():
        for backend_label, digits in values:
            rows = [
                row
                for row in jac_rows
                if row["case"] == case
                and row["backend_label"] == backend_label
                and row["sweep_axis"] == "grad_vars"
            ]
            representative = max(rows, key=lambda row: int(float(row["active_tau_layers"])))
            expected.append(f"{_fixed(float(representative['mean_total_s']), digits)} s")

    for row in validation:
        expected.extend(
            [
                _sci(float(row["max_abs_fd"]), coeff_digits=2),
                _sci(float(row["max_abs_error"]), coeff_digits=2),
                _sci(float(row["max_rel_error"]), coeff_digits=2),
            ]
        )

    expected.extend(
        [
            _range_text(_range(overhead, case="TIR", categories=("tau_1_layer", "tau_all_layers"))),
            _range_text(_range(overhead, case="UV", categories=("tau_1_layer", "tau_all_layers"))),
            _range_text(
                _range(overhead, case="TIR", categories=("omega_1_layer", "omega_all_layers"))
            ),
            _range_text(
                _range(overhead, case="UV", categories=("omega_1_layer", "omega_all_layers"))
            ),
        ]
    )

    return sorted(set(expected))


def audit(
    *,
    paper_tex: Path,
    full_summary_csv: Path,
    spectrum_csv: Path,
    validation_csv: Path,
    overhead_csv: Path,
    combined_summary_csv: Path,
) -> list[str]:
    text = _norm(paper_tex.read_text(encoding="utf-8"))
    missing = []
    for needle in expected_strings(
        full_summary_csv=full_summary_csv,
        spectrum_csv=spectrum_csv,
        validation_csv=validation_csv,
        overhead_csv=overhead_csv,
        combined_summary_csv=combined_summary_csv,
    ):
        if needle not in text:
            missing.append(needle)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        type=Path,
        default=ROOT / "docs" / "py2sess_rt_benchmark_paper.tex",
    )
    parser.add_argument(
        "--full-summary",
        type=Path,
        default=ROOT / "docs" / "assets" / "full_spectrum_paper_summary.csv",
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        default=ROOT / "docs" / "assets" / "full_spectrum_spectrum_comparison.csv",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "docs" / "assets" / "jacobian_gradient_validation_summary.csv",
    )
    parser.add_argument(
        "--overhead",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_synthetic_jacobian_overhead_summary.csv",
    )
    parser.add_argument(
        "--combined-summary",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv",
    )
    args = parser.parse_args()
    missing = audit(
        paper_tex=args.paper,
        full_summary_csv=args.full_summary,
        spectrum_csv=args.spectrum,
        validation_csv=args.validation,
        overhead_csv=args.overhead,
        combined_summary_csv=args.combined_summary,
    )
    if missing:
        print("Missing or stale manuscript numbers:")
        for value in missing:
            print(f"  {value}")
        raise SystemExit(1)
    print("paper RT numerical claim audit passed")


if __name__ == "__main__":
    main()
