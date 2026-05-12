#!/usr/bin/env python3
"""Build one normalized paper RT timing summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "experiment",
    "case",
    "mode",
    "system",
    "backend",
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
    "levels",
    "active_tau_layers",
    "n_grad_vars",
    "n_repeats",
    "best_s",
    "mean_total_s",
    "median_s",
    "std_s",
    "min_s",
    "max_s",
    "mean_fo_s",
    "mean_2s_s",
    "mean_setup_s",
    "mean_forward_s",
    "mean_backward_s",
    "backward_fraction",
    "throughput_best_rows_per_s",
    "cuda_peak_bytes_max",
    "checksum",
    "grad_checksum",
    "grad_l2",
    "max_abs_diff",
    "max_rel_diff_pct",
    "status",
    "source_csv",
    "source_run",
    "notes",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _simplify_hardware(value: str, row: dict[str, str]) -> str:
    source = row.get("source", "") or row.get("source_run", "")
    text = f"{value} {source}".lower()
    if "a100" in text:
        return "A100"
    if "tesla t4" in text or source in {"colab", "colab_t4"}:
        return "Tesla T4"
    if "Apple M2 Pro" in value or source == "local":
        return "Apple M2 Pro"
    return value


def _backend_group(row: dict[str, str]) -> str:
    if row.get("system") == "Fortran":
        return "fortran"
    backend = row.get("backend", "")
    if backend == "NumPy":
        return "numpy"
    if backend.startswith("Torch CPU"):
        return "torch_cpu"
    if backend.startswith("Torch CUDA"):
        return "torch_cuda"
    return backend.lower().replace(" ", "_")


def _backend_label(row: dict[str, str], hardware: str) -> str:
    if row.get("system") == "Fortran":
        return f"Fortran ({hardware})" if hardware else "Fortran"
    backend = row.get("backend", "")
    if backend == "NumPy":
        return f"NumPy ({hardware})" if hardware else "NumPy"
    if backend.startswith("Torch CPU") or backend.startswith("Torch CUDA"):
        suffix = ", torch.compile" if "torch.compile" in backend else ""
        return f"Torch ({hardware}{suffix})" if hardware else f"Torch{suffix}"
    return f"{backend} ({hardware})" if hardware else backend


def _pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return value
    return ""


def _source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _timing_kind(row: dict[str, str], experiment: str) -> str:
    if row.get("timing_kind", ""):
        return row["timing_kind"]
    if experiment == "synthetic-forward":
        return "forward"
    if experiment == "synthetic-jacobian":
        return "forward_backward"
    return ""


def _gradient_target(row: dict[str, str], experiment: str) -> str:
    target = row.get("gradient_target", "")
    if target:
        return target
    sweep_axis = row.get("sweep_axis", "")
    if sweep_axis == "surface_albedo":
        return "surface_albedo"
    if sweep_axis.startswith("omega"):
        return "omega"
    if experiment == "synthetic-jacobian":
        return "tau"
    return ""


def _levels(row: dict[str, str]) -> str:
    if row.get("levels", ""):
        return row["levels"]
    layers = row.get("layers", "")
    if layers:
        return str(int(layers) + 1)
    return ""


def _base_output(
    *,
    row: dict[str, str],
    experiment: str,
    source_csv: Path,
    notes: str = "",
) -> dict[str, str]:
    hardware = _simplify_hardware(row.get("hardware", ""), row)
    out = {field: "" for field in FIELDS}
    out.update(
        {
            "experiment": experiment,
            "case": row.get("case", ""),
            "mode": row.get("mode", ""),
            "system": row.get("system", "py2sess") or "py2sess",
            "backend": row.get("backend", ""),
            "backend_group": _backend_group(row),
            "backend_label": _backend_label(row, hardware),
            "hardware": hardware,
            "device": row.get("device", ""),
            "dtype": row.get("dtype", ""),
            "timing_kind": _timing_kind(row, experiment),
            "sweep_axis": row.get("sweep_axis", "full_spectrum"),
            "gradient_target": _gradient_target(row, experiment),
            "wavelengths": row.get("wavelengths", ""),
            "layers": row.get("layers", ""),
            "levels": _levels(row),
            "active_tau_layers": row.get("active_tau_layers", ""),
            "n_grad_vars": row.get("n_grad_vars", ""),
            "n_repeats": row.get("n_repeats", ""),
            "best_s": row.get("best_s", ""),
            "mean_total_s": _pick(row, "total_mean_s", "mean_s"),
            "median_s": row.get("median_s", ""),
            "std_s": row.get("std_s", ""),
            "min_s": row.get("min_s", ""),
            "max_s": row.get("max_s", ""),
            "mean_fo_s": row.get("fo_mean_s", ""),
            "mean_2s_s": row.get("two_stream_mean_s", ""),
            "mean_setup_s": row.get("setup_mean_s", ""),
            "mean_forward_s": row.get("forward_mean_s", ""),
            "mean_backward_s": row.get("backward_mean_s", ""),
            "backward_fraction": row.get("backward_fraction", ""),
            "throughput_best_rows_per_s": _pick(row, "rows_per_second_best", "rows_per_second"),
            "cuda_peak_bytes_max": row.get("cuda_peak_bytes_max", ""),
            "checksum": row.get("checksum", ""),
            "grad_checksum": row.get("grad_checksum", ""),
            "grad_l2": row.get("grad_l2", ""),
            "max_abs_diff": row.get("max_abs_diff", ""),
            "max_rel_diff_pct": row.get("max_rel_diff_pct", ""),
            "status": row.get("status", ""),
            "source_csv": _source_label(source_csv),
            "source_run": row.get("source", ""),
            "notes": notes,
        }
    )
    return out


def build(
    *,
    full_spectrum: Path,
    synthetic_forward: Path,
    synthetic_jacobian: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_rows(full_spectrum):
        note = "full-spectrum benchmark"
        if row.get("backend") == "NumPy" and "threads=1" in row.get("hardware", ""):
            note = "full-spectrum benchmark; BLAS/OpenMP threads=1"
        rows.append(
            _base_output(
                row=row,
                experiment="full-spectrum-benchmark",
                source_csv=full_spectrum,
                notes=note,
            )
        )

    for row in _read_rows(synthetic_forward):
        rows.append(
            _base_output(
                row=row,
                experiment="synthetic-forward",
                source_csv=synthetic_forward,
                notes="analytic inhomogeneous synthetic optical profiles",
            )
        )

    for row in _read_rows(synthetic_jacobian):
        rows.append(
            _base_output(
                row=row,
                experiment="synthetic-jacobian",
                source_csv=synthetic_jacobian,
                notes="torch autograd; NumPy not applicable",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-spectrum",
        type=Path,
        default=ROOT / "docs" / "assets" / "full_spectrum_paper_summary.csv",
    )
    parser.add_argument(
        "--synthetic-forward",
        type=Path,
        default=ROOT / "docs" / "assets" / "synthetic_forward_scaling_summary.csv",
    )
    parser.add_argument(
        "--synthetic-jacobian",
        type=Path,
        default=ROOT / "docs" / "assets" / "synthetic_jacobian_scaling_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv",
    )
    args = parser.parse_args()

    rows = build(
        full_spectrum=args.full_spectrum,
        synthetic_forward=args.synthetic_forward,
        synthetic_jacobian=args.synthetic_jacobian,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
