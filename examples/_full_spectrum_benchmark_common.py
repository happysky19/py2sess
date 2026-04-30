"""Shared helpers for the full-spectrum benchmark examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    wavelengths: int
    layers: int
    chunk_size: int
    wall_seconds: float
    rt_seconds: float
    fo_seconds: float | None = None
    two_stream_seconds: float | None = None
    max_abs_diff: float | None = None
    max_rel_diff_pct: float | None = None

    @property
    def rows_per_second_rt(self) -> float:
        if self.rt_seconds <= 0.0:
            return 0.0
        return self.wavelengths / self.rt_seconds


def add_common_benchmark_arguments(
    parser: argparse.ArgumentParser,
    *,
    torch_bvp_choices: tuple[str, ...] = ("auto", "block"),
) -> None:
    parser.add_argument(
        "--profile", type=Path, required=True, help="Atmospheric profile text file."
    )
    parser.add_argument("--scene", type=Path, required=True, help="Benchmark scene YAML file.")
    parser.add_argument("--backend", choices=["numpy", "torch", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Optional spectral-row limit.")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Optional chunk size override."
    )
    parser.add_argument(
        "--numpy-bvp-engine", choices=["auto", "block", "pentadiagonal"], default="auto"
    )
    parser.add_argument("--torch-bvp-engine", choices=torch_bvp_choices, default="auto")
    parser.add_argument("--torch-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--torch-dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output-levels",
        action="store_true",
        help="Benchmark the public forward profile path instead of endpoint-only output.",
    )
    parser.add_argument(
        "--require-python-generated-inputs",
        action="store_true",
        help="Fail if the scene would use direct HITRAN instead of a saved gas table.",
    )


def public_bvp_solver(engine: str) -> str:
    if engine == "auto":
        return "auto"
    if engine == "block":
        return "banded"
    if engine == "pentadiagonal":
        return "pentadiag"
    return engine


def slice_spectral_rows(
    bundle: dict[str, np.ndarray],
    keys: tuple[str, ...],
    start: int,
    stop: int,
    *,
    optional_keys: tuple[str, ...] = ("ref_total",),
) -> dict[str, np.ndarray]:
    chunk = {key: bundle[key][start:stop] for key in keys}
    for key in optional_keys:
        if key in bundle:
            chunk[key] = bundle[key][start:stop]
    return chunk


def recommended_chunk_size(
    *,
    total_rows: int,
    nlayers: int,
    backend: str,
    workload: str,
) -> int:
    if total_rows <= 0:
        return 0
    nlay = max(int(nlayers), 1)
    if workload == "solar_obs":
        row_floats = (48 if backend == "torch" else 40) * nlay + 64
        target_mib = 512 if backend == "torch" else 1400
        target_bytes = target_mib * 1024 * 1024
    elif workload == "thermal":
        row_floats = (6 if backend == "torch" else 4) * nlay + 32
        target_bytes = 384 * 1024 * 1024
    else:  # pragma: no cover
        raise ValueError(f"unsupported workload: {workload}")
    chunk = target_bytes // (8 * row_floats)
    granularity = 2000 if backend == "torch" else 1000
    minimum = granularity
    chunk = max(minimum, int(chunk))
    chunk = min(total_rows, ((chunk + granularity - 1) // granularity) * granularity)
    return max(1, chunk)


def print_problem_header(
    *,
    title: str,
    input_path: Path,
    input_kind: str,
    wavelengths: int,
    layers: int,
    load_seconds: float | None = None,
    note: str | None = None,
) -> None:
    print(title)
    print(f"  input: {input_path}")
    print(f"  input kind: {input_kind}")
    print(f"  wavelengths: {wavelengths}")
    print(f"  layers: {layers}")
    if load_seconds is not None:
        print(f"  load (s): {load_seconds:.3f}")
    if note is not None:
        print(f"  note: {note}")


def print_preprocessing_summary(rows: tuple[tuple[str, str, float], ...]) -> None:
    for label, mode, seconds in rows:
        print(f"  {label}: {mode}, {seconds:.3f} s")
    total = sum(seconds for _, _, seconds in rows)
    labels = " + ".join(label for label, _, _ in rows)
    print(f"  preprocessing total: {total:.3f} s ({labels})")


def _format_optional(value: float | None) -> str:
    return f"{'-':>10}" if value is None else f"{value:10.3f}"


def _format_accuracy(value: float | None) -> str:
    return f"{'-':>14}" if value is None else f"{value:14.6e}"


def scalar_value(value: np.ndarray | float | int) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def accuracy_summary(value: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    abs_diff = float(np.max(np.abs(value - reference)))
    scale = np.maximum(np.abs(reference), 1.0e-15)
    rel_diff_pct = float(np.max(np.abs(value - reference) / scale) * 100.0)
    return abs_diff, rel_diff_pct


def print_rows(rows: list[BenchmarkRow]) -> None:
    has_accuracy = any(
        row.max_abs_diff is not None or row.max_rel_diff_pct is not None for row in rows
    )
    backend_width = max(18, *(len(row.backend) for row in rows))
    header = (
        f"{'backend':<{backend_width}} {'wall (s)':>10} {'rt (s)':>10} "
        f"{'fo (s)':>10} {'2s (s)':>10} {'#wavelength/s':>14} {'chunk':>8}"
    )
    if has_accuracy:
        header += f" {'max abs diff':>14} {'max rel (%)':>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row.backend:<{backend_width}} "
            f"{row.wall_seconds:10.3f} "
            f"{row.rt_seconds:10.3f} "
            f"{_format_optional(row.fo_seconds)} "
            f"{_format_optional(row.two_stream_seconds)} "
            f"{row.rows_per_second_rt:14.0f} "
            f"{row.chunk_size:8d}"
        )
        if has_accuracy:
            line += (
                f" {_format_accuracy(row.max_abs_diff)} {_format_accuracy(row.max_rel_diff_pct)}"
            )
        print(line)
