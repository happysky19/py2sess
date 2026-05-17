#!/usr/bin/env python3
"""Prepare a manifest for the final py2sess paper benchmark archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]

TRACKED_ARCHIVE_INPUTS = (
    "docs/py2sess_rt_benchmark_paper.tex",
    "docs/py2sess_rt_benchmark_paper.pdf",
    "docs/assets",
    "docs/full_spectrum_benchmarks.md",
    "docs/paper_rt_benchmarks.md",
    "docs/paper_rt_release_checklist.md",
    "scripts/benchmark_full_spectrum_rt.py",
    "scripts/benchmark_paper_rt.py",
    "scripts/build_paper_rt_timing_summary.py",
    "scripts/export_full_spectrum_benchmark_bundle.py",
    "scripts/export_full_spectrum_spectrum_comparison.py",
    "scripts/export_jacobian_overhead_summary.py",
    "scripts/generate_jacobian_validation_summary.py",
    "scripts/generate_paper_artifact_manifest.py",
    "scripts/prepare_paper_archive_manifest.py",
    "scripts/audit_paper_rt_claims.py",
    "scripts/plot_full_spectrum_rt_benchmarks.py",
    "scripts/plot_paper_rt_benchmarks.py",
    "tests/test_paper_rt_benchmark.py",
    "tests/test_full_spectrum_benchmark.py",
)

RAW_BENCHMARK_ARCHIVE_INPUTS = (
    "outputs/full_spectrum_m2pro_cpu/raw_full_spectrum_timings.csv",
    "outputs/full_spectrum_m2pro_cpu/summary_full_spectrum.csv",
    "outputs/full_spectrum_m2pro_cpu/manifest_full_spectrum.csv",
    "outputs/full_spectrum_m2pro_cpu/fortran_timings",
    "outputs/full_spectrum_m2pro_cpu/fortran_bundle/README.md",
    "outputs/full_spectrum_m2pro_cpu/fortran_bundle/TIR/2S-ESS/Results_Exact_Opt",
    "outputs/full_spectrum_m2pro_cpu/fortran_bundle/UVVSWIR/2S-ESS/Results_Exact_Opt",
    "outputs/full_spectrum_benchmark/input_bundle",
    "outputs/synthetic_rt_m2pro_cpu_50k/raw_timings.csv",
    "outputs/synthetic_rt_m2pro_cpu_50k/summary.csv",
    "outputs/synthetic_rt_m2pro_cpu_50k/manifest.csv",
    "outputs/full_spectrum_benchmark_colab/raw_full_spectrum_timings.csv",
    "outputs/full_spectrum_benchmark_colab/summary_full_spectrum.csv",
    "outputs/full_spectrum_benchmark_colab/manifest_full_spectrum.csv",
    "outputs/full_spectrum_benchmark_colab_a100/raw_full_spectrum_timings.csv",
    "outputs/full_spectrum_benchmark_colab_a100/summary_full_spectrum.csv",
    "outputs/full_spectrum_benchmark_colab_a100/manifest_full_spectrum.csv",
    "outputs/synthetic_rt_cuda_colab/raw_timings.csv",
    "outputs/synthetic_rt_cuda_colab/summary.csv",
    "outputs/synthetic_rt_cuda_colab/manifest.csv",
    "outputs/synthetic_rt_cuda_colab_a100/raw_timings.csv",
    "outputs/synthetic_rt_cuda_colab_a100/summary.csv",
    "outputs/synthetic_rt_cuda_colab_a100/manifest.csv",
)

ARCHIVE_INPUTS = TRACKED_ARCHIVE_INPUTS + RAW_BENCHMARK_ARCHIVE_INPUTS


FIELDNAMES = (
    "generated_utc",
    "git_revision",
    "git_dirty",
    "path",
    "kind",
    "bytes",
    "sha256",
    "required_for_submission",
    "archive_note",
)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty() -> bool:
    return bool(_git(["status", "--porcelain"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [] if path.name == ".DS_Store" else [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and item.name != ".DS_Store")


def _kind(path: Path) -> str:
    parts = set(path.parts)
    if "fortran_timings" in parts:
        return "fortran-timing"
    if "fortran_bundle" in parts:
        return "fortran-bundle"
    if "input_bundle" in parts:
        return "input-bundle"
    if "outputs" in parts:
        return "raw-benchmark-output"
    if path.suffix == ".csv":
        return "paper-table"
    if path.suffix in {".png", ".eps", ".pdf"}:
        return "paper-asset"
    if path.suffix == ".py":
        return "script"
    return "document"


def build_rows(
    paths: tuple[str, ...] | None = None,
    *,
    include_raw_outputs: bool = False,
    require_clean_git: bool = False,
) -> list[dict[str, str]]:
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    revision = _git(["rev-parse", "--short", "HEAD"])
    is_dirty = _git_dirty()
    if require_clean_git and is_dirty:
        raise RuntimeError("paper archive manifest must be generated from a clean git tree")
    dirty = "true" if is_dirty else "false"
    if paths is None:
        paths = TRACKED_ARCHIVE_INPUTS
        if include_raw_outputs:
            paths = paths + RAW_BENCHMARK_ARCHIVE_INPUTS
    rows: list[dict[str, str]] = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"missing archive input: {relative}")
        for file_path in _iter_files(path):
            rel = file_path.relative_to(ROOT).as_posix()
            rows.append(
                {
                    "generated_utc": generated_utc,
                    "git_revision": revision,
                    "git_dirty": dirty,
                    "path": rel,
                    "kind": _kind(file_path),
                    "bytes": str(file_path.stat().st_size),
                    "sha256": _sha256(file_path),
                    "required_for_submission": "true",
                    "archive_note": "Include in the persistent paper benchmark archive.",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path, rows: list[dict[str, str]], *, include_raw_outputs: bool = False
) -> None:
    kinds = sorted({row["kind"] for row in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# py2sess paper benchmark archive manifest",
                "",
                "This manifest lists the local files that should be copied into the",
                "persistent paper archive before submission. It is a preparation aid;",
                "the manuscript DOI should point to the final archive, not to this",
                "working-tree draft.",
                "",
                "Mode: "
                + (
                    "tracked paper assets plus raw benchmark outputs"
                    if include_raw_outputs
                    else "tracked paper assets only"
                ),
                ""
                if include_raw_outputs
                else "Run again with --include-raw-outputs before submission to add ignored local timing outputs.",
                "",
                f"File count: {len(rows)}",
                f"Total bytes: {sum(int(row['bytes']) for row in rows)}",
                f"Content kinds: {', '.join(kinds)}",
                "",
                "Required final additions before submission:",
                "- Replace draft archive language in the manuscript with the archive DOI.",
                "- Include the full external 2S-ESS source/build tree or a citable public",
                "  reference for it, including compiler, flags, executable hashes, and",
                "  the full-spectrum input dump files.",
                "- Confirm that the manifest is regenerated from a clean git revision.",
                "- Use --require-clean-git for the final release manifest.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "paper_rt_archive_manifest.csv",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "outputs" / "paper_rt_archive_README.md",
    )
    parser.add_argument(
        "--include-raw-outputs",
        action="store_true",
        help=(
            "Include ignored raw benchmark output directories in the archive manifest. "
            "Use this for final local archive preparation, not clean CI."
        ),
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if the git worktree is dirty. Use with --include-raw-outputs for the final release manifest.",
    )
    args = parser.parse_args()
    rows = build_rows(
        include_raw_outputs=args.include_raw_outputs,
        require_clean_git=args.require_clean_git,
    )
    write_csv(args.output, rows)
    write_readme(args.readme, rows, include_raw_outputs=args.include_raw_outputs)
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.readme}")


if __name__ == "__main__":
    main()
