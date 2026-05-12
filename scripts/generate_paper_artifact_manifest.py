#!/usr/bin/env python3
"""Generate a reproducibility manifest for paper benchmark assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = (
    "docs/py2sess_rt_benchmark_paper.tex",
    "docs/full_spectrum_benchmarks.md",
    "docs/paper_rt_benchmarks.md",
    "docs/paper_rt_release_checklist.md",
    "docs/assets/full_spectrum_paper_summary.csv",
    "docs/assets/full_spectrum_spectrum_comparison.csv",
    "docs/assets/full_spectrum_rt_runtime.png",
    "docs/assets/full_spectrum_rt_runtime.eps",
    "docs/assets/full_spectrum_spectrum_comparison.png",
    "docs/assets/full_spectrum_spectrum_comparison.eps",
    "docs/assets/synthetic_forward_scaling_summary.csv",
    "docs/assets/synthetic_jacobian_scaling_summary.csv",
    "docs/assets/jacobian_gradient_validation_summary.csv",
    "docs/assets/paper_rt_all_timing_summary.csv",
    "docs/assets/paper_rt_synthetic_jacobian_overhead_summary.csv",
    "docs/assets/paper_rt_synthetic_forward_publication.png",
    "docs/assets/paper_rt_synthetic_forward_publication.eps",
    "docs/assets/paper_rt_synthetic_jacobian_publication.png",
    "docs/assets/paper_rt_synthetic_jacobian_publication.eps",
    "docs/assets/paper_rt_synthetic_jacobian_overhead_publication.png",
    "docs/assets/paper_rt_synthetic_jacobian_overhead_publication.eps",
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
)


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_dirty() -> bool:
    return bool(_run_git(["status", "--porcelain"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "table"
    if path.suffix in {".png", ".eps", ".pdf", ".svg"}:
        return "figure"
    if path.suffix == ".py":
        return "script"
    if path.suffix in {".tex", ".md"}:
        return "document"
    return "file"


def build_rows(
    paths: tuple[str, ...] = DEFAULT_PATHS,
    *,
    allow_missing: bool = False,
    require_clean_git: bool = False,
) -> list[dict[str, str]]:
    revision = _run_git(["rev-parse", "--short", "HEAD"])
    is_dirty = _git_dirty()
    if require_clean_git and is_dirty:
        raise RuntimeError("paper artifact manifest must be generated from a clean git tree")
    dirty = "true" if is_dirty else "false"
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, str]] = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"paper artifact is missing: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"paper artifact is not a file: {relative}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"paper artifact is empty: {relative}")
        stat = path.stat()
        rows.append(
            {
                "generated_utc": generated_utc,
                "git_revision": revision,
                "git_dirty": dirty,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "path": relative,
                "kind": _kind(path),
                "bytes": str(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "sha256": _sha256(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generated_utc",
        "git_revision",
        "git_dirty",
        "python",
        "platform",
        "path",
        "kind",
        "bytes",
        "mtime_utc",
        "sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "assets" / "paper_rt_artifact_manifest.csv",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing listed artifacts instead of failing.",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if the git worktree is dirty. Use this for final paper release manifests.",
    )
    args = parser.parse_args()
    rows = build_rows(
        allow_missing=args.allow_missing,
        require_clean_git=args.require_clean_git,
    )
    write_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
