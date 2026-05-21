#!/usr/bin/env python3
"""Run the fixed OCO-3 replay workflow used for paper figures.

Diagnostic variants belong in ``run_oco3_threeband_py2sess_replay.py``.  This
wrapper intentionally exposes only run-control arguments and locks the physics
choices used by the paper-facing replay.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_oco3_threeband_replay_cases.py"
REPLAY_SCRIPT = ROOT / "scripts" / "run_oco3_threeband_py2sess_replay.py"

DEFAULT_DATA_DIR = ROOT / "outputs" / "oco3_joint_official_downloads" / "20220624_17767a"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "oco3_paper_replay" / "20220624_17767a"
DEFAULT_RT_RETRIEVAL_SUPPORT_DIR = ROOT / "scripts" / "oco3_paper_support" / "rt_retrieval"
DEFAULT_OCO_L2FP_AEROSOL_FILE = DEFAULT_RT_RETRIEVAL_SUPPORT_DIR / "l2_aerosol_combined.h5"

ST_AOD_MAX = 0.01
PAPER_REPLAY_SETTINGS = (
    "--aerosol-treatment",
    "oco-l2fp",
    "--aerosol-type-set",
    "tropospheric",
    "--surface-spectrum",
    "l2-linear",
    "--surface-angular",
    "rpv-brdf",
    "--surface-brdf-retrieval",
    "continuum-linearized-slope",
    "--polarization-correction",
    "rayleigh-aerosol-fo",
    "--stokes-projection",
    "l1b-normalized",
    "--ocean-coxmunk-stokes-scope",
    "all",
    "--gas-doppler",
    "l2-los",
    "--solar-doppler",
    "l2-solar",
    "--fluorescence-treatment",
    "none",
    "--eof-treatment",
    "none",
)


@dataclass(frozen=True)
class ReplayGroup:
    name: str
    operation_mode: str
    land_fraction_min: float
    land_fraction_max: float | None


REPLAY_GROUPS = {
    "land_nd": ReplayGroup("land_nd", "ND", 95.0, None),
    "land_am": ReplayGroup("land_am", "AM", 95.0, None),
    "ocean_gl": ReplayGroup("ocean_gl", "GL", 0.0, 5.0),
}


def _group_list(value: str) -> list[str]:
    groups = [part.strip() for part in value.split(",") if part.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("at least one replay group is required")
    unknown = sorted(set(groups) - set(REPLAY_GROUPS))
    if unknown:
        allowed = ", ".join(REPLAY_GROUPS)
        raise argparse.ArgumentTypeError(f"unknown group(s): {', '.join(unknown)}; use {allowed}")
    return groups


def _prepare_command(
    *,
    group: ReplayGroup,
    data_dir: Path,
    case_dir: Path,
    count: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(case_dir),
        "--count",
        str(count),
        "--operation-mode",
        group.operation_mode,
        "--land-fraction-min",
        f"{group.land_fraction_min:g}",
        "--st-aod-max",
        f"{ST_AOD_MAX:g}",
        "--skip-plot",
    ]
    if group.land_fraction_max is not None:
        command.extend(["--land-fraction-max", f"{group.land_fraction_max:g}"])
    return command


def _replay_command(
    *,
    data_dir: Path,
    case_dir: Path,
    count: int,
    max_colors_per_band: int,
    surface_brdf_max_iterations: int,
    aerosol_file: Path,
    skip_plot: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--case-dir",
        str(case_dir),
        "--count",
        str(count),
        "--max-colors-per-band",
        str(max_colors_per_band),
        "--surface-brdf-max-iterations",
        str(surface_brdf_max_iterations),
        "--oco-l2fp-aerosol-file",
        str(aerosol_file),
        *PAPER_REPLAY_SETTINGS,
    ]
    if skip_plot:
        command.append("--skip-plot")
    return command


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--groups", type=_group_list, default=_group_list("land_nd,land_am,ocean_gl")
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-colors-per-band", type=int, default=120)
    parser.add_argument("--surface-brdf-max-iterations", type=int, default=3)
    parser.add_argument("--oco-l2fp-aerosol-file", type=Path, default=DEFAULT_OCO_L2FP_AEROSOL_FILE)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.max_colors_per_band <= 0:
        raise ValueError("--max-colors-per-band must be positive")
    if args.surface_brdf_max_iterations <= 0:
        raise ValueError("--surface-brdf-max-iterations must be positive")
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.oco_l2fp_aerosol_file.exists():
        raise FileNotFoundError(args.oco_l2fp_aerosol_file)

    for group_name in args.groups:
        group = REPLAY_GROUPS[group_name]
        case_dir = args.output_root / group.name
        _run(
            _prepare_command(
                group=group,
                data_dir=args.data_dir,
                case_dir=case_dir,
                count=args.count,
            ),
            dry_run=args.dry_run,
        )
        _run(
            _replay_command(
                data_dir=args.data_dir,
                case_dir=case_dir,
                count=args.count,
                max_colors_per_band=args.max_colors_per_band,
                surface_brdf_max_iterations=args.surface_brdf_max_iterations,
                aerosol_file=args.oco_l2fp_aerosol_file,
                skip_plot=args.skip_plot,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
