#!/usr/bin/env python3
"""Run the paper OCO-3 replay workflow across downloaded granules."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from combine_oco3_replay_to_netcdf import combine  # noqa: E402
from prepare_oco3_threeband_replay_cases import (  # noqa: E402
    DEFAULT_ST_AOD_MAX,
    _select_candidate_indices,
    _single_data_file,
)
from run_oco3_paper_replay import (  # noqa: E402
    DEFAULT_OCO_L2FP_AEROSOL_FILE,
    PAPER_REPLAY_SETTINGS,
    PREPARE_SCRIPT,
    REPLAY_GROUPS,
    REPLAY_SCRIPT,
)


DEFAULT_DATA_ROOT = ROOT / "outputs" / "oco3_joint_official_downloads"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "oco3_replay_1000"


@dataclass(frozen=True)
class ReplayTask:
    granule: str
    group_key: str
    count: int
    data_dir: Path
    case_dir: Path


def _group_list(value: str) -> list[str]:
    groups = [item.strip() for item in value.split(",") if item.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("at least one group is required")
    unknown = sorted(set(groups) - set(REPLAY_GROUPS))
    if unknown:
        allowed = ", ".join(REPLAY_GROUPS)
        raise argparse.ArgumentTypeError(f"unknown group(s): {', '.join(unknown)}; use {allowed}")
    return groups


def _granule_dirs(data_root: Path, granules: str) -> list[Path]:
    if granules == "auto":
        dirs = sorted(path for path in data_root.glob("20*_*") if path.is_dir())
    else:
        dirs = [data_root / item.strip() for item in granules.split(",") if item.strip()]
    if not dirs:
        raise FileNotFoundError(f"no granule directories found under {data_root}")
    for data_dir in dirs:
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
    return dirs


def _candidate_count(data_dir: Path, group_key: str, *, st_aod_max: float) -> int:
    group = REPLAY_GROUPS[group_key]
    l2std_path = _single_data_file(data_dir, "oco3_L2StdSC_*.h5")
    l2dia_path = _single_data_file(data_dir, "oco3_L2DiaSC_*.h5")
    candidates, _data = _select_candidate_indices(
        l2std_path=l2std_path,
        l2dia_path=l2dia_path,
        land_fraction_min=group.land_fraction_min,
        land_fraction_max=group.land_fraction_max,
        operation_mode=group.operation_mode,
        chi_square_max=1.4,
        snr_o2_min=100.0,
        snr_wco2_min=100.0,
        snr_sco2_min=50.0,
        aod_max=0.30,
        st_aod_max=st_aod_max,
    )
    return int(candidates.size)


def _allocate_counts(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total < target:
        raise ValueError(f"requested {target} cases but only {total} candidates are available")
    raw = {granule: target * count / total for granule, count in counts.items()}
    allocation = {granule: min(counts[granule], int(raw[granule])) for granule in counts}
    remaining = target - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda granule: (raw[granule] - int(raw[granule]), counts[granule]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for granule in order:
            if allocation[granule] < counts[granule]:
                allocation[granule] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise RuntimeError("failed to allocate requested cases")
    return {granule: count for granule, count in allocation.items() if count > 0}


def _write_plan(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("granule", "group_key", "candidate_count", "selected_count"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _prepare_command(task: ReplayTask, *, st_aod_max: float) -> list[str]:
    group = REPLAY_GROUPS[task.group_key]
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--data-dir",
        str(task.data_dir),
        "--output-dir",
        str(task.case_dir),
        "--count",
        str(task.count),
        "--operation-mode",
        group.operation_mode,
        "--land-fraction-min",
        f"{group.land_fraction_min:g}",
        "--st-aod-max",
        f"{st_aod_max:g}",
        "--skip-plot",
    ]
    if group.land_fraction_max is not None:
        command.extend(["--land-fraction-max", f"{group.land_fraction_max:g}"])
    return command


def _replay_command(
    task: ReplayTask,
    *,
    max_colors_per_band: int,
    surface_brdf_max_iterations: int,
    aerosol_file: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--data-dir",
        str(task.data_dir),
        "--case-dir",
        str(task.case_dir),
        "--count",
        str(task.count),
        "--max-colors-per-band",
        str(max_colors_per_band),
        "--surface-brdf-max-iterations",
        str(surface_brdf_max_iterations),
        "--oco-l2fp-aerosol-file",
        str(aerosol_file),
        *PAPER_REPLAY_SETTINGS,
        "--skip-plot",
    ]


def _run_task(
    task: ReplayTask,
    *,
    max_colors_per_band: int,
    surface_brdf_max_iterations: int,
    aerosol_file: Path,
    st_aod_max: float,
    dry_run: bool,
) -> None:
    commands = (
        _prepare_command(task, st_aod_max=st_aod_max),
        _replay_command(
            task,
            max_colors_per_band=max_colors_per_band,
            surface_brdf_max_iterations=surface_brdf_max_iterations,
            aerosol_file=aerosol_file,
        ),
    )
    for command in commands:
        print(" ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def _tasks(args: argparse.Namespace) -> tuple[list[ReplayTask], list[dict[str, object]]]:
    data_dirs = _granule_dirs(args.data_root, args.granules)
    plan_rows: list[dict[str, object]] = []
    tasks: list[ReplayTask] = []
    for group_key in args.groups:
        counts = {
            data_dir.name: _candidate_count(data_dir, group_key, st_aod_max=args.st_aod_max)
            for data_dir in data_dirs
        }
        allocation = _allocate_counts(counts, args.count_per_group)
        for data_dir in data_dirs:
            granule = data_dir.name
            selected_count = allocation.get(granule, 0)
            plan_rows.append(
                {
                    "granule": granule,
                    "group_key": group_key,
                    "candidate_count": counts[granule],
                    "selected_count": selected_count,
                }
            )
            if selected_count > 0:
                tasks.append(
                    ReplayTask(
                        granule=granule,
                        group_key=group_key,
                        count=selected_count,
                        data_dir=data_dir,
                        case_dir=args.output_root / "raw" / granule / group_key,
                    )
                )
    return tasks, plan_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--granules", default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--groups", type=_group_list, default=_group_list("land_nd,land_am,ocean_gl")
    )
    parser.add_argument("--count-per-group", type=int, default=1000)
    parser.add_argument("--max-colors-per-band", type=int, default=120)
    parser.add_argument("--surface-brdf-max-iterations", type=int, default=3)
    parser.add_argument("--st-aod-max", type=float, default=DEFAULT_ST_AOD_MAX)
    parser.add_argument("--oco-l2fp-aerosol-file", type=Path, default=DEFAULT_OCO_L2FP_AEROSOL_FILE)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-combine", action="store_true")
    args = parser.parse_args()

    if args.count_per_group <= 0:
        raise ValueError("--count-per-group must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.max_colors_per_band <= 0:
        raise ValueError("--max-colors-per-band must be positive")
    if args.surface_brdf_max_iterations <= 0:
        raise ValueError("--surface-brdf-max-iterations must be positive")
    if not args.oco_l2fp_aerosol_file.exists():
        raise FileNotFoundError(args.oco_l2fp_aerosol_file)

    tasks, plan_rows = _tasks(args)
    _write_plan(args.output_root / "quota_plan.csv", plan_rows)
    print(args.output_root / "quota_plan.csv", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                _run_task,
                task,
                max_colors_per_band=args.max_colors_per_band,
                surface_brdf_max_iterations=args.surface_brdf_max_iterations,
                aerosol_file=args.oco_l2fp_aerosol_file,
                st_aod_max=args.st_aod_max,
                dry_run=args.dry_run,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            future.result()

    if not args.dry_run and not args.no_combine:
        output = args.output_root / "oco3_replay.nc"
        dataset = combine(args.output_root / "raw", output)
        print(output, flush=True)
        print(f"n_soundings={dataset.sizes['sounding']}", flush=True)


if __name__ == "__main__":
    main()
