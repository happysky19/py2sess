#!/usr/bin/env python3
"""Plot py2sess paper RT benchmark summary CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py2sess_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "py2sess_cache"))

SERIES_STYLES = {
    "numpy_m2pro": {
        "label": "NumPy (Apple M2 Pro)",
        "color": "#4A4A4A",
        "marker": "o",
    },
    "torch_cpu_m2pro": {
        "label": "Torch (Apple M2 Pro)",
        "color": "#3F6C9E",
        "marker": "^",
    },
    "torch_cuda": {
        "label": "Torch (CUDA)",
        "color": "#C46A1A",
        "marker": "s",
    },
    "torch_cuda_t4": {
        "label": "Torch (Tesla T4)",
        "color": "#C46A1A",
        "marker": "s",
    },
    "torch_cuda_a100": {
        "label": "Torch (A100)",
        "color": "#2A8C6A",
        "marker": "D",
    },
}
SERIES_ORDER = (
    "numpy_m2pro",
    "torch_cpu_m2pro",
    "torch_cuda",
    "torch_cuda_t4",
    "torch_cuda_a100",
)
JACOBIAN_SERIES_ORDER = ("torch_cpu_m2pro", "torch_cuda", "torch_cuda_t4", "torch_cuda_a100")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _select(
    rows: list[dict[str, str]],
    *,
    experiment: str,
    sweep_axis: str | None = None,
) -> list[dict[str, str]]:
    out = [row for row in rows if row["experiment"] == experiment and row["status"] == "ok"]
    if sweep_axis is not None:
        out = [row for row in out if row["sweep_axis"] == sweep_axis]
    return out


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float("nan") if value == "" else float(value)


def _int(row: dict[str, str], key: str) -> int:
    if key == "levels" and row.get("levels", "") == "":
        return int(row["layers"]) + 1
    return int(row[key])


def _backend_group(row: dict[str, str]) -> str:
    backend_group = row.get("backend_group", "")
    if backend_group:
        return backend_group
    if row.get("backend") == "NumPy":
        return "numpy"
    if row.get("backend", "").startswith("Torch CPU"):
        return "torch_cpu"
    if row.get("backend", "").startswith("Torch CUDA"):
        return "torch_cuda"
    return row.get("backend", "").lower().replace(" ", "_")


def _hardware(row: dict[str, str]) -> str:
    hardware = row.get("hardware", "")
    if not hardware and row.get("backend_label", ""):
        label = row["backend_label"]
        if "(" in label and label.endswith(")"):
            hardware = label.rsplit("(", 1)[1][:-1]
    source = row.get("source_run", "") or row.get("source", "")
    text = f"{hardware} {source}".lower()
    if "a100" in text:
        return "A100"
    if "tesla t4" in text or source in {"colab", "colab_t4"}:
        return "Tesla T4"
    if "apple m2 pro" in text or source == "local":
        return "Apple M2 Pro"
    return hardware


def _series_key(row: dict[str, str]) -> str:
    group = _backend_group(row)
    hardware = _hardware(row).lower()
    if group == "numpy":
        return "numpy_m2pro"
    if group == "torch_cpu":
        return "torch_cpu_m2pro"
    if group == "torch_cuda":
        if "a100" in hardware:
            return "torch_cuda_a100"
        if "t4" in hardware:
            return "torch_cuda_t4"
    return group


def _series_style(row_or_key: dict[str, str] | str) -> dict[str, str] | None:
    key = row_or_key if isinstance(row_or_key, str) else _series_key(row_or_key)
    return SERIES_STYLES.get(key)


def _present_series_keys(rows: list[dict[str, str]], order: tuple[str, ...]) -> tuple[str, ...]:
    present = {row.get("series_key", _series_key(row)) for row in rows}
    return tuple(key for key in order if key in present)


def _label(row: dict[str, str]) -> str:
    style = _series_style(row)
    if style is not None:
        return style["label"]
    if row.get("backend_label", ""):
        return row["backend_label"]
    return row.get("backend", "")


def _case_display(case: str) -> str:
    if case == "UV":
        return "Solar"
    if case == "TIR":
        return "Thermal"
    return case


def _gradient_target(row: dict[str, str]) -> str:
    target = row.get("gradient_target", "")
    if target:
        return target
    sweep_axis = row.get("sweep_axis", "")
    if sweep_axis == "surface_albedo":
        return "surface_albedo"
    if sweep_axis.startswith("omega"):
        return "omega"
    if (
        row.get("experiment") == "synthetic-jacobian"
        or row.get("timing_kind") == "forward_backward"
    ):
        return "tau"
    return ""


def _configure_publication_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "lines.linewidth": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "py2sess-paper-rt-benchmark-publication",
        }
    )
    return plt


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save(fig, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"pdf", "png", "svg"}:
            save_kwargs["metadata"] = {"Date": None}
        fig.savefig(path, **save_kwargs)
        if fmt == "svg":
            _normalize_svg(path)
        paths.append(path)
    return paths


def _mean_runtime(row: dict[str, str]) -> float:
    if row.get("mean_total_s", ""):
        return _float(row, "mean_total_s")
    return _float(row, "mean_s")


def _optional_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        if row.get(key, ""):
            return _float(row, key)
    return 0.0


def _case_setting(rows: list[dict[str, str]], case: str, sweep_axis: str) -> str:
    subset = [row for row in rows if row["case"] == case and row["sweep_axis"] == sweep_axis]
    if not subset:
        return ""
    first = subset[0]
    if sweep_axis == "wavelengths":
        return f"{int(_float(first, 'layers'))} RT layers"
    if sweep_axis == "layers":
        return f"{int(_float(first, 'wavelengths')):,} wavelengths"
    if sweep_axis == "grad_vars":
        return (
            f"{int(_float(first, 'wavelengths')):,} wavelengths, "
            f"{int(_float(first, 'layers'))} RT layers"
        )
    if sweep_axis == "omega_grad_vars":
        return (
            f"{int(_float(first, 'wavelengths')):,} wavelengths, "
            f"{int(_float(first, 'layers'))} RT layers"
        )
    if sweep_axis == "surface_albedo":
        return f"{int(_float(first, 'wavelengths')):,} wavelengths"
    return ""


def _jacobian_forward_ratio_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    forward_rows = {
        (
            row["case"],
            row["mode"],
            _series_key(row),
            row["dtype"],
            row["wavelengths"],
            row["layers"],
        ): row
        for row in rows
        if row["experiment"] == "synthetic-forward"
        and row["status"] == "ok"
        and _backend_group(row).startswith("torch")
    }
    ratio_rows = []
    for row in rows:
        if row["experiment"] != "synthetic-jacobian" or row["status"] != "ok":
            continue
        key = (
            row["case"],
            row["mode"],
            _series_key(row),
            row["dtype"],
            row["wavelengths"],
            row["layers"],
        )
        forward = forward_rows.get(key)
        if forward is None:
            continue
        forward_mean = _mean_runtime(forward)
        jacobian_mean = _mean_runtime(row)
        if forward_mean <= 0.0 or jacobian_mean <= 0.0:
            continue
        ratio = jacobian_mean / forward_mean
        backward_mean = _optional_float(row, "mean_backward_s", "backward_mean_s")
        backward_ratio = backward_mean / forward_mean if backward_mean > 0.0 else 0.0
        forward_std = _float(forward, "std_s") if forward.get("std_s", "") else 0.0
        jacobian_std = _float(row, "std_s") if row.get("std_s", "") else 0.0
        ratio_std = 0.0
        if forward_std > 0.0 or jacobian_std > 0.0:
            ratio_std = (
                ratio
                * ((forward_std / forward_mean) ** 2 + (jacobian_std / jacobian_mean) ** 2) ** 0.5
            )
        ratio_rows.append(
            {
                "case": row["case"],
                "mode": row["mode"],
                "backend": row["backend"],
                "backend_group": _backend_group(row),
                "series_key": _series_key(row),
                "backend_label": _label(row),
                "device": row.get("device", ""),
                "dtype": row["dtype"],
                "sweep_axis": row["sweep_axis"],
                "gradient_target": _gradient_target(row),
                "wavelengths": row["wavelengths"],
                "layers": row["layers"],
                "levels": row["levels"],
                "active_tau_layers": row["active_tau_layers"],
                "n_grad_vars": row["n_grad_vars"],
                "forward_mean_s": f"{forward_mean:.12g}",
                "jacobian_mean_s": f"{jacobian_mean:.12g}",
                "jacobian_forward_ratio": f"{ratio:.12g}",
                "backward_mean_s": f"{backward_mean:.12g}",
                "backward_forward_ratio": f"{backward_ratio:.12g}",
                "jacobian_forward_ratio_std_approx": f"{ratio_std:.12g}",
            }
        )
    return ratio_rows


def _add_case_header(ax, *, panel: str, case: str, setting: str) -> None:
    ax.text(
        -0.12,
        1.14,
        panel,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.6,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.13,
        case,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10.3,
        fontweight="bold",
    )
    if setting:
        ax.text(
            0.0,
            1.035,
            setting,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=6.8,
            color="0.42",
        )
        ax.plot(
            [0.0, 1.0],
            [1.01, 1.01],
            transform=ax.transAxes,
            color="0.82",
            lw=0.65,
            clip_on=False,
        )


def _plot_forward_publication(
    rows: list[dict[str, str]], output_dir: Path, formats: tuple[str, ...]
) -> list[Path]:
    selected = _select(rows, experiment="synthetic-forward")
    if not selected:
        return []

    plt = _configure_publication_matplotlib()
    cases = ("TIR", "UV")
    sweeps = (
        ("wavelengths", "wavelengths", "Spectral grid", "Number of wavelengths"),
        ("layers", "layers", "Vertical grid", "RT layers"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 4.25), dpi=300)

    column_limits: list[tuple[float, float]] = []
    column_x_limits: list[tuple[float, float]] = []
    for sweep_axis, x_key, _, _ in sweeps:
        y_values = [
            _mean_runtime(row) + _float(row, "std_s")
            for row in selected
            if row["sweep_axis"] == sweep_axis and _series_style(row) is not None
        ]
        positive_y = [value for value in y_values if value > 0.0]
        if positive_y:
            ymin = min(
                _mean_runtime(row)
                for row in selected
                if row["sweep_axis"] == sweep_axis
                and _series_style(row) is not None
                and _mean_runtime(row) > 0.0
            )
            column_limits.append((ymin / 1.65, max(positive_y) * 1.45))
        else:
            column_limits.append((1.0e-3, 1.0))

        x_values = [
            _int(row, x_key)
            for row in selected
            if row["sweep_axis"] == sweep_axis and _series_style(row) is not None
        ]
        if x_values:
            column_x_limits.append((min(x_values) / 1.25, max(x_values) * 1.25))
        else:
            column_x_limits.append((1.0, 10.0))

    panel = 0
    for row_index, case in enumerate(cases):
        for col_index, (sweep_axis, x_key, title, xlabel) in enumerate(sweeps):
            ax = axes[row_index][col_index]
            panel += 1
            subset = [
                row
                for row in selected
                if row["case"] == case
                and row["sweep_axis"] == sweep_axis
                and _series_style(row) is not None
            ]
            for series_key in SERIES_ORDER:
                series_rows = [row for row in subset if _series_key(row) == series_key]
                series_rows = sorted(series_rows, key=lambda row: _int(row, x_key))
                style = _series_style(series_key)
                if not series_rows or style is None:
                    continue
                x = [_int(row, x_key) for row in series_rows]
                y = [_mean_runtime(row) for row in series_rows]
                yerr = [_float(row, "std_s") for row in series_rows]
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    color=style["color"],
                    marker=style["marker"],
                    ms=3.0,
                    mfc=style["color"],
                    mec=style["color"],
                    mew=0.5,
                    capsize=1.6,
                    elinewidth=0.55,
                    label=style["label"],
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(*column_limits[col_index])
            ax.set_xlim(*column_x_limits[col_index])
            ax.grid(axis="y", which="major", color="0.93", lw=0.35)
            ax.tick_params(labelbottom=row_index == 1)
            _add_case_header(
                ax,
                panel=chr(ord("a") + panel - 1),
                case=_case_display(case),
                setting=_case_setting(selected, case, sweep_axis),
            )
            if row_index == 0:
                ax.set_title(title, fontsize=8.0, fontweight="bold", pad=29.0)
                ax.set_xlabel("")
            else:
                ax.set_xlabel(xlabel, labelpad=2.0)
            if col_index == 0:
                ax.set_ylabel("Runtime (s)", labelpad=4.0)
            else:
                ax.set_ylabel("")

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=min(4, len(handles)),
        handlelength=1.6,
        columnspacing=1.35,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.13, top=0.82, wspace=0.25, hspace=0.36)
    paths = _save(fig, output_dir, "paper_rt_synthetic_forward_publication", formats)
    plt.close(fig)
    return paths


def _plot_jacobian_publication(
    rows: list[dict[str, str]], output_dir: Path, formats: tuple[str, ...]
) -> list[Path]:
    selected = [
        row
        for row in _select(rows, experiment="synthetic-jacobian")
        if _gradient_target(row) == "tau"
    ]
    if not selected:
        return []

    plt = _configure_publication_matplotlib()
    cases = ("TIR", "UV")
    sweeps = [
        ("wavelengths", "wavelengths", "Spectral grid", "Number of wavelengths"),
        ("layers", "layers", "Vertical grid", "RT layers"),
        ("grad_vars", "n_grad_vars", "Gradient state", "Active tau-gradient variables"),
    ]
    fig, axes = plt.subplots(2, len(sweeps), figsize=(2.25 * len(sweeps), 4.45), dpi=300)

    column_limits: list[tuple[float, float]] = []
    for sweep_axis, _, _, _ in sweeps:
        values = [
            _mean_runtime(row) + _float(row, "std_s")
            for row in selected
            if row["sweep_axis"] == sweep_axis
        ]
        positive = [value for value in values if value > 0.0]
        if not positive:
            column_limits.append((1.0e-2, 1.0))
            continue
        ymin = min(
            _mean_runtime(row)
            for row in selected
            if row["sweep_axis"] == sweep_axis and _mean_runtime(row) > 0.0
        )
        ymax = max(positive)
        column_limits.append((ymin / 1.65, ymax * 1.45))

    column_x_limits = []
    for sweep_axis, x_key, _, _ in sweeps:
        values = [
            _int(row, x_key)
            for row in selected
            if _series_style(row) is not None and row["sweep_axis"] == sweep_axis
        ]
        column_x_limits.append((min(values) / 1.25, max(values) * 1.25))

    panel = 0
    for row_index, case in enumerate(cases):
        for col_index, (sweep_axis, x_key, title, xlabel) in enumerate(sweeps):
            ax = axes[row_index][col_index]
            panel += 1
            subset = [
                row
                for row in selected
                if row["case"] == case
                and row["sweep_axis"] == sweep_axis
                and _series_style(row) is not None
            ]
            for series_key in JACOBIAN_SERIES_ORDER:
                series_rows = [row for row in subset if _series_key(row) == series_key]
                series_rows = sorted(series_rows, key=lambda row: _int(row, x_key))
                style = _series_style(series_key)
                if not series_rows or style is None:
                    continue
                x = [_int(row, x_key) for row in series_rows]
                y = [_mean_runtime(row) for row in series_rows]
                yerr = [_float(row, "std_s") for row in series_rows]
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    color=style["color"],
                    marker=style["marker"],
                    ms=3.0,
                    mfc=style["color"],
                    mec=style["color"],
                    mew=0.5,
                    capsize=1.6,
                    elinewidth=0.55,
                    label=style["label"],
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(*column_limits[col_index])
            ax.set_xlim(*column_x_limits[col_index])
            ax.grid(axis="y", which="major", color="0.93", lw=0.35)
            ax.tick_params(labelbottom=row_index == 1)
            _add_case_header(
                ax,
                panel=chr(ord("a") + panel - 1),
                case=_case_display(case),
                setting=_case_setting(selected, case, sweep_axis),
            )
            if row_index == 0:
                ax.set_title(title, fontsize=8.0, fontweight="bold", pad=29.0)
                ax.set_xlabel("")
            else:
                ax.set_xlabel(xlabel, labelpad=2.0)
            if col_index == 0:
                ax.set_ylabel("Runtime (s)", labelpad=4.0)
            else:
                ax.set_ylabel("")

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=min(3, len(handles)),
        handlelength=1.6,
        columnspacing=1.5,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.82, wspace=0.30, hspace=0.36)
    paths = _save(fig, output_dir, "paper_rt_synthetic_jacobian_publication", formats)
    plt.close(fig)
    return paths


def _plot_jacobian_overhead_publication(
    rows: list[dict[str, str]], output_dir: Path, formats: tuple[str, ...]
) -> list[Path]:
    selected = _jacobian_forward_ratio_rows(rows)
    if not selected:
        return []

    plt = _configure_publication_matplotlib()
    cases = ("TIR", "UV")
    categories = (
        ("surface_albedo", "surface_albedo", 0, "Surface\nalbedo"),
        ("grad_vars", "tau", 1, "Tau\n1 layer"),
        ("grad_vars", "tau", None, "Tau\nall layers"),
        ("omega_grad_vars", "omega", 1, "Omega\n1 layer"),
        ("omega_grad_vars", "omega", None, "Omega\nall layers"),
        ("g_grad_vars", "g", 1, "g\n1 layer"),
        ("g_grad_vars", "g", None, "g\nall layers"),
        ("surface_emissivity", "surface_emissivity", 0, "Surface\nemissivity"),
    )
    from matplotlib import patheffects as path_effects

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.05), dpi=300, sharey=True)

    y_values = [
        _float(row, "jacobian_forward_ratio") + _float(row, "jacobian_forward_ratio_std_approx")
        for row in selected
        if _series_key(row) in JACOBIAN_SERIES_ORDER
    ]
    y_top = max(12.0, max(y_values, default=12.0) * 1.25)
    plot_series = _present_series_keys(selected, JACOBIAN_SERIES_ORDER)
    bar_width = min(0.22, 0.68 / max(len(plot_series), 1))

    for panel_index, (case, ax) in enumerate(zip(cases, axes, strict=True)):
        x_base = list(range(len(categories)))
        plotted_any = False
        reference_dims = _representative_overhead_dims(selected, case=case)
        label_records: dict[int, list[tuple[float, float, float, str]]] = {
            category_index: [] for category_index in range(len(categories))
        }
        for series_index, series_key in enumerate(plot_series):
            style = _series_style(series_key)
            if style is None:
                continue
            values: list[float] = []
            errors: list[float] = []
            x: list[float] = []
            offset = (series_index - (len(plot_series) - 1) / 2.0) * (bar_width + 0.035)
            for category_index, (sweep_axis, target, active_layers, _) in enumerate(categories):
                match = _representative_overhead_row(
                    selected,
                    case=case,
                    series_key=series_key,
                    sweep_axis=sweep_axis,
                    gradient_target=target,
                    active_layers=active_layers,
                    reference_dims=reference_dims,
                )
                if match is None:
                    continue
                x.append(category_index + offset)
                value = _float(match, "jacobian_forward_ratio")
                values.append(value)
                error = _float(match, "jacobian_forward_ratio_std_approx")
                errors.append(error)
                label_records[category_index].append(
                    (category_index + offset, value, error, style["color"])
                )
            if not values:
                continue
            plotted_any = True
            ax.bar(
                x,
                values,
                yerr=errors,
                width=bar_width,
                color=style["color"],
                edgecolor="white",
                linewidth=0.35,
                error_kw={"elinewidth": 0.6, "capsize": 1.8, "capthick": 0.6},
                label=style["label"],
                zorder=3,
            )
        for ref in (1.0, 3.0, 10.0):
            ax.axhline(ref, color="0.82", lw=0.45, ls=(0, (2.0, 2.0)), zorder=0)
            ax.text(
                len(categories) - 0.15,
                ref,
                f"{ref:g}x",
                ha="left",
                va="center",
                fontsize=5.9,
                color="0.45",
            )
        label_base_offset = max(0.45, y_top * 0.014)
        label_min_sep = max(0.55, y_top * 0.020)
        for category_records in label_records.values():
            placed_y: list[float] = []
            for x_i, value, error, color in sorted(category_records, key=lambda item: item[0]):
                y_i = value + error + label_base_offset
                while any(abs(y_i - other_y) < label_min_sep for other_y in placed_y):
                    y_i += label_min_sep
                placed_y.append(y_i)
                ax.text(
                    x_i,
                    y_i,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    fontweight="bold",
                    color=color,
                    path_effects=[path_effects.withStroke(linewidth=1.25, foreground="white")],
                    clip_on=False,
                    zorder=5,
                )

        ax.set_xticks(x_base, [label for *_, label in categories])
        ax.set_ylim(0.0, y_top)
        ax.set_xlim(-0.65, len(categories) - 0.35)
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=5.7)
        _add_case_header(
            ax,
            panel=chr(ord("a") + panel_index),
            case=_case_display(case),
            setting=_format_reference_dims(reference_dims),
        )
        if panel_index == 0:
            ax.set_ylabel("VJP / forward runtime (x)", labelpad=4.0)
        if not plotted_any:
            ax.text(0.5, 0.5, "No matching rows", transform=ax.transAxes, ha="center", va="center")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.025),
        ncol=min(3, len(handles)),
        handlelength=1.6,
        columnspacing=1.5,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.065, right=0.988, bottom=0.24, top=0.74, wspace=0.15)
    paths = _save(fig, output_dir, "paper_rt_synthetic_jacobian_overhead_publication", formats)
    plt.close(fig)
    return paths


def _representative_overhead_dims(
    rows: list[dict[str, str]], *, case: str
) -> tuple[int, int] | None:
    candidates = [
        (int(float(row["wavelengths"])), int(float(row["layers"])))
        for row in rows
        if row["case"] == case
        and row["sweep_axis"] in {"surface_albedo", "grad_vars", "omega_grad_vars"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))


def _format_reference_dims(reference_dims: tuple[int, int] | None) -> str:
    if reference_dims is None:
        return ""
    wavelengths, layers = reference_dims
    return f"{wavelengths:,} wavelengths, {layers} RT layers"


def _representative_overhead_row(
    rows: list[dict[str, str]],
    *,
    case: str,
    series_key: str,
    sweep_axis: str,
    gradient_target: str,
    active_layers: int | None,
    reference_dims: tuple[int, int] | None,
) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if row["case"] == case
        and row.get("series_key", _series_key(row)) == series_key
        and row["sweep_axis"] == sweep_axis
        and _gradient_target(row) == gradient_target
    ]
    if reference_dims is not None:
        wavelengths, layers = reference_dims
        matches = [
            row
            for row in matches
            if int(float(row["wavelengths"])) == wavelengths and int(float(row["layers"])) == layers
        ]
    if active_layers is not None:
        matches = [row for row in matches if int(float(row["active_tau_layers"])) == active_layers]
    elif gradient_target in {"tau", "omega"}:
        matches = [
            row
            for row in matches
            if int(float(row["active_tau_layers"])) == int(float(row["layers"]))
        ]
    if not matches:
        return None
    return sorted(matches, key=lambda row: int(float(row["active_tau_layers"])))[-1]


def plot(summary_csv: Path, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    rows = _load_rows(summary_csv)
    outputs = []
    outputs.extend(_plot_forward_publication(rows, output_dir, formats))
    outputs.extend(_plot_jacobian_publication(rows, output_dir, formats))
    outputs.extend(_plot_jacobian_overhead_publication(rows, output_dir, formats))
    return outputs


def _formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"eps", "pdf", "png", "svg"}
    if not formats or any(fmt not in allowed for fmt in formats):
        raise ValueError("--formats must contain eps, pdf, png, and/or svg")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs" / "forward_scaling_benchmark" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "forward_scaling_benchmark",
    )
    parser.add_argument("--formats", default="png,eps")
    args = parser.parse_args()
    outputs = plot(args.summary, args.output_dir, _formats(args.formats))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
