"""Plot the Colab Tesla T4 CUDA scaling reference chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["wavelengths"] = int(row["wavelengths"])
        row["best_s"] = float(row["best_s"])
        row["speedup"] = float(row["speedup"])
    return rows


def _select(
    rows: list[dict[str, object]],
    *,
    case: str | None = None,
    backend: str | None = None,
) -> list[dict[str, object]]:
    selected = rows
    if case is not None:
        selected = [row for row in selected if row["case"] == case]
    if backend is not None:
        selected = [row for row in selected if row["backend"] == backend]
    return selected


def _kfmt(value, _position):
    if value >= 1000:
        return f"{int(value / 1000)}k"
    return str(int(value))


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def plot(input_csv: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows = _load_rows(input_csv)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "py2sess-cuda-colab-t4-scaling",
        }
    )

    colors = {"NumPy": "#333333", "Torch CPU": "#9A9A9A", "Torch CUDA": "#0072B2"}
    styles = {"UV": "-", "TIR": "--"}
    markers = {"UV": "o", "TIR": "s"}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.25), dpi=260)

    for backend in ("NumPy", "Torch CPU", "Torch CUDA"):
        for case in ("UV", "TIR"):
            subset = _select(rows, case=case, backend=backend)
            ax0.plot(
                [row["wavelengths"] for row in subset],
                [row["best_s"] for row in subset],
                color=colors[backend],
                ls=styles[case],
                marker=markers[case],
                lw=1.55,
                ms=3.8,
                label=f"{case} {backend}",
            )

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("Number of spectral wavelengths")
    ax0.set_ylabel("RT runtime (s)")
    ax0.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax0.grid(axis="y", color="0.90", lw=0.55)
    ax0.text(-0.14, 1.04, "a", transform=ax0.transAxes, fontsize=10, weight="bold")
    ax0.set_title("RT runtime", fontsize=9.2, pad=4)

    for case in ("UV", "TIR"):
        subset = _select(rows, case=case, backend="Torch CUDA")
        ax1.plot(
            [row["wavelengths"] for row in subset],
            [row["speedup"] for row in subset],
            color="#0072B2",
            ls=styles[case],
            marker=markers[case],
            lw=1.8,
            ms=4.2,
            label=case,
        )
        last = subset[-1]
        ax1.text(
            last["wavelengths"] * 1.08,
            last["speedup"],
            f"{case} {last['speedup']:.1f}x",
            color="#0072B2",
            va="center",
            fontsize=7.5,
        )

    ax1.axhline(1, color="0.35", lw=0.75, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of spectral wavelengths")
    ax1.set_ylabel("CUDA speedup vs NumPy")
    ax1.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax1.set_ylim(0, 9.6)
    ax1.grid(axis="y", color="0.90", lw=0.55)
    ax1.text(-0.14, 1.04, "b", transform=ax1.transAxes, fontsize=10, weight="bold")
    ax1.set_title("CUDA acceleration", fontsize=9.2, pad=4)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.suptitle(
        "py2sess RT runtime scaling on Colab Tesla T4",
        fontsize=10.5,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        (
            "Float64 public API timing; checked-in 1,000-wavelength fixtures tiled "
            "along the spectral dimension; CUDA timing includes host-to-device transfer."
        ),
        ha="center",
        fontsize=6.8,
        color="0.35",
    )

    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Date": None})
    if output.suffix.lower() == ".svg":
        _normalize_svg(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "docs" / "assets" / "cuda_colab_t4_scaling.csv",
        help="CSV file containing CUDA scaling reference data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "assets" / "cuda_colab_t4_scaling.svg",
        help="Output figure path.",
    )
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
