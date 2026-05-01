#!/usr/bin/env python3
"""Create the compact Fortran Jacobian reference fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fortran-results",
        type=Path,
        default=Path("../2S-ESS-Lin-Optimization-Test/2S_MASTERS/Results_Lin_Exact_saved"),
        help="Directory containing Fortran linearized output files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/thermal_jacobian_profile1/fortran_jacobian_reference.npz"),
    )
    parser.add_argument("--case", default="SI19_20041231_18N_61W")
    args = parser.parse_args()

    arrays = _reference_arrays(args.fortran_results, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {args.output}")


def _reference_arrays(root: Path, case: str) -> dict[str, np.ndarray]:
    exact = _load(root / f"Exact_{case}.Out")
    surf = _load(root / f"SurfEmiss_Jac_Exact_{case}.Out")
    tsurf = _load(root / f"Tsurf_Jac_Exact_{case}.Out")
    _require_columns("Exact", exact, 6)
    _require_columns("SurfEmiss_Jac_Exact", surf, 5)
    _require_columns("Tsurf_Jac_Exact", tsurf, 5)
    _require_nonzero("Tsurf_Jac_Exact", tsurf[:, 2:])
    out: dict[str, np.ndarray] = {
        "wavelength_nm": exact[:, 1],
        "wavenumber_cm_inv": 1.0e7 / exact[:, 1],
        "radiance_total": exact[:, 2],
        "surface_emissivity_jacobian_total": surf[:, 2],
        "surface_temperature_jacobian_total": tsurf[:, 2],
    }
    if exact.shape[1] >= 6:
        out["radiance_2s"] = exact[:, 4]
        out["radiance_fo"] = exact[:, 5]
    if surf.shape[1] >= 5:
        out["surface_emissivity_jacobian_2s"] = surf[:, 3]
        out["surface_emissivity_jacobian_fo"] = surf[:, 4]
    if tsurf.shape[1] >= 5:
        out["surface_temperature_jacobian_2s"] = tsurf[:, 3]
        out["surface_temperature_jacobian_fo"] = tsurf[:, 4]

    for name, filename in (
        ("temperature", f"T_Jac_Exact_{case}.Out"),
        ("H2O", f"H2O_Jac_Exact_{case}.Out"),
        ("CH4", f"CH4_Jac_Exact_{case}.Out"),
        ("CO2", f"CO2_Jac_Exact_{case}.Out"),
        ("O2", f"O2_Jac_Exact_{case}.Out"),
    ):
        path = root / filename
        if path.exists():
            values, levels = _profile_jacobian(path)
            out[f"{name}_profile_jacobian_total"] = values
            out[f"{name}_profile_jacobian_levels"] = levels
    return out


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path)
    return np.atleast_2d(data)


def _require_columns(name: str, data: np.ndarray, columns: int) -> None:
    if data.shape[1] < columns:
        raise ValueError(
            f"{name} output has {data.shape[1]} columns; expected at least {columns}. "
            "Use a Fortran output that includes total, 2S, and FO columns."
        )


def _require_nonzero(name: str, data: np.ndarray) -> None:
    if not np.any(np.abs(data) > 0.0):
        raise ValueError(
            f"{name} output is all zero. Use a Fortran run with blackbody/source "
            "Jacobian support enabled."
        )


def _profile_jacobian(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _load(path)
    rows = np.unique(data[:, 0].astype(int))
    levels = np.unique(data[:, 1].astype(int))
    out = np.full((rows.size, levels.size), np.nan, dtype=float)
    row_index = {value: index for index, value in enumerate(rows)}
    level_index = {value: index for index, value in enumerate(levels)}
    for row in data:
        out[row_index[int(row[0])], level_index[int(row[1])]] = row[4]
    return out, levels


if __name__ == "__main__":
    main()
