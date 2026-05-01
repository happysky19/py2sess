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
    return {
        "wavelength_nm": exact[:, 1],
        "wavenumber_cm_inv": 1.0e7 / exact[:, 1],
        "radiance_total": exact[:, 2],
        "surface_emissivity_jacobian_total": surf[:, 2],
        "surface_temperature_jacobian_total": tsurf[:, 2],
    }


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
        raise ValueError(f"{name} output must contain nonzero surface-temperature Jacobian values.")


if __name__ == "__main__":
    main()
