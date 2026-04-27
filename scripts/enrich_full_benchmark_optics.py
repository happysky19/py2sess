#!/usr/bin/env python3
"""Add Python-regenerable optical inputs to a local full-spectrum bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from py2sess.optical.createprops import parse_createprops_dump


def _subset_rows(
    optics: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = int(optics["depol"].shape[0])
    return {
        key: (value[indices] if np.asarray(value).shape[:1] == (n_rows,) else value)
        for key, value in optics.items()
    }


def _copy_bundle_with_optics(
    *,
    bundle_path: Path,
    output_path: Path,
    optics: dict[str, np.ndarray],
) -> None:
    with np.load(bundle_path) as source:
        arrays = {key: np.array(source[key]) for key in source.files}
    n_rows = int(np.asarray(arrays["wavelengths"]).shape[0])
    if int(optics["depol"].shape[0]) != n_rows:
        if "selected_indices" not in arrays:
            raise ValueError("bundle row count does not match dump and has no selected_indices")
        optics = _subset_rows(optics, np.asarray(arrays["selected_indices"], dtype=int) - 1)
    arrays.update(optics)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("uv", "tir"))
    parser.add_argument("dump", type=Path, help="Original Fortran CreateProps text dump.")
    parser.add_argument("bundle", type=Path, help="Existing local benchmark bundle.")
    parser.add_argument("output", type=Path, help="Output enriched benchmark bundle.")
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=None,
        help="TIR GEOCAPE profile file. Defaults to the path implied by the dump name.",
    )
    args = parser.parse_args()

    if args.output.resolve() == args.bundle.resolve():
        raise ValueError("output must be a different path; keep the original bundle intact")
    if not args.dump.is_file():
        raise FileNotFoundError(args.dump)
    if not args.bundle.is_file():
        raise FileNotFoundError(args.bundle)

    optics = parse_createprops_dump(
        args.dump,
        kind=args.kind,
        profile_file=args.profile_file,
    )
    _copy_bundle_with_optics(
        bundle_path=args.bundle,
        output_path=args.output,
        optics=optics,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
