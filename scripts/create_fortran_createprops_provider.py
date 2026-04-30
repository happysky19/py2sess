#!/usr/bin/env python3
"""Create a local optical-provider directory from a Fortran CreateProps dump."""

from __future__ import annotations

import argparse
from pathlib import Path

from py2sess.optical.createprops import parse_createprops_dump, write_createprops_provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("uv", "tir"))
    parser.add_argument("dump", type=Path, help="Fortran CreateProps text dump.")
    parser.add_argument("output_dir", type=Path, help="Output provider array directory.")
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=None,
        help="TIR GEOCAPE profile file. Defaults to the path implied by the dump name.",
    )
    args = parser.parse_args()

    arrays = parse_createprops_dump(
        args.dump,
        kind=args.kind,
        profile_file=args.profile_file,
    )
    write_createprops_provider(arrays, args.output_dir, kind=args.kind)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
