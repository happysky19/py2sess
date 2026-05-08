#!/usr/bin/env python3
"""Export portable full-spectrum benchmark inputs for Colab/local reruns."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_ROOT = Path("/Users/thl/MyFolder/Research/2S-ESS")
DEFAULT_OUTPUT = ROOT / "outputs" / "full_spectrum_benchmark" / "input_bundle"


def _copy_file(source: Path, target: Path, *, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover
        raise ValueError(f"unsupported link mode: {mode}")


def _copy_tree_files(source: Path, target: Path, *, mode: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _copy_file(path, target / path.relative_to(source), mode=mode)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _rewrite_scene_paths(scene: dict[str, Any], *, time_name: str) -> dict[str, Any]:
    scene = dict(scene)
    scene["surface"]["albedo"]["geocape_emissivity"]["path"] = (
        "../geocape_data/Surface_Data/Emissivity_1.asc"
    )
    if "solar" in scene:
        scene["solar"]["flux_factor"]["geocape_solar_spectrum"]["path"] = (
            "../geocape_data/newkur.dat"
        )
    loadings = scene["opacity"]["aerosol"]["loadings"]
    names = ("OC", "SEASACC", "SEASCOA", "BC", "SO4")
    loadings["paths"] = [
        f"../geocape_data/Aerosol_Data/TZ7/{name}_1_2006726_{time_name}.dat" for name in names
    ]
    scene["opacity"]["aerosol"]["ssprops"]["path"] = "../geocape_data/SSprops"
    return scene


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "is_symlink": path.is_symlink(),
                }
            )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "is_symlink"))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path) -> None:
    path.write_text(
        "# py2sess Full-Spectrum Benchmark Input Bundle\n\n"
        "This folder contains the large UV 280k and TIR 200k benchmark inputs that are not tracked in git.\n\n"
        "Colab usage after cloning py2sess and mounting or uploading this bundle:\n\n"
        "```bash\n"
        "PYTHONPATH=src python scripts/benchmark_full_spectrum_rt.py \\\n"
        "  --input-root /content/py2sess_full_spectrum_inputs \\\n"
        "  --systems python \\\n"
        "  --backend-set all \\\n"
        "  --torch-dtypes float64 \\\n"
        "  --repeats 5 \\\n"
        "  --output-dir outputs/full_spectrum_benchmark_colab\n"
        "```\n\n"
        "The script preserves Colab's existing CUDA PyTorch wheel; install py2sess with `pip install -e .`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-bundles", type=Path, default=ROOT / "benchmark_bundles")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    args = parser.parse_args()

    output = args.output_dir
    bundle_dir = output / "benchmark_bundles"
    profile_dir = output / "profiles"
    geocape_dir = output / "geocape_data"
    geocape_source = args.external_root / "geocape_data"

    for name in (
        "uv_gas_xsec.nc",
        "tir_gas_xsec.nc",
        "uv_reference_outputs.npz",
        "tir_reference_outputs.npz",
    ):
        _copy_file(args.benchmark_bundles / name, bundle_dir / name, mode=args.link_mode)

    uv_scene = _rewrite_scene_paths(
        _load_yaml(args.benchmark_bundles / "uv_scene_python.yaml"),
        time_name="1500",
    )
    tir_scene = _rewrite_scene_paths(
        _load_yaml(args.benchmark_bundles / "tir_scene_python.yaml"),
        time_name="0000",
    )
    _write_yaml(bundle_dir / "uv_scene_python.yaml", uv_scene)
    _write_yaml(bundle_dir / "tir_scene_python.yaml", tir_scene)

    for name in ("Profiles_1_2006726_0000.dat", "Profiles_1_2006726_1500.dat"):
        _copy_file(geocape_source / "Profile_Data" / name, profile_dir / name, mode=args.link_mode)
    _copy_file(
        geocape_source / "Surface_Data" / "Emissivity_1.asc",
        geocape_dir / "Surface_Data" / "Emissivity_1.asc",
        mode=args.link_mode,
    )
    _copy_file(geocape_source / "newkur.dat", geocape_dir / "newkur.dat", mode=args.link_mode)
    for time_name in ("0000", "1500"):
        for name in ("OC", "SEASACC", "SEASCOA", "BC", "SO4"):
            file_name = f"{name}_1_2006726_{time_name}.dat"
            _copy_file(
                geocape_source / "Aerosol_Data" / "TZ7" / file_name,
                geocape_dir / "Aerosol_Data" / "TZ7" / file_name,
                mode=args.link_mode,
            )
    _copy_tree_files(geocape_source / "SSprops", geocape_dir / "SSprops", mode=args.link_mode)

    _write_readme(output / "README.md")
    _write_manifest(output / "manifest.csv", _manifest_rows(output))
    print(output)


if __name__ == "__main__":
    main()
