#!/usr/bin/env python3
"""Build py2sess native CUDA extension on Colab with explicit diagnostics."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import traceback
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def _install_build_tools() -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "setuptools>=68",
        "wheel",
        "ninja",
    ]
    code = _run(command)
    if code:
        raise SystemExit(f"build-tool install failed with exit code {code}")


def _probe_cuda() -> tuple[str, str]:
    print(f"python={sys.executable}")
    print(f"cwd={ROOT}")
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception as exc:
        raise SystemExit(f"could not import torch and cpp_extension: {exc}") from exc

    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"CUDA_HOME={CUDA_HOME}")
    nvcc = shutil.which("nvcc")
    print(f"nvcc={nvcc}")
    if nvcc:
        _run([nvcc, "--version"])

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. In Colab, switch Runtime -> Change runtime type -> GPU."
        )
    if CUDA_HOME is None:
        raise SystemExit("torch could not find CUDA_HOME. Check the Colab GPU runtime and nvcc.")
    if nvcc is None:
        raise SystemExit("nvcc is not on PATH. Check the Colab GPU runtime.")

    major, minor = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name(0)
    arch = f"{major}.{minor}"
    print(f"cuda_device={device_name}")
    print(f"TORCH_CUDA_ARCH_LIST={arch}")
    return arch, str(CUDA_HOME)


def _verify_import() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    extension_files = sorted((ROOT / "src" / "py2sess").glob("_native*.so"))
    print("native_extension_files=" + ", ".join(str(path) for path in extension_files))
    try:
        native = importlib.import_module("py2sess._native")
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(
            f"build finished, but direct import of py2sess._native failed: {exc}"
        ) from exc

    direct_info = dict(native.backend_info())
    print(f"direct_native_backend_info={direct_info}")

    from py2sess import native_backend_info

    info = native_backend_info()
    print(f"native_backend_info={info}")
    if not info.get("available") or not info.get("cuda"):
        raise SystemExit("build finished, but py2sess native backend does not report cuda=True")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-build-tools", action="store_true")
    parser.add_argument("--max-jobs", default="2")
    parser.add_argument("--clean-first", action="store_true")
    args = parser.parse_args()

    if args.install_build_tools:
        _install_build_tools()

    arch, _cuda_home = _probe_cuda()
    env = os.environ.copy()
    env.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    env["PY2SESS_BUILD_CUDA"] = "1"
    env.setdefault("MAX_JOBS", args.max_jobs)

    if args.clean_first:
        for pattern in ("build",):
            path = ROOT / pattern
            if path.exists():
                print(f"remove {path}")
                shutil.rmtree(path)
        for path in (ROOT / "src" / "py2sess").glob("_native*.so"):
            print(f"remove {path}")
            path.unlink()

    code = _run([sys.executable, "setup.py", "build_ext", "--inplace", "-v"], env=env)
    if code:
        raise SystemExit(
            "native CUDA build failed. Copy the compiler error lines above this message; "
            "they are the useful part, not the final CalledProcessError wrapper."
        )

    _verify_import()
    print("native CUDA build ok")


if __name__ == "__main__":
    main()
