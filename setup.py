from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

_build_ext_cls = None


def _positive_int_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return str(parsed)


def _native_extension():
    global _build_ext_cls
    try:
        from torch.utils import cpp_extension
    except Exception:
        return None

    _build_ext_cls = cpp_extension.BuildExtension
    sources = [
        "native/csrc/native_dispatch.cpp",
        "native/csrc/native_module.cpp",
        "native/csrc/native_bindings.cpp",
    ]
    include_dirs = ["native/csrc"]
    build_cuda = os.environ.get("PY2SESS_BUILD_CUDA", "0") == "1"
    define_macros = []
    extra_compile_args = {"cxx": ["-std=c++17"]}
    extra_link_args = []
    if build_cuda and cpp_extension.CUDA_HOME is None:
        raise RuntimeError("PY2SESS_BUILD_CUDA=1 requested, but torch could not find CUDA_HOME")
    if build_cuda:
        sources.append("native/csrc/native_dispatch_cuda.cu")
        extension_cls = cpp_extension.CUDAExtension
        define_macros.append(("PY2SESS_WITH_CUDA", "1"))
        for name in ("PY2SESS_NATIVE_2S_CHUNKS", "PY2SESS_NATIVE_BLOCK_SIZE"):
            value = _positive_int_env(name)
            if value is not None:
                define_macros.append((name, value))
        extra_compile_args["nvcc"] = ["-std=c++17", "--extended-lambda"]
        torch_lib = Path(cpp_extension.__file__).resolve().parents[1] / "lib"
        extra_link_args.append(f"-Wl,-rpath,{torch_lib}")
    else:
        extension_cls = cpp_extension.CppExtension

    return extension_cls(
        name="py2sess._native",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


extension = _native_extension()
setup(
    ext_modules=[] if extension is None else [extension],
    cmdclass={} if _build_ext_cls is None else {"build_ext": _build_ext_cls},
)
