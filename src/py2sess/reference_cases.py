"""Load packaged UV and TIR benchmark fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np


_STREAM_VALUE = 1.0 / np.sqrt(3.0)


@dataclass(frozen=True)
class TirBenchmarkCase:
    """Packaged thermal benchmark case."""

    selected_indices: np.ndarray
    wavelengths: np.ndarray
    heights: np.ndarray
    user_angle: float
    tau_arr: np.ndarray
    omega_arr: np.ndarray
    asymm_arr: np.ndarray
    d2s_scaling: np.ndarray
    thermal_bb_input: np.ndarray
    surfbb: np.ndarray
    albedo: np.ndarray
    emissivity: np.ndarray
    ref_2s: np.ndarray
    ref_fo: np.ndarray
    ref_total: np.ndarray
    stream_value: float = _STREAM_VALUE

    @property
    def n_layers(self) -> int:
        """Returns the number of atmospheric layers."""
        return int(self.tau_arr.shape[1])

    @property
    def n_wavelengths(self) -> int:
        """Returns the number of spectral rows in the fixture."""
        return int(self.tau_arr.shape[0])


@dataclass(frozen=True)
class UvBenchmarkCase:
    """Packaged solar benchmark case."""

    selected_indices: np.ndarray
    wavelengths: np.ndarray
    user_obsgeom: np.ndarray
    heights: np.ndarray
    tau: np.ndarray
    omega: np.ndarray
    asymm: np.ndarray
    scaling: np.ndarray
    albedo: np.ndarray
    flux_factor: np.ndarray
    chapman: np.ndarray
    x0: float
    user_stream: float
    user_secant: float
    azmfac: float
    px11: float
    pxsq: np.ndarray
    px0x: np.ndarray
    ulp: float
    ref_2s: np.ndarray
    ref_fo: np.ndarray
    ref_total: np.ndarray
    stream_value: float = _STREAM_VALUE

    @property
    def n_layers(self) -> int:
        """Returns the number of atmospheric layers."""
        return int(self.tau.shape[1])

    @property
    def n_wavelengths(self) -> int:
        """Returns the number of spectral rows in the fixture."""
        return int(self.tau.shape[0])


def _load_npz(name: str) -> dict[str, np.ndarray]:
    """Loads one packaged NumPy fixture into memory."""
    resource = files("py2sess.data.benchmark").joinpath(name)
    with as_file(resource) as path:
        with np.load(path) as data:
            return {key: np.array(data[key]) for key in data.files}


def load_tir_benchmark_case() -> TirBenchmarkCase:
    """Returns the packaged thermal benchmark case."""
    data = _load_npz("tir_benchmark_fixture.npz")
    return TirBenchmarkCase(
        selected_indices=data["selected_indices"],
        wavelengths=data["wavelengths"],
        heights=data["heights"],
        user_angle=float(data["user_angle"][0]),
        tau_arr=data["tau_arr"],
        omega_arr=data["omega_arr"],
        asymm_arr=data["asymm_arr"],
        d2s_scaling=data["d2s_scaling"],
        thermal_bb_input=data["thermal_bb_input"],
        surfbb=data["surfbb"],
        albedo=data["albedo"],
        emissivity=data["emissivity"],
        ref_2s=data["ref_2s"],
        ref_fo=data["ref_fo"],
        ref_total=data["ref_total"],
    )


def load_uv_benchmark_case() -> UvBenchmarkCase:
    """Returns the packaged solar benchmark case."""
    data = _load_npz("uv_benchmark_fixture.npz")
    return UvBenchmarkCase(
        selected_indices=data["selected_indices"],
        wavelengths=data["wavelengths"],
        user_obsgeom=data["user_obsgeom"],
        heights=data["heights"],
        tau=data["tau"],
        omega=data["omega"],
        asymm=data["asymm"],
        scaling=data["scaling"],
        albedo=data["albedo"],
        flux_factor=data["flux_factor"],
        chapman=data["chapman"],
        x0=float(data["x0"][0]),
        user_stream=float(data["user_stream"][0]),
        user_secant=float(data["user_secant"][0]),
        azmfac=float(data["azmfac"][0]),
        px11=float(data["px11"][0]),
        pxsq=data["pxsq"],
        px0x=data["px0x"],
        ulp=float(data["ulp"][0]),
        ref_2s=data["ref_2s"],
        ref_fo=data["ref_fo"],
        ref_total=data["ref_total"],
    )
