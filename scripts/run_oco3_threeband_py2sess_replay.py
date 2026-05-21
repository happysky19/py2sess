#!/usr/bin/env python3
"""Replay OCO-3 three-band soundings with py2sess.

By default, the driver uses official posterior state fields, OCO ABSCO tables,
the OCO solar model, L1B ILS sampling, L2 posterior gaussian-log aerosol
loading, and L1B instrument Stokes coefficients. It does not fit pressure,
spectroscopy, wavelength, gas columns, or aerosol loading. It applies a
continuum-constrained surface brightness adjustment for each sounding and band;
pass --surface-brdf-retrieval none for a strict fixed-L2-surface replay. OCO
photon radiances and py2sess replay spectra are both reported as energy
spectral radiance.

The default polarization treatment uses the OCO L1B normalized-radiance
convention, L = I + (m12/m11) Q + (m13/m11) U. A raw detector projection,
L = m11 I + m12 Q + m13 U, is retained only as a convention check.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from functools import lru_cache
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py2sess_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "py2sess_cache"))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "oco3_joint_official_downloads" / "20220624_17767a"
DEFAULT_CASE_DIR = ROOT / "outputs" / "oco3_threeband_replay" / "20220624_17767a"
DEFAULT_ABSCO_DIR = ROOT / "outputs" / "oco3_joint_official_downloads" / "absco_v52"
DEFAULT_CO2_ABSCO = DEFAULT_ABSCO_DIR / "co2_v52.hdf"
DEFAULT_O2_ABSCO = DEFAULT_ABSCO_DIR / "o2_v52.hdf"
DEFAULT_H2O_ABSCO = DEFAULT_ABSCO_DIR / "h2o_v52.hdf"
DEFAULT_OCO_SOLAR_MODEL = (
    ROOT / "outputs" / "oco3_joint_official_downloads" / "solar" / "l2_solar_model.h5"
)
DEFAULT_RT_RETRIEVAL_SUPPORT_DIR = ROOT / "scripts" / "oco3_paper_support" / "rt_retrieval"
DEFAULT_OCO3_EOF_FILE = DEFAULT_RT_RETRIEVAL_SUPPORT_DIR / "l2_oco3_eof.h5"

BANDS = ("o2", "wco2", "sco2")
BAND_LABELS = {"o2": "O2 A", "wco2": "weak CO2", "sco2": "strong CO2"}
BAND_INDEX = {"o2": 0, "wco2": 1, "sco2": 2}
BAND_REFERENCE_WAVELENGTH_UM = {"o2": 0.77, "wco2": 1.615, "sco2": 2.06}
OCO_CONTINUUM_FIELD = {
    "o2": "L1bScSpectralParameters/rad_continuum_o2",
    "wco2": "L1bScSpectralParameters/rad_continuum_weak_co2",
    "sco2": "L1bScSpectralParameters/rad_continuum_strong_co2",
}
RPV_KERNEL_NORMALIZATION = 20.0
OCEAN_SURFACE_TYPE = "Coxmunk,Lambertian"
OCO_COXMUNK_REFRACTIVE_INDEX = {"o2": 1.331, "wco2": 1.318, "sco2": 1.303}
# solar_obs_brdf_from_kernels follows the local py2sess Fourier convention;
# its direct_brf term is twice the OCO L2 BRDF reflectance kernel used here.
SOLAR_OBS_DIRECT_BRF_TO_OCO_BRF = 0.5
# RtRetrieval AerosolOptical normalizes aerosol extinction at reference_wn=1e4/0.755.
AEROSOL_REFERENCE_WAVELENGTH_UM = 0.755
AEROSOL_HG_MOMENT_ORDER = 80
PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299792458.0
OCO_NOISE_MAX_MS = (7.00e20, 2.45e20, 1.25e20)
OCO3_EOF_SCALE_TO_STDDEV = 1.0e19
OCO_FLUORESCENCE_REFERENCE_WAVELENGTH_UM = 0.757
AEROSOL_TYPE_HG_DEFAULTS = {
    "DU": {"ssa": 0.94, "g": 0.75, "angstrom": 0.20},
    "SS": {"ssa": 0.99, "g": 0.75, "angstrom": 0.10},
    "BC": {"ssa": 0.75, "g": 0.55, "angstrom": 1.20},
    "OC": {"ssa": 0.90, "g": 0.65, "angstrom": 1.20},
    "SO": {"ssa": 0.98, "g": 0.65, "angstrom": 1.50},
    "Ice": {"ssa": 0.999, "g": 0.85, "angstrom": 0.00},
    "Water": {"ssa": 0.999, "g": 0.85, "angstrom": 0.00},
    "ST": {"ssa": 0.94, "g": 0.75, "angstrom": 0.20},
}
OCO_L2FP_AEROSOL_PROPERTY_GROUPS = {
    "DU": "DU",
    "SS": "SS",
    "BC": "BC",
    "OC": "OC",
    "SO": "SO",
    "Ice": "ice_cloud_MODIS6_deltaM_1000",
    "Water": "wc_008",
    "ST": "strat",
}
OCO_SCALAR_REPLAY_AEROSOL_TYPES = frozenset(("DU", "SS", "BC", "OC", "SO"))
ABSCO_GAS_DATASET = {
    "o2": "Gas_07_Absorption",
    "co2": "Gas_02_Absorption",
    "h2o": "Gas_01_Absorption",
}
O2_DRY_AIR_MOLE_FRACTION = 0.2095
M_AIR = 28.9647e-3
M_H2O = 18.01528e-3
M2_TO_CM2 = 1.0e-4
AVOGADRO_PER_MOL = 6.02214076e23
STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class AbscoTable:
    path: Path
    dataset: str
    wavenumber: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    broadener: np.ndarray

    @classmethod
    def open(cls, path: Path, dataset: str) -> "AbscoTable":
        with h5py.File(path, "r") as handle:
            return cls(
                path=path,
                dataset=dataset,
                wavenumber=handle["Wavenumber"][...],
                pressure=handle["Pressure"][...],
                temperature=handle["Temperature"][...],
                broadener=handle["Broadener_01_VMR"][...],
            )

    def cross_section_cm2(
        self,
        *,
        wavelength_um: np.ndarray,
        pressure_pa: np.ndarray,
        temperature_k: np.ndarray,
        h2o_vmr: np.ndarray,
    ) -> np.ndarray:
        """Interpolate ABSCO cross sections to wavelength/layer points."""
        wavelength = np.asarray(wavelength_um, dtype=float)
        p_layer = np.asarray(pressure_pa, dtype=float)
        t_layer = np.asarray(temperature_k, dtype=float)
        h2o_layer = np.asarray(h2o_vmr, dtype=float)
        if p_layer.shape != t_layer.shape or p_layer.shape != h2o_layer.shape:
            raise ValueError("pressure, temperature, and h2o_vmr must have the same shape")

        wn = 1.0e4 / wavelength
        if wn.min() < self.wavenumber[0] or wn.max() > self.wavenumber[-1]:
            raise ValueError(
                f"{self.path.name} does not cover requested wavenumbers "
                f"{wn.min():.2f}-{wn.max():.2f} cm^-1"
            )
        lo = max(int(np.searchsorted(self.wavenumber, wn.min(), side="left")) - 1, 0)
        hi = min(
            int(np.searchsorted(self.wavenumber, wn.max(), side="right")) + 1, self.wavenumber.size
        )
        grid = self.wavenumber[lo:hi]
        with h5py.File(self.path, "r") as handle:
            cube = handle[self.dataset][:, :, :, lo:hi]

        wn_lo, wn_hi, wn_w = _linear_interp_indices(grid, wn)
        p_lo, p_hi, p_w = _bracket(self.pressure, p_layer)
        b_lo, b_hi, b_w = _bracket(
            self.broadener, np.clip(h2o_layer, self.broadener[0], self.broadener[-1])
        )
        out = np.zeros((wavelength.size, p_layer.size), dtype=float)
        for layer in range(p_layer.size):
            for p_choice, pw in ((p_lo[layer], 1.0 - p_w[layer]), (p_hi[layer], p_w[layer])):
                t_lo, t_hi, t_w = _bracket(
                    self.temperature[p_choice],
                    np.array(
                        [
                            np.clip(
                                t_layer[layer],
                                self.temperature[p_choice, 0],
                                self.temperature[p_choice, -1],
                            )
                        ]
                    ),
                )
                for t_choice, tw in (
                    (int(t_lo[0]), 1.0 - float(t_w[0])),
                    (int(t_hi[0]), float(t_w[0])),
                ):
                    for b_choice, bw in (
                        (b_lo[layer], 1.0 - b_w[layer]),
                        (b_hi[layer], b_w[layer]),
                    ):
                        weight = float(pw * tw * bw)
                        if weight == 0.0:
                            continue
                        out[:, layer] += weight * _interp_with_indices(
                            cube[int(p_choice), int(t_choice), int(b_choice)],
                            wn_lo,
                            wn_hi,
                            wn_w,
                        )
        return out


def _linear_interp_indices(
    grid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_arr = np.asarray(grid, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    hi = np.searchsorted(grid_arr, value_arr, side="right")
    hi = np.clip(hi, 1, grid_arr.size - 1)
    lo = hi - 1
    span = grid_arr[hi] - grid_arr[lo]
    weight = np.where(span > 0.0, (value_arr - grid_arr[lo]) / span, 0.0)
    return lo, hi, weight


def _interp_with_indices(
    values: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    value_arr = np.asarray(values, dtype=float)
    return value_arr[lo] * (1.0 - weight) + value_arr[hi] * weight


def _bracket(grid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    hi = np.searchsorted(grid, values, side="right")
    hi = np.clip(hi, 1, grid.size - 1)
    lo = hi - 1
    span = grid[hi] - grid[lo]
    weight = np.where(span > 0.0, (values - grid[lo]) / span, 0.0)
    return lo, hi, np.clip(weight, 0.0, 1.0)


@dataclass(frozen=True)
class AerosolInputs:
    extinction_tau: np.ndarray
    scattering_tau: np.ndarray
    moments: np.ndarray
    polarization_moments: np.ndarray
    interp_fraction: np.ndarray
    total_aod_used: float
    phase_model: str


@dataclass(frozen=True)
class OcoL2fpAerosolProperty:
    wave_number_cm: np.ndarray
    extinction_coefficient: np.ndarray
    scattering_coefficient: np.ndarray
    phase_moments: np.ndarray
    polarization_moments: np.ndarray


@dataclass(frozen=True)
class EofCorrection:
    values: np.ndarray
    scale_values: np.ndarray
    basis_model: str


@dataclass(frozen=True)
class Py2sessReplayResult:
    scalar_radiance: np.ndarray
    polarization_correction: np.ndarray
    radiance: np.ndarray


@dataclass(frozen=True)
class SurfaceBrdfRetrieval:
    scale: float
    tilt: float
    n_points: int
    fit_rmse_percent: float
    status: str
    iterations: int = 1


@dataclass(frozen=True)
class StokesProjection:
    scalar_factor: float
    analyzer_q: float
    analyzer_u: float
    description: str


@dataclass(frozen=True)
class Py2sessRtContext:
    tau: np.ndarray
    ssa: np.ndarray
    g: np.ndarray
    delta_m_truncation_factor: np.ndarray
    fo_scatter_term: np.ndarray
    rayleigh_scattering_tau: np.ndarray
    aerosol_scattering_tau: np.ndarray
    aerosol_polarization_moments: np.ndarray
    aerosol_interp_fraction: np.ndarray
    scattering_tau: np.ndarray
    depol: np.ndarray
    height_grid: np.ndarray
    angles: np.ndarray
    solar_reference_factor: np.ndarray | None
    stokes_projection: StokesProjection
    polarization_correction: str
    polarization_sign: int
    polarization_diffuse_azimuths: int
    stream_value: float


def _band_slices(counts: np.ndarray) -> dict[str, slice]:
    start = 0
    out: dict[str, slice] = {}
    for band, count in zip(BANDS, counts, strict=True):
        stop = start + int(count)
        out[band] = slice(start, stop)
        start = stop
    return out


def _load_selected_cases(case_dir: Path, count: int) -> list[dict[str, str]]:
    path = case_dir / "selected_soundings.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:count]


def _single_data_file(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        preview = ", ".join(path.name for path in matches[:5])
        if len(matches) > 5:
            preview += ", ..."
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} file in {data_dir}; "
            f"found {len(matches)}" + (f": {preview}" if preview else "")
        )
    return matches[0]


def _default_oco_solar_model() -> Path | None:
    if DEFAULT_OCO_SOLAR_MODEL.exists():
        return DEFAULT_OCO_SOLAR_MODEL
    return None


def _default_oco3_eof_file() -> Path | None:
    env_path = os.environ.get("RTRF_OCO3_EOF_FILE")
    if env_path:
        return Path(env_path)
    if DEFAULT_OCO3_EOF_FILE.exists():
        return DEFAULT_OCO3_EOF_FILE
    return None


def _surface_relative_azimuth(solar_azimuth: float, view_azimuth: float) -> float:
    delta = abs(float(solar_azimuth) - float(view_azimuth)) % 360.0
    return 360.0 - delta if delta > 180.0 else delta


def _rt_relative_azimuth(solar_azimuth: float, view_azimuth: float) -> float:
    # Match RtRetrievalFramework Level1b::relative_azimuth: "follow the
    # photons" convention with the observation azimuth reversed.
    return (180.0 + float(view_azimuth) - float(solar_azimuth)) % 360.0


def _gas_lookup_wavelength_in_atmosphere_frame(
    *,
    wavelength_um: np.ndarray,
    gas_doppler: str,
    relative_velocity_m_s: float,
) -> tuple[np.ndarray, float]:
    wavelength = np.asarray(wavelength_um, dtype=float)
    if gas_doppler == "off":
        return wavelength, 0.0
    if gas_doppler != "l2-los":
        raise ValueError(f"unknown gas Doppler treatment: {gas_doppler!r}")
    velocity = float(relative_velocity_m_s)
    beta = velocity / SPEED_OF_LIGHT_M_S
    return wavelength * (1.0 + beta), velocity


def _solar_reference_lookup_wavelength(
    *,
    wavelength_um: np.ndarray,
    solar_doppler: str,
    solar_relative_velocity_m_s: float,
    los_relative_velocity_m_s: float,
) -> tuple[np.ndarray, float]:
    wavelength = np.asarray(wavelength_um, dtype=float)
    if solar_doppler == "off":
        return wavelength, 0.0
    if solar_doppler == "l2-solar":
        velocity = float(solar_relative_velocity_m_s)
        instrument_beta = float(los_relative_velocity_m_s) / SPEED_OF_LIGHT_M_S
        solar_beta = velocity / SPEED_OF_LIGHT_M_S
        return wavelength * (1.0 + instrument_beta) / (1.0 + solar_beta), velocity
    elif solar_doppler == "l2-los":
        velocity = float(los_relative_velocity_m_s)
    else:
        raise ValueError(f"unknown solar Doppler treatment: {solar_doppler!r}")
    beta = velocity / SPEED_OF_LIGHT_M_S
    return wavelength / (1.0 + beta), velocity


def _interp_profile_to_retrieval_levels(
    *,
    target_pressure_pa: np.ndarray,
    met_pressure_pa: np.ndarray,
    met_values: np.ndarray,
) -> np.ndarray:
    xp = np.asarray(met_pressure_pa, dtype=float)
    fp = np.asarray(met_values, dtype=float)
    x = np.asarray(target_pressure_pa, dtype=float)
    return np.interp(x, xp, fp)


def _specific_humidity_to_vmr(q: np.ndarray) -> np.ndarray:
    q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 0.95)
    return q_arr / np.maximum(1.0 - q_arr, 1.0e-12) * (M_AIR / M_H2O)


def _hydrostatic_column_cm2(
    pressure_pa: np.ndarray,
    layer_h2o_vmr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure_pa, dtype=float)
    h2o = np.asarray(layer_h2o_vmr, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2 or np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure levels must increase from top to bottom")
    if h2o.shape != (pressure.size - 1,):
        raise ValueError("layer H2O VMR must have one value per pressure layer")
    if np.any(h2o < 0.0) or not np.all(np.isfinite(h2o)):
        raise ValueError("layer H2O VMR must be finite and nonnegative")
    delta_pressure = np.diff(pressure)
    dry_air = (
        delta_pressure
        / STANDARD_GRAVITY_M_S2
        * AVOGADRO_PER_MOL
        / (M_AIR + h2o * M_H2O)
        * M2_TO_CM2
    )
    h2o_column = dry_air * h2o
    wet_air = dry_air + h2o_column
    return dry_air, wet_air, h2o_column


def _hydrostatic_heights_km(
    *,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    h2o_vmr: np.ndarray,
    surface_height_km: float,
) -> np.ndarray:
    pressure = np.asarray(pressure_pa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    h2o = np.asarray(h2o_vmr, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2 or np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure levels must increase from top to bottom")
    if temperature.shape != pressure.shape or h2o.shape != pressure.shape:
        raise ValueError("temperature and H2O VMR must match pressure levels")
    layer_temperature = 0.5 * (temperature[:-1] + temperature[1:])
    layer_h2o = 0.5 * (h2o[:-1] + h2o[1:])
    moist_molar_mass = (M_AIR + layer_h2o * M_H2O) / (1.0 + layer_h2o)
    gas_constant = 8.31446261815324
    thickness_km = (
        gas_constant
        * layer_temperature
        / (moist_molar_mass * STANDARD_GRAVITY_M_S2)
        * np.log(pressure[1:] / pressure[:-1])
        / 1000.0
    )
    heights = np.empty_like(pressure, dtype=float)
    heights[-1] = float(surface_height_km)
    for layer in range(pressure.size - 2, -1, -1):
        heights[layer] = heights[layer + 1] + thickness_km[layer]
    return heights


def _metadata_o2_vmr(std: h5py.File) -> float:
    if "Metadata/VMRO2" not in std:
        return O2_DRY_AIR_MOLE_FRACTION
    value = float(np.asarray(std["Metadata/VMRO2"][...], dtype=float).reshape(-1)[0])
    return value if np.isfinite(value) and value > 0.0 else O2_DRY_AIR_MOLE_FRACTION


def _state_for_retrieval(
    std: h5py.File,
    index: int,
    *,
    layer_pressure_method: str = "geometric",
    surface_pressure_offset_hpa: float = 0.0,
    surface_pressure_column_mode: str = "fixed-columns",
) -> dict[str, np.ndarray | float]:
    rr = std["RetrievalResults"]
    pressure_pa = rr["vector_pressure_levels"][index].astype(float)
    original_pressure_pa = pressure_pa.copy()
    original_surface_pressure_pa = float(pressure_pa[-1])
    if surface_pressure_offset_hpa != 0.0:
        offset_pa = float(surface_pressure_offset_hpa) * 100.0
        surface_pressure_pa = original_surface_pressure_pa + offset_pa
        if surface_pressure_pa <= 0.0:
            raise ValueError("diagnostic surface pressure offset produced non-positive pressure")
        pressure_pa = pressure_pa * (surface_pressure_pa / original_surface_pressure_pa)
    original_heights_km = rr["vector_altitude_levels"][index].astype(float) / 1000.0
    heights_km = original_heights_km.copy()
    met_pressure = rr["vector_pressure_levels_met"][index].astype(float)
    met_temperature = rr["temperature_profile_met"][index].astype(float)
    met_q = rr["specific_humidity_profile_met"][index].astype(float)
    temperature_offset = float(rr["temperature_offset_fph"][index])
    h2o_scale_factor = float(rr["h2o_scale_factor"][index])
    temperature = _interp_profile_to_retrieval_levels(
        target_pressure_pa=pressure_pa,
        met_pressure_pa=met_pressure,
        met_values=met_temperature,
    )
    temperature = temperature + temperature_offset
    met_temperature = met_temperature + temperature_offset
    met_h2o_vmr = _specific_humidity_to_vmr(met_q) * h2o_scale_factor
    h2o_vmr = _interp_profile_to_retrieval_levels(
        target_pressure_pa=pressure_pa,
        met_pressure_pa=met_pressure,
        met_values=met_h2o_vmr,
    )
    co2_vmr = rr["co2_profile"][index].astype(float)

    if layer_pressure_method == "geometric":
        layer_pressure = np.sqrt(pressure_pa[:-1] * pressure_pa[1:])
    elif layer_pressure_method == "arithmetic":
        layer_pressure = 0.5 * (pressure_pa[:-1] + pressure_pa[1:])
    else:
        raise ValueError(f"unknown layer pressure method: {layer_pressure_method!r}")
    layer_temperature = 0.5 * (temperature[:-1] + temperature[1:])
    layer_h2o_vmr = 0.5 * (h2o_vmr[:-1] + h2o_vmr[1:])
    layer_co2_vmr = 0.5 * (co2_vmr[:-1] + co2_vmr[1:])
    dry_air_col_cm2 = (
        rr["retrieved_dry_air_column_layer_thickness"][index].astype(float) * M2_TO_CM2
    )
    wet_air_col_cm2 = (
        rr["retrieved_wet_air_column_layer_thickness"][index].astype(float) * M2_TO_CM2
    )
    h2o_col_cm2 = rr["retrieved_h2o_column_layer_thickness"][index].astype(float) * M2_TO_CM2
    if surface_pressure_column_mode == "fixed-columns":
        pass
    elif surface_pressure_column_mode == "hydrostatic-columns":
        dry_air_col_cm2, wet_air_col_cm2, h2o_col_cm2 = _hydrostatic_column_cm2(
            pressure_pa,
            layer_h2o_vmr,
        )
        heights_km = _hydrostatic_heights_km(
            pressure_pa=pressure_pa,
            temperature_k=temperature,
            h2o_vmr=h2o_vmr,
            surface_height_km=float(original_heights_km[-1]),
        )
    else:
        raise ValueError(f"unknown surface pressure column mode: {surface_pressure_column_mode!r}")
    o2_vmr = _metadata_o2_vmr(std)

    return {
        "pressure_pa": pressure_pa,
        "original_pressure_pa": original_pressure_pa,
        "heights_km": heights_km,
        "original_heights_km": original_heights_km,
        "temperature_k": temperature,
        "h2o_vmr": h2o_vmr,
        "co2_vmr": co2_vmr,
        "met_pressure_pa": met_pressure,
        "met_temperature_k": met_temperature,
        "met_h2o_vmr": met_h2o_vmr,
        "layer_pressure_pa": layer_pressure,
        "layer_temperature_k": layer_temperature,
        "layer_h2o_vmr": layer_h2o_vmr,
        "o2_vmr": o2_vmr,
        "o2_col_cm2": dry_air_col_cm2 * o2_vmr,
        "co2_col_cm2": dry_air_col_cm2 * layer_co2_vmr,
        "h2o_col_cm2": h2o_col_cm2,
        "dry_air_col_cm2": dry_air_col_cm2,
        "wet_air_col_cm2": wet_air_col_cm2,
        "xco2_ppm": float(rr["xco2"][index]) * 1.0e6,
        "surface_pressure_original_hpa": original_surface_pressure_pa / 100.0,
        "surface_pressure_used_hpa": float(pressure_pa[-1]) / 100.0,
        "surface_pressure_offset_hpa": float(surface_pressure_offset_hpa),
        "surface_pressure_column_mode": surface_pressure_column_mode,
    }


def _profile_at_pressure(
    *,
    pressure_levels_pa: np.ndarray,
    values: np.ndarray,
    target_pressure_pa: np.ndarray,
) -> np.ndarray:
    pressure = np.asarray(pressure_levels_pa, dtype=float)
    profile = np.asarray(values, dtype=float)
    target = np.asarray(target_pressure_pa, dtype=float)
    return np.interp(target, pressure, profile)


def _simpson_layer_pressure_grid(
    pressure_levels_pa: np.ndarray,
    *,
    nsub: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if nsub <= 0 or nsub % 2 != 0:
        raise ValueError("Simpson sublayer count must be a positive even integer")
    pressure = np.asarray(pressure_levels_pa, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2 or np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure levels must be one-dimensional and increasing")
    n_layers = pressure.size - 1
    fractions = np.linspace(0.0, 1.0, nsub + 1)
    subpressure = (
        pressure[:-1, np.newaxis] * (1.0 - fractions) + pressure[1:, np.newaxis] * fractions
    )
    spacing = (pressure[1:] - pressure[:-1])[:, np.newaxis] / float(nsub)
    coeff = np.ones(nsub + 1, dtype=float)
    coeff[1:-1:2] = 4.0
    coeff[2:-1:2] = 2.0
    weights = spacing * coeff[np.newaxis, :] / 3.0
    if subpressure.shape != (n_layers, nsub + 1):
        raise AssertionError("unexpected Simpson pressure grid shape")
    return subpressure, weights


def _simpson_layer_pressure_samples(
    pressure_levels_pa: np.ndarray,
    *,
    important_pressure_levels_pa: np.ndarray | None = None,
    nsub: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if important_pressure_levels_pa is None:
        subpressure, weights = _simpson_layer_pressure_grid(pressure_levels_pa, nsub=nsub)
        layer_index = np.repeat(np.arange(subpressure.shape[0]), subpressure.shape[1])
        return subpressure.reshape(-1), weights.reshape(-1), layer_index
    if nsub <= 0 or nsub % 2 != 0:
        raise ValueError("Simpson sublayer count must be a positive even integer")
    pressure = np.asarray(pressure_levels_pa, dtype=float)
    important = np.asarray(important_pressure_levels_pa, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2 or np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure levels must be one-dimensional and increasing")
    if important.ndim != 1:
        raise ValueError("important pressure levels must be one-dimensional")
    n_regions = nsub // 2
    points: list[float] = []
    weights: list[float] = []
    layers: list[int] = []
    for layer, (p_top, p_bottom) in enumerate(zip(pressure[:-1], pressure[1:], strict=True)):
        uniform = p_top + (p_bottom - p_top) * np.arange(1, n_regions + 1) / n_regions
        inside = important[(important > p_top) & (important < p_bottom)]
        endpoints = np.unique(np.concatenate(([p_top], uniform, inside, [p_bottom])))
        endpoints.sort()
        for left, right in zip(endpoints[:-1], endpoints[1:], strict=True):
            delta = right - left
            if delta <= 0.0:
                continue
            midpoint = 0.5 * (left + right)
            points.extend((left, midpoint, right))
            weights.extend((delta / 6.0, 4.0 * delta / 6.0, delta / 6.0))
            layers.extend((layer, layer, layer))
    return (
        np.asarray(points, dtype=float),
        np.asarray(weights, dtype=float),
        np.asarray(layers, dtype=int),
    )


def _column_weighted_absco_cross_section_cm2(
    *,
    absco: AbscoTable,
    wavelength_um: np.ndarray,
    pressure_levels_pa: np.ndarray,
    temperature_levels_k: np.ndarray,
    h2o_vmr_levels: np.ndarray,
    temperature_pressure_levels_pa: np.ndarray | None = None,
    h2o_vmr_pressure_levels_pa: np.ndarray | None = None,
    species_vmr_levels: np.ndarray | None = None,
    species_vmr_pressure_levels_pa: np.ndarray | None = None,
    important_pressure_levels_pa: np.ndarray | None = None,
    nsub: int = 10,
) -> np.ndarray:
    pressure_flat, pressure_weights_flat, layer_index = _simpson_layer_pressure_samples(
        pressure_levels_pa,
        important_pressure_levels_pa=important_pressure_levels_pa,
        nsub=nsub,
    )
    temperature_flat = _profile_at_pressure(
        pressure_levels_pa=(
            pressure_levels_pa
            if temperature_pressure_levels_pa is None
            else temperature_pressure_levels_pa
        ),
        values=temperature_levels_k,
        target_pressure_pa=pressure_flat,
    )
    h2o_flat = _profile_at_pressure(
        pressure_levels_pa=(
            pressure_levels_pa if h2o_vmr_pressure_levels_pa is None else h2o_vmr_pressure_levels_pa
        ),
        values=h2o_vmr_levels,
        target_pressure_pa=pressure_flat,
    )
    xsec_flat = absco.cross_section_cm2(
        wavelength_um=wavelength_um,
        pressure_pa=pressure_flat,
        temperature_k=temperature_flat,
        h2o_vmr=h2o_flat,
    )
    n_layers = np.asarray(pressure_levels_pa).size - 1
    weights = pressure_weights_flat / (1.0 + h2o_flat * M_H2O / M_AIR)
    if species_vmr_levels is not None:
        species_flat = _profile_at_pressure(
            pressure_levels_pa=(
                pressure_levels_pa
                if species_vmr_pressure_levels_pa is None
                else species_vmr_pressure_levels_pa
            ),
            values=species_vmr_levels,
            target_pressure_pa=pressure_flat,
        )
        weights = weights * np.clip(species_flat, 0.0, np.inf)
    denom = np.bincount(layer_index, weights=weights, minlength=n_layers)
    if np.any(denom <= 0.0):
        raise ValueError("non-positive gas integration weights")
    out = np.zeros((np.asarray(wavelength_um).size, n_layers), dtype=float)
    weighted_xsec = xsec_flat * weights[np.newaxis, :]
    for layer in range(n_layers):
        out[:, layer] = np.sum(weighted_xsec[:, layer_index == layer], axis=1) / denom[layer]
    return out


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8", errors="replace").strip()
        if isinstance(value, bytes)
        else str(value).strip()
        for value in np.asarray(values).tolist()
    ]


def _empty_aerosol_inputs(wavelength_um: np.ndarray, n_layers: int) -> AerosolInputs:
    return AerosolInputs(
        extinction_tau=np.zeros((wavelength_um.size, n_layers, 0), dtype=float),
        scattering_tau=np.zeros((wavelength_um.size, n_layers, 0), dtype=float),
        moments=np.zeros((2, 3, 0), dtype=float),
        polarization_moments=np.zeros((2, 3, 0), dtype=float),
        interp_fraction=np.zeros(wavelength_um.shape, dtype=float),
        total_aod_used=0.0,
        phase_model="none",
    )


def _aerosol_type_defaults(aerosol_type: str) -> dict[str, float]:
    return AEROSOL_TYPE_HG_DEFAULTS.get(
        aerosol_type,
        {"ssa": 0.94, "g": 0.70, "angstrom": 1.0},
    )


def _parse_aerosol_type_filter(value: str | None) -> frozenset[str] | None:
    if value is None or not value.strip():
        return None
    parsed = frozenset(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def _default_aerosol_type_filter(kind: str) -> frozenset[str] | None:
    if kind == "tropospheric":
        return OCO_SCALAR_REPLAY_AEROSOL_TYPES
    if kind == "all":
        return None
    raise ValueError(f"unknown aerosol type set: {kind!r}")


def _validate_aerosol_type_filter(
    aerosol_types: list[str],
    aerosol_type_filter: frozenset[str] | None,
) -> None:
    if aerosol_type_filter is None:
        return
    available = set(aerosol_types)
    unknown = sorted(aerosol_type_filter - available)
    if unknown:
        raise ValueError(
            "unknown aerosol type(s) in --diagnostic-aerosol-types: "
            f"{', '.join(unknown)}; available types are: {', '.join(sorted(available))}"
        )


def _hg_phase_moments(g: float, order: int = AEROSOL_HG_MOMENT_ORDER) -> np.ndarray:
    """Return Legendre coefficients for a Henyey-Greenstein phase function."""
    ell = np.arange(int(order) + 1, dtype=float)
    return (2.0 * ell + 1.0) * float(g) ** ell


def _resolve_oco_l2fp_aerosol_file(path: Path | None) -> Path:
    if path is not None:
        return path
    env_path = os.environ.get("RTRF_AEROSOL_FILE")
    if env_path:
        return Path(env_path)
    raise FileNotFoundError(
        "OCO L2FP aerosol treatment requires --oco-l2fp-aerosol-file or RTRF_AEROSOL_FILE"
    )


def _load_oco_l2fp_aerosol_property(
    handle: h5py.File,
    aerosol_type: str,
) -> OcoL2fpAerosolProperty:
    group_name = OCO_L2FP_AEROSOL_PROPERTY_GROUPS.get(aerosol_type)
    if group_name is None:
        raise ValueError(f"OCO L2FP aerosol property mapping is missing {aerosol_type!r}")
    if group_name not in handle:
        raise ValueError(f"OCO L2FP aerosol property file is missing group {group_name!r}")
    group = handle[f"{group_name}/Properties"]
    required = (
        "wave_number",
        "extinction_coefficient",
        "scattering_coefficient",
        "phase_function_moment",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} is missing {missing}")
    wave_number = np.asarray(group["wave_number"][...], dtype=float)
    extinction = np.asarray(group["extinction_coefficient"][...], dtype=float)
    scattering = np.asarray(group["scattering_coefficient"][...], dtype=float)
    moments = np.asarray(group["phase_function_moment"][...], dtype=float)
    if (
        wave_number.ndim != 1
        or extinction.shape != wave_number.shape
        or scattering.shape != wave_number.shape
        or moments.ndim != 3
        or moments.shape[0] != wave_number.size
        or moments.shape[1] < 3
        or moments.shape[2] < 1
    ):
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} has unexpected shapes")
    if not (
        np.all(np.isfinite(wave_number))
        and np.all(np.isfinite(extinction))
        and np.all(np.isfinite(scattering))
        and np.all(np.isfinite(moments[:, :, 0]))
    ):
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} contains non-finite data")
    if np.any(np.diff(wave_number) <= 0.0):
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} wave_number must increase")
    if np.any(extinction <= 0.0) or np.any(scattering < 0.0):
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} has invalid optical coefficients")
    return OcoL2fpAerosolProperty(
        wave_number_cm=wave_number,
        extinction_coefficient=extinction,
        scattering_coefficient=scattering,
        phase_moments=moments[:, :, 0],
        polarization_moments=(
            moments[:, :, 4] if moments.shape[2] > 4 else np.zeros_like(moments[:, :, 0])
        ),
    )


@lru_cache(maxsize=None)
def _load_oco_l2fp_aerosol_property_file(
    property_file: str,
    aerosol_type: str,
) -> OcoL2fpAerosolProperty:
    with h5py.File(property_file, "r") as handle:
        return _load_oco_l2fp_aerosol_property(handle, aerosol_type)


def _interp_oco_l2fp_extinction_scattering(
    property_table: OcoL2fpAerosolProperty,
    wavelength_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wave_number = 1.0e4 / np.asarray(wavelength_um, dtype=float)
    source = property_table.wave_number_cm
    if wave_number.min() < source[0] or wave_number.max() > source[-1]:
        raise ValueError(
            "OCO L2FP aerosol property table does not cover requested wavelength range"
        )
    extinction = np.interp(wave_number, source, property_table.extinction_coefficient)
    scattering = np.interp(wave_number, source, property_table.scattering_coefficient)
    return extinction, scattering


def _interp_oco_l2fp_property(
    property_table: OcoL2fpAerosolProperty,
    wavelength_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave_number = 1.0e4 / np.asarray(wavelength_um, dtype=float)
    source = property_table.wave_number_cm
    if wave_number.min() < source[0] or wave_number.max() > source[-1]:
        raise ValueError(
            "OCO L2FP aerosol property table does not cover requested wavelength range"
        )
    extinction = np.interp(wave_number, source, property_table.extinction_coefficient)
    scattering = np.interp(wave_number, source, property_table.scattering_coefficient)
    moments = np.column_stack(
        [
            np.interp(wave_number, source, property_table.phase_moments[:, moment])
            for moment in range(property_table.phase_moments.shape[1])
        ]
    )
    polarization_moments = np.column_stack(
        [
            np.interp(wave_number, source, property_table.polarization_moments[:, moment])
            for moment in range(property_table.polarization_moments.shape[1])
        ]
    )
    return extinction, scattering, moments, polarization_moments


def _stack_oco_l2fp_endpoint_moments(
    endpoint_moment_list: list[tuple[int, np.ndarray]],
    n_active: int,
) -> np.ndarray:
    if not endpoint_moment_list:
        return np.zeros((2, 3, n_active), dtype=float)
    # RtRetrieval keeps the longest requested moment table and pads shorter
    # aerosol tables with zeros; truncating to the shortest table suppresses
    # forward-scattering aerosols such as sea salt.
    n_moments = max(endpoint.shape[1] for _, endpoint in endpoint_moment_list)
    moments = np.zeros((2, n_moments, n_active), dtype=float)
    for out_index, endpoint_moments in endpoint_moment_list:
        moments[:, : endpoint_moments.shape[1], out_index] = endpoint_moments
    return moments


def _aerosol_profile_tau_from_gaussian(
    *,
    pressure_levels_pa: np.ndarray,
    log_aod: float,
    center_pressure_ratio: float,
    sigma_pressure_ratio: float,
) -> np.ndarray:
    if sigma_pressure_ratio <= 0.0:
        return np.zeros(pressure_levels_pa.size - 1, dtype=float)
    pressure = np.asarray(pressure_levels_pa, dtype=float)
    surface_pressure = float(pressure[-1])
    pressure_ratio = pressure / surface_pressure
    shape = np.exp(-0.5 * ((pressure_ratio - center_pressure_ratio) / sigma_pressure_ratio) ** 2)
    delta_pressure = np.diff(pressure)
    weights = 0.5 * (shape[:-1] + shape[1:]) * delta_pressure
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return np.zeros(pressure.size - 1, dtype=float)
    return float(np.exp(log_aod)) * weights / total_weight


def _posterior_oco_l2fp_aerosol_inputs(
    *,
    std: h5py.File,
    property_file: Path,
    index: int,
    state: dict[str, np.ndarray | float],
    wavelength_um: np.ndarray,
    aerosol_type_filter: frozenset[str] | None,
    aerosol_scale: float,
) -> AerosolInputs:
    wavelength = np.asarray(wavelength_um, dtype=float)
    aerosol_types = _decode_h5_strings(std["Metadata/AllAerosolTypes"][...])
    _validate_aerosol_type_filter(aerosol_types, aerosol_type_filter)
    aerosol_model = _decode_h5_strings(std["AerosolResults/aerosol_model"][index])
    aerosol_param = std["AerosolResults/aerosol_param"][index].astype(float)
    aerosol_retrieved = std["AerosolResults/aerosol_type_retrieved"][index].astype(bool)
    pressure_levels = np.asarray(state["pressure_pa"], dtype=float)
    n_layers = pressure_levels.size - 1

    active = [
        type_index
        for type_index, retrieved in enumerate(aerosol_retrieved)
        if retrieved
        and aerosol_model[type_index].strip()
        and (aerosol_type_filter is None or aerosol_types[type_index] in aerosol_type_filter)
    ]
    if not active:
        return _empty_aerosol_inputs(wavelength, n_layers)

    extinction = np.zeros((wavelength.size, n_layers, len(active)), dtype=float)
    scattering = np.zeros_like(extinction)
    endpoint_moment_list: list[tuple[int, np.ndarray]] = []
    endpoint_polarization_moment_list: list[tuple[int, np.ndarray]] = []
    total_ref_aod = 0.0
    if wavelength.size == 1:
        interp_fraction = np.zeros(wavelength.shape, dtype=float)
        endpoint_wavelengths = np.array([wavelength[0], wavelength[0]], dtype=float)
    else:
        span = wavelength[-1] - wavelength[0]
        interp_fraction = (
            (wavelength - wavelength[0]) / span if span != 0.0 else np.zeros_like(wavelength)
        )
        endpoint_wavelengths = np.array([wavelength[0], wavelength[-1]], dtype=float)

    property_file_key = str(property_file)
    for out_index, type_index in enumerate(active):
        aerosol_type = aerosol_types[type_index]
        model = aerosol_model[type_index].strip()
        if model != "gaussian_log":
            raise ValueError(
                "OCO L2FP aerosol treatment currently supports gaussian_log only; "
                f"{aerosol_type} uses {model!r}"
            )
        log_aod, center_ratio, sigma_ratio = aerosol_param[type_index]
        if not np.all(np.isfinite((log_aod, center_ratio, sigma_ratio))):
            continue
        tau_ref = _aerosol_profile_tau_from_gaussian(
            pressure_levels_pa=pressure_levels,
            log_aod=float(log_aod),
            center_pressure_ratio=float(center_ratio),
            sigma_pressure_ratio=float(sigma_ratio),
        ) * float(aerosol_scale)
        total_ref_aod += float(np.sum(tau_ref))
        property_table = _load_oco_l2fp_aerosol_property_file(property_file_key, aerosol_type)
        qext, qsca = _interp_oco_l2fp_extinction_scattering(property_table, wavelength)
        qext_ref = float(
            np.interp(
                1.0e4 / AEROSOL_REFERENCE_WAVELENGTH_UM,
                property_table.wave_number_cm,
                property_table.extinction_coefficient,
            )
        )
        if qext_ref <= 0.0:
            raise ValueError(f"OCO L2FP aerosol property {aerosol_type} has bad reference Qext")
        extinction[:, :, out_index] = (qext / qext_ref)[:, np.newaxis] * tau_ref[np.newaxis, :]
        scattering[:, :, out_index] = (qsca / qext_ref)[:, np.newaxis] * tau_ref[np.newaxis, :]
        _, _, endpoint_moments, endpoint_polarization_moments = _interp_oco_l2fp_property(
            property_table, endpoint_wavelengths
        )
        endpoint_moment_list.append((out_index, endpoint_moments))
        endpoint_polarization_moment_list.append((out_index, endpoint_polarization_moments))

    moments = _stack_oco_l2fp_endpoint_moments(endpoint_moment_list, len(active))
    polarization_moments = _stack_oco_l2fp_endpoint_moments(
        endpoint_polarization_moment_list,
        len(active),
    )

    return AerosolInputs(
        extinction_tau=extinction,
        scattering_tau=scattering,
        moments=moments,
        polarization_moments=polarization_moments,
        interp_fraction=interp_fraction,
        total_aod_used=total_ref_aod,
        phase_model=f"OCO L2FP aerosol properties from {property_file.name}",
    )


def _posterior_aerosol_inputs(
    *,
    std: h5py.File,
    oco_l2fp_property_file: Path | None,
    index: int,
    state: dict[str, np.ndarray | float],
    wavelength_um: np.ndarray,
    treatment: str,
    aerosol_type_filter: frozenset[str] | None,
    aerosol_scale: float,
) -> AerosolInputs:
    wavelength = np.asarray(wavelength_um, dtype=float)
    n_layers = np.asarray(state["layer_pressure_pa"], dtype=float).size
    if treatment == "none":
        return _empty_aerosol_inputs(wavelength, n_layers)
    if treatment == "oco-l2fp":
        property_file = _resolve_oco_l2fp_aerosol_file(oco_l2fp_property_file)
        if not property_file.exists():
            raise FileNotFoundError(
                "OCO L2FP aerosol property file not found. Pass "
                "--oco-l2fp-aerosol-file or set RTRF_AEROSOL_FILE."
            )
        return _posterior_oco_l2fp_aerosol_inputs(
            std=std,
            property_file=property_file,
            index=index,
            state=state,
            wavelength_um=wavelength,
            aerosol_type_filter=aerosol_type_filter,
            aerosol_scale=aerosol_scale,
        )
    if treatment not in {"l2-posterior-hg", "l2-posterior-gaussian-hg"}:
        raise ValueError(f"unknown aerosol treatment: {treatment!r}")

    aerosol_types = _decode_h5_strings(std["Metadata/AllAerosolTypes"][...])
    _validate_aerosol_type_filter(aerosol_types, aerosol_type_filter)
    aod = std["AerosolResults/aerosol_aod"][index].astype(float)
    if aod.shape[0] != len(aerosol_types) or aod.shape[1] < 4:
        raise ValueError("unexpected L2 aerosol_aod shape")
    aerosol_model = _decode_h5_strings(std["AerosolResults/aerosol_model"][index])
    aerosol_param = std["AerosolResults/aerosol_param"][index].astype(float)
    aerosol_retrieved = std["AerosolResults/aerosol_type_retrieved"][index].astype(bool)

    layer_pressure = np.asarray(state["layer_pressure_pa"], dtype=float)
    layer_air = np.asarray(state["wet_air_col_cm2"], dtype=float)
    tau_ref = np.zeros((n_layers, len(aerosol_types)), dtype=float)
    if treatment == "l2-posterior-hg":
        subcolumn_masks = (
            layer_pressure >= 80000.0,
            (layer_pressure >= 50000.0) & (layer_pressure < 80000.0),
            layer_pressure < 50000.0,
        )
        all_positive = layer_air > 0.0
        for type_index in range(len(aerosol_types)):
            if not aerosol_retrieved[type_index]:
                continue
            if (
                aerosol_type_filter is not None
                and aerosol_types[type_index] not in aerosol_type_filter
            ):
                continue
            subcolumn_sum = 0.0
            for sub_index, mask in enumerate(subcolumn_masks, start=1):
                sub_aod = max(float(aod[type_index, sub_index]), 0.0)
                subcolumn_sum += sub_aod
                if sub_aod == 0.0 or not np.any(mask):
                    continue
                weights = np.where(mask, layer_air, 0.0)
                total_weight = float(np.sum(weights))
                if total_weight <= 0.0:
                    weights = mask.astype(float)
                    total_weight = float(np.sum(weights))
                tau_ref[:, type_index] += sub_aod * weights / total_weight

            total_aod = max(float(aod[type_index, 0]), 0.0)
            if subcolumn_sum == 0.0 and total_aod > 0.0 and np.any(all_positive):
                weights = np.where(all_positive, layer_air, 0.0)
                tau_ref[:, type_index] += total_aod * weights / float(np.sum(weights))
        phase_model = "L2 posterior AOD, pressure-subcolumn profile, HG type defaults"
    else:
        surface_pressure = float(np.asarray(state["pressure_pa"], dtype=float)[-1])
        pressure_ratio = layer_pressure / surface_pressure
        positive_air = layer_air > 0.0
        for type_index, model in enumerate(aerosol_model):
            if not aerosol_retrieved[type_index]:
                continue
            if (
                aerosol_type_filter is not None
                and aerosol_types[type_index] not in aerosol_type_filter
            ):
                continue
            if model.strip() != "gaussian_log":
                raise ValueError(
                    f"aerosol treatment {treatment!r} only supports gaussian_log; "
                    f"{aerosol_types[type_index]} uses {model!r}"
                )
            log_aod, center_ratio, sigma_ratio = aerosol_param[type_index]
            if not np.all(np.isfinite((log_aod, center_ratio, sigma_ratio))):
                continue
            total_aod = float(np.exp(log_aod))
            if total_aod <= 0.0 or sigma_ratio <= 0.0:
                continue
            weights = np.exp(-0.5 * ((pressure_ratio - center_ratio) / sigma_ratio) ** 2)
            weights = np.where(positive_air, weights * layer_air, 0.0)
            total_weight = float(np.sum(weights))
            if total_weight <= 0.0:
                continue
            tau_ref[:, type_index] = total_aod * weights / total_weight
        phase_model = "L2 posterior Gaussian-log aerosol profile, HG type defaults"

    tau_ref *= float(aerosol_scale)

    extinction = np.zeros((wavelength.size, n_layers, len(aerosol_types)), dtype=float)
    scattering = np.zeros_like(extinction)
    moments = np.zeros((2, AEROSOL_HG_MOMENT_ORDER + 1, len(aerosol_types)), dtype=float)
    polarization_moments = np.zeros_like(moments)
    for type_index, aerosol_type in enumerate(aerosol_types):
        defaults = _aerosol_type_defaults(aerosol_type)
        scale = (wavelength / AEROSOL_REFERENCE_WAVELENGTH_UM) ** (-defaults["angstrom"])
        extinction[:, :, type_index] = scale[:, np.newaxis] * tau_ref[np.newaxis, :, type_index]
        scattering[:, :, type_index] = extinction[:, :, type_index] * defaults["ssa"]
        moments[:, :, type_index] = _hg_phase_moments(defaults["g"])

    return AerosolInputs(
        extinction_tau=extinction,
        scattering_tau=scattering,
        moments=moments,
        polarization_moments=polarization_moments,
        interp_fraction=np.zeros(wavelength.shape, dtype=float),
        total_aod_used=float(np.sum(tau_ref)),
        phase_model=phase_model,
    )


def _band_l2_eof_name(band: str) -> str:
    return {"o2": "o2", "wco2": "weak_co2", "sco2": "strong_co2"}[band]


def _band_l2_brdf_name(band: str) -> str:
    return {"o2": "o2", "wco2": "weak_co2", "sco2": "strong_co2"}[band]


def _band_l1b_eof_name(band: str) -> str:
    return {"o2": "o2", "wco2": "weak_co2", "sco2": "strong_co2"}[band]


def _attach_l2_brdf_parameters(case: dict[str, str], std: h5py.File, index: int) -> dict[str, str]:
    """Add OCO surface parameters missing from older selected-case CSVs."""
    out = dict(case)
    out.setdefault("wind_speed", f"{float(std['RetrievalResults/wind_speed'][index]):.12g}")
    out.setdefault(
        "wind_speed_apriori",
        f"{float(std['RetrievalResults/wind_speed_apriori'][index]):.12g}",
    )
    for band in BANDS:
        l2_name = _band_l2_brdf_name(band)
        out.setdefault(
            f"albedo_{band}",
            f"{float(std[f'AlbedoResults/albedo_{l2_name}_fph'][index]):.12g}",
        )
        out.setdefault(
            f"albedo_slope_{band}",
            f"{float(std[f'AlbedoResults/albedo_slope_{l2_name}'][index]):.12g}",
        )
        fields = {
            "weight_slope": f"BRDFResults/brdf_weight_slope_{l2_name}",
            "weight_quadratic": f"BRDFResults/brdf_weight_quadratic_{l2_name}",
            "rahman_factor": f"BRDFResults/brdf_rahman_factor_{l2_name}",
            "hotspot_parameter": f"BRDFResults/brdf_hotspot_parameter_{l2_name}",
            "asymmetry_parameter": f"BRDFResults/brdf_asymmetry_parameter_{l2_name}",
            "anisotropy_parameter": f"BRDFResults/brdf_anisotropy_parameter_{l2_name}",
            "breon_factor": f"BRDFResults/brdf_breon_factor_{l2_name}",
        }
        for name, field in fields.items():
            out.setdefault(f"brdf_{name}_{band}", f"{float(std[field][index]):.12g}")
    return out


def _eof_detector_correction(
    *,
    l1b: h5py.File,
    eof_static: h5py.File | None,
    std: h5py.File,
    index: int,
    frame: int,
    footprint: int,
    band: str,
    sample_indexes: np.ndarray,
    surface_type: str,
    treatment: str,
) -> EofCorrection:
    sample = np.asarray(sample_indexes, dtype=int)
    if treatment == "none":
        return EofCorrection(
            values=np.zeros(sample.shape, dtype=float),
            scale_values=np.zeros(4, dtype=float),
            basis_model="none",
        )
    if treatment == "oco3-static":
        if eof_static is None:
            raise ValueError("--eof-treatment oco3-static requires --oco3-eof-file")
        return _oco3_static_eof_detector_correction(
            l1b=l1b,
            eof_static=eof_static,
            std=std,
            index=index,
            frame=frame,
            footprint=footprint,
            band=band,
            sample_indexes=sample,
            surface_type=surface_type,
        )
    raise ValueError(f"unknown EOF treatment: {treatment!r}")


def _l1b_full_uncertainty_photon(
    *,
    l1b: h5py.File,
    frame: int,
    footprint: int,
    band: str,
) -> np.ndarray:
    band_index = BAND_INDEX[band]
    radiance_field = {
        "o2": "SoundingMeasurements/radiance_o2",
        "wco2": "SoundingMeasurements/radiance_weak_co2",
        "sco2": "SoundingMeasurements/radiance_strong_co2",
    }[band]
    radiance = l1b[radiance_field][frame, footprint, :].astype(float)
    coefs = l1b["InstrumentHeader/snr_coef"][band_index, footprint, :, :].astype(float)
    photon = coefs[:, 0]
    background = coefs[:, 1]
    max_ms = OCO_NOISE_MAX_MS[band_index]
    variance = (100.0 * np.where(radiance > 0.0, radiance, 0.0) / max_ms) * photon * photon
    variance = variance + background * background
    return np.sqrt(variance) * max_ms / 100.0


def _oco3_static_eof_detector_correction(
    *,
    l1b: h5py.File,
    eof_static: h5py.File,
    std: h5py.File,
    index: int,
    frame: int,
    footprint: int,
    band: str,
    sample_indexes: np.ndarray,
    surface_type: str,
) -> EofCorrection:
    band_index = BAND_INDEX[band]
    l2_name = _band_l2_eof_name(band)
    sample = np.asarray(sample_indexes, dtype=int)
    group = "Water" if surface_type == "Coxmunk,Lambertian" else "Land"
    uncertainty = _l1b_full_uncertainty_photon(
        l1b=l1b,
        frame=frame,
        footprint=footprint,
        band=band,
    )
    correction = np.zeros(sample.shape, dtype=float)
    max_order = 4 if band == "sco2" else 3
    scales = np.array(
        [
            float(std[f"RetrievalResults/eof_{order}_scale_{l2_name}"][index])
            for order in range(1, 5)
        ],
        dtype=float,
    )
    for order in range(1, max_order + 1):
        dataset = (
            f"Instrument/EmpiricalOrthogonalFunction/{group}/EOF_{order}_waveform_{band_index + 1}"
        )
        waveform = eof_static[dataset][footprint, :].astype(float)
        eof_full = waveform * uncertainty
        stddev = float(np.sqrt(np.sum((eof_full - np.mean(eof_full)) ** 2) / (eof_full.size - 1)))
        if not np.isfinite(stddev) or stddev <= 0.0:
            raise ValueError(f"OCO3 EOF {dataset} has zero scaled standard deviation")
        eof_full = eof_full * (OCO3_EOF_SCALE_TO_STDDEV / stddev)
        correction = correction + scales[order - 1] * eof_full[sample]
    return EofCorrection(
        values=correction,
        scale_values=scales,
        basis_model=(
            f"OCO3 static {group} EOF waveforms times L1B noise uncertainty, "
            f"orders 1-{max_order}, photon radiance"
        ),
    )


def _fluorescence_photon_radiance(
    *,
    wavelength_um: np.ndarray,
    o2_column_tau: np.ndarray,
    view_zenith_deg: float,
    stokes_coefficients: np.ndarray,
    fluorescence_at_reference: float,
    fluorescence_slope: float,
) -> np.ndarray:
    f_ref = float(fluorescence_at_reference)
    slope = float(fluorescence_slope)
    if not np.isfinite(f_ref) or f_ref <= 0.0:
        return np.zeros(np.asarray(wavelength_um).shape, dtype=float)
    wavelength = np.asarray(wavelength_um, dtype=float)
    tau = np.asarray(o2_column_tau, dtype=float)
    if wavelength.shape != tau.shape:
        raise ValueError("fluorescence wavelength and O2 optical depth arrays must match")
    mu_view = math.cos(math.radians(float(view_zenith_deg)))
    if mu_view <= 0.0:
        raise ValueError(f"invalid view zenith for fluorescence: {view_zenith_deg}")
    wavenumber = 1.0e4 / wavelength
    reference_wn = 1.0e4 / OCO_FLUORESCENCE_REFERENCE_WAVELENGTH_UM
    surface_radiance = f_ref * (1.0 + slope * (wavenumber - reference_wn))
    return float(stokes_coefficients[0]) * surface_radiance * np.exp(-tau / mu_view)


def _sample_detector_colors(section: slice, max_colors_per_band: int) -> np.ndarray:
    colors = np.arange(section.start, section.stop, dtype=int)
    if max_colors_per_band <= 0 or colors.size <= max_colors_per_band:
        return colors
    positions = np.linspace(0, colors.size - 1, max_colors_per_band)
    return colors[np.round(positions).astype(int)]


def _detector_average(
    values: np.ndarray,
    *,
    detector_id: np.ndarray,
    response_flat: np.ndarray,
    n_detector: int,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    detector = np.asarray(detector_id, dtype=int)
    weights = np.asarray(response_flat, dtype=float)
    if values_arr.shape != detector.shape or values_arr.shape != weights.shape:
        raise ValueError("values, detector_id, and response_flat must have the same shape")
    if np.any(detector < 0) or np.any(detector >= int(n_detector)):
        raise ValueError("detector_id contains values outside n_detector")

    weight_sum = np.bincount(detector, weights=weights, minlength=int(n_detector))
    weighted_sum = np.bincount(detector, weights=values_arr * weights, minlength=int(n_detector))
    detector_values = np.full(int(n_detector), np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=detector_values, where=weight_sum > 0.0)
    return detector_values


def _land_surface_spectrum(
    *,
    case: dict[str, str],
    band: str,
    wavelength_um: np.ndarray,
    surface_spectrum: str,
    base_field: str,
) -> np.ndarray:
    base = float(case[f"{base_field}_{band}"])
    if not np.isfinite(base) or base <= 0.0:
        surface_type = case.get("surface_type", "unknown")
        raise ValueError(
            "selected case does not provide a positive Lambertian/land BRDF "
            f"{base_field} for {band}; surface_type={surface_type!r}. "
            "The replay driver should not be used for Coxmunk ocean cases "
            "until an ocean-surface BRDF path is implemented."
        )
    wavelength = np.asarray(wavelength_um, dtype=float)
    if surface_spectrum == "constant":
        reflectance = np.full(wavelength.shape, base, dtype=float)
    elif surface_spectrum in {"l2-linear", "l2-polynomial"}:
        slope = float(case[f"{base_field}_slope_{band}"])
        if not np.isfinite(slope):
            raise ValueError(f"L2 BRDF spectral coefficients are not finite for {band}")
        reference_wn = 1.0e4 / BAND_REFERENCE_WAVELENGTH_UM[band]
        wavenumber = 1.0e4 / wavelength
        delta_wn = wavenumber - reference_wn
        reflectance = base + slope * delta_wn
        if surface_spectrum == "l2-polynomial":
            quadratic = float(case[f"{base_field}_quadratic_{band}"])
            if not np.isfinite(quadratic):
                raise ValueError(f"L2 BRDF spectral coefficients are not finite for {band}")
            reflectance = reflectance + quadratic * delta_wn * delta_wn
    else:
        raise ValueError(f"unknown surface_spectrum={surface_spectrum!r}")
    return reflectance


def _is_ocean_surface(case: dict[str, str]) -> bool:
    return case.get("surface_type", "") == OCEAN_SURFACE_TYPE


def _ocean_lambertian_spectrum(
    *,
    case: dict[str, str],
    band: str,
    wavelength_um: np.ndarray,
    surface_spectrum: str,
) -> np.ndarray:
    base = float(case[f"albedo_{band}"])
    if not np.isfinite(base) or base < 0.0:
        raise ValueError(f"selected ocean case does not provide a valid albedo for {band}")
    wavelength = np.asarray(wavelength_um, dtype=float)
    if surface_spectrum == "constant":
        albedo = np.full(wavelength.shape, base, dtype=float)
    elif surface_spectrum in {"l2-linear", "l2-polynomial"}:
        slope = float(case[f"albedo_slope_{band}"])
        if not np.isfinite(slope):
            raise ValueError(f"L2 ocean albedo slope is not finite for {band}")
        reference_wn = 1.0e4 / BAND_REFERENCE_WAVELENGTH_UM[band]
        albedo = base + slope * (1.0e4 / wavelength - reference_wn)
    else:
        raise ValueError(f"unknown surface_spectrum={surface_spectrum!r}")
    if not np.all(np.isfinite(albedo)) or np.any(albedo < 0.0):
        raise ValueError(
            f"computed negative ocean Lambertian albedo for {band}; "
            f"min={np.nanmin(albedo):.6g}, max={np.nanmax(albedo):.6g}"
        )
    return albedo


def _land_surface_reflectance(
    *,
    case: dict[str, str],
    band: str,
    wavelength_um: np.ndarray,
    surface_spectrum: str,
    surface_angular: str,
    angles: np.ndarray,
) -> np.ndarray:
    if surface_angular == "l2-reflectance":
        reflectance = _land_surface_spectrum(
            case=case,
            band=band,
            wavelength_um=wavelength_um,
            surface_spectrum=surface_spectrum,
            base_field="brdf_reflectance",
        )
    elif surface_angular in {"rpv-weight", "rpv-brdf"}:
        reflectance = _land_surface_spectrum(
            case=case,
            band=band,
            wavelength_um=wavelength_um,
            surface_spectrum=surface_spectrum,
            base_field="brdf_weight",
        )
    else:
        raise ValueError(f"unknown surface_angular={surface_angular!r}")

    if surface_angular in {"rpv-weight", "rpv-brdf"}:
        reflectance = reflectance * _oco_rpv_kernel(case=case, band=band, angles=angles)

    if not np.all(np.isfinite(reflectance)) or np.any(reflectance <= 0.0):
        raise ValueError(
            f"computed non-positive land surface reflectance for {band}; "
            f"min={np.nanmin(reflectance):.6g}, max={np.nanmax(reflectance):.6g}"
        )
    return reflectance


def _oco_rpv_kernel(*, case: dict[str, str], band: str, angles: np.ndarray) -> float:
    """Return the normalized OCO land RPV angular kernel for one geometry."""
    sza, vza, relaz = (float(value) for value in np.asarray(angles, dtype=float))
    hotspot = float(case[f"brdf_hotspot_parameter_{band}"])
    asymmetry = float(case[f"brdf_asymmetry_parameter_{band}"])
    anisotropy = float(case[f"brdf_anisotropy_parameter_{band}"])
    if not all(np.isfinite([hotspot, asymmetry, anisotropy])):
        raise ValueError(f"L2 RPV parameters are not finite for {band}")

    mu_i = math.cos(math.radians(sza))
    mu_r = math.cos(math.radians(vza))
    if mu_i <= 0.0 or mu_r <= 0.0:
        raise ValueError(f"invalid RPV geometry for {band}: sza={sza}, vza={vza}")
    sin_i = math.sqrt(max(0.0, 1.0 - mu_i * mu_i))
    sin_r = math.sqrt(max(0.0, 1.0 - mu_r * mu_r))
    cos_phi = math.cos(math.radians(relaz))
    cos_g = max(-1.0, min(1.0, mu_i * mu_r + sin_i * sin_r * cos_phi))
    minnaert = (mu_i * mu_r) ** (anisotropy - 1.0) / (mu_i + mu_r) ** (1.0 - anisotropy)
    phase = (1.0 - asymmetry * asymmetry) / (
        1.0 + asymmetry * asymmetry + 2.0 * asymmetry * cos_g
    ) ** 1.5
    tan_i = math.tan(math.radians(sza))
    tan_r = math.tan(math.radians(vza))
    geom = math.sqrt(max(0.0, tan_i * tan_i + tan_r * tan_r - 2.0 * tan_i * tan_r * cos_phi))
    hotspot_term = 1.0 + (1.0 - hotspot) / (1.0 + geom)
    return minnaert * phase * hotspot_term / RPV_KERNEL_NORMALIZATION


def _oco_rpv_brdf(
    *,
    case: dict[str, str],
    band: str,
    wavelength_um: np.ndarray,
    surface_spectrum: str,
    angles: np.ndarray,
    stream_value: float,
    brdf_quadrature_streams: int,
    fo_direct_brf_factor: float = 1.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build wavelength-dependent OCO RPV BRDF coefficients for py2sess."""
    from py2sess.optical.brdf_solar_obs import RPV_IDX, solar_obs_brdf_from_kernels

    weights = _land_surface_spectrum(
        case=case,
        band=band,
        wavelength_um=wavelength_um,
        surface_spectrum=surface_spectrum,
        base_field="brdf_weight",
    )
    rahman_factor = float(case.get(f"brdf_rahman_factor_{band}", 1.0))
    breon_factor = float(case.get(f"brdf_breon_factor_{band}", 0.0))
    if not np.isfinite(rahman_factor):
        raise ValueError(f"L2 RPV Rahman factor is not finite for {band}")
    if np.isfinite(breon_factor) and abs(breon_factor) > 1.0e-8:
        raise NotImplementedError(
            "py2sess OCO replay currently implements the RPV/Rahman land-BRDF "
            f"component, but this sounding has non-negligible Breon factor for {band}: "
            f"{breon_factor:.6g}"
        )
    weights = weights * rahman_factor
    coeffs = solar_obs_brdf_from_kernels(
        kernel_specs=[
            {
                "which_brdf": RPV_IDX,
                "factor": 1.0,
                "hotspot": float(case[f"brdf_hotspot_parameter_{band}"]),
                "asymmetry": float(case[f"brdf_asymmetry_parameter_{band}"]),
                "anisotropy": float(case[f"brdf_anisotropy_parameter_{band}"]),
                "normalization": RPV_KERNEL_NORMALIZATION,
                "nstreams_brdf": brdf_quadrature_streams,
            }
        ],
        user_obsgeoms=np.asarray(angles, dtype=float).reshape(1, 3),
        stream_value=stream_value,
        n_geoms=1,
    )
    direct_brf = (
        float(fo_direct_brf_factor) * weights[:, np.newaxis] * coeffs.direct_brf[np.newaxis, :]
    )
    brdf = {
        "brdf_f_0": weights[:, np.newaxis, np.newaxis] * coeffs.brdf_f_0[np.newaxis, :, :],
        "brdf_f": weights[:, np.newaxis] * coeffs.brdf_f[np.newaxis, :],
        "ubrdf_f": weights[:, np.newaxis, np.newaxis] * coeffs.ubrdf_f[np.newaxis, :, :],
        "direct_brf": direct_brf,
    }
    return brdf, direct_brf[:, 0]


def _oco_coxmunk_lambertian_brdf(
    *,
    case: dict[str, str],
    band: str,
    wavelength_um: np.ndarray,
    surface_spectrum: str,
    angles: np.ndarray,
    stream_value: float,
    brdf_quadrature_streams: int,
    stokes_projection: StokesProjection,
    coxmunk_stokes_scope: str,
    fo_direct_brf_factor: float = 1.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    from py2sess.optical.brdf_solar_obs import (
        COXMUNK_IDX,
        LAMBERTIAN_IDX,
        coxmunk_giss_stokes_direct_kernel,
        solar_obs_brdf_from_kernels,
    )

    wind_speed = float(case["wind_speed"])
    if not np.isfinite(wind_speed) or wind_speed < 0.0:
        raise ValueError(f"invalid Cox-Munk wind speed for {band}: {wind_speed}")
    refractive_index = OCO_COXMUNK_REFRACTIVE_INDEX[band]
    lambertian = _ocean_lambertian_spectrum(
        case=case,
        band=band,
        wavelength_um=wavelength_um,
        surface_spectrum=surface_spectrum,
    )
    common = {
        "user_obsgeoms": np.asarray(angles, dtype=float).reshape(1, 3),
        "stream_value": stream_value,
        "n_geoms": 1,
    }
    coxmunk = solar_obs_brdf_from_kernels(
        kernel_specs=[
            {
                "which_brdf": COXMUNK_IDX,
                "factor": 1.0,
                "wind_speed": wind_speed,
                "refractive_index": refractive_index,
                "nstreams_brdf": brdf_quadrature_streams,
            }
        ],
        **common,
    )
    lambert = solar_obs_brdf_from_kernels(
        kernel_specs=[
            {
                "which_brdf": LAMBERTIAN_IDX,
                "factor": 1.0,
                "nstreams_brdf": brdf_quadrature_streams,
            }
        ],
        **common,
    )
    coxmunk_stokes = coxmunk_giss_stokes_direct_kernel(
        sza_deg=float(angles[0]),
        vza_deg=float(angles[1]),
        relative_azimuth_deg=float(angles[2]),
        wind_speed=wind_speed,
        refractive_index=refractive_index,
    )
    scalar_direct_brf = float(coxmunk.direct_brf[0])
    scalar_factor = float(stokes_projection.scalar_factor)
    if abs(scalar_factor) <= 1.0e-12:
        raise ValueError("Stokes scalar factor is too small for Cox-Munk projection")
    analyzer_q = float(stokes_projection.analyzer_q) / scalar_factor
    analyzer_u = float(stokes_projection.analyzer_u) / scalar_factor
    projected_direct_brf = float(
        coxmunk_stokes[0] + analyzer_q * coxmunk_stokes[1] + analyzer_u * coxmunk_stokes[2]
    )
    coxmunk_scale = 1.0
    if scalar_direct_brf > 1.0e-14 and projected_direct_brf > 0.0:
        coxmunk_scale = projected_direct_brf / scalar_direct_brf
    if coxmunk_stokes_scope == "all":
        coxmunk_direct_scale = coxmunk_scale
        coxmunk_fourier_scale = coxmunk_scale
    elif coxmunk_stokes_scope == "direct":
        coxmunk_direct_scale = coxmunk_scale
        coxmunk_fourier_scale = 1.0
    elif coxmunk_stokes_scope == "none":
        coxmunk_direct_scale = 1.0
        coxmunk_fourier_scale = 1.0
    else:
        raise ValueError(f"unknown Cox-Munk Stokes scope: {coxmunk_stokes_scope!r}")

    direct_brf = coxmunk_direct_scale * coxmunk.direct_brf[np.newaxis, :] + (
        lambertian[:, np.newaxis] * lambert.direct_brf[np.newaxis, :]
    )
    brdf = {
        "brdf_f_0": coxmunk_fourier_scale * coxmunk.brdf_f_0[np.newaxis, :, :]
        + lambertian[:, np.newaxis, np.newaxis] * lambert.brdf_f_0[np.newaxis, :, :],
        "brdf_f": coxmunk_fourier_scale * coxmunk.brdf_f[np.newaxis, :]
        + lambertian[:, np.newaxis] * lambert.brdf_f[np.newaxis, :],
        "ubrdf_f": coxmunk_fourier_scale * coxmunk.ubrdf_f[np.newaxis, :, :]
        + lambertian[:, np.newaxis, np.newaxis] * lambert.ubrdf_f[np.newaxis, :, :],
        "direct_brf": float(fo_direct_brf_factor) * direct_brf,
    }
    metadata = {
        "wind_speed": wind_speed,
        "refractive_index": refractive_index,
        "lambertian_albedo_reference": float(case[f"albedo_{band}"]),
        "coxmunk_direct_brf": scalar_direct_brf,
        "coxmunk_stokes_i": float(coxmunk_stokes[0]),
        "coxmunk_stokes_scale": float(coxmunk_scale),
        "coxmunk_direct_scale": float(coxmunk_direct_scale),
        "coxmunk_fourier_scale": float(coxmunk_fourier_scale),
    }
    return brdf, direct_brf[:, 0], metadata


def _ils_integration_weights(delta_lambda: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Return response weights including the nonuniform ILS wavelength grid."""
    delta = np.asarray(delta_lambda, dtype=float)
    resp = np.asarray(response, dtype=float)
    if delta.shape != resp.shape or delta.ndim != 2:
        raise ValueError("delta_lambda and response must have shape (n_detector, n_ils)")
    spacing = np.empty_like(delta, dtype=float)
    spacing[:, 0] = np.abs(delta[:, 1] - delta[:, 0])
    spacing[:, -1] = np.abs(delta[:, -1] - delta[:, -2])
    spacing[:, 1:-1] = 0.5 * np.abs(delta[:, 2:] - delta[:, :-2])
    return resp * spacing


def _sample_spacing(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("sample spacing requires at least two one-dimensional points")
    spacing = np.empty_like(arr, dtype=float)
    spacing[0] = abs(arr[1] - arr[0])
    spacing[-1] = abs(arr[-1] - arr[-2])
    spacing[1:-1] = 0.5 * np.abs(arr[2:] - arr[:-2])
    return spacing


def _build_ils_eval_grid(
    *,
    center_wavelength_um: np.ndarray,
    delta_lambda_um: np.ndarray,
    response: np.ndarray,
    grid_spacing_cm_inv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(center_wavelength_um, dtype=float)
    delta = np.asarray(delta_lambda_um, dtype=float)
    resp = np.asarray(response, dtype=float)
    if delta.shape != resp.shape or delta.ndim != 2 or delta.shape[0] != center.size:
        raise ValueError("ILS arrays have inconsistent detector dimensions")
    if grid_spacing_cm_inv <= 0.0:
        weights = _ils_integration_weights(delta, resp)
        detector = np.repeat(np.arange(center.size), resp.shape[1])
        return (center[:, np.newaxis] + delta).reshape(-1), weights.reshape(-1), detector

    eval_wavelength: list[np.ndarray] = []
    eval_weight: list[np.ndarray] = []
    detector_id: list[np.ndarray] = []
    spacing = float(grid_spacing_cm_inv)
    for det, center_wl in enumerate(center):
        wl_table = center_wl + delta[det]
        resp_table = resp[det]
        valid = np.isfinite(wl_table) & np.isfinite(resp_table) & (resp_table > 0.0)
        if np.count_nonzero(valid) < 2:
            continue
        wl_table = wl_table[valid]
        resp_table = resp_table[valid]
        order = np.argsort(wl_table)
        wl_table = wl_table[order]
        resp_table = resp_table[order]
        wn_min = 1.0e4 / wl_table[-1]
        wn_max = 1.0e4 / wl_table[0]
        wn_start = math.ceil(wn_min / spacing) * spacing
        wn_stop = math.floor(wn_max / spacing) * spacing
        if wn_stop < wn_start:
            wn_grid = np.array([(wn_min + wn_max) * 0.5], dtype=float)
        else:
            n_grid = int(round((wn_stop - wn_start) / spacing)) + 1
            wn_grid = wn_start + spacing * np.arange(n_grid, dtype=float)
        wl_grid = 1.0e4 / wn_grid
        resp_grid = np.interp(wl_grid, wl_table, resp_table, left=0.0, right=0.0)
        weights = resp_grid * _sample_spacing(wl_grid)
        keep = np.isfinite(weights) & (weights > 0.0)
        eval_wavelength.append(wl_grid[keep])
        eval_weight.append(weights[keep])
        detector_id.append(np.full(np.count_nonzero(keep), det, dtype=int))
    if not eval_wavelength:
        raise ValueError("no valid ILS evaluation points")
    return (
        np.concatenate(eval_wavelength),
        np.concatenate(eval_weight),
        np.concatenate(detector_id),
    )


def _rayleigh_tau_cm2(
    wavelength_um: np.ndarray, air_col_cm2: np.ndarray, co2_ppm: float
) -> tuple[np.ndarray, np.ndarray]:
    from py2sess.optical.rayleigh import rayleigh_bodhaine

    rayleigh = rayleigh_bodhaine(wavelength_um * 1000.0, co2_ppmv=co2_ppm)
    return rayleigh.cross_section[:, np.newaxis] * air_col_cm2[
        np.newaxis, :
    ], rayleigh.depolarization


def _solar_obs_scattering_cosine(angles: np.ndarray) -> float:
    geometry = np.asarray(angles, dtype=float).reshape(3)
    sza = math.radians(float(geometry[0]))
    vza = math.radians(float(geometry[1]))
    raz = math.radians(float(geometry[2]))
    mu0 = math.cos(sza)
    mu = math.cos(vza)
    if math.isclose(float(geometry[0]), 0.0):
        return -mu if not math.isclose(mu, 0.0) else 0.0
    return -(mu * mu0) + math.sin(vza) * math.sin(sza) * math.cos(raz)


def _stokes_rotation_from_scattering_plane(angles: np.ndarray) -> tuple[float, float, float]:
    """Return cos(2 chi), sin(2 chi), and scattering cosine for LRad-style FO."""
    geometry = np.asarray(angles, dtype=float).reshape(3)
    sza = math.radians(float(geometry[0]))
    vza = math.radians(float(geometry[1]))
    phi = math.radians(float(geometry[2]) % 360.0)
    mu0 = math.cos(sza)
    mu = math.cos(vza)
    costhm = _solar_obs_scattering_cosine(geometry)
    pure_nadir = math.isclose(float(geometry[1]), 0.0, abs_tol=1.0e-10)
    if (
        pure_nadir
        or math.isclose(phi, 0.0, abs_tol=1.0e-10)
        or math.isclose(phi, math.pi, abs_tol=1.0e-10)
        or math.isclose(phi, 2.0 * math.pi, abs_tol=1.0e-10)
    ):
        return 1.0, 0.0, costhm

    cosphi = math.cos(phi)
    denom = math.sqrt(max(1.0 - costhm * costhm, 0.0))
    if denom <= 1.0e-12:
        return 1.0, 0.0, costhm
    rmu0 = math.sqrt(max(1.0 - mu0 * mu0, 0.0)) * mu
    rmu = math.sqrt(max(1.0 - mu * mu, 0.0)) * mu0
    cosi2m = (-rmu - rmu0 * cosphi) / denom
    cosi2m = min(1.0, max(-1.0, cosi2m))
    sin2 = max(1.0 - cosi2m * cosi2m, 0.0)
    sign = 2.0 if phi >= math.pi else -2.0
    return 1.0 - 2.0 * sin2, sign * math.sqrt(sin2) * cosi2m, costhm


def _spin2_spherical_function(cos_scatter: float, n_moments: int) -> np.ndarray:
    """LRad gsfmi(:,2) basis for scattering-matrix polarization moments."""
    if n_moments <= 0:
        return np.zeros(0, dtype=float)
    basis = np.zeros(int(n_moments), dtype=float)
    if n_moments <= 2:
        return basis
    u = min(1.0, max(-1.0, float(cos_scatter)))
    qroot6 = 0.25 * math.sqrt(6.0)
    basis[2] = -qroot6 * (1.0 - u * u)
    if n_moments <= 3:
        return basis
    sql4 = np.zeros(n_moments, dtype=float)
    for order in range(3, n_moments):
        sql4[order] = math.sqrt(float(order * order - 4))
    for order in range(2, n_moments - 1):
        basis[order + 1] = (
            float(2 * order + 1) * u * basis[order] - sql4[order] * basis[order - 1]
        ) / sql4[order + 1]
    return basis


def _direction_vector(mu: float, azimuth_deg: float) -> np.ndarray:
    rho = math.sqrt(max(1.0 - float(mu) * float(mu), 0.0))
    azimuth = math.radians(float(azimuth_deg))
    return np.array([rho * math.cos(azimuth), rho * math.sin(azimuth), float(mu)])


def _projected_scattering_geometry(
    *,
    incoming_mu: float,
    incoming_azimuth_deg: float,
    outgoing_mu: float,
    outgoing_azimuth_deg: float,
    stokes_projection: StokesProjection,
) -> tuple[float, float]:
    k_in = _direction_vector(incoming_mu, incoming_azimuth_deg)
    k_out = _direction_vector(outgoing_mu, outgoing_azimuth_deg)
    cos_scatter = float(np.dot(k_in, k_out))
    cos_scatter = min(1.0, max(-1.0, cos_scatter))

    z_axis = np.array([0.0, 0.0, 1.0])
    scattering_normal = np.cross(k_in, k_out)
    scattering_norm = float(np.linalg.norm(scattering_normal))
    reference_normal = np.cross(z_axis, k_out)
    reference_norm = float(np.linalg.norm(reference_normal))
    if scattering_norm <= 1.0e-12 or reference_norm <= 1.0e-12:
        c2i2m = 1.0
        s2i2m = 0.0
    else:
        scattering_normal /= scattering_norm
        reference_normal /= reference_norm
        cos_chi = float(np.dot(scattering_normal, reference_normal))
        sin_chi = -float(np.dot(k_out, np.cross(reference_normal, scattering_normal)))
        c2i2m = cos_chi * cos_chi - sin_chi * sin_chi
        s2i2m = 2.0 * sin_chi * cos_chi

    analyzer_projection = (
        stokes_projection.analyzer_q * c2i2m + stokes_projection.analyzer_u * s2i2m
    )
    return analyzer_projection, cos_scatter


def _projected_rayleigh_p12(
    *,
    incoming_mu: float,
    incoming_azimuth_deg: float,
    outgoing_mu: float,
    outgoing_azimuth_deg: float,
    depol: np.ndarray,
    stokes_projection: StokesProjection,
    sign: int,
) -> np.ndarray:
    """Return instrument-projected Rayleigh P12 for one incoming/outgoing pair."""
    analyzer_projection, cos_scatter = _projected_scattering_geometry(
        incoming_mu=incoming_mu,
        incoming_azimuth_deg=incoming_azimuth_deg,
        outgoing_mu=outgoing_mu,
        outgoing_azimuth_deg=outgoing_azimuth_deg,
        stokes_projection=stokes_projection,
    )
    delta = 2.0 * (1.0 - np.asarray(depol, dtype=float)) / (2.0 + np.asarray(depol, dtype=float))
    rayleigh_p12 = -0.75 * delta * (1.0 - cos_scatter * cos_scatter)
    return float(sign) * analyzer_projection * rayleigh_p12


def _rayleigh_p12_from_cosine(cos_scatter: float, depol: np.ndarray) -> np.ndarray:
    delta = 2.0 * (1.0 - np.asarray(depol, dtype=float)) / (2.0 + np.asarray(depol, dtype=float))
    return -0.75 * delta * (1.0 - float(cos_scatter) * float(cos_scatter))


def _scattering_plane_cos2(
    *,
    first_incoming_mu: float,
    first_incoming_azimuth_deg: float,
    shared_mu: float,
    shared_azimuth_deg: float,
    second_outgoing_mu: float,
    second_outgoing_azimuth_deg: float,
) -> float:
    first_in = _direction_vector(first_incoming_mu, first_incoming_azimuth_deg)
    shared = _direction_vector(shared_mu, shared_azimuth_deg)
    second_out = _direction_vector(second_outgoing_mu, second_outgoing_azimuth_deg)
    first_normal = np.cross(first_in, shared)
    second_normal = np.cross(shared, second_out)
    first_norm = np.linalg.norm(first_normal)
    second_norm = np.linalg.norm(second_normal)
    if first_norm <= 1.0e-14 or second_norm <= 1.0e-14:
        return 1.0
    first_normal /= first_norm
    second_normal /= second_norm
    cos_chi = float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))
    return 2.0 * cos_chi * cos_chi - 1.0


def _projected_aerosol_p12(
    *,
    incoming_mu: float,
    incoming_azimuth_deg: float,
    outgoing_mu: float,
    outgoing_azimuth_deg: float,
    aerosol_polarization_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    stokes_projection: StokesProjection,
    sign: int,
) -> np.ndarray:
    moments = np.asarray(aerosol_polarization_moments, dtype=float)
    fac = np.asarray(aerosol_interp_fraction, dtype=float)
    if moments.ndim != 3 or moments.shape[0] != 2:
        raise ValueError("aerosol polarization moments must have shape (2, nmom, naerosol)")
    if moments.shape[2] == 0:
        return np.zeros((fac.size, 0), dtype=float)
    analyzer_projection, cos_scatter = _projected_scattering_geometry(
        incoming_mu=incoming_mu,
        incoming_azimuth_deg=incoming_azimuth_deg,
        outgoing_mu=outgoing_mu,
        outgoing_azimuth_deg=outgoing_azimuth_deg,
        stokes_projection=stokes_projection,
    )
    basis = _spin2_spherical_function(cos_scatter, moments.shape[1])
    endpoint_p12 = np.matmul(np.moveaxis(moments, 1, 2), basis)
    p12 = endpoint_p12[0] + fac[:, np.newaxis] * (endpoint_p12[1] - endpoint_p12[0])
    return float(sign) * analyzer_projection * p12


def _l1b_stokes_coefficients(
    l1b: h5py.File, *, frame: int, footprint: int, band_index: int
) -> np.ndarray:
    if "FootprintGeometry/footprint_stokes_coefficients" in l1b:
        return l1b["FootprintGeometry/footprint_stokes_coefficients"][
            frame, footprint, band_index, :
        ].astype(float)
    if "FootprintGeometry/footprint_polarization_angle" not in l1b:
        return np.array([0.5, -0.5, 0.0, 0.0], dtype=float)
    angle = math.radians(
        float(l1b["FootprintGeometry/footprint_polarization_angle"][frame, footprint, band_index])
    )
    return np.array([0.5, 0.5 * math.cos(2.0 * angle), 0.5 * math.sin(2.0 * angle), 0.0])


def _stokes_projection(stokes_coefficients: np.ndarray, mode: str) -> StokesProjection:
    """Return the scalar and Q/U analyzer factors for one Stokes convention.

    OCO L1B provides raw instrument Stokes response coefficients. Published L1B
    radiances are normally used as unpolarized-radiance equivalents, so the
    polarization response should be normalized by m11:

        L = I + (m12/m11) Q + (m13/m11) U.

    The raw-detector option applies the literal detector projection:

        L = m11 I + m12 Q + m13 U.
    """
    stokes = np.asarray(stokes_coefficients, dtype=float)
    if stokes.shape[0] < 3:
        raise ValueError("stokes_coefficients must contain at least I, Q, and U terms")
    if not np.all(np.isfinite(stokes[:3])):
        raise ValueError("stokes I/Q/U coefficients must be finite")
    if mode == "l1b-normalized":
        if abs(stokes[0]) <= np.finfo(float).eps:
            raise ValueError("stokes m11 coefficient is zero; cannot normalize Q/U response")
        return StokesProjection(
            scalar_factor=1.0,
            analyzer_q=float(stokes[1] / stokes[0]),
            analyzer_u=float(stokes[2] / stokes[0]),
            description="L1B-normalized: I + (m12/m11) Q + (m13/m11) U",
        )
    if mode == "raw-detector":
        return StokesProjection(
            scalar_factor=float(stokes[0]),
            analyzer_q=float(stokes[1]),
            analyzer_u=float(stokes[2]),
            description="raw detector: m11 I + m12 Q + m13 U",
        )
    raise ValueError(f"unknown Stokes projection mode: {mode!r}")


def _rayleigh_projected_polarization_scatter_term(
    *,
    ssa: np.ndarray,
    rayleigh_scattering_tau: np.ndarray,
    scattering_tau: np.ndarray,
    depol: np.ndarray,
    delta_m_truncation_factor: np.ndarray,
    angles: np.ndarray,
    stokes_projection: StokesProjection,
    sign: int,
) -> np.ndarray:
    """Build the FO source for instrument-projected Rayleigh Q/U scattering."""
    ssa_arr, ray_arr, scattering_arr, factor_arr = np.broadcast_arrays(
        np.asarray(ssa, dtype=float),
        np.asarray(rayleigh_scattering_tau, dtype=float),
        np.asarray(scattering_tau, dtype=float),
        np.asarray(delta_m_truncation_factor, dtype=float),
    )
    if ssa_arr.ndim == 0:
        raise ValueError("ssa must include a layer axis")
    lead_shape = ssa_arr.shape[:-1]
    depol_arr = np.broadcast_to(np.asarray(depol, dtype=float), lead_shape)

    inv_scattering = np.zeros_like(scattering_arr, dtype=float)
    np.divide(1.0, scattering_arr, out=inv_scattering, where=scattering_arr > 0.0)
    denominator = 1.0 - factor_arr * ssa_arr
    c2i2m, s2i2m, cos_scatter = _stokes_rotation_from_scattering_plane(angles)
    delta = 2.0 * (1.0 - depol_arr) / (2.0 + depol_arr)
    rayleigh_p12 = -0.75 * delta * (1.0 - cos_scatter * cos_scatter)
    analyzer_projection = (
        stokes_projection.analyzer_q * c2i2m + stokes_projection.analyzer_u * s2i2m
    )
    source = (
        float(sign)
        * analyzer_projection
        * rayleigh_p12[..., np.newaxis]
        * ray_arr
        * inv_scattering
        * ssa_arr
        / denominator
    )
    source = np.where(np.isfinite(source), source, 0.0)
    return np.ascontiguousarray(source, dtype=float)


def _aerosol_projected_polarization_scatter_term(
    *,
    ssa: np.ndarray,
    aerosol_scattering_tau: np.ndarray,
    scattering_tau: np.ndarray,
    aerosol_polarization_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    delta_m_truncation_factor: np.ndarray,
    angles: np.ndarray,
    stokes_projection: StokesProjection,
    sign: int,
) -> np.ndarray:
    """Build a direct-FO aerosol M12 source projected through OCO Stokes response."""
    aerosol_tau = np.asarray(aerosol_scattering_tau, dtype=float)
    moments = np.asarray(aerosol_polarization_moments, dtype=float)
    if aerosol_tau.shape[-1] == 0:
        return np.zeros(np.asarray(scattering_tau, dtype=float).shape, dtype=float)
    if moments.ndim != 3 or moments.shape[0] != 2 or moments.shape[2] != aerosol_tau.shape[-1]:
        raise ValueError("aerosol polarization moments must have shape (2, nmom, naerosol)")

    ssa_arr, scattering_arr, factor_arr = np.broadcast_arrays(
        np.asarray(ssa, dtype=float),
        np.asarray(scattering_tau, dtype=float),
        np.asarray(delta_m_truncation_factor, dtype=float),
    )
    if aerosol_tau.shape[:2] != scattering_arr.shape:
        raise ValueError("aerosol scattering tau must have shape (nwave, nlayer, naerosol)")
    fac = np.asarray(aerosol_interp_fraction, dtype=float)
    if fac.shape != scattering_arr.shape[:1]:
        raise ValueError("aerosol interpolation fraction must have shape (nwave,)")

    c2i2m, s2i2m, cos_scatter = _stokes_rotation_from_scattering_plane(angles)
    analyzer_projection = (
        stokes_projection.analyzer_q * c2i2m + stokes_projection.analyzer_u * s2i2m
    )
    basis = _spin2_spherical_function(cos_scatter, moments.shape[1])
    endpoint_p12 = np.matmul(np.moveaxis(moments, 1, 2), basis)
    aerosol_p12 = endpoint_p12[0] + fac[:, np.newaxis] * (endpoint_p12[1] - endpoint_p12[0])
    aerosol_p12_scattering = np.einsum("wla,wa->wl", aerosol_tau, aerosol_p12)

    inv_scattering = np.zeros_like(scattering_arr, dtype=float)
    np.divide(1.0, scattering_arr, out=inv_scattering, where=scattering_arr > 0.0)
    denominator = 1.0 - factor_arr * ssa_arr
    source = (
        float(sign)
        * analyzer_projection
        * aerosol_p12_scattering
        * inv_scattering
        * ssa_arr
        / denominator
    )
    source = np.where(np.isfinite(source), source, 0.0)
    return np.ascontiguousarray(source, dtype=float)


def _rayleigh_diffuse_source_iteration_correction(
    *,
    solver,
    tau: np.ndarray,
    ssa: np.ndarray,
    g: np.ndarray,
    height_grid: np.ndarray,
    observer_angles: np.ndarray,
    diffuse_albedo: np.ndarray,
    delta_m_truncation_factor: np.ndarray,
    rayleigh_scattering_tau: np.ndarray,
    depol: np.ndarray,
    stokes_projection: StokesProjection,
    first_order_correction: np.ndarray,
    sign: int,
    n_azimuths: int,
    stream_value: float,
) -> np.ndarray:
    """Approximate Rayleigh Q/U from scalar 2S upwelling diffuse radiance.

    py2sess supplies the scalar diffuse upwelling field, and Rayleigh P12
    scatters that field once into the OCO line of sight. The result is
    normalized to the existing FO correction using the same line-of-sight
    integration for the direct solar beam; no OCO observed radiance is used.
    """
    tau_arr = np.asarray(tau, dtype=float)
    ssa_arr = np.asarray(ssa, dtype=float)
    factor_arr = np.asarray(delta_m_truncation_factor, dtype=float)
    ray_arr = np.asarray(rayleigh_scattering_tau, dtype=float)
    tau_eff = tau_arr * (1.0 - factor_arr * ssa_arr)
    ray_per_tau_eff = np.divide(ray_arr, tau_eff, out=np.zeros_like(ray_arr), where=tau_eff > 0.0)
    mu0 = math.cos(math.radians(float(observer_angles[0])))
    mu_obs = math.cos(math.radians(float(observer_angles[1])))
    if mu0 <= 0.0 or mu_obs <= 0.0:
        return np.zeros(tau_arr.shape[0], dtype=float)

    top_tau = np.concatenate(
        (np.zeros((tau_eff.shape[0], 1), dtype=float), np.cumsum(tau_eff, axis=1)[:, :-1]),
        axis=1,
    )
    mid_tau = top_tau + 0.5 * tau_eff
    los_weight = np.exp(-top_tau / mu_obs) * (1.0 - np.exp(-tau_eff / mu_obs))

    direct_p12 = _projected_rayleigh_p12(
        incoming_mu=-mu0,
        incoming_azimuth_deg=0.0,
        outgoing_mu=mu_obs,
        outgoing_azimuth_deg=float(observer_angles[2]),
        depol=depol,
        stokes_projection=stokes_projection,
        sign=sign,
    )
    direct_incident = np.exp(-mid_tau / mu0)
    manual_first = np.sum(
        direct_incident * direct_p12[:, np.newaxis] * ray_per_tau_eff * los_weight,
        axis=1,
    )
    scale = np.divide(
        np.asarray(first_order_correction, dtype=float),
        manual_first,
        out=np.zeros_like(manual_first),
        where=np.abs(manual_first) > 1.0e-14,
    )

    mu_quad = float(stream_value)
    vza_quad = math.degrees(math.acos(mu_quad))
    azimuths = np.linspace(0.0, 360.0, int(n_azimuths), endpoint=False)
    diffuse_angles = np.column_stack(
        (
            np.full(azimuths.size, float(observer_angles[0])),
            np.full(azimuths.size, vza_quad),
            azimuths,
        )
    )
    diffuse_result = solver.forward(
        tau=tau_arr,
        ssa=ssa_arr,
        g=g,
        z=height_grid,
        angles=diffuse_angles,
        fbeam=1.0,
        albedo=diffuse_albedo,
        stream=stream_value,
        delta_m_truncation_factor=factor_arr,
        include_fo=False,
    )
    up_profile = np.asarray(diffuse_result.radiance_profile_2s, dtype=float)
    if up_profile.ndim != 3:
        raise ValueError("diffuse radiance profile must have shape (nwave, nazimuth, nlevel)")
    layer_incident = 0.5 * (up_profile[:, :, :-1] + up_profile[:, :, 1:])
    diffuse_source = np.zeros_like(tau_arr, dtype=float)
    direction_weight = 0.5 / float(azimuths.size)
    for geom_index, azimuth in enumerate(azimuths):
        diffuse_p12 = _projected_rayleigh_p12(
            incoming_mu=mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            depol=depol,
            stokes_projection=stokes_projection,
            sign=sign,
        )
        diffuse_source += (
            direction_weight
            * layer_incident[:, geom_index, :]
            * diffuse_p12[:, np.newaxis]
            * ray_per_tau_eff
        )
    manual_diffuse = np.sum(diffuse_source * los_weight, axis=1)
    # ``scale`` maps the dimensionless manual direct-beam source to py2sess FO
    # radiance, so it contains the solar-beam F/(4*pi) normalization.  The
    # scalar diffuse profile is already a radiance; keep only the residual
    # geometry/source-integration correction when applying it to diffuse light.
    diffuse_scale = scale / (0.25 / math.pi)
    correction = diffuse_scale * manual_diffuse
    return np.where(np.isfinite(correction), correction, 0.0)


def _polarized_diffuse_source_iteration_correction(
    *,
    solver,
    tau: np.ndarray,
    ssa: np.ndarray,
    g: np.ndarray,
    height_grid: np.ndarray,
    observer_angles: np.ndarray,
    diffuse_albedo: np.ndarray,
    delta_m_truncation_factor: np.ndarray,
    rayleigh_scattering_tau: np.ndarray,
    aerosol_scattering_tau: np.ndarray,
    aerosol_polarization_moments: np.ndarray,
    aerosol_interp_fraction: np.ndarray,
    depol: np.ndarray,
    stokes_projection: StokesProjection,
    direct_scatter_source: np.ndarray,
    first_order_correction: np.ndarray,
    sign: int,
    n_azimuths: int,
    stream_value: float,
) -> np.ndarray:
    tau_arr = np.asarray(tau, dtype=float)
    ssa_arr = np.asarray(ssa, dtype=float)
    factor_arr = np.asarray(delta_m_truncation_factor, dtype=float)
    ray_arr = np.asarray(rayleigh_scattering_tau, dtype=float)
    aer_arr = np.asarray(aerosol_scattering_tau, dtype=float)
    direct_source_arr = np.asarray(direct_scatter_source, dtype=float)
    tau_eff = tau_arr * (1.0 - factor_arr * ssa_arr)
    mu0 = math.cos(math.radians(float(observer_angles[0])))
    mu_obs = math.cos(math.radians(float(observer_angles[1])))
    if mu0 <= 0.0 or mu_obs <= 0.0:
        return np.zeros(tau_arr.shape[0], dtype=float)

    top_tau = np.concatenate(
        (np.zeros((tau_eff.shape[0], 1), dtype=float), np.cumsum(tau_eff, axis=1)[:, :-1]),
        axis=1,
    )
    mid_tau = top_tau + 0.5 * tau_eff
    los_weight = np.exp(-top_tau / mu_obs) * (1.0 - np.exp(-tau_eff / mu_obs))
    direct_incident = np.exp(-mid_tau / mu0)
    manual_first = np.sum(direct_incident * direct_source_arr * los_weight, axis=1)
    scale = np.divide(
        np.asarray(first_order_correction, dtype=float),
        manual_first,
        out=np.zeros_like(manual_first),
        where=np.abs(manual_first) > 1.0e-14,
    )

    mu_quad = float(stream_value)
    vza_quad = math.degrees(math.acos(mu_quad))
    azimuths = np.linspace(0.0, 360.0, int(n_azimuths), endpoint=False)
    diffuse_angles = np.column_stack(
        (
            np.full(azimuths.size, float(observer_angles[0])),
            np.full(azimuths.size, vza_quad),
            azimuths,
        )
    )
    diffuse_result = solver.forward(
        tau=tau_arr,
        ssa=ssa_arr,
        g=g,
        z=height_grid,
        angles=diffuse_angles,
        fbeam=1.0,
        albedo=diffuse_albedo,
        stream=stream_value,
        delta_m_truncation_factor=factor_arr,
        include_fo=False,
    )
    up_profile = np.asarray(diffuse_result.radiance_profile_2s, dtype=float)
    if up_profile.ndim != 3:
        raise ValueError("diffuse radiance profile must have shape (nwave, nazimuth, nlevel)")
    layer_incident = 0.5 * (up_profile[:, :, :-1] + up_profile[:, :, 1:])

    inv_tau_eff = np.zeros_like(tau_eff, dtype=float)
    np.divide(1.0, tau_eff, out=inv_tau_eff, where=tau_eff > 0.0)
    diffuse_source = np.zeros_like(tau_arr, dtype=float)
    direction_weight = 0.5 / float(azimuths.size)
    for geom_index, azimuth in enumerate(azimuths):
        rayleigh_p12 = _projected_rayleigh_p12(
            incoming_mu=mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            depol=depol,
            stokes_projection=stokes_projection,
            sign=sign,
        )
        scatter_source = rayleigh_p12[:, np.newaxis] * ray_arr * inv_tau_eff
        if aer_arr.shape[-1] > 0:
            aerosol_p12 = _projected_aerosol_p12(
                incoming_mu=mu_quad,
                incoming_azimuth_deg=float(azimuth),
                outgoing_mu=mu_obs,
                outgoing_azimuth_deg=float(observer_angles[2]),
                aerosol_polarization_moments=aerosol_polarization_moments,
                aerosol_interp_fraction=aerosol_interp_fraction,
                stokes_projection=stokes_projection,
                sign=sign,
            )
            scatter_source = scatter_source + np.einsum(
                "wa,wla,wl->wl",
                aerosol_p12,
                aer_arr,
                inv_tau_eff,
            )
        diffuse_source += direction_weight * layer_incident[:, geom_index, :] * scatter_source

    manual_diffuse = np.sum(diffuse_source * los_weight, axis=1)
    # ``scale`` maps the dimensionless manual direct-beam source to py2sess FO
    # radiance, so it contains the solar-beam F/(4*pi) normalization.  The
    # scalar diffuse profile is already a radiance; keep only the residual
    # geometry/source-integration correction when applying it to diffuse light.
    diffuse_scale = scale / (0.25 / math.pi)
    correction = diffuse_scale * manual_diffuse
    return np.where(np.isfinite(correction), correction, 0.0)


def _polarized_second_order_source_iteration_correction(
    *,
    solver,
    tau: np.ndarray,
    ssa: np.ndarray,
    g: np.ndarray,
    height_grid: np.ndarray,
    observer_angles: np.ndarray,
    delta_m_truncation_factor: np.ndarray,
    rayleigh_scattering_tau: np.ndarray,
    depol: np.ndarray,
    stokes_projection: StokesProjection,
    direct_scatter_source: np.ndarray,
    first_order_correction: np.ndarray,
    sign: int,
    n_azimuths: int,
    stream_value: float,
) -> np.ndarray:
    """Approximate Rayleigh-Rayleigh second-order polarization.

    This is a diagnostic 2OS-like source iteration.  The first scattering is
    scalar direct-solar Rayleigh FO into the two-stream upward direction.  The
    second scattering projects Rayleigh P12 into the OCO detector Stokes
    response.  Aerosol direct-FO polarization remains in the base correction,
    but aerosol second order is not included here because the aerosol phase
    forward peak needs a higher-order angular quadrature than this diagnostic.
    """
    tau_arr = np.asarray(tau, dtype=float)
    ssa_arr = np.asarray(ssa, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    factor_arr = np.asarray(delta_m_truncation_factor, dtype=float)
    ray_arr = np.asarray(rayleigh_scattering_tau, dtype=float)
    direct_source_arr = np.asarray(direct_scatter_source, dtype=float)
    tau_eff = tau_arr * (1.0 - factor_arr * ssa_arr)
    mu0 = math.cos(math.radians(float(observer_angles[0])))
    mu_obs = math.cos(math.radians(float(observer_angles[1])))
    if mu0 <= 0.0 or mu_obs <= 0.0:
        return np.zeros(tau_arr.shape[0], dtype=float)

    top_tau = np.concatenate(
        (np.zeros((tau_eff.shape[0], 1), dtype=float), np.cumsum(tau_eff, axis=1)[:, :-1]),
        axis=1,
    )
    mid_tau = top_tau + 0.5 * tau_eff
    los_weight = np.exp(-top_tau / mu_obs) * (1.0 - np.exp(-tau_eff / mu_obs))
    direct_incident = np.exp(-mid_tau / mu0)
    manual_first = np.sum(direct_incident * direct_source_arr * los_weight, axis=1)
    scale = np.divide(
        np.asarray(first_order_correction, dtype=float),
        manual_first,
        out=np.zeros_like(manual_first),
        where=np.abs(manual_first) > 1.0e-14,
    )
    source_scale = scale / (0.25 / math.pi)

    mu_quad = float(stream_value)
    vza_quad = math.degrees(math.acos(mu_quad))
    azimuths = np.linspace(0.0, 360.0, int(n_azimuths), endpoint=False)
    first_order_angles = np.column_stack(
        (
            np.full(azimuths.size, float(observer_angles[0])),
            np.full(azimuths.size, vza_quad),
            azimuths,
        )
    )
    sza = np.deg2rad(first_order_angles[:, 0])
    vza = np.deg2rad(first_order_angles[:, 1])
    raz = np.deg2rad(first_order_angles[:, 2])
    mu1 = np.cos(vza)
    cos_scatter = -(np.cos(vza) * np.cos(sza)) + np.sin(vza) * np.sin(sza) * np.cos(raz)
    overhead = np.isclose(first_order_angles[:, 0], 0.0)
    if np.any(overhead):
        cos_scatter = cos_scatter.copy()
        cos_scatter[overhead] = np.where(np.isclose(mu1[overhead], 0.0), 0.0, -mu1[overhead])
    delta = 2.0 * (1.0 - np.asarray(depol, dtype=float)) / (2.0 + np.asarray(depol, dtype=float))
    rayleigh_phase = delta[:, np.newaxis] * 0.75 * (1.0 + cos_scatter * cos_scatter) + (
        1.0 - delta[:, np.newaxis]
    )
    rayleigh_p12_up = -0.75 * delta[:, np.newaxis] * (1.0 - cos_scatter * cos_scatter)
    first_order_scatter = np.divide(
        rayleigh_phase[:, np.newaxis, :] * ray_arr[:, :, np.newaxis],
        tau_eff[:, :, np.newaxis],
        out=np.zeros(tau_eff.shape + (azimuths.size,), dtype=float),
        where=tau_eff[:, :, np.newaxis] > 0.0,
    )
    first_order_q_scatter = np.divide(
        rayleigh_p12_up[:, np.newaxis, :] * ray_arr[:, :, np.newaxis],
        tau_eff[:, :, np.newaxis],
        out=np.zeros(tau_eff.shape + (azimuths.size,), dtype=float),
        where=tau_eff[:, :, np.newaxis] > 0.0,
    )
    first_order = solver.forward_fo(
        tau=tau_arr,
        ssa=ssa_arr,
        g=g_arr,
        z=height_grid,
        angles=first_order_angles,
        fbeam=1.0,
        albedo=np.zeros(tau_arr.shape[0], dtype=float),
        stream=stream_value,
        delta_m_truncation_factor=factor_arr,
        fo_scatter_term=first_order_scatter,
        n_moments=0,
    )
    first_order_q = solver.forward_fo(
        tau=tau_arr,
        ssa=ssa_arr,
        g=g_arr,
        z=height_grid,
        angles=first_order_angles,
        fbeam=1.0,
        albedo=np.zeros(tau_arr.shape[0], dtype=float),
        stream=stream_value,
        delta_m_truncation_factor=factor_arr,
        fo_scatter_term=first_order_q_scatter,
        n_moments=0,
    )
    first_profile = np.asarray(first_order.intensity_ss_profile, dtype=float)
    first_q_profile = np.asarray(first_order_q.intensity_ss_profile, dtype=float)
    expected_shape = (tau_arr.shape[0], azimuths.size, tau_arr.shape[1] + 1)
    if first_profile.shape != expected_shape or first_q_profile.shape != expected_shape:
        raise ValueError(
            "first-order profiles must have shape "
            f"{expected_shape}; got {first_profile.shape} and {first_q_profile.shape}"
        )
    up_layer_incident = 0.5 * (first_profile[:, :, :-1] + first_profile[:, :, 1:])
    up_q_layer = 0.5 * (first_q_profile[:, :, :-1] + first_q_profile[:, :, 1:])

    cos_scatter_down = mu0 * mu_quad + math.sin(math.radians(float(observer_angles[0]))) * math.sin(
        math.acos(mu_quad)
    ) * np.cos(np.deg2rad(azimuths))
    rayleigh_phase_down = delta[:, np.newaxis] * 0.75 * (
        1.0 + cos_scatter_down * cos_scatter_down
    ) + (1.0 - delta[:, np.newaxis])
    rayleigh_p12_down = -0.75 * delta[:, np.newaxis] * (1.0 - cos_scatter_down * cos_scatter_down)
    down_scatter = np.divide(
        rayleigh_phase_down[:, np.newaxis, :] * ray_arr[:, :, np.newaxis],
        tau_eff[:, :, np.newaxis],
        out=np.zeros(tau_eff.shape + (azimuths.size,), dtype=float),
        where=tau_eff[:, :, np.newaxis] > 0.0,
    )
    down_q_scatter = np.divide(
        rayleigh_p12_down[:, np.newaxis, :] * ray_arr[:, :, np.newaxis],
        tau_eff[:, :, np.newaxis],
        out=np.zeros(tau_eff.shape + (azimuths.size,), dtype=float),
        where=tau_eff[:, :, np.newaxis] > 0.0,
    )
    down_source = (0.25 / math.pi) * down_scatter * np.exp(-mid_tau[:, :, np.newaxis] / mu0)
    down_q_source = (0.25 / math.pi) * down_q_scatter * np.exp(-mid_tau[:, :, np.newaxis] / mu0)
    down_trans = np.exp(-tau_eff / mu_quad)
    down_profile = np.zeros(expected_shape, dtype=float)
    down_q_profile = np.zeros(expected_shape, dtype=float)
    for layer_index in range(tau_arr.shape[1]):
        down_profile[:, :, layer_index + 1] = down_profile[:, :, layer_index] * down_trans[
            :, layer_index, np.newaxis
        ] + down_source[:, layer_index, :] * (1.0 - down_trans[:, layer_index, np.newaxis])
        down_q_profile[:, :, layer_index + 1] = down_q_profile[:, :, layer_index] * down_trans[
            :, layer_index, np.newaxis
        ] + down_q_source[:, layer_index, :] * (1.0 - down_trans[:, layer_index, np.newaxis])
    down_layer_incident = 0.5 * (down_profile[:, :, :-1] + down_profile[:, :, 1:])
    down_q_layer = 0.5 * (down_q_profile[:, :, :-1] + down_q_profile[:, :, 1:])

    inv_tau_eff = np.zeros_like(tau_eff, dtype=float)
    np.divide(1.0, tau_eff, out=inv_tau_eff, where=tau_eff > 0.0)
    second_source = np.zeros_like(tau_arr, dtype=float)
    icorr_source = np.zeros_like(tau_arr, dtype=float)
    direction_weight = 0.5 / float(azimuths.size)
    for geom_index, azimuth in enumerate(azimuths):
        rayleigh_p12 = _projected_rayleigh_p12(
            incoming_mu=mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            depol=depol,
            stokes_projection=stokes_projection,
            sign=sign,
        )
        scatter_source = rayleigh_p12[:, np.newaxis] * ray_arr * inv_tau_eff
        second_source += direction_weight * up_layer_incident[:, geom_index, :] * scatter_source
        _, up_second_cos = _projected_scattering_geometry(
            incoming_mu=mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            stokes_projection=stokes_projection,
        )
        up_second_p12 = _rayleigh_p12_from_cosine(up_second_cos, depol)
        up_cos2 = _scattering_plane_cos2(
            first_incoming_mu=-mu0,
            first_incoming_azimuth_deg=0.0,
            shared_mu=mu_quad,
            shared_azimuth_deg=float(azimuth),
            second_outgoing_mu=mu_obs,
            second_outgoing_azimuth_deg=float(observer_angles[2]),
        )
        icorr_source += (
            direction_weight
            * up_q_layer[:, geom_index, :]
            * up_cos2
            * up_second_p12[:, np.newaxis]
            * ray_arr
            * inv_tau_eff
        )

        down_rayleigh_p12 = _projected_rayleigh_p12(
            incoming_mu=-mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            depol=depol,
            stokes_projection=stokes_projection,
            sign=sign,
        )
        down_scatter_source = down_rayleigh_p12[:, np.newaxis] * ray_arr * inv_tau_eff
        second_source += (
            direction_weight * down_layer_incident[:, geom_index, :] * down_scatter_source
        )
        _, down_second_cos = _projected_scattering_geometry(
            incoming_mu=-mu_quad,
            incoming_azimuth_deg=float(azimuth),
            outgoing_mu=mu_obs,
            outgoing_azimuth_deg=float(observer_angles[2]),
            stokes_projection=stokes_projection,
        )
        down_second_p12 = _rayleigh_p12_from_cosine(down_second_cos, depol)
        down_cos2 = _scattering_plane_cos2(
            first_incoming_mu=-mu0,
            first_incoming_azimuth_deg=0.0,
            shared_mu=-mu_quad,
            shared_azimuth_deg=float(azimuth),
            second_outgoing_mu=mu_obs,
            second_outgoing_azimuth_deg=float(observer_angles[2]),
        )
        icorr_source += (
            direction_weight
            * down_q_layer[:, geom_index, :]
            * down_cos2
            * down_second_p12[:, np.newaxis]
            * ray_arr
            * inv_tau_eff
        )

    manual_second = np.sum(second_source * los_weight, axis=1)
    manual_icorr = np.sum(icorr_source * los_weight, axis=1)
    correction = source_scale * (
        manual_second + float(stokes_projection.scalar_factor) * manual_icorr
    )
    return np.where(np.isfinite(correction), correction, 0.0)


def _prepare_py2sess_rt_context(
    *,
    wavelength_um: np.ndarray,
    state: dict[str, np.ndarray | float],
    gas_tau: np.ndarray,
    angles: np.ndarray,
    aerosol: AerosolInputs,
    solar_reference_factor: np.ndarray | None,
    stokes_coefficients: np.ndarray,
    stokes_projection_mode: str,
    polarization_correction: str,
    polarization_sign: int,
    polarization_diffuse_azimuths: int,
    stream_value: float,
) -> Py2sessRtContext:
    from py2sess.optical.phase import build_solar_phase_inputs_from_scattering_tau

    ray_tau, depol = _rayleigh_tau_cm2(
        wavelength_um,
        np.asarray(state["dry_air_col_cm2"], dtype=float),
        float(state["xco2_ppm"]),
    )
    stokes_projection = _stokes_projection(stokes_coefficients, stokes_projection_mode)
    aerosol_extinction = np.sum(aerosol.extinction_tau, axis=-1)
    aerosol_scattering = aerosol.scattering_tau
    scattering_tau = ray_tau + np.sum(aerosol_scattering, axis=-1)
    tau = gas_tau + ray_tau + aerosol_extinction
    ssa = np.divide(scattering_tau, tau, out=np.zeros_like(tau), where=tau > 0.0)
    phase = build_solar_phase_inputs_from_scattering_tau(
        ssa=ssa,
        depol=depol,
        rayleigh_scattering_tau=ray_tau,
        aerosol_scattering_tau=aerosol_scattering,
        aerosol_moments=aerosol.moments,
        aerosol_interp_fraction=aerosol.interp_fraction,
        angles=angles,
        validate_inputs=False,
    )
    return Py2sessRtContext(
        tau=tau,
        ssa=ssa,
        g=phase.g,
        delta_m_truncation_factor=phase.delta_m_truncation_factor,
        fo_scatter_term=phase.fo_scatter_term,
        rayleigh_scattering_tau=ray_tau,
        aerosol_scattering_tau=aerosol_scattering,
        aerosol_polarization_moments=aerosol.polarization_moments,
        aerosol_interp_fraction=aerosol.interp_fraction,
        scattering_tau=scattering_tau,
        depol=depol,
        height_grid=np.asarray(state["heights_km"], dtype=float),
        angles=np.asarray(angles, dtype=float),
        solar_reference_factor=solar_reference_factor,
        stokes_projection=stokes_projection,
        polarization_correction=polarization_correction,
        polarization_sign=polarization_sign,
        polarization_diffuse_azimuths=polarization_diffuse_azimuths,
        stream_value=stream_value,
    )


def _compute_py2sess_polarization_correction(
    *,
    context: Py2sessRtContext,
    diffuse_albedo: np.ndarray,
) -> np.ndarray:
    from py2sess import TwoStreamEss, TwoStreamEssOptions

    polarization_radiance = np.zeros(context.tau.shape[0], dtype=float)
    if context.polarization_correction in {
        "rayleigh-fo",
        "rayleigh-aerosol-fo",
        "rayleigh-fo-updiffuse",
        "rayleigh-aerosol-fo-updiffuse",
        "rayleigh-aerosol-fo-rayleigh-2os-diagnostic",
    }:
        rayleigh_polarization_scatter = _rayleigh_projected_polarization_scatter_term(
            ssa=context.ssa,
            rayleigh_scattering_tau=context.rayleigh_scattering_tau,
            scattering_tau=context.scattering_tau,
            depol=context.depol,
            delta_m_truncation_factor=context.delta_m_truncation_factor,
            angles=context.angles,
            stokes_projection=context.stokes_projection,
            sign=context.polarization_sign,
        )
        polarization_scatter = rayleigh_polarization_scatter
        if context.polarization_correction in {
            "rayleigh-aerosol-fo",
            "rayleigh-aerosol-fo-updiffuse",
            "rayleigh-aerosol-fo-rayleigh-2os-diagnostic",
        }:
            polarization_scatter = (
                polarization_scatter
                + _aerosol_projected_polarization_scatter_term(
                    ssa=context.ssa,
                    aerosol_scattering_tau=context.aerosol_scattering_tau,
                    scattering_tau=context.scattering_tau,
                    aerosol_polarization_moments=context.aerosol_polarization_moments,
                    aerosol_interp_fraction=context.aerosol_interp_fraction,
                    delta_m_truncation_factor=context.delta_m_truncation_factor,
                    angles=context.angles,
                    stokes_projection=context.stokes_projection,
                    sign=context.polarization_sign,
                )
            )
        fo_solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=context.tau.shape[1],
                mode="solar",
                backend="numpy",
                output_levels=False,
                brdf_surface=False,
            )
        )
        polarization_fo = fo_solver.forward_fo(
            tau=context.tau,
            ssa=context.ssa,
            g=context.g,
            z=context.height_grid,
            angles=context.angles,
            fbeam=1.0,
            albedo=np.zeros(context.tau.shape[0], dtype=float),
            stream=context.stream_value,
            delta_m_truncation_factor=context.delta_m_truncation_factor,
            fo_scatter_term=polarization_scatter,
            n_moments=0,
        )
        polarization_radiance = np.asarray(polarization_fo.intensity_ss, dtype=float)
        if context.polarization_correction == "rayleigh-fo-updiffuse":
            diffuse_solver = TwoStreamEss(
                TwoStreamEssOptions(
                    nlyr=context.tau.shape[1],
                    mode="solar",
                    backend="numpy",
                    output_levels=True,
                    brdf_surface=False,
                )
            )
            polarization_radiance = (
                polarization_radiance
                + _rayleigh_diffuse_source_iteration_correction(
                    solver=diffuse_solver,
                    tau=context.tau,
                    ssa=context.ssa,
                    g=context.g,
                    height_grid=context.height_grid,
                    observer_angles=context.angles,
                    diffuse_albedo=np.asarray(diffuse_albedo, dtype=float),
                    delta_m_truncation_factor=context.delta_m_truncation_factor,
                    rayleigh_scattering_tau=context.rayleigh_scattering_tau,
                    depol=context.depol,
                    stokes_projection=context.stokes_projection,
                    first_order_correction=np.asarray(polarization_fo.intensity_ss, dtype=float),
                    sign=context.polarization_sign,
                    n_azimuths=context.polarization_diffuse_azimuths,
                    stream_value=context.stream_value,
                )
            )
        elif context.polarization_correction == "rayleigh-aerosol-fo-updiffuse":
            diffuse_solver = TwoStreamEss(
                TwoStreamEssOptions(
                    nlyr=context.tau.shape[1],
                    mode="solar",
                    backend="numpy",
                    output_levels=True,
                    brdf_surface=False,
                )
            )
            polarization_radiance = (
                polarization_radiance
                + _polarized_diffuse_source_iteration_correction(
                    solver=diffuse_solver,
                    tau=context.tau,
                    ssa=context.ssa,
                    g=context.g,
                    height_grid=context.height_grid,
                    observer_angles=context.angles,
                    diffuse_albedo=np.asarray(diffuse_albedo, dtype=float),
                    delta_m_truncation_factor=context.delta_m_truncation_factor,
                    rayleigh_scattering_tau=context.rayleigh_scattering_tau,
                    aerosol_scattering_tau=context.aerosol_scattering_tau,
                    aerosol_polarization_moments=context.aerosol_polarization_moments,
                    aerosol_interp_fraction=context.aerosol_interp_fraction,
                    depol=context.depol,
                    stokes_projection=context.stokes_projection,
                    direct_scatter_source=polarization_scatter,
                    first_order_correction=np.asarray(polarization_fo.intensity_ss, dtype=float),
                    sign=context.polarization_sign,
                    n_azimuths=context.polarization_diffuse_azimuths,
                    stream_value=context.stream_value,
                )
            )
        elif context.polarization_correction == "rayleigh-aerosol-fo-rayleigh-2os-diagnostic":
            rayleigh_fo = fo_solver.forward_fo(
                tau=context.tau,
                ssa=context.ssa,
                g=context.g,
                z=context.height_grid,
                angles=context.angles,
                fbeam=1.0,
                albedo=np.zeros(context.tau.shape[0], dtype=float),
                stream=context.stream_value,
                delta_m_truncation_factor=context.delta_m_truncation_factor,
                fo_scatter_term=rayleigh_polarization_scatter,
                n_moments=0,
            )
            second_order_solver = TwoStreamEss(
                TwoStreamEssOptions(
                    nlyr=context.tau.shape[1],
                    mode="solar",
                    backend="numpy",
                    output_levels=True,
                    brdf_surface=False,
                )
            )
            polarization_radiance = (
                polarization_radiance
                + _polarized_second_order_source_iteration_correction(
                    solver=second_order_solver,
                    tau=context.tau,
                    ssa=context.ssa,
                    g=context.g,
                    height_grid=context.height_grid,
                    observer_angles=context.angles,
                    delta_m_truncation_factor=context.delta_m_truncation_factor,
                    rayleigh_scattering_tau=context.rayleigh_scattering_tau,
                    depol=context.depol,
                    stokes_projection=context.stokes_projection,
                    direct_scatter_source=rayleigh_polarization_scatter,
                    first_order_correction=np.asarray(rayleigh_fo.intensity_ss, dtype=float),
                    sign=context.polarization_sign,
                    n_azimuths=context.polarization_diffuse_azimuths,
                    stream_value=context.stream_value,
                )
            )
    elif context.polarization_correction != "none":
        raise ValueError(f"unknown polarization correction: {context.polarization_correction!r}")
    return polarization_radiance


def _run_py2sess_prepared(
    *,
    context: Py2sessRtContext,
    albedo: np.ndarray,
    diffuse_albedo: np.ndarray,
    brdf: dict[str, np.ndarray] | None,
    polarization_correction_cache: np.ndarray | None = None,
) -> Py2sessReplayResult:
    from py2sess import TwoStreamEss, TwoStreamEssOptions

    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=context.tau.shape[1],
            mode="solar",
            backend="numpy",
            output_levels=False,
            brdf_surface=brdf is not None,
        )
    )
    result = solver.forward(
        tau=context.tau,
        ssa=context.ssa,
        g=context.g,
        z=context.height_grid,
        angles=context.angles,
        fbeam=1.0,
        albedo=albedo,
        brdf=brdf,
        stream=context.stream_value,
        delta_m_truncation_factor=context.delta_m_truncation_factor,
        fo_scatter_term=context.fo_scatter_term,
        include_fo=True,
    )
    scalar_radiance = (
        np.asarray(result.radiance_total, dtype=float) * context.stokes_projection.scalar_factor
    )
    if polarization_correction_cache is None:
        polarization_radiance = _compute_py2sess_polarization_correction(
            context=context,
            diffuse_albedo=diffuse_albedo,
        )
    else:
        polarization_radiance = np.asarray(polarization_correction_cache, dtype=float)
    if context.solar_reference_factor is not None:
        factor = np.asarray(context.solar_reference_factor, dtype=float)
        scalar_radiance = scalar_radiance * factor
        polarization_radiance = polarization_radiance * factor
    return Py2sessReplayResult(
        scalar_radiance=scalar_radiance,
        polarization_correction=polarization_radiance,
        radiance=scalar_radiance + polarization_radiance,
    )


def _run_py2sess(
    *,
    wavelength_um: np.ndarray,
    state: dict[str, np.ndarray | float],
    gas_tau: np.ndarray,
    albedo: np.ndarray,
    diffuse_albedo: np.ndarray,
    brdf: dict[str, np.ndarray] | None,
    angles: np.ndarray,
    aerosol: AerosolInputs,
    solar_reference_factor: np.ndarray | None,
    stokes_coefficients: np.ndarray,
    stokes_projection_mode: str,
    polarization_correction: str,
    polarization_sign: int,
    polarization_diffuse_azimuths: int,
    stream_value: float,
) -> Py2sessReplayResult:
    context = _prepare_py2sess_rt_context(
        wavelength_um=wavelength_um,
        state=state,
        gas_tau=gas_tau,
        angles=angles,
        aerosol=aerosol,
        solar_reference_factor=solar_reference_factor,
        stokes_coefficients=stokes_coefficients,
        stokes_projection_mode=stokes_projection_mode,
        polarization_correction=polarization_correction,
        polarization_sign=polarization_sign,
        polarization_diffuse_azimuths=polarization_diffuse_azimuths,
        stream_value=stream_value,
    )
    return _run_py2sess_prepared(
        context=context,
        albedo=albedo,
        diffuse_albedo=diffuse_albedo,
        brdf=brdf,
    )


def _photon_to_energy_spectral_radiance(
    photon_radiance: np.ndarray | float,
    wavelength_um: np.ndarray | float,
) -> np.ndarray:
    """Convert OCO photon spectral radiance to W m^-2 sr^-1 um^-1."""
    wavelength_m = np.asarray(wavelength_um, dtype=float) * 1.0e-6
    if np.any(~np.isfinite(wavelength_m)) or np.any(wavelength_m <= 0.0):
        raise ValueError("wavelength_um must be positive and finite")
    return (
        np.asarray(photon_radiance, dtype=float)
        * PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_S
        / wavelength_m
    )


def _sample_continuum_level(values: np.ndarray) -> float:
    """Estimate a continuum level using OCO's high-signal percentile idea."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    lo = float(np.nanpercentile(finite, 98.0))
    hi = float(np.nanpercentile(finite, 99.0))
    window = finite[(finite >= lo) & (finite <= hi)]
    if window.size == 0:
        window = finite[finite >= lo]
    return float(np.nanmean(window))


def _continuum_fit_mask(
    values: np.ndarray,
    *,
    fraction: float,
    min_points: int,
) -> np.ndarray:
    """Select high-signal detector colors for a continuum-only surface fit."""
    radiance = np.asarray(values, dtype=float)
    finite = np.isfinite(radiance)
    mask = np.zeros(radiance.shape, dtype=bool)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        return mask
    n_keep = int(math.ceil(float(fraction) * finite_count))
    n_keep = min(finite_count, max(int(min_points), n_keep))
    finite_index = np.flatnonzero(finite)
    ordered = finite_index[np.argsort(radiance[finite_index])]
    mask[ordered[-n_keep:]] = True
    return mask


def _scale_brdf(
    brdf: dict[str, np.ndarray] | None,
    scale: np.ndarray | float,
) -> dict[str, np.ndarray] | None:
    if brdf is None:
        return None
    factor = np.asarray(scale, dtype=float)
    if factor.ndim == 0:
        return {
            name: float(factor) * np.asarray(value, dtype=float) for name, value in brdf.items()
        }
    return {
        name: factor.reshape((factor.size,) + (1,) * (np.asarray(value).ndim - 1))
        * np.asarray(value, dtype=float)
        for name, value in brdf.items()
    }


def _surface_brdf_tilt_axis(*, wavelength_um: np.ndarray, band: str) -> np.ndarray:
    """Dimensionless in-band wavenumber axis for the BRDF weight slope."""
    wavenumber = 1.0e4 / np.asarray(wavelength_um, dtype=float)
    delta = wavenumber - 1.0e4 / BAND_REFERENCE_WAVELENGTH_UM[band]
    scale = float(np.nanmax(np.abs(delta)))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(delta, dtype=float)
    return delta / scale


def _solve_linearized_surface_brdf_scale(
    *,
    measured_radiance: np.ndarray,
    base_radiance: np.ndarray,
    probe_radiance: np.ndarray,
    continuum_radiance: float,
    continuum_mask: np.ndarray,
    probe_scale: float,
    current_scale: float = 1.0,
    scale_min: float,
    scale_max: float,
    prior_sigma: float,
) -> SurfaceBrdfRetrieval:
    mask = np.asarray(continuum_mask, dtype=bool)
    n_points = int(np.count_nonzero(mask))
    if n_points == 0:
        return SurfaceBrdfRetrieval(1.0, 0.0, 0, math.nan, "no_continuum_points")

    delta_scale = float(probe_scale) - float(current_scale)
    if not np.isfinite(delta_scale) or abs(delta_scale) < 1.0e-12:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_points, math.nan, "bad_probe_scale")

    continuum = float(continuum_radiance)
    if not np.isfinite(continuum) or continuum <= 0.0:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_points, math.nan, "bad_continuum")

    jac = (np.asarray(probe_radiance, dtype=float) - np.asarray(base_radiance, dtype=float)) / (
        delta_scale * continuum
    )
    residual = (
        np.asarray(measured_radiance, dtype=float) - np.asarray(base_radiance, dtype=float)
    ) / continuum
    valid = mask & np.isfinite(jac) & np.isfinite(residual)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return SurfaceBrdfRetrieval(1.0, 0.0, 0, math.nan, "no_valid_continuum_points")

    numerator = float(np.sum(jac[valid] * residual[valid]))
    denominator = float(np.sum(jac[valid] * jac[valid]))
    if prior_sigma > 0.0:
        prior_weight = 1.0 / (float(prior_sigma) * float(prior_sigma))
        numerator -= (float(current_scale) - 1.0) * prior_weight
        denominator += prior_weight
    if denominator <= 0.0 or not np.isfinite(denominator):
        return SurfaceBrdfRetrieval(1.0, 0.0, n_valid, math.nan, "zero_surface_response")

    scale = float(current_scale) + numerator / denominator
    scale = float(np.clip(scale, float(scale_min), float(scale_max)))
    linearized = np.asarray(base_radiance, dtype=float) + (
        np.asarray(probe_radiance, dtype=float) - np.asarray(base_radiance, dtype=float)
    ) * ((scale - float(current_scale)) / delta_scale)
    fit_residual = 100.0 * (linearized[valid] - np.asarray(measured_radiance)[valid]) / continuum
    fit_rmse = float(np.sqrt(np.mean(fit_residual * fit_residual)))
    return SurfaceBrdfRetrieval(scale, 0.0, n_valid, fit_rmse, "continuum_linearized")


def _solve_linearized_surface_brdf_scale_and_tilt(
    *,
    measured_radiance: np.ndarray,
    base_radiance: np.ndarray,
    scale_probe_radiance: np.ndarray,
    tilt_probe_radiance: np.ndarray,
    continuum_radiance: float,
    continuum_mask: np.ndarray,
    probe_scale: float,
    probe_tilt: float,
    current_scale: float = 1.0,
    current_tilt: float = 0.0,
    scale_min: float,
    scale_max: float,
    tilt_min: float,
    tilt_max: float,
) -> SurfaceBrdfRetrieval:
    mask = np.asarray(continuum_mask, dtype=bool)
    n_points = int(np.count_nonzero(mask))
    if n_points == 0:
        return SurfaceBrdfRetrieval(1.0, 0.0, 0, math.nan, "no_continuum_points")
    continuum = float(continuum_radiance)
    if not np.isfinite(continuum) or continuum <= 0.0:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_points, math.nan, "bad_continuum")
    delta_scale = float(probe_scale) - float(current_scale)
    delta_tilt = float(probe_tilt) - float(current_tilt)
    if abs(delta_scale) < 1.0e-12 or abs(delta_tilt) < 1.0e-12:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_points, math.nan, "bad_probe")

    base = np.asarray(base_radiance, dtype=float)
    scale_jac = (np.asarray(scale_probe_radiance, dtype=float) - base) / (delta_scale * continuum)
    tilt_jac = (np.asarray(tilt_probe_radiance, dtype=float) - base) / (delta_tilt * continuum)
    residual = (np.asarray(measured_radiance, dtype=float) - base) / continuum
    valid = mask & np.isfinite(scale_jac) & np.isfinite(tilt_jac) & np.isfinite(residual)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_valid, math.nan, "too_few_points")

    design = np.column_stack((scale_jac[valid], tilt_jac[valid]))
    try:
        solution, *_ = np.linalg.lstsq(design, residual[valid], rcond=None)
    except np.linalg.LinAlgError:
        return SurfaceBrdfRetrieval(1.0, 0.0, n_valid, math.nan, "singular_surface_response")
    scale = float(np.clip(float(current_scale) + solution[0], scale_min, scale_max))
    tilt = float(np.clip(float(current_tilt) + solution[1], tilt_min, tilt_max))
    linearized = (
        base
        + (scale - float(current_scale)) * continuum * scale_jac
        + (tilt - float(current_tilt)) * continuum * tilt_jac
    )
    fit_residual = 100.0 * (linearized[valid] - np.asarray(measured_radiance)[valid]) / continuum
    fit_rmse = float(np.sqrt(np.mean(fit_residual * fit_residual)))
    return SurfaceBrdfRetrieval(scale, tilt, n_valid, fit_rmse, "continuum_linearized_slope")


def _continuum_residual_stats(
    py_radiance: np.ndarray,
    measured_radiance: np.ndarray,
    continuum_radiance: float,
) -> dict[str, float]:
    residual_percent = (
        100.0
        * (np.asarray(py_radiance, dtype=float) - np.asarray(measured_radiance, dtype=float))
        / float(continuum_radiance)
    )
    return {
        "continuum_referenced_bias_percent": float(np.nanmean(residual_percent)),
        "continuum_referenced_rmse_percent": float(
            np.sqrt(np.nanmean(residual_percent * residual_percent))
        ),
        "continuum_referenced_max_abs_percent": float(np.nanmax(np.abs(residual_percent))),
        "corr": float(np.corrcoef(py_radiance, measured_radiance)[0, 1]),
    }


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    soundings = list(dict.fromkeys(int(row["sounding_id"]) for row in rows))
    n_cases = len(soundings)
    height_ratios = []
    for _ in soundings:
        height_ratios.extend((1.8, 1.0))
    fig, axes = plt.subplots(
        2 * n_cases,
        3,
        figsize=(9.8, max(3.8, 3.0 * n_cases)),
        dpi=220,
        squeeze=False,
        sharex="col",
        sharey=False,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.16},
    )
    for r, sid in enumerate(soundings):
        spectrum_row = 2 * r
        diff_row = spectrum_row + 1
        for c, band in enumerate(BANDS):
            spectrum_axis = axes[spectrum_row, c]
            diff_axis = axes[diff_row, c]
            sub = [row for row in rows if int(row["sounding_id"]) == sid and row["band"] == band]
            if not sub:
                spectrum_axis.set_axis_off()
                diff_axis.set_axis_off()
                continue
            x = np.array([float(row["wavelength_um"]) for row in sub])
            measured = np.array([float(row["measured_radiance_w_m2_sr_um"]) for row in sub])
            py2sess = np.array([float(row["py2sess_radiance_w_m2_sr_um"]) for row in sub])
            rel_diff = np.array(
                [float(row["py2sess_minus_measured_continuum_percent"]) for row in sub]
            )
            spectrum_axis.plot(x, measured, color="#111111", lw=0.8, label="measured")
            spectrum_axis.plot(x, py2sess, color="#D55E00", lw=0.85, ls="--", label="py2sess")
            diff_axis.axhline(0.0, color="#9ca3af", lw=0.55)
            diff_axis.plot(x, rel_diff, color="#D55E00", lw=0.75)
            for axis in (spectrum_axis, diff_axis):
                axis.grid(True, color="#e5e7eb", lw=0.45)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.tick_params(labelsize=7)
            if r == 0:
                spectrum_axis.set_title(BAND_LABELS[band], fontsize=8.5)
            if c == 0:
                spectrum_axis.set_ylabel(
                    f"{sid}\nRadiance\n(W m$^{{-2}}$ sr$^{{-1}}$ um$^{{-1}}$)",
                    fontsize=7.5,
                )
                diff_axis.set_ylabel(
                    "Relative diff\n(% continuum)",
                    fontsize=7.5,
                )
            if r == n_cases - 1:
                diff_axis.set_xlabel("Wavelength (um)", fontsize=7.5)
            spectrum_axis.tick_params(labelbottom=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=7,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.88, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--absco-dir", type=Path, default=DEFAULT_ABSCO_DIR)
    parser.add_argument("--co2-absco", type=Path, default=DEFAULT_CO2_ABSCO)
    parser.add_argument("--o2-absco", type=Path, default=DEFAULT_O2_ABSCO)
    parser.add_argument("--h2o-absco", type=Path, default=DEFAULT_H2O_ABSCO)
    parser.add_argument(
        "--oco-solar-model",
        type=Path,
        default=_default_oco_solar_model(),
        help=(
            "OCO/RtRetrievalFramework l2_solar_model.h5 containing the official "
            "solar continuum photon table and absorption spectra."
        ),
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-colors-per-band", type=int, default=120)
    parser.add_argument(
        "--stream-value",
        type=float,
        default=1.0 / math.sqrt(3.0),
        help=(
            "Two-stream quadrature cosine used by py2sess and by the BRDF "
            "Fourier coefficients. The py2sess public default is 1/sqrt(3); "
            "RtRetrievalFramework's twostream/LIDORT-comparison mode uses 0.5."
        ),
    )
    parser.add_argument(
        "--brdf-quadrature-streams",
        type=int,
        default=50,
        help="Number of azimuth quadrature points used to generate BRDF Fourier terms.",
    )
    parser.add_argument(
        "--ils-grid-spacing-cm-1",
        type=float,
        default=0.0,
        help=(
            "High-resolution wavenumber spacing used to resample each ILS before "
            "RT evaluation. The default 0 uses the native L1B ILS table nodes."
        ),
    )
    parser.add_argument(
        "--surface-spectrum",
        choices=("l2-polynomial", "l2-linear", "constant"),
        default="l2-linear",
        help=(
            "Land surface spectral reflectance model. 'l2-linear' uses the L2 "
            "band-center reflectance plus the wavenumber slope term. "
            "'l2-polynomial' also applies the L2 quadratic term; 'constant' "
            "drops both in-band spectral terms."
        ),
    )
    parser.add_argument(
        "--surface-angular",
        choices=("rpv-weight", "l2-reflectance", "rpv-brdf"),
        default="rpv-brdf",
        help=(
            "'l2-reflectance' uses the L2 reflectance field already evaluated "
            "at the observation geometry. 'rpv-weight' reconstructs OCO land "
            "reflectance from the retrieved RPV BRDF weight and fixed angular "
            "kernel. 'rpv-brdf' passes the retrieved OCO "
            "RPV kernel coefficients to the py2sess BRDF surface path."
        ),
    )
    parser.add_argument(
        "--surface-brdf-retrieval",
        choices=("none", "continuum-linearized", "continuum-linearized-slope"),
        default="continuum-linearized-slope",
        help=(
            "Continuum-constrained retrieval of a multiplicative scale on the "
            "L2 land BRDF weight for each sounding and band. The default "
            "'continuum-linearized-slope' also fits an in-band BRDF weight-slope "
            "perturbation. Use 'none' for a strict fixed-L2-surface replay. "
            "'continuum-linearized' fits high-signal continuum colors using a "
            "finite-difference surface Jacobian around the L2 value."
        ),
    )
    parser.add_argument(
        "--surface-brdf-continuum-fraction",
        type=float,
        default=0.40,
        help="Fraction of highest-radiance detector colors used for the surface BRDF fit.",
    )
    parser.add_argument(
        "--surface-brdf-continuum-min-points",
        type=int,
        default=8,
        help="Minimum number of detector colors used for the surface BRDF continuum fit.",
    )
    parser.add_argument(
        "--surface-brdf-scale-min",
        type=float,
        default=0.5,
        help="Lower bound for the retrieved multiplicative L2 BRDF weight scale.",
    )
    parser.add_argument(
        "--surface-brdf-scale-max",
        type=float,
        default=1.5,
        help="Upper bound for the retrieved multiplicative L2 BRDF weight scale.",
    )
    parser.add_argument(
        "--surface-brdf-probe-step",
        type=float,
        default=0.02,
        help="Finite-difference step for the linearized surface BRDF retrieval.",
    )
    parser.add_argument(
        "--surface-brdf-max-iterations",
        type=int,
        default=3,
        help=(
            "Maximum number of continuum surface BRDF linearization iterations. "
            "Use 1 to reproduce the original single-step replay."
        ),
    )
    parser.add_argument(
        "--surface-brdf-prior-sigma",
        type=float,
        default=0.0,
        help=(
            "Optional Gaussian prior sigma on the multiplicative BRDF scale. "
            "The default 0 disables the prior."
        ),
    )
    parser.add_argument(
        "--surface-brdf-tilt-min",
        type=float,
        default=-0.25,
        help="Lower bound for the BRDF weight-slope tilt parameter.",
    )
    parser.add_argument(
        "--surface-brdf-tilt-max",
        type=float,
        default=0.25,
        help="Upper bound for the BRDF weight-slope tilt parameter.",
    )
    parser.add_argument(
        "--aerosol-treatment",
        choices=("none", "l2-posterior-hg", "l2-posterior-gaussian-hg", "oco-l2fp"),
        default="l2-posterior-gaussian-hg",
        help=(
            "'none' uses gas plus Rayleigh scattering only. "
            "'l2-posterior-hg' inserts L2 posterior aerosol AOD using the "
            "official pressure subcolumns and simple HG optical defaults. "
            "'l2-posterior-gaussian-hg' is the default and uses L2 gaussian_log "
            "aerosol parameters for the vertical profile. 'oco-l2fp' uses the "
            "RtRetrievalFramework L2FP aerosol optical-property table."
        ),
    )
    parser.add_argument(
        "--oco-l2fp-aerosol-file",
        type=Path,
        default=None,
        help=(
            "Path to RtRetrievalFramework l2_aerosol_combined.h5. Required for "
            "--aerosol-treatment oco-l2fp unless RTRF_AEROSOL_FILE is set."
        ),
    )
    parser.add_argument(
        "--diagnostic-aerosol-types",
        default=None,
        help=(
            "Comma-separated aerosol types to keep for aerosol diagnostics, e.g. SS,SO. "
            "When set, this overrides --aerosol-type-set."
        ),
    )
    parser.add_argument(
        "--aerosol-type-set",
        choices=("tropospheric", "all"),
        default="tropospheric",
        help=(
            "Aerosol type set used when --diagnostic-aerosol-types is not supplied. "
            "'tropospheric' keeps DU, SS, BC, OC, and SO. 'all' also keeps the "
            "OCO cloud/stratospheric proxy types Ice, Water, and ST."
        ),
    )
    parser.add_argument(
        "--diagnostic-aerosol-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplicative scale applied to aerosol optical depth.",
    )
    parser.add_argument(
        "--eof-treatment",
        choices=("none", "oco3-static"),
        default="none",
        help=(
            "'oco3-static' applies the OCO3 RtRetrievalFramework static EOF "
            "waveforms with L2 EOF scales and L1B noise scaling."
        ),
    )
    parser.add_argument(
        "--oco3-eof-file",
        type=Path,
        default=_default_oco3_eof_file(),
        help=(
            "RtRetrievalFramework l2_oco3_eof.h5 file used by "
            "--eof-treatment oco3-static. Can also be set with RTRF_OCO3_EOF_FILE."
        ),
    )
    parser.add_argument(
        "--gas-doppler",
        choices=("l2-los", "off"),
        default="l2-los",
        help=(
            "Transform ABSCO gas lookup wavelengths with the L2 line-of-sight relative velocity."
        ),
    )
    parser.add_argument(
        "--solar-doppler",
        choices=("l2-solar", "l2-los", "off"),
        default="l2-solar",
        help=(
            "Transform solar pseudo-transmittance lookup wavelengths with the OCO "
            "solar relative velocity after undoing the instrument Doppler domain "
            "shift. 'l2-los' is a diagnostic using only the line-of-sight gas velocity."
        ),
    )
    parser.add_argument(
        "--fluorescence-treatment",
        choices=("none", "l2-posterior"),
        default="none",
        help=(
            "O2 A-band land fluorescence spectrum effect. 'l2-posterior' uses "
            "RetrievalResults/fluorescence_at_reference and fluorescence_slope; "
            "it is ignored for the two CO2 bands."
        ),
    )
    parser.add_argument(
        "--polarization-correction",
        choices=(
            "none",
            "rayleigh-fo",
            "rayleigh-aerosol-fo",
            "rayleigh-fo-updiffuse",
            "rayleigh-aerosol-fo-updiffuse",
            "rayleigh-aerosol-fo-rayleigh-2os-diagnostic",
        ),
        default="rayleigh-aerosol-fo",
        help=(
            "Optional correction that projects Q/U scattering "
            "through the L1B instrument Stokes coefficients. The default "
            "'rayleigh-aerosol-fo' uses the direct-solar Rayleigh term and "
            "also includes direct-solar aerosol polarization from the OCO "
            "L2FP phase matrix when available; 'rayleigh-fo' keeps only "
            "Rayleigh for diagnostics; "
            "'rayleigh-aerosol-fo-updiffuse' also adds one source-iteration "
            "term from py2sess scalar upwelling diffuse radiance; "
            "'rayleigh-aerosol-fo-rayleigh-2os-diagnostic' adds a diagnostic "
            "Rayleigh-Rayleigh second-order source iteration from direct-solar "
            "FO scalar radiance. Use 'none' for scalar-only replay. This "
            "leaves the py2sess scalar solver unchanged."
        ),
    )
    parser.add_argument(
        "--stokes-projection",
        choices=("l1b-normalized", "raw-detector"),
        default="l1b-normalized",
        help=(
            "How to apply L1B instrument Stokes coefficients. 'l1b-normalized' "
            "uses the published-radiance convention L = I + (m12/m11)Q + "
            "(m13/m11)U. 'raw-detector' uses the literal detector projection "
            "L = m11 I + m12 Q + m13 U."
        ),
    )
    parser.add_argument(
        "--ocean-coxmunk-stokes-scope",
        choices=("all", "direct", "none"),
        default="all",
        help=(
            "Diagnostic scope for applying the GISS Cox-Munk Stokes projection. "
            "'all' applies it to both direct glint and py2sess BRDF Fourier terms; "
            "'direct' applies it only to the direct surface BRF; 'none' leaves "
            "the scalar Cox-Munk surface unprojected."
        ),
    )
    parser.add_argument(
        "--diagnostic-gas-tau-scale-o2",
        type=float,
        default=1.0,
        help="Diagnostic multiplier on O2-band O2 gas optical depth.",
    )
    parser.add_argument(
        "--diagnostic-gas-tau-scale-wco2",
        type=float,
        default=1.0,
        help="Diagnostic multiplier on weak-CO2-band CO2 gas optical depth.",
    )
    parser.add_argument(
        "--diagnostic-gas-tau-scale-sco2",
        type=float,
        default=1.0,
        help="Diagnostic multiplier on strong-CO2-band CO2 gas optical depth.",
    )
    parser.add_argument(
        "--diagnostic-layer-pressure-method",
        choices=("geometric", "arithmetic"),
        default="geometric",
        help="Diagnostic layer-pressure sampling used for ABSCO lookup.",
    )
    parser.add_argument(
        "--diagnostic-surface-pressure-offset-hpa",
        type=float,
        default=0.0,
        help=(
            "Diagnostic offset added to the L2 retrieved surface pressure before "
            "ABSCO lookup. The retrieval pressure grid is scaled to the shifted "
            "surface pressure."
        ),
    )
    parser.add_argument(
        "--diagnostic-surface-pressure-column-mode",
        choices=("fixed-columns", "hydrostatic-columns"),
        default="fixed-columns",
        help=(
            "How to update layer columns after a diagnostic surface-pressure shift. "
            "'fixed-columns' preserves the L2 retrieved gas columns and isolates "
            "pressure-broadening sensitivity. 'hydrostatic-columns' recomputes "
            "dry/wet/H2O columns from delta-p and H2O VMR, so O2, CO2, Rayleigh, "
            "and height inputs follow the shifted pressure grid."
        ),
    )
    parser.add_argument(
        "--diagnostic-gas-integration",
        choices=("single-point", "simpson10", "simpson10-metgrid"),
        default="single-point",
        help=(
            "Diagnostic ABSCO layer integration. 'single-point' uses the current "
            "one pressure/temperature sample per layer. 'simpson10' preserves the "
            "L2 gas columns but averages cross sections over ten pressure "
            "subintervals in each layer. 'simpson10-metgrid' also inserts the "
            "meteorology pressure levels as Simpson segment endpoints, matching "
            "the RtRetrieval AbsorberAbsco treatment more closely."
        ),
    )
    parser.add_argument(
        "--polarization-diffuse-azimuths",
        type=int,
        default=8,
        help=(
            "Number of azimuth quadrature points for the updiffuse polarization modes. "
            "Ignored by other polarization modes."
        ),
    )
    parser.add_argument(
        "--polarization-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help=("Sign convention switch for the Rayleigh Q/U source."),
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write CSV outputs without generating the spectrum/residual figure.",
    )
    args = parser.parse_args()
    if args.polarization_diffuse_azimuths <= 0:
        raise ValueError("--polarization-diffuse-azimuths must be positive")
    if not math.isfinite(args.stream_value) or args.stream_value <= 0.0 or args.stream_value > 1.0:
        raise ValueError("--stream-value must satisfy 0 < value <= 1")
    if args.brdf_quadrature_streams <= 0 or args.brdf_quadrature_streams % 2 != 0:
        raise ValueError("--brdf-quadrature-streams must be a positive even integer")
    if not np.isfinite(args.ils_grid_spacing_cm_1) or args.ils_grid_spacing_cm_1 < 0.0:
        raise ValueError("--ils-grid-spacing-cm-1 must be nonnegative and finite")
    if not (0.0 < args.surface_brdf_continuum_fraction <= 1.0):
        raise ValueError("--surface-brdf-continuum-fraction must satisfy 0 < value <= 1")
    if args.surface_brdf_continuum_min_points <= 0:
        raise ValueError("--surface-brdf-continuum-min-points must be positive")
    if (
        not np.isfinite(args.surface_brdf_scale_min)
        or not np.isfinite(args.surface_brdf_scale_max)
        or args.surface_brdf_scale_min <= 0.0
        or args.surface_brdf_scale_min >= args.surface_brdf_scale_max
    ):
        raise ValueError("--surface-brdf-scale-min/max must be finite positive bounds")
    if not (args.surface_brdf_scale_min < 1.0 < args.surface_brdf_scale_max):
        raise ValueError("--surface-brdf-scale bounds must contain 1")
    if not np.isfinite(args.surface_brdf_probe_step) or args.surface_brdf_probe_step <= 0.0:
        raise ValueError("--surface-brdf-probe-step must be positive and finite")
    if args.surface_brdf_max_iterations <= 0:
        raise ValueError("--surface-brdf-max-iterations must be positive")
    if 1.0 + args.surface_brdf_probe_step > args.surface_brdf_scale_max:
        raise ValueError("--surface-brdf-probe-step must stay within --surface-brdf-scale-max")
    if args.surface_brdf_prior_sigma < 0.0 or not np.isfinite(args.surface_brdf_prior_sigma):
        raise ValueError("--surface-brdf-prior-sigma must be nonnegative and finite")
    if (
        not np.isfinite(args.surface_brdf_tilt_min)
        or not np.isfinite(args.surface_brdf_tilt_max)
        or args.surface_brdf_tilt_min >= args.surface_brdf_tilt_max
    ):
        raise ValueError("--surface-brdf-tilt-min/max must be finite ordered bounds")
    if args.eof_treatment == "oco3-static" and args.oco3_eof_file is None:
        raise FileNotFoundError(
            "--eof-treatment oco3-static requires --oco3-eof-file or RTRF_OCO3_EOF_FILE"
        )
    if args.diagnostic_aerosol_scale < 0.0:
        raise ValueError("--diagnostic-aerosol-scale must be nonnegative")
    aerosol_type_filter = _parse_aerosol_type_filter(args.diagnostic_aerosol_types)
    if aerosol_type_filter is None:
        aerosol_type_filter = _default_aerosol_type_filter(args.aerosol_type_set)
    aerosol_type_filter_label = (
        ",".join(sorted(aerosol_type_filter))
        if aerosol_type_filter is not None
        else "all_retrieved"
    )

    l1b_path = _single_data_file(args.data_dir, "oco3_L1bScSC_*.h5")
    l2std_path = _single_data_file(args.data_dir, "oco3_L2StdSC_*.h5")
    l2dia_path = _single_data_file(args.data_dir, "oco3_L2DiaSC_*.h5")
    for path in (l1b_path, l2std_path, l2dia_path, args.co2_absco, args.o2_absco, args.h2o_absco):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.oco_solar_model is None:
        raise FileNotFoundError(
            "OCO replay requires --oco-solar-model pointing to l2_solar_model.h5"
        )
    if not args.oco_solar_model.exists():
        raise FileNotFoundError(args.oco_solar_model)
    if args.eof_treatment == "oco3-static" and not args.oco3_eof_file.exists():
        raise FileNotFoundError(args.oco3_eof_file)
    from py2sess.optical.solar_reference import OcoSolarModel

    oco_solar_model = OcoSolarModel.from_hdf(args.oco_solar_model)

    selected = _load_selected_cases(args.case_dir, args.count)
    absco = {
        "o2": AbscoTable.open(args.o2_absco, ABSCO_GAS_DATASET["o2"]),
        "co2": AbscoTable.open(args.co2_absco, ABSCO_GAS_DATASET["co2"]),
        "h2o": AbscoTable.open(args.h2o_absco, ABSCO_GAS_DATASET["h2o"]),
    }

    summary_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    eof_context = (
        h5py.File(args.oco3_eof_file, "r")
        if args.eof_treatment == "oco3-static"
        else nullcontext(None)
    )
    with (
        h5py.File(l1b_path, "r") as l1b,
        h5py.File(l2std_path, "r") as std,
        h5py.File(l2dia_path, "r") as dia,
        eof_context as eof_static,
    ):
        l1b_sounding_id = l1b["SoundingGeometry/sounding_id"][...]
        dia_sounding_id = dia["RetrievalHeader/sounding_id"][...]
        counts = dia["SpectralParameters/num_colors_per_band"]
        sample_indexes = dia["SpectralParameters/sample_indexes"]
        wavelength = dia["SpectralParameters/wavelength"]
        measured = dia["SpectralParameters/measured_radiance"]
        o2_scale = std["Metadata/AbscoO2Scale"][...].astype(float)
        co2_scale = std["Metadata/AbscoCO2Scale"][...].astype(float)
        h2o_scale = std["Metadata/AbscoH2OScale"][...].astype(float)
        diagnostic_gas_tau_scale = {
            "o2": float(args.diagnostic_gas_tau_scale_o2),
            "wco2": float(args.diagnostic_gas_tau_scale_wco2),
            "sco2": float(args.diagnostic_gas_tau_scale_sco2),
        }

        for case in selected:
            index = int(case["retrieval_index"])
            sid = int(case["sounding_id"])
            case = _attach_l2_brdf_parameters(case, std, index)
            l1b_pos = np.argwhere(l1b_sounding_id == sid)
            if l1b_pos.size == 0:
                raise ValueError(f"sounding {sid} not found in L1B")
            frame, footprint = map(int, l1b_pos[0])
            dia_index = int(np.argwhere(dia_sounding_id == sid)[0, 0])
            if dia_index != index:
                raise ValueError(f"L2Std/L2Dia retrieval index mismatch for {sid}")
            state = _state_for_retrieval(
                std,
                index,
                layer_pressure_method=args.diagnostic_layer_pressure_method,
                surface_pressure_offset_hpa=args.diagnostic_surface_pressure_offset_hpa,
                surface_pressure_column_mode=args.diagnostic_surface_pressure_column_mode,
            )
            relative_velocity = float(std["RetrievalGeometry/retrieval_relative_velocity"][index])
            solar_relative_velocity = float(
                std["RetrievalGeometry/retrieval_solar_relative_velocity"][index]
            )
            solar_distance_m = float(std["RetrievalGeometry/retrieval_solar_distance"][index])
            band_slices = _band_slices(counts[index])
            for band in BANDS:
                selected_colors = _sample_detector_colors(
                    band_slices[band],
                    args.max_colors_per_band,
                )
                sample = sample_indexes[index, selected_colors].astype(int)
                l1b_sample = sample - 1
                band_index = BAND_INDEX[band]
                solar_azimuth = float(
                    l1b["FootprintGeometry/footprint_solar_azimuth"][frame, footprint, band_index]
                )
                view_azimuth = float(
                    l1b["FootprintGeometry/footprint_azimuth"][frame, footprint, band_index]
                )
                surface_angles = np.array(
                    [
                        float(
                            l1b["FootprintGeometry/footprint_solar_zenith"][
                                frame, footprint, band_index
                            ]
                        ),
                        float(
                            l1b["FootprintGeometry/footprint_zenith"][frame, footprint, band_index]
                        ),
                        _surface_relative_azimuth(solar_azimuth, view_azimuth),
                    ],
                    dtype=float,
                )
                rt_angles = np.array(
                    [
                        surface_angles[0],
                        surface_angles[1],
                        _rt_relative_azimuth(solar_azimuth, view_azimuth),
                    ],
                    dtype=float,
                )
                stokes_coefficients = _l1b_stokes_coefficients(
                    l1b, frame=frame, footprint=footprint, band_index=band_index
                )
                stokes_projection = _stokes_projection(stokes_coefficients, args.stokes_projection)
                fo_direct_brf_factor = SOLAR_OBS_DIRECT_BRF_TO_OCO_BRF
                center_wavelength = wavelength[index, selected_colors].astype(float)
                delta = l1b["InstrumentHeader/ils_delta_lambda"][
                    band_index, footprint, l1b_sample, :
                ].astype(float)
                response = l1b["InstrumentHeader/ils_relative_response"][
                    band_index, footprint, l1b_sample, :
                ].astype(float)
                eval_wavelength, response_flat, detector_id = _build_ils_eval_grid(
                    center_wavelength_um=center_wavelength,
                    delta_lambda_um=delta,
                    response=response,
                    grid_spacing_cm_inv=args.ils_grid_spacing_cm_1,
                )
                valid = (
                    np.isfinite(eval_wavelength)
                    & np.isfinite(response_flat)
                    & (response_flat > 0.0)
                )
                if not np.all(valid):
                    eval_wavelength = eval_wavelength[valid]
                    response_flat = response_flat[valid]
                    detector_id = detector_id[valid]
                gas_wavelength, doppler_velocity = _gas_lookup_wavelength_in_atmosphere_frame(
                    wavelength_um=eval_wavelength,
                    gas_doppler=args.gas_doppler,
                    relative_velocity_m_s=relative_velocity,
                )

                important_gas_pressure = (
                    np.asarray(state["met_pressure_pa"], dtype=float)
                    if args.diagnostic_gas_integration == "simpson10-metgrid"
                    else None
                )
                if args.diagnostic_gas_integration in {"simpson10", "simpson10-metgrid"}:
                    x_h2o = _column_weighted_absco_cross_section_cm2(
                        absco=absco["h2o"],
                        wavelength_um=gas_wavelength,
                        pressure_levels_pa=np.asarray(state["pressure_pa"], dtype=float),
                        important_pressure_levels_pa=important_gas_pressure,
                        temperature_pressure_levels_pa=np.asarray(
                            state["met_pressure_pa"], dtype=float
                        ),
                        temperature_levels_k=np.asarray(state["met_temperature_k"], dtype=float),
                        h2o_vmr_pressure_levels_pa=np.asarray(
                            state["met_pressure_pa"], dtype=float
                        ),
                        h2o_vmr_levels=np.asarray(state["met_h2o_vmr"], dtype=float),
                        species_vmr_pressure_levels_pa=np.asarray(
                            state["met_pressure_pa"], dtype=float
                        ),
                        species_vmr_levels=np.asarray(state["met_h2o_vmr"], dtype=float),
                    )
                    if band == "o2":
                        x_o2 = _column_weighted_absco_cross_section_cm2(
                            absco=absco["o2"],
                            wavelength_um=gas_wavelength,
                            pressure_levels_pa=np.asarray(state["pressure_pa"], dtype=float),
                            important_pressure_levels_pa=important_gas_pressure,
                            temperature_pressure_levels_pa=np.asarray(
                                state["met_pressure_pa"], dtype=float
                            ),
                            temperature_levels_k=np.asarray(
                                state["met_temperature_k"], dtype=float
                            ),
                            h2o_vmr_pressure_levels_pa=np.asarray(
                                state["met_pressure_pa"], dtype=float
                            ),
                            h2o_vmr_levels=np.asarray(state["met_h2o_vmr"], dtype=float),
                        )
                        x_co2 = 0.0
                    else:
                        x_o2 = 0.0
                        x_co2 = _column_weighted_absco_cross_section_cm2(
                            absco=absco["co2"],
                            wavelength_um=gas_wavelength,
                            pressure_levels_pa=np.asarray(state["pressure_pa"], dtype=float),
                            important_pressure_levels_pa=important_gas_pressure,
                            temperature_pressure_levels_pa=np.asarray(
                                state["met_pressure_pa"], dtype=float
                            ),
                            temperature_levels_k=np.asarray(
                                state["met_temperature_k"], dtype=float
                            ),
                            h2o_vmr_pressure_levels_pa=np.asarray(
                                state["met_pressure_pa"], dtype=float
                            ),
                            h2o_vmr_levels=np.asarray(state["met_h2o_vmr"], dtype=float),
                            species_vmr_levels=np.asarray(state["co2_vmr"], dtype=float),
                        )
                else:
                    x_o2 = (
                        absco["o2"].cross_section_cm2(
                            wavelength_um=gas_wavelength,
                            pressure_pa=np.asarray(state["layer_pressure_pa"], dtype=float),
                            temperature_k=np.asarray(state["layer_temperature_k"], dtype=float),
                            h2o_vmr=np.asarray(state["layer_h2o_vmr"], dtype=float),
                        )
                        if band == "o2"
                        else 0.0
                    )
                    x_co2 = (
                        absco["co2"].cross_section_cm2(
                            wavelength_um=gas_wavelength,
                            pressure_pa=np.asarray(state["layer_pressure_pa"], dtype=float),
                            temperature_k=np.asarray(state["layer_temperature_k"], dtype=float),
                            h2o_vmr=np.asarray(state["layer_h2o_vmr"], dtype=float),
                        )
                        if band != "o2"
                        else 0.0
                    )
                    x_h2o = absco["h2o"].cross_section_cm2(
                        wavelength_um=gas_wavelength,
                        pressure_pa=np.asarray(state["layer_pressure_pa"], dtype=float),
                        temperature_k=np.asarray(state["layer_temperature_k"], dtype=float),
                        h2o_vmr=np.asarray(state["layer_h2o_vmr"], dtype=float),
                    )
                x_h2o = x_h2o * h2o_scale[band_index]
                if band == "o2":
                    x_o2 = x_o2 * o2_scale[band_index] * diagnostic_gas_tau_scale[band]
                else:
                    x_co2 = x_co2 * co2_scale[band_index] * diagnostic_gas_tau_scale[band]
                gas_tau = x_h2o * np.asarray(state["h2o_col_cm2"], dtype=float)[np.newaxis, :]
                o2_column_tau = np.zeros(eval_wavelength.shape, dtype=float)
                if band == "o2":
                    o2_column_tau = np.sum(
                        x_o2 * np.asarray(state["o2_col_cm2"], dtype=float)[np.newaxis, :],
                        axis=1,
                    )
                    gas_tau = (
                        gas_tau + x_o2 * np.asarray(state["o2_col_cm2"], dtype=float)[np.newaxis, :]
                    )
                else:
                    gas_tau = (
                        gas_tau
                        + x_co2 * np.asarray(state["co2_col_cm2"], dtype=float)[np.newaxis, :]
                    )

                surface_model_family = "land-rpv"
                surface_angular_model = args.surface_angular
                surface_wind_speed = math.nan
                surface_refractive_index = math.nan
                surface_lambertian_albedo_l2 = math.nan
                surface_coxmunk_direct_brf = math.nan
                surface_coxmunk_stokes_i = math.nan
                surface_coxmunk_stokes_scale = math.nan
                surface_coxmunk_direct_scale = math.nan
                surface_coxmunk_fourier_scale = math.nan
                surface_l2_weight = math.nan
                surface_l2_weight_at_reference = math.nan
                brdf = None
                if _is_ocean_surface(case):
                    surface_model_family = "ocean-coxmunk-lambertian"
                    surface_angular_model = "coxmunk-lambertian"
                    brdf, reflectance, ocean_surface = _oco_coxmunk_lambertian_brdf(
                        case=case,
                        band=band,
                        wavelength_um=eval_wavelength,
                        surface_spectrum=args.surface_spectrum,
                        angles=rt_angles,
                        stream_value=args.stream_value,
                        brdf_quadrature_streams=args.brdf_quadrature_streams,
                        stokes_projection=stokes_projection,
                        coxmunk_stokes_scope=args.ocean_coxmunk_stokes_scope,
                        fo_direct_brf_factor=fo_direct_brf_factor,
                    )
                    solver_albedo = np.zeros_like(reflectance)
                    surface_wind_speed = ocean_surface["wind_speed"]
                    surface_refractive_index = ocean_surface["refractive_index"]
                    surface_lambertian_albedo_l2 = ocean_surface["lambertian_albedo_reference"]
                    surface_coxmunk_direct_brf = ocean_surface["coxmunk_direct_brf"]
                    surface_coxmunk_stokes_i = ocean_surface["coxmunk_stokes_i"]
                    surface_coxmunk_stokes_scale = ocean_surface["coxmunk_stokes_scale"]
                    surface_coxmunk_direct_scale = ocean_surface["coxmunk_direct_scale"]
                    surface_coxmunk_fourier_scale = ocean_surface["coxmunk_fourier_scale"]
                else:
                    reflectance = _land_surface_reflectance(
                        case=case,
                        band=band,
                        wavelength_um=eval_wavelength,
                        surface_spectrum=args.surface_spectrum,
                        surface_angular=args.surface_angular,
                        angles=surface_angles,
                    )
                    solver_albedo = reflectance
                if args.surface_angular == "rpv-brdf" and not _is_ocean_surface(case):
                    brdf, reflectance = _oco_rpv_brdf(
                        case=case,
                        band=band,
                        wavelength_um=eval_wavelength,
                        surface_spectrum=args.surface_spectrum,
                        angles=surface_angles,
                        stream_value=args.stream_value,
                        brdf_quadrature_streams=args.brdf_quadrature_streams,
                        fo_direct_brf_factor=fo_direct_brf_factor,
                    )
                    solver_albedo = np.zeros_like(reflectance)
                    surface_l2_weight = float(case[f"brdf_weight_{band}"])
                aerosol = _posterior_aerosol_inputs(
                    std=std,
                    oco_l2fp_property_file=args.oco_l2fp_aerosol_file,
                    index=index,
                    state=state,
                    wavelength_um=eval_wavelength,
                    treatment=args.aerosol_treatment,
                    aerosol_type_filter=aerosol_type_filter,
                    aerosol_scale=args.diagnostic_aerosol_scale,
                )
                rpv_fields = (
                    f"brdf_hotspot_parameter_{band}",
                    f"brdf_asymmetry_parameter_{band}",
                    f"brdf_anisotropy_parameter_{band}",
                )
                rpv_kernel = (
                    _oco_rpv_kernel(case=case, band=band, angles=surface_angles)
                    if (not _is_ocean_surface(case) and all(field in case for field in rpv_fields))
                    else math.nan
                )
                if not _is_ocean_surface(case) and f"brdf_weight_{band}" in case:
                    surface_l2_weight = float(case[f"brdf_weight_{band}"])
                solar_doppler_velocity = 0.0
                solar_wavelength, solar_doppler_velocity = _solar_reference_lookup_wavelength(
                    wavelength_um=eval_wavelength,
                    solar_doppler=args.solar_doppler,
                    solar_relative_velocity_m_s=solar_relative_velocity,
                    los_relative_velocity_m_s=relative_velocity,
                )
                solar_reference_factor = oco_solar_model.energy_irradiance_w_m2_um(
                    solar_wavelength,
                    band_index=band_index + 1,
                    observer_distance_m=solar_distance_m,
                    energy_wavelength_um=eval_wavelength,
                )
                solar_irradiance_reference = float(
                    oco_solar_model.energy_irradiance_w_m2_um(
                        np.array([BAND_REFERENCE_WAVELENGTH_UM[band]]),
                        band_index=band_index + 1,
                        observer_distance_m=solar_distance_m,
                    )[0]
                )
                obs = measured[index, selected_colors].astype(float)
                rt_context = _prepare_py2sess_rt_context(
                    wavelength_um=eval_wavelength,
                    state=state,
                    gas_tau=gas_tau,
                    angles=rt_angles,
                    aerosol=aerosol,
                    solar_reference_factor=solar_reference_factor,
                    stokes_coefficients=stokes_coefficients,
                    stokes_projection_mode=args.stokes_projection,
                    polarization_correction=args.polarization_correction,
                    polarization_sign=args.polarization_sign,
                    polarization_diffuse_azimuths=args.polarization_diffuse_azimuths,
                    stream_value=args.stream_value,
                )
                polarization_correction_cache = None
                if args.polarization_correction in {"none", "rayleigh-fo", "rayleigh-aerosol-fo"}:
                    polarization_correction_cache = _compute_py2sess_polarization_correction(
                        context=rt_context,
                        diffuse_albedo=reflectance,
                    )
                py_run = _run_py2sess_prepared(
                    context=rt_context,
                    albedo=solver_albedo,
                    diffuse_albedo=reflectance,
                    brdf=brdf,
                    polarization_correction_cache=polarization_correction_cache,
                )
                py_detector = _detector_average(
                    py_run.radiance,
                    detector_id=detector_id,
                    response_flat=response_flat,
                    n_detector=center_wavelength.size,
                )
                py_scalar_detector = _detector_average(
                    py_run.scalar_radiance,
                    detector_id=detector_id,
                    response_flat=response_flat,
                    n_detector=center_wavelength.size,
                )
                py_polarization_detector = _detector_average(
                    py_run.polarization_correction,
                    detector_id=detector_id,
                    response_flat=response_flat,
                    n_detector=center_wavelength.size,
                )

                fluorescence_detector_photon = np.zeros(center_wavelength.shape, dtype=float)
                if args.fluorescence_treatment == "l2-posterior" and band == "o2":
                    fluorescence_eval = _fluorescence_photon_radiance(
                        wavelength_um=gas_wavelength,
                        o2_column_tau=o2_column_tau,
                        view_zenith_deg=surface_angles[1],
                        stokes_coefficients=stokes_coefficients,
                        fluorescence_at_reference=float(
                            std["RetrievalResults/fluorescence_at_reference"][index]
                        ),
                        fluorescence_slope=float(std["RetrievalResults/fluorescence_slope"][index]),
                    )
                    fluorescence_detector_photon = _detector_average(
                        fluorescence_eval,
                        detector_id=detector_id,
                        response_flat=response_flat,
                        n_detector=center_wavelength.size,
                    )
                fluorescence_energy = _photon_to_energy_spectral_radiance(
                    fluorescence_detector_photon,
                    center_wavelength,
                )

                eof = _eof_detector_correction(
                    l1b=l1b,
                    eof_static=eof_static,
                    std=std,
                    index=index,
                    frame=frame,
                    footprint=footprint,
                    band=band,
                    sample_indexes=l1b_sample,
                    surface_type=case.get("surface_type", ""),
                    treatment=args.eof_treatment,
                )
                eof_energy = (
                    _photon_to_energy_spectral_radiance(eof.values, center_wavelength)
                    if args.eof_treatment == "oco3-static"
                    else np.zeros(center_wavelength.shape, dtype=float)
                )
                continuum_signal = float(std[OCO_CONTINUUM_FIELD[band]][index])
                if not np.isfinite(continuum_signal) or continuum_signal <= 0.0:
                    raise ValueError(f"bad OCO continuum signal for sounding {sid} band {band}")
                continuum_signal_energy = float(
                    _photon_to_energy_spectral_radiance(
                        continuum_signal,
                        BAND_REFERENCE_WAVELENGTH_UM[band],
                    )
                )
                fixed_detector_energy = fluorescence_energy + eof_energy
                py_detector = py_detector + fixed_detector_energy
                py_scalar_detector = py_detector - py_polarization_detector
                obs_energy = _photon_to_energy_spectral_radiance(obs, center_wavelength)
                unadjusted_stats = _continuum_residual_stats(
                    py_detector,
                    obs_energy,
                    continuum_signal_energy,
                )
                surface_brdf_retrieval = SurfaceBrdfRetrieval(
                    scale=1.0,
                    tilt=0.0,
                    n_points=0,
                    fit_rmse_percent=math.nan,
                    status="not_requested",
                    iterations=0,
                )
                tilt_axis = _surface_brdf_tilt_axis(wavelength_um=eval_wavelength, band=band)

                def _run_scaled_surface_detector(
                    surface_scale: float,
                    surface_tilt: float = 0.0,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    surface_factor = float(surface_scale) + float(surface_tilt) * tilt_axis
                    if np.any(~np.isfinite(surface_factor)) or np.any(surface_factor <= 0.0):
                        raise ValueError("surface BRDF scale/tilt produced non-positive weights")
                    scaled_reflectance = surface_factor * reflectance
                    scaled_run = _run_py2sess_prepared(
                        context=rt_context,
                        albedo=solver_albedo
                        if brdf is not None
                        else surface_factor * solver_albedo,
                        diffuse_albedo=scaled_reflectance,
                        brdf=_scale_brdf(brdf, surface_factor),
                        polarization_correction_cache=polarization_correction_cache,
                    )
                    detector = _detector_average(
                        scaled_run.radiance,
                        detector_id=detector_id,
                        response_flat=response_flat,
                        n_detector=center_wavelength.size,
                    )
                    polarization_detector = _detector_average(
                        scaled_run.polarization_correction,
                        detector_id=detector_id,
                        response_flat=response_flat,
                        n_detector=center_wavelength.size,
                    )
                    detector = detector + fixed_detector_energy
                    return detector, detector - polarization_detector, polarization_detector

                if args.surface_brdf_retrieval in {
                    "continuum-linearized",
                    "continuum-linearized-slope",
                }:
                    continuum_mask = _continuum_fit_mask(
                        obs_energy,
                        fraction=args.surface_brdf_continuum_fraction,
                        min_points=args.surface_brdf_continuum_min_points,
                    )
                    current_scale = 1.0
                    current_tilt = 0.0
                    for surface_iteration in range(args.surface_brdf_max_iterations):
                        probe_scale = current_scale + args.surface_brdf_probe_step
                        probe_detector, _, _ = _run_scaled_surface_detector(
                            probe_scale,
                            current_tilt,
                        )
                        if args.surface_brdf_retrieval == "continuum-linearized":
                            candidate = _solve_linearized_surface_brdf_scale(
                                measured_radiance=obs_energy,
                                base_radiance=py_detector,
                                probe_radiance=probe_detector,
                                continuum_radiance=continuum_signal_energy,
                                continuum_mask=continuum_mask,
                                probe_scale=probe_scale,
                                current_scale=current_scale,
                                scale_min=args.surface_brdf_scale_min,
                                scale_max=args.surface_brdf_scale_max,
                                prior_sigma=args.surface_brdf_prior_sigma,
                            )
                        else:
                            probe_tilt = current_tilt + args.surface_brdf_probe_step
                            tilt_probe_detector, _, _ = _run_scaled_surface_detector(
                                current_scale,
                                probe_tilt,
                            )
                            candidate = _solve_linearized_surface_brdf_scale_and_tilt(
                                measured_radiance=obs_energy,
                                base_radiance=py_detector,
                                scale_probe_radiance=probe_detector,
                                tilt_probe_radiance=tilt_probe_detector,
                                continuum_radiance=continuum_signal_energy,
                                continuum_mask=continuum_mask,
                                probe_scale=probe_scale,
                                probe_tilt=probe_tilt,
                                current_scale=current_scale,
                                current_tilt=current_tilt,
                                scale_min=args.surface_brdf_scale_min,
                                scale_max=args.surface_brdf_scale_max,
                                tilt_min=args.surface_brdf_tilt_min,
                                tilt_max=args.surface_brdf_tilt_max,
                            )
                        if not candidate.status.startswith("continuum_linearized"):
                            if surface_iteration == 0:
                                surface_brdf_retrieval = SurfaceBrdfRetrieval(
                                    candidate.scale,
                                    candidate.tilt,
                                    candidate.n_points,
                                    candidate.fit_rmse_percent,
                                    candidate.status,
                                    0,
                                )
                            break
                        current_scale = candidate.scale
                        current_tilt = candidate.tilt
                        py_detector, py_scalar_detector, py_polarization_detector = (
                            _run_scaled_surface_detector(current_scale, current_tilt)
                        )
                        surface_brdf_retrieval = SurfaceBrdfRetrieval(
                            candidate.scale,
                            candidate.tilt,
                            candidate.n_points,
                            candidate.fit_rmse_percent,
                            candidate.status,
                            surface_iteration + 1,
                        )
                final_surface_factor = (
                    surface_brdf_retrieval.scale + surface_brdf_retrieval.tilt * tilt_axis
                )
                final_reflectance = final_surface_factor * reflectance
                if np.isfinite(surface_l2_weight):
                    surface_l2_weight_at_reference = (
                        surface_l2_weight * surface_brdf_retrieval.scale
                    )
                is_ocean_surface = _is_ocean_surface(case)
                surface_adjustment_target = "none"
                surface_adjustment_components = "none"
                if args.surface_brdf_retrieval != "none":
                    if is_ocean_surface:
                        surface_adjustment_target = "ocean_coxmunk_lambertian_effective_continuum"
                        surface_adjustment_components = (
                            "coxmunk_fourier_terms;lambertian_albedo;diffuse_albedo"
                        )
                    else:
                        surface_adjustment_target = "land_rpv_brdf_weight"
                        surface_adjustment_components = "rpv_brdf_weight;diffuse_albedo"
                land_brdf_weight_scale = (
                    surface_brdf_retrieval.scale
                    if (not is_ocean_surface and np.isfinite(surface_l2_weight))
                    else math.nan
                )
                land_brdf_weight_tilt = (
                    surface_brdf_retrieval.tilt
                    if (not is_ocean_surface and np.isfinite(surface_l2_weight))
                    else math.nan
                )
                ocean_surface_continuum_scale = (
                    surface_brdf_retrieval.scale if is_ocean_surface else math.nan
                )
                ocean_surface_continuum_tilt = (
                    surface_brdf_retrieval.tilt if is_ocean_surface else math.nan
                )
                land_brdf_weight_fit_points = (
                    surface_brdf_retrieval.n_points if not is_ocean_surface else 0
                )
                land_brdf_weight_fit_rmse = (
                    surface_brdf_retrieval.fit_rmse_percent if not is_ocean_surface else math.nan
                )
                land_brdf_weight_status = (
                    surface_brdf_retrieval.status
                    if not is_ocean_surface
                    else "not_land_brdf_weight"
                )
                ocean_surface_continuum_fit_points = (
                    surface_brdf_retrieval.n_points if is_ocean_surface else 0
                )
                ocean_surface_continuum_fit_rmse = (
                    surface_brdf_retrieval.fit_rmse_percent if is_ocean_surface else math.nan
                )

                py_unit_continuum_signal = _sample_continuum_level(py_detector)
                if not np.isfinite(py_unit_continuum_signal) or py_unit_continuum_signal <= 0.0:
                    raise ValueError(f"bad py2sess continuum signal for sounding {sid} band {band}")
                py2sess_posthoc_scale = 1.0
                py2sess_effective_fbeam = solar_irradiance_reference
                py_energy = py_detector * py2sess_posthoc_scale
                py_scalar_energy = py_scalar_detector * py2sess_posthoc_scale
                py_polarization_energy = py_polarization_detector * py2sess_posthoc_scale
                residual_continuum_percent = (
                    100.0 * (py_energy - obs_energy) / continuum_signal_energy
                )
                polarization_median_continuum_percent = (
                    100.0 * float(np.nanmedian(py_polarization_energy)) / continuum_signal_energy
                )
                polarization_max_abs_continuum_percent = (
                    100.0
                    * float(np.nanmax(np.abs(py_polarization_energy)))
                    / continuum_signal_energy
                )
                fluorescence_median_continuum_percent = (
                    100.0 * float(np.nanmedian(fluorescence_energy)) / continuum_signal_energy
                )
                eof_median_continuum_percent = (
                    100.0 * float(np.nanmedian(eof_energy)) / continuum_signal_energy
                )
                stats = _continuum_residual_stats(
                    py_energy,
                    obs_energy,
                    continuum_signal_energy,
                )
                summary_rows.append(
                    {
                        "sounding_id": sid,
                        "retrieval_index": index,
                        "band": band,
                        "n_detector_colors": center_wavelength.size,
                        "n_ils_eval_wavelengths": eval_wavelength.size,
                        "max_colors_per_band": args.max_colors_per_band,
                        "ils_grid_spacing_cm_1": f"{args.ils_grid_spacing_cm_1:.9g}",
                        "relative_azimuth_deg": f"{surface_angles[2]:.6f}",
                        "rt_relative_azimuth_deg": f"{rt_angles[2]:.6f}",
                        "surface_model_family": surface_model_family,
                        "surface_reflectance_model": args.surface_spectrum,
                        "surface_angular_model": surface_angular_model,
                        "surface_rpv_kernel": f"{rpv_kernel:.9g}",
                        "surface_reflectance_used": (
                            f"{float(np.nanmedian(final_reflectance)):.9g}"
                        ),
                        "surface_reflectance_min": f"{float(np.nanmin(final_reflectance)):.9g}",
                        "surface_reflectance_max": f"{float(np.nanmax(final_reflectance)):.9g}",
                        "surface_ocean_wind_speed_m_s": f"{surface_wind_speed:.9g}",
                        "surface_ocean_refractive_index": f"{surface_refractive_index:.9g}",
                        "surface_lambertian_albedo_l2": f"{surface_lambertian_albedo_l2:.9g}",
                        "surface_coxmunk_direct_brf": f"{surface_coxmunk_direct_brf:.9g}",
                        "surface_coxmunk_stokes_i": f"{surface_coxmunk_stokes_i:.9g}",
                        "surface_coxmunk_stokes_scale": f"{surface_coxmunk_stokes_scale:.9g}",
                        "surface_coxmunk_stokes_scope": args.ocean_coxmunk_stokes_scope,
                        "surface_coxmunk_direct_scale": f"{surface_coxmunk_direct_scale:.9g}",
                        "surface_coxmunk_fourier_scale": f"{surface_coxmunk_fourier_scale:.9g}",
                        "surface_adjustment_target": surface_adjustment_target,
                        "surface_adjustment_components": surface_adjustment_components,
                        "surface_adjustment_method": args.surface_brdf_retrieval,
                        "surface_adjustment_status": surface_brdf_retrieval.status,
                        "surface_adjustment_iterations": surface_brdf_retrieval.iterations,
                        "surface_adjustment_scale": f"{surface_brdf_retrieval.scale:.9g}",
                        "surface_adjustment_tilt": f"{surface_brdf_retrieval.tilt:.9g}",
                        "surface_adjustment_fit_points": surface_brdf_retrieval.n_points,
                        "surface_adjustment_fit_rmse_percent": (
                            f"{surface_brdf_retrieval.fit_rmse_percent:.9g}"
                        ),
                        "surface_adjustment_scale_bounds": (
                            f"{args.surface_brdf_scale_min:.9g};{args.surface_brdf_scale_max:.9g}"
                        ),
                        "surface_unadjusted_continuum_referenced_bias_percent": (
                            f"{unadjusted_stats['continuum_referenced_bias_percent']:.9g}"
                        ),
                        "surface_unadjusted_continuum_referenced_rmse_percent": (
                            f"{unadjusted_stats['continuum_referenced_rmse_percent']:.9g}"
                        ),
                        "surface_unadjusted_continuum_referenced_max_abs_percent": (
                            f"{unadjusted_stats['continuum_referenced_max_abs_percent']:.9g}"
                        ),
                        "surface_brdf_retrieval": args.surface_brdf_retrieval,
                        "surface_brdf_retrieval_status": land_brdf_weight_status,
                        "surface_brdf_weight_l2": f"{surface_l2_weight:.9g}",
                        "surface_brdf_weight_scale": f"{land_brdf_weight_scale:.9g}",
                        "surface_brdf_weight_tilt": f"{land_brdf_weight_tilt:.9g}",
                        "surface_brdf_weight_at_reference": f"{surface_l2_weight_at_reference:.9g}",
                        "land_brdf_weight_l2": f"{surface_l2_weight:.9g}",
                        "land_brdf_weight_scale": f"{land_brdf_weight_scale:.9g}",
                        "land_brdf_weight_tilt": f"{land_brdf_weight_tilt:.9g}",
                        "land_brdf_weight_at_reference": f"{surface_l2_weight_at_reference:.9g}",
                        "land_brdf_weight_fit_points": land_brdf_weight_fit_points,
                        "land_brdf_weight_fit_rmse_percent": (f"{land_brdf_weight_fit_rmse:.9g}"),
                        "land_brdf_weight_status": land_brdf_weight_status,
                        "ocean_surface_continuum_status": (
                            surface_brdf_retrieval.status if is_ocean_surface else "not_ocean"
                        ),
                        "ocean_surface_continuum_scale": (f"{ocean_surface_continuum_scale:.9g}"),
                        "ocean_surface_continuum_tilt": (f"{ocean_surface_continuum_tilt:.9g}"),
                        "ocean_surface_continuum_fit_points": (ocean_surface_continuum_fit_points),
                        "ocean_surface_continuum_fit_rmse_percent": (
                            f"{ocean_surface_continuum_fit_rmse:.9g}"
                        ),
                        "ocean_surface_unadjusted_bias_percent": (
                            f"{unadjusted_stats['continuum_referenced_bias_percent']:.9g}"
                            if is_ocean_surface
                            else "nan"
                        ),
                        "ocean_surface_unadjusted_rmse_percent": (
                            f"{unadjusted_stats['continuum_referenced_rmse_percent']:.9g}"
                            if is_ocean_surface
                            else "nan"
                        ),
                        "surface_brdf_fit_points": land_brdf_weight_fit_points,
                        "surface_brdf_fit_rmse_percent": (f"{land_brdf_weight_fit_rmse:.9g}"),
                        "surface_brdf_scale_bounds": (
                            f"{args.surface_brdf_scale_min:.9g};{args.surface_brdf_scale_max:.9g}"
                        ),
                        "stream_value": f"{args.stream_value:.9g}",
                        "brdf_quadrature_streams": args.brdf_quadrature_streams,
                        "absco_o2_scale": f"{o2_scale[band_index]:.9g}",
                        "absco_co2_scale": f"{co2_scale[band_index]:.9g}",
                        "absco_h2o_scale": f"{h2o_scale[band_index]:.9g}",
                        "diagnostic_gas_tau_scale": f"{diagnostic_gas_tau_scale[band]:.9g}",
                        "diagnostic_layer_pressure_method": (args.diagnostic_layer_pressure_method),
                        "diagnostic_surface_pressure_offset_hpa": (
                            f"{args.diagnostic_surface_pressure_offset_hpa:.9g}"
                        ),
                        "surface_pressure_original_hpa": (
                            f"{float(state['surface_pressure_original_hpa']):.9g}"
                        ),
                        "surface_pressure_used_hpa": (
                            f"{float(state['surface_pressure_used_hpa']):.9g}"
                        ),
                        "diagnostic_surface_pressure_column_mode": (
                            str(state["surface_pressure_column_mode"])
                        ),
                        "diagnostic_gas_integration": args.diagnostic_gas_integration,
                        "gas_doppler": args.gas_doppler,
                        "relative_velocity_m_s": f"{relative_velocity:.9g}",
                        "doppler_lookup_velocity_m_s": f"{doppler_velocity:.9g}",
                        "solar_doppler": args.solar_doppler,
                        "solar_relative_velocity_m_s": f"{solar_relative_velocity:.9g}",
                        "solar_doppler_lookup_velocity_m_s": (f"{solar_doppler_velocity:.9g}"),
                        "oco_solar_model": args.oco_solar_model.name,
                        "solar_distance_m": f"{solar_distance_m:.9g}",
                        "solar_irradiance_reference_w_m2_um": (
                            f"{solar_irradiance_reference:.9e}"
                            if np.isfinite(solar_irradiance_reference)
                            else "not_applied"
                        ),
                        "aerosol_treatment": args.aerosol_treatment,
                        "aerosol_type_filter": aerosol_type_filter_label,
                        "aerosol_scale": f"{args.diagnostic_aerosol_scale:.9g}",
                        "aerosol_total_aod_used": f"{aerosol.total_aod_used:.9g}",
                        "aerosol_phase_model": aerosol.phase_model,
                        "polarization_correction": args.polarization_correction,
                        "stokes_projection": args.stokes_projection,
                        "stokes_projection_description": stokes_projection.description,
                        "stokes_scalar_factor": f"{stokes_projection.scalar_factor:.9g}",
                        "stokes_analyzer_q": f"{stokes_projection.analyzer_q:.9g}",
                        "stokes_analyzer_u": f"{stokes_projection.analyzer_u:.9g}",
                        "fo_direct_brf_factor": f"{fo_direct_brf_factor:.9g}",
                        "polarization_sign": args.polarization_sign,
                        "polarization_diffuse_azimuths": args.polarization_diffuse_azimuths,
                        "stokes_m11": f"{stokes_coefficients[0]:.9g}",
                        "stokes_m12": f"{stokes_coefficients[1]:.9g}",
                        "stokes_m13": f"{stokes_coefficients[2]:.9g}",
                        "stokes_m14": f"{stokes_coefficients[3]:.9g}",
                        "polarization_median_continuum_percent": (
                            f"{polarization_median_continuum_percent:.9g}"
                        ),
                        "polarization_max_abs_continuum_percent": (
                            f"{polarization_max_abs_continuum_percent:.9g}"
                        ),
                        "fluorescence_treatment": args.fluorescence_treatment,
                        "fluorescence_median_continuum_percent": (
                            f"{fluorescence_median_continuum_percent:.9g}"
                        ),
                        "eof_treatment": args.eof_treatment,
                        "eof_basis_model": eof.basis_model,
                        "eof_scales": ";".join(f"{value:.9g}" for value in eof.scale_values),
                        "eof_correction_min": f"{float(np.nanmin(eof.values)):.9g}",
                        "eof_correction_max": f"{float(np.nanmax(eof.values)):.9g}",
                        "eof_median_continuum_percent": f"{eof_median_continuum_percent:.9g}",
                        "oco_continuum_signal_source": OCO_CONTINUUM_FIELD[band],
                        "oco_continuum_signal_radiance_w_m2_sr_um": (
                            f"{continuum_signal_energy:.9e}"
                        ),
                        "oco_continuum_reference_wavelength_um": (
                            f"{BAND_REFERENCE_WAVELENGTH_UM[band]:.9g}"
                        ),
                        "py2sess_unit_continuum_signal": f"{py_unit_continuum_signal:.9e}",
                        "py2sess_unit_continuum_signal_source": (
                            "98-99 percentile window of sampled detector signal"
                        ),
                        "py2sess_effective_fbeam_w_m2_um": f"{py2sess_effective_fbeam:.9e}",
                        "py2sess_posthoc_scale": f"{py2sess_posthoc_scale:.9e}",
                        **{key: f"{value:.9g}" for key, value in stats.items()},
                    }
                )
                for local, packed in enumerate(selected_colors):
                    spectrum_rows.append(
                        {
                            "sounding_id": sid,
                            "retrieval_index": index,
                            "band": band,
                            "packed_color_index": int(packed),
                            "sample_index": int(sample_indexes[index, packed]),
                            "wavelength_um": f"{float(wavelength[index, packed]):.9f}",
                            "measured_radiance_w_m2_sr_um": f"{obs_energy[local]:.9e}",
                            "py2sess_scalar_radiance_w_m2_sr_um": (
                                f"{py_scalar_energy[local]:.9e}"
                            ),
                            "py2sess_polarization_correction_w_m2_sr_um": (
                                f"{py_polarization_energy[local]:.9e}"
                            ),
                            "py2sess_radiance_w_m2_sr_um": f"{py_energy[local]:.9e}",
                            "py2sess_fluorescence_w_m2_sr_um": (
                                f"{fluorescence_energy[local]:.9e}"
                            ),
                            "py2sess_eof_correction_w_m2_sr_um": f"{eof_energy[local]:.9e}",
                            "solar_irradiance_reference_w_m2_um": (
                                f"{solar_irradiance_reference:.9e}"
                                if np.isfinite(solar_irradiance_reference)
                                else "not_applied"
                            ),
                            "surface_adjustment_target": surface_adjustment_target,
                            "surface_adjustment_status": surface_brdf_retrieval.status,
                            "surface_adjustment_iterations": surface_brdf_retrieval.iterations,
                            "surface_adjustment_scale": f"{surface_brdf_retrieval.scale:.9e}",
                            "surface_adjustment_tilt": f"{surface_brdf_retrieval.tilt:.9e}",
                            "surface_brdf_weight_scale": f"{land_brdf_weight_scale:.9e}",
                            "surface_brdf_weight_tilt": f"{land_brdf_weight_tilt:.9e}",
                            "land_brdf_weight_scale": f"{land_brdf_weight_scale:.9e}",
                            "land_brdf_weight_tilt": f"{land_brdf_weight_tilt:.9e}",
                            "ocean_surface_continuum_scale": (
                                f"{ocean_surface_continuum_scale:.9e}"
                            ),
                            "ocean_surface_continuum_tilt": (f"{ocean_surface_continuum_tilt:.9e}"),
                            "oco_continuum_signal_radiance_w_m2_sr_um": (
                                f"{continuum_signal_energy:.9e}"
                            ),
                            "py2sess_unit_solver_signal": f"{py_detector[local]:.9e}",
                            "py2sess_effective_fbeam_w_m2_um": (f"{py2sess_effective_fbeam:.9e}"),
                            "py2sess_posthoc_scale": f"{py2sess_posthoc_scale:.9e}",
                            "py2sess_minus_measured_continuum_percent": (
                                f"{residual_continuum_percent[local]:.9e}"
                            ),
                        }
                    )

    summary_path = args.case_dir / "py2sess_replay_summary.csv"
    spectrum_path = args.case_dir / "py2sess_replay_spectrum.csv"
    plot_path = args.case_dir / "py2sess_replay_spectrum_and_relative_diff.png"
    _write_rows(
        summary_path,
        [
            "sounding_id",
            "retrieval_index",
            "band",
            "n_detector_colors",
            "n_ils_eval_wavelengths",
            "max_colors_per_band",
            "ils_grid_spacing_cm_1",
            "relative_azimuth_deg",
            "rt_relative_azimuth_deg",
            "surface_model_family",
            "surface_reflectance_model",
            "surface_angular_model",
            "surface_rpv_kernel",
            "surface_reflectance_used",
            "surface_reflectance_min",
            "surface_reflectance_max",
            "surface_ocean_wind_speed_m_s",
            "surface_ocean_refractive_index",
            "surface_lambertian_albedo_l2",
            "surface_coxmunk_direct_brf",
            "surface_coxmunk_stokes_i",
            "surface_coxmunk_stokes_scale",
            "surface_coxmunk_stokes_scope",
            "surface_coxmunk_direct_scale",
            "surface_coxmunk_fourier_scale",
            "surface_adjustment_target",
            "surface_adjustment_components",
            "surface_adjustment_method",
            "surface_adjustment_status",
            "surface_adjustment_iterations",
            "surface_adjustment_scale",
            "surface_adjustment_tilt",
            "surface_adjustment_fit_points",
            "surface_adjustment_fit_rmse_percent",
            "surface_adjustment_scale_bounds",
            "surface_unadjusted_continuum_referenced_bias_percent",
            "surface_unadjusted_continuum_referenced_rmse_percent",
            "surface_unadjusted_continuum_referenced_max_abs_percent",
            "surface_brdf_retrieval",
            "surface_brdf_retrieval_status",
            "surface_brdf_weight_l2",
            "surface_brdf_weight_scale",
            "surface_brdf_weight_tilt",
            "surface_brdf_weight_at_reference",
            "land_brdf_weight_l2",
            "land_brdf_weight_scale",
            "land_brdf_weight_tilt",
            "land_brdf_weight_at_reference",
            "land_brdf_weight_fit_points",
            "land_brdf_weight_fit_rmse_percent",
            "land_brdf_weight_status",
            "ocean_surface_continuum_status",
            "ocean_surface_continuum_scale",
            "ocean_surface_continuum_tilt",
            "ocean_surface_continuum_fit_points",
            "ocean_surface_continuum_fit_rmse_percent",
            "ocean_surface_unadjusted_bias_percent",
            "ocean_surface_unadjusted_rmse_percent",
            "surface_brdf_fit_points",
            "surface_brdf_fit_rmse_percent",
            "surface_brdf_scale_bounds",
            "stream_value",
            "brdf_quadrature_streams",
            "absco_o2_scale",
            "absco_co2_scale",
            "absco_h2o_scale",
            "diagnostic_gas_tau_scale",
            "diagnostic_layer_pressure_method",
            "diagnostic_surface_pressure_offset_hpa",
            "surface_pressure_original_hpa",
            "surface_pressure_used_hpa",
            "diagnostic_surface_pressure_column_mode",
            "diagnostic_gas_integration",
            "gas_doppler",
            "relative_velocity_m_s",
            "doppler_lookup_velocity_m_s",
            "solar_doppler",
            "solar_relative_velocity_m_s",
            "solar_doppler_lookup_velocity_m_s",
            "oco_solar_model",
            "solar_distance_m",
            "solar_irradiance_reference_w_m2_um",
            "aerosol_treatment",
            "aerosol_type_filter",
            "aerosol_scale",
            "aerosol_total_aod_used",
            "aerosol_phase_model",
            "polarization_correction",
            "stokes_projection",
            "stokes_projection_description",
            "stokes_scalar_factor",
            "stokes_analyzer_q",
            "stokes_analyzer_u",
            "fo_direct_brf_factor",
            "polarization_sign",
            "polarization_diffuse_azimuths",
            "stokes_m11",
            "stokes_m12",
            "stokes_m13",
            "stokes_m14",
            "polarization_median_continuum_percent",
            "polarization_max_abs_continuum_percent",
            "fluorescence_treatment",
            "fluorescence_median_continuum_percent",
            "eof_treatment",
            "eof_basis_model",
            "eof_scales",
            "eof_correction_min",
            "eof_correction_max",
            "eof_median_continuum_percent",
            "oco_continuum_signal_source",
            "oco_continuum_signal_radiance_w_m2_sr_um",
            "oco_continuum_reference_wavelength_um",
            "py2sess_unit_continuum_signal",
            "py2sess_unit_continuum_signal_source",
            "py2sess_effective_fbeam_w_m2_um",
            "py2sess_posthoc_scale",
            "continuum_referenced_bias_percent",
            "continuum_referenced_rmse_percent",
            "continuum_referenced_max_abs_percent",
            "corr",
        ],
        summary_rows,
    )
    _write_rows(
        spectrum_path,
        [
            "sounding_id",
            "retrieval_index",
            "band",
            "packed_color_index",
            "sample_index",
            "wavelength_um",
            "measured_radiance_w_m2_sr_um",
            "py2sess_scalar_radiance_w_m2_sr_um",
            "py2sess_polarization_correction_w_m2_sr_um",
            "py2sess_radiance_w_m2_sr_um",
            "py2sess_fluorescence_w_m2_sr_um",
            "py2sess_eof_correction_w_m2_sr_um",
            "solar_irradiance_reference_w_m2_um",
            "surface_adjustment_target",
            "surface_adjustment_status",
            "surface_adjustment_iterations",
            "surface_adjustment_scale",
            "surface_adjustment_tilt",
            "surface_brdf_weight_scale",
            "surface_brdf_weight_tilt",
            "land_brdf_weight_scale",
            "land_brdf_weight_tilt",
            "ocean_surface_continuum_scale",
            "ocean_surface_continuum_tilt",
            "oco_continuum_signal_radiance_w_m2_sr_um",
            "py2sess_unit_solver_signal",
            "py2sess_effective_fbeam_w_m2_um",
            "py2sess_posthoc_scale",
            "py2sess_minus_measured_continuum_percent",
        ],
        spectrum_rows,
    )
    if not args.skip_plot:
        _plot(plot_path, spectrum_rows)
    print(f"summary={summary_path}")
    print(f"spectrum={spectrum_path}")
    if not args.skip_plot:
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
