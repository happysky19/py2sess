#!/usr/bin/env python3
"""Replay OCO-3 three-band soundings with py2sess.

The driver uses official posterior state fields, OCO ABSCO tables, the OCO
solar model, L1B ILS sampling, and L1B instrument Stokes coefficients. It does
not fit pressure, spectroscopy, wavelength, gas columns, aerosol loading, or
surface brightness. OCO photon radiances and py2sess replay spectra are both
reported as energy spectral radiance.

The default polarization treatment uses the OCO L1B normalized-radiance
convention, L = I + (m12/m11) Q + (m13/m11) U. A raw detector projection,
L = m11 I + m12 Q + m13 U, is retained only as a convention check.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
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
DEFAULT_O2_ABSCO = ROOT.parent / "bayesrt" / "data" / "external" / "absco" / "o2_v52.hdf"
DEFAULT_H2O_ABSCO = ROOT.parent / "bayesrt" / "data" / "external" / "absco" / "h2o_v52.hdf"
DEFAULT_OCO_SOLAR_MODEL = (
    ROOT / "outputs" / "oco3_joint_official_downloads" / "solar" / "l2_solar_model.h5"
)
DEFAULT_OCO3_EOF_FILE = Path("/tmp/RtRetrievalFramework/input/oco/input/l2_oco3_eof.h5")

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
# solar_obs_brdf_from_kernels follows the local py2sess Fourier convention;
# its direct_brf term is twice the OCO L2 BRDF reflectance kernel used here.
SOLAR_OBS_DIRECT_BRF_TO_OCO_BRF = 0.5
AEROSOL_REFERENCE_WAVELENGTH_UM = 0.76
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
ABSCO_GAS_DATASET = {
    "o2": "Gas_07_Absorption",
    "co2": "Gas_02_Absorption",
    "h2o": "Gas_01_Absorption",
}
O2_DRY_AIR_MOLE_FRACTION = 0.2095
M_AIR = 28.9647e-3
M_H2O = 18.01528e-3
M2_TO_CM2 = 1.0e-4


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
                        out[:, layer] += weight * np.interp(
                            wn,
                            grid,
                            cube[int(p_choice), int(t_choice), int(b_choice)],
                        )
        return out


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
    interp_fraction: np.ndarray
    total_aod_used: float
    phase_model: str


@dataclass(frozen=True)
class OcoL2fpAerosolProperty:
    wave_number_cm: np.ndarray
    extinction_coefficient: np.ndarray
    scattering_coefficient: np.ndarray
    phase_moments: np.ndarray


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
class StokesProjection:
    scalar_factor: float
    analyzer_q: float
    analyzer_u: float
    description: str


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
    xp = np.log(np.asarray(met_pressure_pa, dtype=float))
    fp = np.asarray(met_values, dtype=float)
    x = np.log(np.asarray(target_pressure_pa, dtype=float))
    return np.interp(x, xp, fp)


def _specific_humidity_to_vmr(q: np.ndarray) -> np.ndarray:
    q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 0.95)
    mol_ratio = q_arr / np.maximum(1.0 - q_arr, 1.0e-12) * (M_AIR / M_H2O)
    return mol_ratio / (1.0 + mol_ratio)


def _state_for_retrieval(std: h5py.File, index: int) -> dict[str, np.ndarray | float]:
    rr = std["RetrievalResults"]
    pressure_pa = rr["vector_pressure_levels"][index].astype(float)
    heights_km = rr["vector_altitude_levels"][index].astype(float) / 1000.0
    met_pressure = rr["vector_pressure_levels_met"][index].astype(float)
    met_temperature = rr["temperature_profile_met"][index].astype(float)
    met_q = rr["specific_humidity_profile_met"][index].astype(float)
    temperature = _interp_profile_to_retrieval_levels(
        target_pressure_pa=pressure_pa,
        met_pressure_pa=met_pressure,
        met_values=met_temperature,
    )
    temperature = temperature + float(rr["temperature_offset_fph"][index])
    h2o_vmr = _specific_humidity_to_vmr(
        _interp_profile_to_retrieval_levels(
            target_pressure_pa=pressure_pa,
            met_pressure_pa=met_pressure,
            met_values=met_q,
        )
    )
    h2o_vmr *= float(rr["h2o_scale_factor"][index])
    co2_vmr = rr["co2_profile"][index].astype(float)

    layer_pressure = np.sqrt(pressure_pa[:-1] * pressure_pa[1:])
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

    return {
        "pressure_pa": pressure_pa,
        "heights_km": heights_km,
        "temperature_k": temperature,
        "layer_pressure_pa": layer_pressure,
        "layer_temperature_k": layer_temperature,
        "layer_h2o_vmr": layer_h2o_vmr,
        "o2_col_cm2": dry_air_col_cm2 * O2_DRY_AIR_MOLE_FRACTION,
        "co2_col_cm2": dry_air_col_cm2 * layer_co2_vmr,
        "h2o_col_cm2": h2o_col_cm2,
        "wet_air_col_cm2": wet_air_col_cm2,
        "xco2_ppm": float(rr["xco2"][index]) * 1.0e6,
    }


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
        interp_fraction=np.zeros(wavelength_um.shape, dtype=float),
        total_aod_used=0.0,
        phase_model="none",
    )


def _aerosol_type_defaults(aerosol_type: str) -> dict[str, float]:
    return AEROSOL_TYPE_HG_DEFAULTS.get(
        aerosol_type,
        {"ssa": 0.94, "g": 0.70, "angstrom": 1.0},
    )


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
        or moments.shape[2] < 1
    ):
        raise ValueError(f"OCO L2FP aerosol group {group_name!r} has unexpected shapes")
    if not (
        np.all(np.isfinite(wave_number))
        and np.all(np.isfinite(extinction))
        and np.all(np.isfinite(scattering))
        and np.all(np.isfinite(moments[:, :3, 0]))
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
        phase_moments=moments[:, :3, 0],
    )


def _interp_oco_l2fp_property(
    property_table: OcoL2fpAerosolProperty,
    wavelength_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            for moment in range(3)
        ]
    )
    return extinction, scattering, moments


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
) -> AerosolInputs:
    wavelength = np.asarray(wavelength_um, dtype=float)
    aerosol_types = _decode_h5_strings(std["Metadata/AllAerosolTypes"][...])
    aerosol_model = _decode_h5_strings(std["AerosolResults/aerosol_model"][index])
    aerosol_param = std["AerosolResults/aerosol_param"][index].astype(float)
    aerosol_retrieved = std["AerosolResults/aerosol_type_retrieved"][index].astype(bool)
    pressure_levels = np.asarray(state["pressure_pa"], dtype=float)
    n_layers = pressure_levels.size - 1

    active = [
        type_index
        for type_index, retrieved in enumerate(aerosol_retrieved)
        if retrieved and aerosol_model[type_index].strip()
    ]
    if not active:
        return _empty_aerosol_inputs(wavelength, n_layers)

    extinction = np.zeros((wavelength.size, n_layers, len(active)), dtype=float)
    scattering = np.zeros_like(extinction)
    moments = np.zeros((2, 3, len(active)), dtype=float)
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

    with h5py.File(property_file, "r") as prop_handle:
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
            )
            total_ref_aod += float(np.sum(tau_ref))
            property_table = _load_oco_l2fp_aerosol_property(prop_handle, aerosol_type)
            qext, qsca, _ = _interp_oco_l2fp_property(property_table, wavelength)
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
            _, _, endpoint_moments = _interp_oco_l2fp_property(property_table, endpoint_wavelengths)
            moments[:, :, out_index] = endpoint_moments

    return AerosolInputs(
        extinction_tau=extinction,
        scattering_tau=scattering,
        moments=moments,
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
        )
    if treatment not in {"l2-posterior-hg", "l2-posterior-gaussian-hg"}:
        raise ValueError(f"unknown aerosol treatment: {treatment!r}")

    aerosol_types = _decode_h5_strings(std["Metadata/AllAerosolTypes"][...])
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

    extinction = np.zeros((wavelength.size, n_layers, len(aerosol_types)), dtype=float)
    scattering = np.zeros_like(extinction)
    moments = np.zeros((2, 3, len(aerosol_types)), dtype=float)
    for type_index, aerosol_type in enumerate(aerosol_types):
        defaults = _aerosol_type_defaults(aerosol_type)
        scale = (wavelength / AEROSOL_REFERENCE_WAVELENGTH_UM) ** (-defaults["angstrom"])
        extinction[:, :, type_index] = scale[:, np.newaxis] * tau_ref[np.newaxis, :, type_index]
        scattering[:, :, type_index] = extinction[:, :, type_index] * defaults["ssa"]
        g = defaults["g"]
        moments[:, 0, type_index] = 1.0
        moments[:, 1, type_index] = 3.0 * g
        moments[:, 2, type_index] = 5.0 * g * g

    return AerosolInputs(
        extinction_tau=extinction,
        scattering_tau=scattering,
        moments=moments,
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
    """Add OCO land-BRDF kernel parameters missing from older selected-case CSVs."""
    out = dict(case)
    for band in BANDS:
        l2_name = _band_l2_brdf_name(band)
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
    detector_values = np.full(n_detector, np.nan, dtype=float)
    for det in range(n_detector):
        det_mask = detector_id == det
        weights = response_flat[det_mask]
        weight_sum = np.sum(weights)
        detector_values[det] = np.sum(values[det_mask] * weights) / weight_sum
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
        n_grid = max(2, int(math.ceil((wn_max - wn_min) / spacing)) + 1)
        wn_grid = np.linspace(wn_max, wn_min, n_grid)
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


def _direction_vector(mu: float, azimuth_deg: float) -> np.ndarray:
    rho = math.sqrt(max(1.0 - float(mu) * float(mu), 0.0))
    azimuth = math.radians(float(azimuth_deg))
    return np.array([rho * math.cos(azimuth), rho * math.sin(azimuth), float(mu)])


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
        # Sign convention chosen to reproduce the LRad first-scatter rotation
        # for the direct-solar incoming direction.
        sin_chi = -float(np.dot(k_out, np.cross(reference_normal, scattering_normal)))
        c2i2m = cos_chi * cos_chi - sin_chi * sin_chi
        s2i2m = 2.0 * sin_chi * cos_chi

    delta = 2.0 * (1.0 - np.asarray(depol, dtype=float)) / (2.0 + np.asarray(depol, dtype=float))
    rayleigh_p12 = -0.75 * delta * (1.0 - cos_scatter * cos_scatter)
    analyzer_projection = (
        stokes_projection.analyzer_q * c2i2m + stokes_projection.analyzer_u * s2i2m
    )
    return float(sign) * analyzer_projection * rayleigh_p12


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
    return np.where(np.isfinite(scale * manual_diffuse), scale * manual_diffuse, 0.0)


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
    from py2sess import TwoStreamEss, TwoStreamEssOptions
    from py2sess.optical.phase import build_solar_phase_inputs_from_scattering_tau

    ray_tau, depol = _rayleigh_tau_cm2(
        wavelength_um,
        np.asarray(state["wet_air_col_cm2"], dtype=float),
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
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tau.shape[1],
            mode="solar",
            backend="numpy",
            output_levels=False,
            brdf_surface=brdf is not None,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=ssa,
        g=phase.g,
        z=np.asarray(state["heights_km"], dtype=float),
        angles=angles,
        fbeam=1.0,
        albedo=albedo,
        brdf=brdf,
        stream=stream_value,
        delta_m_truncation_factor=phase.delta_m_truncation_factor,
        fo_scatter_term=phase.fo_scatter_term,
        include_fo=True,
    )
    scalar_radiance = (
        np.asarray(result.radiance_total, dtype=float) * stokes_projection.scalar_factor
    )
    polarization_radiance = np.zeros_like(scalar_radiance, dtype=float)
    if polarization_correction in {"rayleigh-fo", "rayleigh-fo-updiffuse"}:
        polarization_scatter = _rayleigh_projected_polarization_scatter_term(
            ssa=ssa,
            rayleigh_scattering_tau=ray_tau,
            scattering_tau=scattering_tau,
            depol=depol,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
            angles=angles,
            stokes_projection=stokes_projection,
            sign=polarization_sign,
        )
        fo_solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=tau.shape[1],
                mode="solar",
                backend="numpy",
                output_levels=False,
                brdf_surface=False,
            )
        )
        polarization_fo = fo_solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=phase.g,
            z=np.asarray(state["heights_km"], dtype=float),
            angles=angles,
            fbeam=1.0,
            albedo=np.zeros_like(albedo, dtype=float),
            stream=stream_value,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
            fo_scatter_term=polarization_scatter,
            n_moments=0,
        )
        polarization_radiance = np.asarray(polarization_fo.intensity_ss, dtype=float)
        if polarization_correction == "rayleigh-fo-updiffuse":
            diffuse_solver = TwoStreamEss(
                TwoStreamEssOptions(
                    nlyr=tau.shape[1],
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
                    tau=tau,
                    ssa=ssa,
                    g=phase.g,
                    height_grid=np.asarray(state["heights_km"], dtype=float),
                    observer_angles=angles,
                    diffuse_albedo=np.asarray(diffuse_albedo, dtype=float),
                    delta_m_truncation_factor=phase.delta_m_truncation_factor,
                    rayleigh_scattering_tau=ray_tau,
                    depol=depol,
                    stokes_projection=stokes_projection,
                    first_order_correction=np.asarray(polarization_fo.intensity_ss, dtype=float),
                    sign=polarization_sign,
                    n_azimuths=polarization_diffuse_azimuths,
                    stream_value=stream_value,
                )
            )
    elif polarization_correction != "none":
        raise ValueError(f"unknown polarization correction: {polarization_correction!r}")

    if solar_reference_factor is not None:
        factor = np.asarray(solar_reference_factor, dtype=float)
        scalar_radiance = scalar_radiance * factor
        polarization_radiance = polarization_radiance * factor
    return Py2sessReplayResult(
        scalar_radiance=scalar_radiance,
        polarization_correction=polarization_radiance,
        radiance=scalar_radiance + polarization_radiance,
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
        "--aerosol-treatment",
        choices=("none", "l2-posterior-hg", "l2-posterior-gaussian-hg", "oco-l2fp"),
        default="none",
        help=(
            "'none' uses gas plus Rayleigh scattering only. "
            "'l2-posterior-hg' inserts L2 posterior aerosol AOD using the "
            "official pressure subcolumns and simple HG optical defaults. "
            "'l2-posterior-gaussian-hg' uses the L2 gaussian_log aerosol "
            "parameters for the vertical profile. 'oco-l2fp' uses the "
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
            "solar relative velocity. 'l2-los' uses the line-of-sight gas velocity."
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
        choices=("none", "rayleigh-fo", "rayleigh-fo-updiffuse"),
        default="rayleigh-fo",
        help=(
            "Optional correction that projects Rayleigh Q/U scattering "
            "through the L1B instrument Stokes coefficients. The default "
            "'rayleigh-fo' uses only the direct-solar first-order term; "
            "'rayleigh-fo-updiffuse' also adds one source-iteration term from "
            "py2sess scalar upwelling diffuse radiance. Use 'none' for scalar-only "
            "replay. This leaves the py2sess scalar solver unchanged."
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
        "--polarization-diffuse-azimuths",
        type=int,
        default=8,
        help=(
            "Number of azimuth quadrature points for rayleigh-fo-updiffuse. "
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
    args = parser.parse_args()
    if args.polarization_diffuse_azimuths <= 0:
        raise ValueError("--polarization-diffuse-azimuths must be positive")
    if not math.isfinite(args.stream_value) or args.stream_value <= 0.0 or args.stream_value > 1.0:
        raise ValueError("--stream-value must satisfy 0 < value <= 1")
    if args.brdf_quadrature_streams <= 0 or args.brdf_quadrature_streams % 2 != 0:
        raise ValueError("--brdf-quadrature-streams must be a positive even integer")
    if not np.isfinite(args.ils_grid_spacing_cm_1) or args.ils_grid_spacing_cm_1 < 0.0:
        raise ValueError("--ils-grid-spacing-cm-1 must be nonnegative and finite")
    if args.eof_treatment == "oco3-static" and args.oco3_eof_file is None:
        raise FileNotFoundError(
            "--eof-treatment oco3-static requires --oco3-eof-file or RTRF_OCO3_EOF_FILE"
        )

    l1b_path = args.data_dir / "oco3_L1bScSC_17767a_220624_B10313r_220907005244.h5"
    l2std_path = args.data_dir / "oco3_L2StdSC_17767a_220624_B10313r_220919181911.h5"
    l2dia_path = args.data_dir / "oco3_L2DiaSC_17767a_220624_B10313r_220919175057.h5"
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
            state = _state_for_retrieval(std, index)
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
                    band_index, footprint, sample, :
                ].astype(float)
                response = l1b["InstrumentHeader/ils_relative_response"][
                    band_index, footprint, sample, :
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

                x_o2 = (
                    absco["o2"].cross_section_cm2(
                        wavelength_um=gas_wavelength,
                        pressure_pa=np.asarray(state["layer_pressure_pa"], dtype=float),
                        temperature_k=np.asarray(state["layer_temperature_k"], dtype=float),
                        h2o_vmr=np.asarray(state["layer_h2o_vmr"], dtype=float),
                    )
                    * o2_scale[band_index]
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
                    * co2_scale[band_index]
                    if band != "o2"
                    else 0.0
                )
                x_h2o = (
                    absco["h2o"].cross_section_cm2(
                        wavelength_um=gas_wavelength,
                        pressure_pa=np.asarray(state["layer_pressure_pa"], dtype=float),
                        temperature_k=np.asarray(state["layer_temperature_k"], dtype=float),
                        h2o_vmr=np.asarray(state["layer_h2o_vmr"], dtype=float),
                    )
                    * h2o_scale[band_index]
                )
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

                reflectance = _land_surface_reflectance(
                    case=case,
                    band=band,
                    wavelength_um=eval_wavelength,
                    surface_spectrum=args.surface_spectrum,
                    surface_angular=args.surface_angular,
                    angles=surface_angles,
                )
                brdf = None
                solver_albedo = reflectance
                if args.surface_angular == "rpv-brdf":
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
                aerosol = _posterior_aerosol_inputs(
                    std=std,
                    oco_l2fp_property_file=args.oco_l2fp_aerosol_file,
                    index=index,
                    state=state,
                    wavelength_um=eval_wavelength,
                    treatment=args.aerosol_treatment,
                )
                rpv_fields = (
                    f"brdf_hotspot_parameter_{band}",
                    f"brdf_asymmetry_parameter_{band}",
                    f"brdf_anisotropy_parameter_{band}",
                )
                rpv_kernel = (
                    _oco_rpv_kernel(case=case, band=band, angles=surface_angles)
                    if all(field in case for field in rpv_fields)
                    else math.nan
                )
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
                py_run = _run_py2sess(
                    wavelength_um=eval_wavelength,
                    state=state,
                    gas_tau=gas_tau,
                    albedo=solver_albedo,
                    diffuse_albedo=reflectance,
                    brdf=brdf,
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
                    for det in range(center_wavelength.size):
                        det_mask = detector_id == det
                        weights = response_flat[det_mask]
                        weight_sum = np.sum(weights)
                        fluorescence_detector_photon[det] = (
                            np.sum(fluorescence_eval[det_mask] * weights) / weight_sum
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
                    sample_indexes=sample,
                    surface_type=case.get("surface_type", ""),
                    treatment=args.eof_treatment,
                )
                eof_energy = (
                    _photon_to_energy_spectral_radiance(eof.values, center_wavelength)
                    if args.eof_treatment == "oco3-static"
                    else np.zeros(center_wavelength.shape, dtype=float)
                )
                py_detector = py_detector + fluorescence_energy + eof_energy
                py_scalar_detector = py_detector - py_polarization_detector
                continuum_signal = float(std[OCO_CONTINUUM_FIELD[band]][index])
                if not np.isfinite(continuum_signal) or continuum_signal <= 0.0:
                    raise ValueError(f"bad OCO continuum signal for sounding {sid} band {band}")
                continuum_signal_energy = float(
                    _photon_to_energy_spectral_radiance(
                        continuum_signal,
                        BAND_REFERENCE_WAVELENGTH_UM[band],
                    )
                )
                py_unit_continuum_signal = _sample_continuum_level(py_detector)
                if not np.isfinite(py_unit_continuum_signal) or py_unit_continuum_signal <= 0.0:
                    raise ValueError(f"bad py2sess continuum signal for sounding {sid} band {band}")
                py2sess_posthoc_scale = 1.0
                py2sess_effective_fbeam = solar_irradiance_reference
                obs_energy = _photon_to_energy_spectral_radiance(obs, center_wavelength)
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
                        "surface_reflectance_model": args.surface_spectrum,
                        "surface_angular_model": args.surface_angular,
                        "surface_rpv_kernel": f"{rpv_kernel:.9g}",
                        "surface_reflectance_used": f"{float(np.nanmedian(reflectance)):.9g}",
                        "surface_reflectance_min": f"{float(np.nanmin(reflectance)):.9g}",
                        "surface_reflectance_max": f"{float(np.nanmax(reflectance)):.9g}",
                        "stream_value": f"{args.stream_value:.9g}",
                        "brdf_quadrature_streams": args.brdf_quadrature_streams,
                        "absco_o2_scale": f"{o2_scale[band_index]:.9g}",
                        "absco_co2_scale": f"{co2_scale[band_index]:.9g}",
                        "absco_h2o_scale": f"{h2o_scale[band_index]:.9g}",
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
            "surface_reflectance_model",
            "surface_angular_model",
            "surface_rpv_kernel",
            "surface_reflectance_used",
            "surface_reflectance_min",
            "surface_reflectance_max",
            "stream_value",
            "brdf_quadrature_streams",
            "absco_o2_scale",
            "absco_co2_scale",
            "absco_h2o_scale",
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
            "oco_continuum_signal_radiance_w_m2_sr_um",
            "py2sess_unit_solver_signal",
            "py2sess_effective_fbeam_w_m2_um",
            "py2sess_posthoc_scale",
            "py2sess_minus_measured_continuum_percent",
        ],
        spectrum_rows,
    )
    _plot(plot_path, spectrum_rows)
    print(f"summary={summary_path}")
    print(f"spectrum={spectrum_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
