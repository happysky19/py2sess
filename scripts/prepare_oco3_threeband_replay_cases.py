#!/usr/bin/env python3
"""Select clean OCO-3 three-band posterior-replay cases from L2 Diagnostic data.

This helper intentionally does not fit or adjust any physical quantity.  It
only selects official full-physics retrieval records and exports the measured
and posterior-modeled spectra already present in L2 Diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py2sess_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "py2sess_cache"))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "oco3_joint_official_downloads" / "20220624_17767a"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "oco3_threeband_replay" / "20220624_17767a"

BANDS = ("o2", "wco2", "sco2")
BAND_LABELS = {
    "o2": "O2 A",
    "wco2": "weak CO2",
    "sco2": "strong CO2",
}


def _read(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle[dataset][...]


def _decode_array(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8", errors="replace").strip()
        if isinstance(item, bytes)
        else str(item).strip()
        for item in values
    ]


def _band_slices(counts: np.ndarray) -> dict[str, slice]:
    start = 0
    out: dict[str, slice] = {}
    for band, count in zip(BANDS, counts, strict=True):
        stop = start + int(count)
        out[band] = slice(start, stop)
        start = stop
    return out


def _finite_positive_valid_spectrum(
    measured: np.ndarray,
    modeled: np.ndarray,
    wavelength: np.ndarray,
    sample_index: np.ndarray,
    ncolors: int,
) -> bool:
    if ncolors <= 0:
        return False
    valid = slice(0, int(ncolors))
    spectral_arrays = (measured[valid], modeled[valid], wavelength[valid])
    samples = sample_index[valid]
    return (
        all(np.isfinite(arr).all() for arr in (*spectral_arrays, samples))
        and all((arr > 0).all() for arr in spectral_arrays)
        and (samples >= 0).all()
    )


def _select_candidate_indices(
    *,
    l2std_path: Path,
    l2dia_path: Path,
    land_fraction_min: float,
    land_fraction_max: float | None,
    operation_mode: str | None,
    chi_square_max: float,
    snr_o2_min: float,
    snr_wco2_min: float,
    snr_sco2_min: float,
    aod_max: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(l2std_path, "r") as std:
        data = {
            "sounding_id": std["RetrievalHeader/sounding_id"][...],
            "operation_mode": np.asarray(
                _decode_array(std["RetrievalHeader/sounding_operation_mode"][...]), dtype=object
            ),
            "latitude": std["RetrievalGeometry/retrieval_latitude"][...],
            "longitude": std["RetrievalGeometry/retrieval_longitude"][...],
            "land_fraction": std["RetrievalGeometry/retrieval_land_fraction"][...],
            "land_water_indicator": std["RetrievalGeometry/retrieval_land_water_indicator"][...],
            "solar_zenith_deg": std["RetrievalGeometry/retrieval_solar_zenith"][...],
            "view_zenith_deg": std["RetrievalGeometry/retrieval_zenith"][...],
            "outcome_flag": std["RetrievalResults/outcome_flag"][...],
            "diverging_steps": std["RetrievalResults/diverging_steps"][...],
            "iterations": std["RetrievalResults/iterations"][...],
            "cloud_flag_abp": std["PreprocessingResults/cloud_flag_abp"][...],
            "cloud_flag_idp": std["PreprocessingResults/cloud_flag_idp"][...],
            "aerosol_total_aod": std["AerosolResults/aerosol_total_aod"][...],
            "surface_pressure_hpa": std["RetrievalResults/surface_pressure_fph"][...] / 100.0,
            "wind_speed": std["RetrievalResults/wind_speed"][...],
            "wind_speed_apriori": std["RetrievalResults/wind_speed_apriori"][...],
            "xco2_ppm": std["RetrievalResults/xco2"][...] * 1.0e6,
            "h2o_scale_factor": std["RetrievalResults/h2o_scale_factor"][...],
            "temperature_offset_k": std["RetrievalResults/temperature_offset_fph"][...],
            "brdf_reflectance_o2": std["BRDFResults/brdf_reflectance_o2"][...],
            "brdf_reflectance_wco2": std["BRDFResults/brdf_reflectance_weak_co2"][...],
            "brdf_reflectance_sco2": std["BRDFResults/brdf_reflectance_strong_co2"][...],
            "brdf_reflectance_slope_o2": std["BRDFResults/brdf_reflectance_slope_o2"][...],
            "brdf_reflectance_slope_wco2": std["BRDFResults/brdf_reflectance_slope_weak_co2"][...],
            "brdf_reflectance_slope_sco2": std["BRDFResults/brdf_reflectance_slope_strong_co2"][
                ...
            ],
            "brdf_reflectance_quadratic_o2": std["BRDFResults/brdf_reflectance_quadratic_o2"][...],
            "brdf_reflectance_quadratic_wco2": std[
                "BRDFResults/brdf_reflectance_quadratic_weak_co2"
            ][...],
            "brdf_reflectance_quadratic_sco2": std[
                "BRDFResults/brdf_reflectance_quadratic_strong_co2"
            ][...],
            "brdf_weight_o2": std["BRDFResults/brdf_weight_o2"][...],
            "brdf_weight_wco2": std["BRDFResults/brdf_weight_weak_co2"][...],
            "brdf_weight_sco2": std["BRDFResults/brdf_weight_strong_co2"][...],
            "brdf_weight_slope_o2": std["BRDFResults/brdf_weight_slope_o2"][...],
            "brdf_weight_slope_wco2": std["BRDFResults/brdf_weight_slope_weak_co2"][...],
            "brdf_weight_slope_sco2": std["BRDFResults/brdf_weight_slope_strong_co2"][...],
            "brdf_weight_quadratic_o2": std["BRDFResults/brdf_weight_quadratic_o2"][...],
            "brdf_weight_quadratic_wco2": std["BRDFResults/brdf_weight_quadratic_weak_co2"][...],
            "brdf_weight_quadratic_sco2": std["BRDFResults/brdf_weight_quadratic_strong_co2"][...],
            "brdf_rahman_factor_o2": std["BRDFResults/brdf_rahman_factor_o2"][...],
            "brdf_rahman_factor_wco2": std["BRDFResults/brdf_rahman_factor_weak_co2"][...],
            "brdf_rahman_factor_sco2": std["BRDFResults/brdf_rahman_factor_strong_co2"][...],
            "brdf_hotspot_parameter_o2": std["BRDFResults/brdf_hotspot_parameter_o2"][...],
            "brdf_hotspot_parameter_wco2": std["BRDFResults/brdf_hotspot_parameter_weak_co2"][...],
            "brdf_hotspot_parameter_sco2": std["BRDFResults/brdf_hotspot_parameter_strong_co2"][
                ...
            ],
            "brdf_asymmetry_parameter_o2": std["BRDFResults/brdf_asymmetry_parameter_o2"][...],
            "brdf_asymmetry_parameter_wco2": std["BRDFResults/brdf_asymmetry_parameter_weak_co2"][
                ...
            ],
            "brdf_asymmetry_parameter_sco2": std["BRDFResults/brdf_asymmetry_parameter_strong_co2"][
                ...
            ],
            "brdf_anisotropy_parameter_o2": std["BRDFResults/brdf_anisotropy_parameter_o2"][...],
            "brdf_anisotropy_parameter_wco2": std["BRDFResults/brdf_anisotropy_parameter_weak_co2"][
                ...
            ],
            "brdf_anisotropy_parameter_sco2": std[
                "BRDFResults/brdf_anisotropy_parameter_strong_co2"
            ][...],
            "brdf_breon_factor_o2": std["BRDFResults/brdf_breon_factor_o2"][...],
            "brdf_breon_factor_wco2": std["BRDFResults/brdf_breon_factor_weak_co2"][...],
            "brdf_breon_factor_sco2": std["BRDFResults/brdf_breon_factor_strong_co2"][...],
            "dispersion_offset_o2": std["DispersionResults/dispersion_offset_o2"][...],
            "dispersion_offset_wco2": std["DispersionResults/dispersion_offset_weak_co2"][...],
            "dispersion_offset_sco2": std["DispersionResults/dispersion_offset_strong_co2"][...],
            "dispersion_spacing_o2": std["DispersionResults/dispersion_spacing_o2"][...],
            "dispersion_spacing_wco2": std["DispersionResults/dispersion_spacing_weak_co2"][...],
            "dispersion_spacing_sco2": std["DispersionResults/dispersion_spacing_strong_co2"][...],
            "chi2_o2": std["SpectralParameters/reduced_chi_squared_o2_fph"][...],
            "chi2_wco2": std["SpectralParameters/reduced_chi_squared_weak_co2_fph"][...],
            "chi2_sco2": std["SpectralParameters/reduced_chi_squared_strong_co2_fph"][...],
            "snr_o2": std["L1bScSpectralParameters/snr_o2_l1b"][...],
            "snr_wco2": std["L1bScSpectralParameters/snr_weak_co2_l1b"][...],
            "snr_sco2": std["L1bScSpectralParameters/snr_strong_co2_l1b"][...],
        }
        aerosol_model = std["AerosolResults/aerosol_model"][...]
        data["aerosol_models"] = np.asarray(
            [";".join(_decode_array(row)) for row in aerosol_model], dtype=object
        )
        data["surface_type"] = np.asarray(
            _decode_array(std["RetrievalResults/surface_type"][...]), dtype=object
        )

    with h5py.File(l2dia_path, "r") as dia:
        spectral = dia["SpectralParameters"]
        num_colors = spectral["num_colors"][...]
        num_colors_per_band = spectral["num_colors_per_band"][...]
        measured = spectral["measured_radiance"]
        modeled = spectral["modeled_radiance"]
        wavelength = spectral["wavelength"]
        sample_index = spectral["sample_indexes"]
        valid_spectrum = np.zeros(num_colors.shape, dtype=bool)
        for index, ncolors in enumerate(num_colors):
            valid_spectrum[index] = _finite_positive_valid_spectrum(
                measured[index],
                modeled[index],
                wavelength[index],
                sample_index[index],
                int(ncolors),
            )

    data["num_colors"] = num_colors
    for band_index, band in enumerate(BANDS):
        data[f"num_colors_{band}"] = num_colors_per_band[:, band_index]

    finite = np.ones(num_colors.shape, dtype=bool)
    for value in data.values():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            finite &= np.isfinite(value)

    land_surface_ok = (
        (data["surface_type"] != "Coxmunk,Lambertian")
        & (data["brdf_reflectance_o2"] > 0.0)
        & (data["brdf_reflectance_wco2"] > 0.0)
        & (data["brdf_reflectance_sco2"] > 0.0)
        & (data["brdf_weight_o2"] > 0.0)
        & (data["brdf_weight_wco2"] > 0.0)
        & (data["brdf_weight_sco2"] > 0.0)
    )
    ocean_surface_ok = (
        (data["surface_type"] == "Coxmunk,Lambertian")
        & np.isfinite(data["wind_speed"])
        & (data["wind_speed"] > 0.0)
    )

    mask = (
        finite
        & valid_spectrum
        & (data["outcome_flag"] == 1)
        & (data["land_fraction"] >= land_fraction_min)
        & (data["cloud_flag_abp"] == 0)
        & (data["cloud_flag_idp"] >= 2)
        & (data["chi2_o2"] < chi_square_max)
        & (data["chi2_wco2"] < chi_square_max)
        & (data["chi2_sco2"] < chi_square_max)
        & (data["snr_o2"] >= snr_o2_min)
        & (data["snr_wco2"] >= snr_wco2_min)
        & (data["snr_sco2"] >= snr_sco2_min)
        & (data["aerosol_total_aod"] >= 0.0)
        & (data["aerosol_total_aod"] <= aod_max)
        & (land_surface_ok | ocean_surface_ok)
    )
    if land_fraction_max is not None:
        mask &= data["land_fraction"] <= land_fraction_max
    if operation_mode is not None:
        mask &= data["operation_mode"] == operation_mode
    return np.where(mask)[0], data


def _sample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("--count must be positive")
    if indices.size <= count:
        return indices
    positions = np.linspace(0, indices.size - 1, count)
    return indices[np.round(positions).astype(int)]


def _write_selected_cases(path: Path, indices: np.ndarray, data: dict[str, np.ndarray]) -> None:
    fields = [
        "retrieval_index",
        "sounding_id",
        "operation_mode",
        "latitude",
        "longitude",
        "land_fraction",
        "land_water_indicator",
        "solar_zenith_deg",
        "view_zenith_deg",
        "outcome_flag",
        "cloud_flag_abp",
        "cloud_flag_idp",
        "aerosol_total_aod",
        "surface_pressure_hpa",
        "wind_speed",
        "wind_speed_apriori",
        "xco2_ppm",
        "h2o_scale_factor",
        "temperature_offset_k",
        "surface_type",
        "brdf_reflectance_o2",
        "brdf_reflectance_wco2",
        "brdf_reflectance_sco2",
        "brdf_reflectance_slope_o2",
        "brdf_reflectance_slope_wco2",
        "brdf_reflectance_slope_sco2",
        "brdf_reflectance_quadratic_o2",
        "brdf_reflectance_quadratic_wco2",
        "brdf_reflectance_quadratic_sco2",
        "brdf_weight_o2",
        "brdf_weight_wco2",
        "brdf_weight_sco2",
        "brdf_weight_slope_o2",
        "brdf_weight_slope_wco2",
        "brdf_weight_slope_sco2",
        "brdf_weight_quadratic_o2",
        "brdf_weight_quadratic_wco2",
        "brdf_weight_quadratic_sco2",
        "brdf_rahman_factor_o2",
        "brdf_rahman_factor_wco2",
        "brdf_rahman_factor_sco2",
        "brdf_hotspot_parameter_o2",
        "brdf_hotspot_parameter_wco2",
        "brdf_hotspot_parameter_sco2",
        "brdf_asymmetry_parameter_o2",
        "brdf_asymmetry_parameter_wco2",
        "brdf_asymmetry_parameter_sco2",
        "brdf_anisotropy_parameter_o2",
        "brdf_anisotropy_parameter_wco2",
        "brdf_anisotropy_parameter_sco2",
        "brdf_breon_factor_o2",
        "brdf_breon_factor_wco2",
        "brdf_breon_factor_sco2",
        "dispersion_offset_o2",
        "dispersion_offset_wco2",
        "dispersion_offset_sco2",
        "dispersion_spacing_o2",
        "dispersion_spacing_wco2",
        "dispersion_spacing_sco2",
        "chi2_o2",
        "chi2_wco2",
        "chi2_sco2",
        "snr_o2",
        "snr_wco2",
        "snr_sco2",
        "num_colors",
        "num_colors_o2",
        "num_colors_wco2",
        "num_colors_sco2",
        "aerosol_models",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in indices:
            row = {"retrieval_index": str(int(index))}
            for field in fields[1:]:
                value = data[field][index]
                if isinstance(value, np.generic):
                    value = value.item()
                row[field] = str(value)
            writer.writerow(row)


def _write_selected_spectra(path: Path, indices: np.ndarray, l2dia_path: Path) -> None:
    fields = [
        "retrieval_index",
        "sounding_id",
        "band",
        "band_color_index",
        "packed_color_index",
        "sample_index",
        "wavelength_um",
        "measured_radiance",
        "modeled_radiance",
        "measured_radiance_uncert",
        "official_model_minus_measured_percent",
    ]
    with h5py.File(l2dia_path, "r") as dia, path.open("w", encoding="utf-8", newline="") as handle:
        header = dia["RetrievalHeader/sounding_id"][...]
        spectral = dia["SpectralParameters"]
        measured = spectral["measured_radiance"]
        modeled = spectral["modeled_radiance"]
        uncert = spectral["measured_radiance_uncert"]
        wavelength = spectral["wavelength"]
        sample_index = spectral["sample_indexes"]
        counts = spectral["num_colors_per_band"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in indices:
            band_slices = _band_slices(counts[index])
            for band in BANDS:
                section = band_slices[band]
                for local, packed in enumerate(range(section.start, section.stop)):
                    obs = float(measured[index, packed])
                    fit = float(modeled[index, packed])
                    writer.writerow(
                        {
                            "retrieval_index": int(index),
                            "sounding_id": int(header[index]),
                            "band": band,
                            "band_color_index": local,
                            "packed_color_index": packed,
                            "sample_index": int(sample_index[index, packed]),
                            "wavelength_um": f"{float(wavelength[index, packed]):.9f}",
                            "measured_radiance": f"{obs:.12e}",
                            "modeled_radiance": f"{fit:.12e}",
                            "measured_radiance_uncert": (f"{float(uncert[index, packed]):.12e}"),
                            "official_model_minus_measured_percent": (
                                f"{100.0 * (fit - obs) / obs:.9e}"
                            ),
                        }
                    )


def _plot_selected_spectra(path: Path, indices: np.ndarray, l2dia_path: Path) -> None:
    import matplotlib.pyplot as plt

    with h5py.File(l2dia_path, "r") as dia:
        sounding_id = dia["RetrievalHeader/sounding_id"][...]
        spectral = dia["SpectralParameters"]
        measured = spectral["measured_radiance"]
        modeled = spectral["modeled_radiance"]
        wavelength = spectral["wavelength"]
        counts = spectral["num_colors_per_band"]

        n_cases = len(indices)
        fig, axes = plt.subplots(
            n_cases,
            3,
            figsize=(9.0, max(2.2, 1.55 * n_cases)),
            dpi=180,
            squeeze=False,
            sharey=False,
        )
        for row, index in enumerate(indices):
            band_slices = _band_slices(counts[index])
            for col, band in enumerate(BANDS):
                axis = axes[row, col]
                section = band_slices[band]
                x = wavelength[index, section]
                obs = measured[index, section]
                fit = modeled[index, section]
                axis.plot(x, obs, color="#222222", lw=0.75, label="measured")
                axis.plot(x, fit, color="#D55E00", lw=0.65, ls="--", label="official posterior")
                axis.grid(True, color="#e5e7eb", lw=0.45)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if row == 0:
                    axis.set_title(BAND_LABELS[band], fontsize=8.5)
                if col == 0:
                    axis.set_ylabel(f"{int(sounding_id[index])}\nradiance", fontsize=7.5)
                if row == n_cases - 1:
                    axis.set_xlabel("Wavelength (um)", fontsize=7.5)
                axis.tick_params(labelsize=7)
        axes[0, -1].legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--land-fraction-min", type=float, default=95.0)
    parser.add_argument("--land-fraction-max", type=float, default=None)
    parser.add_argument(
        "--operation-mode",
        choices=("AM", "GL", "ND", "TG", "XS"),
        default=None,
    )
    parser.add_argument("--chi-square-max", type=float, default=1.4)
    parser.add_argument("--snr-o2-min", type=float, default=100.0)
    parser.add_argument("--snr-wco2-min", type=float, default=100.0)
    parser.add_argument("--snr-sco2-min", type=float, default=50.0)
    parser.add_argument("--aod-max", type=float, default=0.30)
    args = parser.parse_args()

    l2std_path = args.data_dir / "oco3_L2StdSC_17767a_220624_B10313r_220919181911.h5"
    l2dia_path = args.data_dir / "oco3_L2DiaSC_17767a_220624_B10313r_220919175057.h5"
    for path in (l2std_path, l2dia_path):
        if not path.exists():
            raise FileNotFoundError(path)

    candidates, data = _select_candidate_indices(
        l2std_path=l2std_path,
        l2dia_path=l2dia_path,
        land_fraction_min=args.land_fraction_min,
        land_fraction_max=args.land_fraction_max,
        operation_mode=args.operation_mode,
        chi_square_max=args.chi_square_max,
        snr_o2_min=args.snr_o2_min,
        snr_wco2_min=args.snr_wco2_min,
        snr_sco2_min=args.snr_sco2_min,
        aod_max=args.aod_max,
    )
    selected = _sample_indices(candidates, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_selected_cases(args.output_dir / "selected_soundings.csv", selected, data)
    _write_selected_spectra(args.output_dir / "selected_l2dia_spectra.csv", selected, l2dia_path)
    _plot_selected_spectra(args.output_dir / "selected_l2dia_spectra.png", selected, l2dia_path)

    print(f"candidate_count={candidates.size}")
    print(f"selected_count={selected.size}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
