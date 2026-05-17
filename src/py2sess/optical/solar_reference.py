"""External solar reference spectrum helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K = 5772.0
IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2 = 1361.0
ASTRONOMICAL_UNIT_M = 149597870700.0
PLANCK_CONSTANT_J_S = 6.62607015e-34
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
SPEED_OF_LIGHT_M_S = 299792458.0
STEFAN_BOLTZMANN_CONSTANT_W_M2_K4 = 5.670374419e-8


@dataclass(frozen=True)
class ToonSolarReference:
    """Toon/JPL solar pseudo-transmittance reference spectrum."""

    wavenumber_cm_inv: np.ndarray
    pseudo_transmittance: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "ToonSolarReference":
        """Read a Toon/JPL two-column pseudo-transmittance table."""
        table = np.atleast_2d(np.loadtxt(path, dtype=float, skiprows=3, usecols=(0, 1)))
        if table.shape[1] != 2:
            raise ValueError("Toon solar reference table must have two numeric columns")
        order = np.argsort(table[:, 0])
        wavenumber = table[order, 0]
        pseudo_transmittance = table[order, 1]
        if np.any(np.diff(wavenumber) <= 0.0):
            raise ValueError("Toon solar reference wavenumbers must be unique")
        return cls(wavenumber_cm_inv=wavenumber, pseudo_transmittance=pseudo_transmittance)

    def at_wavelength_um(
        self,
        wavelength_um,
        *,
        planck_continuum_reference_um: float | None = None,
        solar_effective_temperature_k: float = IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
    ) -> np.ndarray:
        """Interpolate the solar reference onto wavelengths in microns.

        By default this returns only the Toon/JPL solar pseudo-transmittance
        spectrum. Passing ``planck_continuum_reference_um`` also multiplies by
        a relative Planck continuum, normalized to one at the supplied reference
        wavelength. This gives a smooth solar-color term without introducing an
        absolute solar irradiance scale.
        """
        wavelength = _finite_1d("wavelength_um", wavelength_um)
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_um values must be positive")

        target_wavenumber = 1.0e4 / wavelength
        if (
            target_wavenumber.min() < self.wavenumber_cm_inv[0]
            or target_wavenumber.max() > self.wavenumber_cm_inv[-1]
        ):
            raise ValueError(
                "Toon solar reference does not cover requested wavelengths "
                f"({wavelength.min():.6g}-{wavelength.max():.6g} um)"
            )
        values = np.interp(target_wavenumber, self.wavenumber_cm_inv, self.pseudo_transmittance)
        if planck_continuum_reference_um is not None:
            values = values * solar_planck_continuum_ratio(
                wavelength,
                reference_wavelength_um=planck_continuum_reference_um,
                temperature_k=solar_effective_temperature_k,
            )
        return values


@dataclass(frozen=True)
class OcoSolarModel:
    """OCO/RtRetrievalFramework solar continuum and absorption model."""

    continuum_wavenumber_cm_inv: tuple[np.ndarray, ...]
    continuum_photon_irradiance_m2_um: tuple[np.ndarray, ...]
    absorption_wavenumber_cm_inv: tuple[np.ndarray, ...]
    absorption_transmittance: tuple[np.ndarray, ...]

    @classmethod
    def from_hdf(cls, path: str | Path) -> "OcoSolarModel":
        """Read the OCO L2 solar model HDF table.

        The RtRetrievalFramework table stores the continuum at 1 AU in
        photons s^-1 m^-2 um^-1 and the high-resolution solar absorption as a
        dimensionless pseudo-transmittance.
        """
        continuum_wn: list[np.ndarray] = []
        continuum_ph: list[np.ndarray] = []
        absorption_wn: list[np.ndarray] = []
        absorption_tx: list[np.ndarray] = []
        with h5py.File(path, "r") as handle:
            for band_index in range(1, 4):
                continuum_group = f"Solar/Continuum/Continuum_{band_index}"
                absorption_group = f"Solar/Absorption/Absorption_{band_index}"
                continuum_wn.append(
                    _strictly_increasing(
                        handle[f"{continuum_group}/wavenumber"][...].astype(float),
                        f"{continuum_group}/wavenumber",
                    )
                )
                continuum_ph.append(handle[f"{continuum_group}/spectrum"][...].astype(float))
                absorption_wn.append(
                    _strictly_increasing(
                        handle[f"{absorption_group}/wavenumber"][...].astype(float),
                        f"{absorption_group}/wavenumber",
                    )
                )
                absorption_tx.append(handle[f"{absorption_group}/spectrum"][...].astype(float))
        return cls(
            continuum_wavenumber_cm_inv=tuple(continuum_wn),
            continuum_photon_irradiance_m2_um=tuple(continuum_ph),
            absorption_wavenumber_cm_inv=tuple(absorption_wn),
            absorption_transmittance=tuple(absorption_tx),
        )

    def photon_irradiance_m2_um(
        self,
        wavelength_um,
        *,
        band_index: int,
        observer_distance_m: float = ASTRONOMICAL_UNIT_M,
    ) -> np.ndarray:
        """Return OCO solar spectral photon irradiance at observer distance."""
        wavelength = _finite_1d("wavelength_um", wavelength_um)
        distance = float(observer_distance_m)
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_um values must be positive")
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("observer_distance_m must be positive and finite")
        band_offset = int(band_index) - 1
        if band_offset not in (0, 1, 2):
            raise ValueError("band_index must be 1, 2, or 3")
        target_wavenumber = 1.0e4 / wavelength
        continuum = _interp_checked(
            target_wavenumber,
            self.continuum_wavenumber_cm_inv[band_offset],
            self.continuum_photon_irradiance_m2_um[band_offset],
            "solar continuum",
        )
        absorption = _interp_checked(
            target_wavenumber,
            self.absorption_wavenumber_cm_inv[band_offset],
            self.absorption_transmittance[band_offset],
            "solar absorption",
        )
        distance_factor = (ASTRONOMICAL_UNIT_M / distance) ** 2
        return continuum * absorption * distance_factor

    def energy_irradiance_w_m2_um(
        self,
        wavelength_um,
        *,
        band_index: int,
        observer_distance_m: float = ASTRONOMICAL_UNIT_M,
        energy_wavelength_um=None,
    ) -> np.ndarray:
        """Return OCO solar spectral irradiance in W m^-2 um^-1."""
        lookup_wavelength = _finite_1d("wavelength_um", wavelength_um)
        energy_wavelength = (
            lookup_wavelength
            if energy_wavelength_um is None
            else _finite_1d("energy_wavelength_um", energy_wavelength_um)
        )
        if energy_wavelength.shape != lookup_wavelength.shape:
            raise ValueError("energy_wavelength_um must match wavelength_um shape")
        photon = self.photon_irradiance_m2_um(
            lookup_wavelength,
            band_index=band_index,
            observer_distance_m=observer_distance_m,
        )
        photon_energy = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / (energy_wavelength * 1.0e-6)
        return photon * photon_energy


def solar_planck_continuum_ratio(
    wavelength_um,
    *,
    reference_wavelength_um: float,
    temperature_k: float = IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
) -> np.ndarray:
    """Return a relative solar Planck continuum per unit wavelength.

    The ratio is normalized to one at ``reference_wavelength_um``. It is meant
    for spectral-shape correction when the absolute beam is calibrated
    separately.
    """
    wavelength = _finite_1d("wavelength_um", wavelength_um)
    reference = float(reference_wavelength_um)
    temperature = float(temperature_k)
    if np.any(wavelength <= 0.0) or not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("wavelengths must be positive and finite")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be positive and finite")

    wavelength_m = wavelength * 1.0e-6
    reference_m = reference * 1.0e-6
    c2 = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / BOLTZMANN_CONSTANT_J_K

    def log_planck_wavelength(lambda_m: np.ndarray | float) -> np.ndarray:
        lambda_arr = np.asarray(lambda_m, dtype=float)
        exponent = c2 / (lambda_arr * temperature)
        return -5.0 * np.log(lambda_arr) - np.log(np.expm1(exponent))

    return np.exp(log_planck_wavelength(wavelength_m) - log_planck_wavelength(reference_m))


def solar_planck_irradiance_w_m2_um(
    wavelength_um,
    *,
    temperature_k: float = IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
    total_solar_irradiance_w_m2: float = IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2,
    observer_distance_m: float = ASTRONOMICAL_UNIT_M,
) -> np.ndarray:
    """Return blackbody solar spectral irradiance at the observer.

    The projected solar solid-angle factor is set by the supplied total solar
    irradiance at 1 AU and the effective temperature. The result is per micron
    and scales as inverse square observer distance.
    """
    wavelength = _finite_1d("wavelength_um", wavelength_um)
    temperature = float(temperature_k)
    total_irradiance = float(total_solar_irradiance_w_m2)
    distance = float(observer_distance_m)
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_um values must be positive")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be positive and finite")
    if not np.isfinite(total_irradiance) or total_irradiance <= 0.0:
        raise ValueError("total_solar_irradiance_w_m2 must be positive and finite")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("observer_distance_m must be positive and finite")

    wavelength_m = wavelength * 1.0e-6
    c2 = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / BOLTZMANN_CONSTANT_J_K
    log_b_lambda = (
        np.log(2.0 * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S**2)
        - 5.0 * np.log(wavelength_m)
        - np.log(np.expm1(c2 / (wavelength_m * temperature)))
    )
    projected_solid_angle = total_irradiance / (STEFAN_BOLTZMANN_CONSTANT_W_M2_K4 * temperature**4)
    distance_factor = (ASTRONOMICAL_UNIT_M / distance) ** 2
    return np.pi * np.exp(log_b_lambda) * projected_solid_angle * distance_factor * 1.0e-6


def _finite_1d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return arr


def _strictly_increasing(values: np.ndarray, name: str) -> np.ndarray:
    arr = _finite_1d(name, values)
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _interp_checked(
    target: np.ndarray, grid: np.ndarray, values: np.ndarray, label: str
) -> np.ndarray:
    if values.shape != grid.shape:
        raise ValueError(f"{label} grid and values must have matching shapes")
    if target.min() < grid[0] or target.max() > grid[-1]:
        raise ValueError(
            f"{label} table does not cover requested wavenumbers "
            f"({target.min():.6g}-{target.max():.6g} cm^-1)"
        )
    return np.interp(target, grid, values)
