"""Scene-level optical preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .properties import LayerOpticalProperties, build_layer_optical_properties
from .rayleigh import rayleigh_bodhaine


@dataclass(frozen=True)
class AtmosphericProfile:
    """Atmospheric profile quantities used by the optical-property builder."""

    heights_km: np.ndarray
    pressure_hpa: np.ndarray
    temperature_k: np.ndarray
    air_columns: np.ndarray
    air_density_per_km: np.ndarray
    gas_density_per_km: np.ndarray


@dataclass(frozen=True)
class AerosolOpticalComponents:
    """Aerosol extinction and scattering optical depths."""

    extinction_tau: np.ndarray
    scattering_tau: np.ndarray


@dataclass(frozen=True)
class SceneOpacityComponents:
    """Component optical depths produced by scene opacity providers.

    These fields are the Python-side replacement for the CreateProps layer
    quantities that later become Fortran ``taug``, ``taudp``, ``omega``, ``fr``,
    and ``fa``.
    """

    gas_absorption_tau: np.ndarray
    rayleigh_scattering_tau: np.ndarray
    aerosol_extinction_tau: np.ndarray
    aerosol_scattering_tau: np.ndarray
    depol: np.ndarray

    def layer_properties(self) -> LayerOpticalProperties:
        """Combine components into RT layer inputs."""
        return build_layer_optical_properties(
            absorption_tau=self.gas_absorption_tau,
            rayleigh_scattering_tau=self.rayleigh_scattering_tau,
            aerosol_extinction_tau=(
                self.aerosol_extinction_tau if self.aerosol_extinction_tau.shape[-1] else None
            ),
            aerosol_scattering_tau=(
                self.aerosol_scattering_tau if self.aerosol_scattering_tau.shape[-1] else None
            ),
        )


@dataclass(frozen=True)
class SceneLayerOpticalProperties:
    """Layer optical inputs generated from scene-level physical inputs."""

    layer: LayerOpticalProperties
    gas_absorption_tau: np.ndarray
    rayleigh_scattering_tau: np.ndarray
    aerosol_extinction_tau: np.ndarray
    aerosol_scattering_tau: np.ndarray
    depol: np.ndarray


def atmospheric_profile_from_levels(
    *,
    pressure_hpa,
    temperature_k,
    gas_vmr=None,
    heights_km=None,
    surface_altitude_m: float = 0.0,
) -> AtmosphericProfile:
    """Build level heights, air columns, and gas densities from a profile.

    The hydrostatic height and air-column calculations follow the GEOCAPE
    ``geocape_profile_setter_2`` convention used by the Fortran benchmark.
    Pressures, temperatures, and heights use top-to-bottom level ordering.
    ``gas_vmr`` is a level quantity with shape ``(nlevel, ngas)`` and unitless
    volume mixing ratio.
    """
    pressure = np.asarray(pressure_hpa, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    if pressure.ndim != 1 or temperature.ndim != 1:
        raise ValueError("pressure_hpa and temperature_k must be one-dimensional")
    if pressure.shape != temperature.shape:
        raise ValueError("pressure_hpa and temperature_k must have the same shape")
    if pressure.size < 2:
        raise ValueError("at least two pressure levels are required")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(temperature)):
        raise ValueError("pressure_hpa and temperature_k must be finite")
    if np.any(pressure <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError("pressure_hpa and temperature_k must be positive")
    if np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure_hpa must increase from top to bottom")

    if heights_km is None:
        heights = _hydrostatic_heights_km(
            pressure_hpa=pressure,
            temperature_k=temperature,
            surface_altitude_m=surface_altitude_m,
        )
    else:
        heights = np.asarray(heights_km, dtype=float)
        if heights.shape != pressure.shape:
            raise ValueError("heights_km must have the same shape as pressure_hpa")
        if not np.all(np.isfinite(heights)):
            raise ValueError("heights_km must be finite")
        if np.any(np.diff(heights) >= 0.0):
            raise ValueError("heights_km must decrease from top to bottom")

    air_density = _air_density_per_km(pressure, temperature)
    air_columns = 0.5 * (air_density[:-1] + air_density[1:]) * (heights[:-1] - heights[1:])

    if gas_vmr is None:
        gas = np.zeros((pressure.size, 0), dtype=float)
    else:
        gas_vmr_arr = np.asarray(gas_vmr, dtype=float)
        if gas_vmr_arr.ndim == 1:
            gas_vmr_arr = gas_vmr_arr[:, np.newaxis]
        if gas_vmr_arr.ndim != 2 or gas_vmr_arr.shape[0] != pressure.size:
            raise ValueError("gas_vmr must have shape (nlevel,) or (nlevel, ngas)")
        if not np.all(np.isfinite(gas_vmr_arr)):
            raise ValueError("gas_vmr must be finite")
        if np.any(gas_vmr_arr < 0.0):
            raise ValueError("gas_vmr must be nonnegative")
        gas = air_density[:, np.newaxis] * gas_vmr_arr

    return AtmosphericProfile(
        heights_km=heights,
        pressure_hpa=pressure,
        temperature_k=temperature,
        air_columns=air_columns,
        air_density_per_km=air_density,
        gas_density_per_km=gas,
    )


def gas_absorption_tau_from_cross_sections(
    *,
    heights_km,
    gas_density_per_km,
    cross_sections,
) -> np.ndarray:
    """Integrate layer gas absorption optical depth.

    ``cross_sections`` may have shape ``(nspec, ngas)`` for level-independent
    cross sections or ``(nspec, nlevel, ngas)`` for level-dependent values.
    """
    heights = np.asarray(heights_km, dtype=float)
    gas_density = np.asarray(gas_density_per_km, dtype=float)
    xsec = np.asarray(cross_sections, dtype=float)
    if heights.ndim != 1:
        raise ValueError("heights_km must be one-dimensional")
    if gas_density.ndim != 2 or gas_density.shape[0] != heights.size:
        raise ValueError("gas_density_per_km must have shape (nlevel, ngas)")
    if not np.all(np.isfinite(gas_density)) or np.any(gas_density < 0.0):
        raise ValueError("gas_density_per_km must be finite and nonnegative")
    if not np.all(np.isfinite(xsec)) or np.any(xsec < 0.0):
        raise ValueError("cross_sections must be finite and nonnegative")
    ngas = gas_density.shape[1]
    if xsec.ndim == 2:
        if xsec.shape[1] != ngas:
            raise ValueError("cross_sections must have ngas columns")
        xsec_upper = xsec[:, np.newaxis, :]
        xsec_lower = xsec[:, np.newaxis, :]
    elif xsec.ndim == 3:
        if xsec.shape[1:] != gas_density.shape:
            raise ValueError("cross_sections must have shape (nspec, nlevel, ngas)")
        xsec_upper = xsec[:, :-1, :]
        xsec_lower = xsec[:, 1:, :]
    else:
        raise ValueError("cross_sections must have shape (nspec, ngas) or (nspec, nlevel, ngas)")

    layer_thickness_km = heights[:-1] - heights[1:]
    if np.any(layer_thickness_km <= 0.0):
        raise ValueError("heights_km must decrease from top to bottom")
    upper = gas_density[:-1, :]
    lower = gas_density[1:, :]
    integrand = (upper[np.newaxis, :, :] * xsec_upper + lower[np.newaxis, :, :] * xsec_lower).sum(
        axis=-1
    )
    return 0.5 * layer_thickness_km[np.newaxis, :] * integrand


def rayleigh_scattering_tau_from_air_columns(
    *,
    wavelengths_nm,
    air_columns,
    co2_ppmv: float = 385.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Rayleigh layer optical depth and depolarization."""
    air = np.asarray(air_columns, dtype=float)
    if air.ndim != 1:
        raise ValueError("air_columns must be one-dimensional")
    if not np.all(np.isfinite(air)) or np.any(air < 0.0):
        raise ValueError("air_columns must be finite and nonnegative")
    rayleigh = rayleigh_bodhaine(wavelengths_nm, co2_ppmv=co2_ppmv)
    tau = rayleigh.cross_section[:, np.newaxis] * air[np.newaxis, :]
    return tau, rayleigh.depolarization


def aerosol_components_from_tables(
    *,
    wavelengths_microns,
    select_wavelength_microns: float,
    aerosol_loadings,
    aerosol_wavelengths_microns,
    aerosol_bulk_iops,
) -> AerosolOpticalComponents:
    """Interpolate GEOCAPE-style aerosol tables to spectral layer components.

    ``aerosol_bulk_iops[0]`` is the extinction-like quantity and
    ``aerosol_bulk_iops[1]`` is the scattering-like quantity. ``aerosol_loadings``
    has shape ``(nlayer, naerosol)``.
    """
    wavelengths = np.asarray(wavelengths_microns, dtype=float)
    select = float(select_wavelength_microns)
    loadings = np.asarray(aerosol_loadings, dtype=float)
    table_wavelengths = np.asarray(aerosol_wavelengths_microns, dtype=float)
    bulk = np.asarray(aerosol_bulk_iops, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_microns must be one-dimensional")
    if loadings.ndim != 2:
        raise ValueError("aerosol_loadings must have shape (nlayer, naerosol)")
    if bulk.ndim != 3 or bulk.shape[0] != 2:
        raise ValueError("aerosol_bulk_iops must have shape (2, nwavelength, naerosol)")
    if bulk.shape[1] != table_wavelengths.size or bulk.shape[2] != loadings.shape[1]:
        raise ValueError("aerosol table shapes do not match aerosol_loadings")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_microns must be finite and positive")
    if not np.isfinite(select) or select <= 0.0:
        raise ValueError("select_wavelength_microns must be finite and positive")
    if not np.all(np.isfinite(table_wavelengths)) or np.any(np.diff(table_wavelengths) <= 0.0):
        raise ValueError("aerosol_wavelengths_microns must be finite and increasing")
    for name, arr in (("aerosol_loadings", loadings), ("aerosol_bulk_iops", bulk)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")

    select_ext = _interp_aerosol_table(select, table_wavelengths, bulk[0])
    if np.any(select_ext <= 0.0):
        raise ValueError("selected aerosol extinction table values must be positive")
    ext = _interp_aerosol_table(wavelengths, table_wavelengths, bulk[0])
    scat = _interp_aerosol_table(wavelengths, table_wavelengths, bulk[1])
    scale = loadings[np.newaxis, :, :] / select_ext[np.newaxis, np.newaxis, :]
    return AerosolOpticalComponents(
        extinction_tau=scale * ext[:, np.newaxis, :],
        scattering_tau=scale * scat[:, np.newaxis, :],
    )


def build_scene_layer_optical_properties(
    *,
    wavelengths_nm,
    profile: AtmosphericProfile,
    gas_cross_sections,
    aerosol_loadings=None,
    aerosol_wavelengths_microns=None,
    aerosol_bulk_iops=None,
    aerosol_select_wavelength_microns=0.4,
    co2_ppmv: float = 385.0,
) -> SceneLayerOpticalProperties:
    """Generate RT layer optical inputs from a physical scene profile."""
    components = build_scene_opacity_components(
        wavelengths_nm=wavelengths_nm,
        profile=profile,
        gas_cross_sections=gas_cross_sections,
        aerosol_loadings=aerosol_loadings,
        aerosol_wavelengths_microns=aerosol_wavelengths_microns,
        aerosol_bulk_iops=aerosol_bulk_iops,
        aerosol_select_wavelength_microns=aerosol_select_wavelength_microns,
        co2_ppmv=co2_ppmv,
    )
    layer = components.layer_properties()
    return SceneLayerOpticalProperties(
        layer=layer,
        gas_absorption_tau=components.gas_absorption_tau,
        rayleigh_scattering_tau=components.rayleigh_scattering_tau,
        aerosol_extinction_tau=components.aerosol_extinction_tau,
        aerosol_scattering_tau=components.aerosol_scattering_tau,
        depol=components.depol,
    )


def build_scene_opacity_components(
    *,
    wavelengths_nm,
    profile: AtmosphericProfile,
    gas_cross_sections,
    aerosol_loadings=None,
    aerosol_wavelengths_microns=None,
    aerosol_bulk_iops=None,
    aerosol_select_wavelength_microns=0.4,
    co2_ppmv: float = 385.0,
) -> SceneOpacityComponents:
    """Build gas, Rayleigh, and aerosol optical-depth components."""
    gas_tau = gas_absorption_tau_from_cross_sections(
        heights_km=profile.heights_km,
        gas_density_per_km=profile.gas_density_per_km,
        cross_sections=gas_cross_sections,
    )
    ray_tau, depol = rayleigh_scattering_tau_from_air_columns(
        wavelengths_nm=wavelengths_nm,
        air_columns=profile.air_columns,
        co2_ppmv=co2_ppmv,
    )
    if aerosol_loadings is None:
        aerosol_ext = np.zeros(gas_tau.shape + (0,), dtype=float)
        aerosol_scat = aerosol_ext
    else:
        if aerosol_wavelengths_microns is None or aerosol_bulk_iops is None:
            raise ValueError(
                "aerosol_loadings requires aerosol_wavelengths_microns and aerosol_bulk_iops"
            )
        aerosol = aerosol_components_from_tables(
            wavelengths_microns=np.asarray(wavelengths_nm, dtype=float) / 1000.0,
            select_wavelength_microns=aerosol_select_wavelength_microns,
            aerosol_loadings=aerosol_loadings,
            aerosol_wavelengths_microns=aerosol_wavelengths_microns,
            aerosol_bulk_iops=aerosol_bulk_iops,
        )
        aerosol_ext = aerosol.extinction_tau
        aerosol_scat = aerosol.scattering_tau

    return SceneOpacityComponents(
        gas_absorption_tau=gas_tau,
        rayleigh_scattering_tau=ray_tau,
        aerosol_extinction_tau=aerosol_ext,
        aerosol_scattering_tau=aerosol_scat,
        depol=depol,
    )


def _hydrostatic_heights_km(
    *,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    surface_altitude_m: float,
) -> np.ndarray:
    heights = np.empty_like(pressure_hpa, dtype=float)
    heights[-1] = surface_altitude_m / 1000.0
    ccon = -9.81 * 28.9 / 8314.0 * 500.0
    for n in range(pressure_hpa.size - 1, 0, -1):
        avit = 1.0 / temperature_k[n - 1] + 1.0 / temperature_k[n]
        heights[n - 1] = heights[n] - np.log(pressure_hpa[n] / pressure_hpa[n - 1]) / avit / ccon
    return heights


def _air_density_per_km(pressure_hpa: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    rho_stand = 2.68675e19
    pzero = 1013.25
    tzero = 273.15
    rho_zero = rho_stand * tzero / pzero
    const = 1.0e5 * rho_zero
    return const * pressure_hpa / temperature_k


def _interp_aerosol_table(x, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if np.any(values <= xp[0]) or np.any(values >= xp[-1]):
        raise ValueError("aerosol interpolation wavelength is out of table bounds")
    flat = values.reshape(-1)
    out = np.empty((flat.size, fp.shape[-1]), dtype=float)
    for j in range(fp.shape[-1]):
        out[:, j] = np.interp(flat, xp, fp[:, j])
    return out.reshape(values.shape + (fp.shape[-1],))
