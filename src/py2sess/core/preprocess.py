"""Input normalization and geometry preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .brdf_solar_obs import solar_obs_brdf_from_kernels
from .brdf_thermal import thermal_brdf_from_kernels
from .geometry import auxgeom_solar_obs, chapman_factors


DEFAULT_EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class PreparedGeometry:
    """Geometry terms derived from validated solver inputs."""

    deg_to_rad: float
    pi4: float
    do_postprocessing: bool
    do_include_mvout: np.ndarray
    n_fouriers: int
    chapman_factors: np.ndarray
    average_secant_pp: np.ndarray
    x0: np.ndarray
    user_streams: np.ndarray
    user_secants: np.ndarray
    azmfac: np.ndarray
    px11: float
    pxsq: np.ndarray
    px0x: np.ndarray
    ulp: np.ndarray
    surface_factor: np.ndarray
    delta_factor: np.ndarray


@dataclass(frozen=True)
class PreparedBrdf:
    """Prepared BRDF coefficients for the active geometry set."""

    brdf_f_0: np.ndarray
    brdf_f: np.ndarray
    ubrdf_f: np.ndarray


@dataclass(frozen=True)
class PreparedSurfaceLeaving:
    """Prepared surface-leaving coefficients for the active geometry set."""

    slterm_isotropic: np.ndarray
    slterm_f_0: np.ndarray


@dataclass(frozen=True)
class PreparedThermal:
    """Prepared thermal source and surface-emission inputs."""

    thermal_bb_input: np.ndarray
    surfbb: float
    emissivity: float


@dataclass(frozen=True)
class PreparedInputs:
    """Normalized inputs consumed by the NumPy and torch solvers."""

    source_mode: str
    tau_arr: np.ndarray
    omega_arr: np.ndarray
    asymm_arr: np.ndarray
    d2s_scaling: np.ndarray
    height_grid: np.ndarray | None
    user_obsgeoms: np.ndarray | None
    stream_value: float
    flux_factor: float
    albedo: float
    earth_radius: float
    geometry: PreparedGeometry
    brdf: PreparedBrdf | None
    surface_leaving: PreparedSurfaceLeaving | None
    thermal: PreparedThermal | None
    lattice_counts: tuple[int, int, int] | None = None
    lattice_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _as_1d(name: str, value: Any, size: int) -> np.ndarray:
    """Converts a value to a fixed-length 1D float array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError(f"{name} must be a 1D array of length {size}")
    return arr


def _as_optional_1d(name: str, value: Any | None, size: int) -> np.ndarray | None:
    """Converts an optional value to a fixed-length 1D float array."""
    if value is None:
        return None
    return _as_1d(name, value, size)


def _validate_inputs(
    *,
    options: Any,
    tau_arr: np.ndarray,
    omega_arr: np.ndarray,
    asymm_arr: np.ndarray,
    d2s_scaling: np.ndarray,
    height_grid: np.ndarray,
    user_obsgeoms: np.ndarray,
    earth_radius: float,
) -> tuple[bool, float]:
    """Validates solar-style inputs and returns normalized flags."""
    if not options.do_upwelling and not options.do_dnwelling:
        raise ValueError("At least one of do_upwelling or do_dnwelling must be true")

    if options.do_mvout_only and options.do_additional_mvout:
        raise ValueError("do_mvout_only and do_additional_mvout cannot both be true")

    if user_obsgeoms.ndim != 2 or user_obsgeoms.shape[1] != 3:
        raise ValueError("user_obsgeoms must have shape (n_geometries, 3)")

    if height_grid.ndim != 1 or height_grid.shape[0] != options.n_layers + 1:
        raise ValueError(f"height_grid must have length {options.n_layers + 1}")

    if np.any(height_grid[:-1] <= height_grid[1:]):
        raise ValueError("height_grid must be strictly decreasing")

    sza = user_obsgeoms[:, 0]
    if np.any((sza < 0.0) | (sza >= 90.0)):
        raise ValueError("solar zenith angles must satisfy 0 <= sza < 90")

    if not options.do_mvout_only:
        vza = user_obsgeoms[:, 1]
        if np.any((vza < 0.0) | (vza > 90.0)):
            raise ValueError("viewing zenith angles must satisfy 0 <= vza <= 90")
        azm = user_obsgeoms[:, 2]
        if np.any((azm < 0.0) | (azm > 360.0)):
            raise ValueError("azimuth angles must satisfy 0 <= azimuth <= 360")

    do_postprocessing = options.do_additional_mvout or (not options.do_mvout_only)

    if not options.do_plane_parallel and (earth_radius < 6320.0 or earth_radius > 6420.0):
        earth_radius = DEFAULT_EARTH_RADIUS_KM

    if not np.all(np.isfinite(tau_arr)):
        raise ValueError("tau_arr must be finite")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega_arr must be finite")
    if not np.all(np.isfinite(asymm_arr)):
        raise ValueError("asymm_arr must be finite")
    if not np.all(np.isfinite(d2s_scaling)):
        raise ValueError("d2s_scaling must be finite")

    return do_postprocessing, earth_radius


def _validate_inputs_thermal(
    *,
    options: Any,
    tau_arr: np.ndarray,
    omega_arr: np.ndarray,
    asymm_arr: np.ndarray,
    d2s_scaling: np.ndarray,
    user_angles: np.ndarray,
    thermal_bb_input: np.ndarray,
) -> bool:
    """Validates thermal-style inputs and returns the postprocessing flag."""
    if not options.do_upwelling and not options.do_dnwelling:
        raise ValueError("At least one of do_upwelling or do_dnwelling must be true")
    if options.do_mvout_only and options.do_additional_mvout:
        raise ValueError("do_mvout_only and do_additional_mvout cannot both be true")
    if np.any((user_angles < 0.0) | (user_angles > 90.0)):
        raise ValueError("viewing zenith angles must satisfy 0 <= vza <= 90")
    if not np.all(np.isfinite(tau_arr)):
        raise ValueError("tau_arr must be finite")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega_arr must be finite")
    if not np.all(np.isfinite(asymm_arr)):
        raise ValueError("asymm_arr must be finite")
    if not np.all(np.isfinite(d2s_scaling)):
        raise ValueError("d2s_scaling must be finite")
    if not np.all(np.isfinite(thermal_bb_input)):
        raise ValueError("thermal_bb_input must be finite")
    return options.do_additional_mvout or (not options.do_mvout_only)


def _prepare_geometry(
    *,
    options: Any,
    height_grid: np.ndarray,
    user_obsgeoms: np.ndarray,
    stream_value: float,
    earth_radius: float,
    do_postprocessing: bool,
) -> PreparedGeometry:
    """Builds prepared geometry terms for solar-style runs."""
    deg_to_rad = math.acos(-1.0) / 180.0
    pi4 = deg_to_rad * 720.0

    x0 = np.cos(user_obsgeoms[:, 0] * deg_to_rad)
    average_secant_pp = np.zeros(user_obsgeoms.shape[0], dtype=float)
    if options.do_plane_parallel:
        average_secant_pp = 1.0 / x0

    if do_postprocessing:
        user_streams = np.cos(user_obsgeoms[:, 1] * deg_to_rad)
        user_secants = 1.0 / user_streams
        azmfac = np.cos(user_obsgeoms[:, 2] * deg_to_rad)
    else:
        user_streams = np.zeros(user_obsgeoms.shape[0], dtype=float)
        user_secants = np.zeros(user_obsgeoms.shape[0], dtype=float)
        azmfac = np.zeros(user_obsgeoms.shape[0], dtype=float)

    px11, pxsq, px0x, ulp = auxgeom_solar_obs(
        x0=x0,
        user_streams=user_streams,
        stream_value=stream_value,
        do_postprocessing=do_postprocessing,
    )

    n_fouriers = 0 if options.do_mvout_only else 1
    if user_obsgeoms.shape[0] == 1 and user_obsgeoms[0, 0] < 1.0e-8:
        n_fouriers = 0

    do_include_mvout = np.array([False, False], dtype=bool)
    if options.do_additional_mvout or options.do_mvout_only:
        do_include_mvout[0] = True

    surface_factor = np.array([2.0, 1.0], dtype=float)
    delta_factor = np.array([1.0, 2.0], dtype=float)

    n_layers = options.n_layers
    chapman = np.zeros((n_layers, n_layers, user_obsgeoms.shape[0]), dtype=float)
    if not options.do_plane_parallel:
        for ib, sza in enumerate(user_obsgeoms[:, 0]):
            chapman[:, :, ib] = chapman_factors(height_grid, earth_radius, float(sza))

    return PreparedGeometry(
        deg_to_rad=deg_to_rad,
        pi4=pi4,
        do_postprocessing=do_postprocessing,
        do_include_mvout=do_include_mvout,
        n_fouriers=n_fouriers,
        chapman_factors=chapman,
        average_secant_pp=average_secant_pp,
        x0=x0,
        user_streams=user_streams,
        user_secants=user_secants,
        azmfac=azmfac,
        px11=px11,
        pxsq=pxsq,
        px0x=px0x,
        ulp=ulp,
        surface_factor=surface_factor,
        delta_factor=delta_factor,
    )


def _prepare_geometry_thermal(
    *,
    options: Any,
    user_angles: np.ndarray,
    stream_value: float,
    do_postprocessing: bool,
) -> PreparedGeometry:
    """Builds prepared geometry terms for thermal runs."""
    deg_to_rad = math.acos(-1.0) / 180.0
    pi4 = deg_to_rad * 720.0
    if do_postprocessing:
        user_streams = np.cos(user_angles * deg_to_rad)
        user_secants = 1.0 / user_streams
    else:
        user_streams = np.zeros(user_angles.size, dtype=float)
        user_secants = np.zeros(user_angles.size, dtype=float)
    do_include_mvout = np.array(
        [options.do_additional_mvout or options.do_mvout_only, False],
        dtype=bool,
    )
    pxsq = np.array([stream_value * stream_value, 0.0], dtype=float)
    return PreparedGeometry(
        deg_to_rad=deg_to_rad,
        pi4=pi4,
        do_postprocessing=do_postprocessing,
        do_include_mvout=do_include_mvout,
        n_fouriers=0,
        chapman_factors=np.zeros((options.n_layers, options.n_layers, 0), dtype=float),
        average_secant_pp=np.zeros(0, dtype=float),
        x0=np.zeros(0, dtype=float),
        user_streams=user_streams,
        user_secants=user_secants,
        azmfac=np.zeros(0, dtype=float),
        px11=0.0,
        pxsq=pxsq,
        px0x=np.zeros((0, 2), dtype=float),
        ulp=np.zeros(0, dtype=float),
        surface_factor=np.array([2.0, 0.0], dtype=float),
        delta_factor=np.array([1.0, 0.0], dtype=float),
    )


def _prepare_brdf(
    *,
    brdf: Any | None,
    n_geoms: int,
    user_obsgeoms: np.ndarray | None = None,
    stream_value: float = 1.0 / math.sqrt(3.0),
) -> PreparedBrdf | None:
    """Normalizes BRDF inputs for solar-style runs."""
    if brdf is None:
        return None
    if not isinstance(brdf, dict):
        raise ValueError("brdf must be a mapping with keys 'brdf_f_0', 'brdf_f', and 'ubrdf_f'")
    if "kernel_specs" in brdf:
        generated = solar_obs_brdf_from_kernels(
            kernel_specs=list(brdf["kernel_specs"]),
            user_obsgeoms=user_obsgeoms if user_obsgeoms is not None else brdf.get("user_obsgeoms"),
            stream_value=float(
                stream_value
                if user_obsgeoms is not None
                else brdf.get("stream_value", 1.0 / math.sqrt(3.0))
            ),
            n_geoms=n_geoms,
        )
        return PreparedBrdf(
            brdf_f_0=np.asarray(generated.brdf_f_0, dtype=float),
            brdf_f=np.asarray(generated.brdf_f, dtype=float),
            ubrdf_f=np.asarray(generated.ubrdf_f, dtype=float),
        )
    brdf_f_0 = np.asarray(brdf.get("brdf_f_0"), dtype=float)
    brdf_f = np.asarray(brdf.get("brdf_f"), dtype=float)
    ubrdf_f = np.asarray(brdf.get("ubrdf_f"), dtype=float)
    if brdf_f_0.shape != (n_geoms, 2):
        raise ValueError("brdf['brdf_f_0'] must have shape (n_geometries, 2)")
    if brdf_f.shape != (2,):
        raise ValueError("brdf['brdf_f'] must have shape (2,)")
    if ubrdf_f.shape != (n_geoms, 2):
        raise ValueError("brdf['ubrdf_f'] must have shape (n_geometries, 2)")
    return PreparedBrdf(brdf_f_0=brdf_f_0, brdf_f=brdf_f, ubrdf_f=ubrdf_f)


def _prepare_surface_leaving(
    *,
    surface_leaving: Any | None,
    n_geoms: int,
) -> PreparedSurfaceLeaving | None:
    if surface_leaving is None:
        return None
    if not isinstance(surface_leaving, dict):
        raise ValueError(
            "surface_leaving must be a mapping with keys 'slterm_isotropic' and 'slterm_f_0'"
        )
    slterm_isotropic = np.asarray(surface_leaving.get("slterm_isotropic"), dtype=float)
    slterm_f_0 = np.asarray(surface_leaving.get("slterm_f_0"), dtype=float)
    if slterm_isotropic.shape != (n_geoms,):
        raise ValueError("surface_leaving['slterm_isotropic'] must have shape (n_geometries,)")
    if slterm_f_0.shape != (n_geoms, 2):
        raise ValueError("surface_leaving['slterm_f_0'] must have shape (n_geometries, 2)")
    return PreparedSurfaceLeaving(
        slterm_isotropic=slterm_isotropic,
        slterm_f_0=slterm_f_0,
    )


def prepare_inputs(
    *,
    options: Any,
    tau_arr: Any,
    omega_arr: Any,
    asymm_arr: Any,
    height_grid: Any,
    user_obsgeoms: Any,
    stream_value: float,
    flux_factor: float,
    albedo: float,
    d2s_scaling: Any | None,
    brdf: Any | None,
    surface_leaving: Any | None,
    user_angles: Any | None,
    beam_szas: Any | None = None,
    user_relazms: Any | None = None,
    thermal_bb_input: Any | None,
    surfbb: float,
    emissivity: float,
    earth_radius: float,
) -> PreparedInputs:
    """Normalizes public API inputs into solver-ready arrays.

    Parameters
    ----------
    options
        Public solver options object.
    tau_arr, omega_arr, asymm_arr
        Layer optical-depth, single-scattering-albedo, and asymmetry arrays.
    height_grid
        Level-height grid for solar and spherical FO paths.
    user_obsgeoms
        Observation-geometry array for solar observation mode.
    stream_value, flux_factor, albedo
        Two-stream quadrature cosine and surface/source scalar inputs.
    d2s_scaling
        Optional delta-M scaling factors; missing values default to zero.
    brdf, surface_leaving
        Optional surface-supplement inputs.
    user_angles, beam_szas, user_relazms
        Thermal or solar-lattice angle inputs.
    thermal_bb_input, surfbb, emissivity
        Thermal layer and surface source inputs.
    earth_radius
        Planetary radius in kilometers.

    Returns
    -------
    PreparedInputs
        Validated and precomputed inputs shared by NumPy and torch solvers.
    """
    tau = _as_1d("tau_arr", tau_arr, options.n_layers)
    omega = _as_1d("omega_arr", omega_arr, options.n_layers)
    asymm = _as_1d("asymm_arr", asymm_arr, options.n_layers)
    d2s = (
        np.zeros(options.n_layers, dtype=float)
        if d2s_scaling is None
        else _as_1d("d2s_scaling", d2s_scaling, options.n_layers)
    )
    if options.source_mode == "thermal":
        angles = _as_1d(
            "user_angles",
            user_angles,
            np.asarray(user_angles, dtype=float).size if user_angles is not None else 0,
        )
        if angles.size == 0:
            raise ValueError("user_angles must be provided for thermal mode")
        thermal_bb = _as_1d("thermal_bb_input", thermal_bb_input, options.n_layers + 1)
        do_postprocessing = _validate_inputs_thermal(
            options=options,
            tau_arr=tau,
            omega_arr=omega,
            asymm_arr=asymm,
            d2s_scaling=d2s,
            user_angles=angles,
            thermal_bb_input=thermal_bb,
        )
        geometry = _prepare_geometry_thermal(
            options=options,
            user_angles=angles,
            stream_value=float(stream_value),
            do_postprocessing=do_postprocessing,
        )
        prepared_brdf = None
        if options.do_brdf_surface:
            if brdf is None or not isinstance(brdf, dict):
                raise ValueError("do_brdf_surface=True requires thermal brdf coefficients")
            if "kernel_specs" in brdf:
                generated = thermal_brdf_from_kernels(
                    kernel_specs=list(brdf["kernel_specs"]),
                    user_angles=angles,
                    do_surface_emission=float(emissivity) != 0.0,
                )
                brdf_f = np.asarray(generated.brdf_f, dtype=float)
                ubrdf_f = np.asarray(generated.ubrdf_f, dtype=float)
                emissivity = generated.emissivity
            else:
                brdf_f = np.asarray(brdf.get("brdf_f"), dtype=float)
                ubrdf_f = np.asarray(brdf.get("ubrdf_f"), dtype=float)
            if brdf_f.shape != ():
                raise ValueError("thermal brdf['brdf_f'] must be a scalar")
            if ubrdf_f.shape != (angles.size,):
                raise ValueError("thermal brdf['ubrdf_f'] must have shape (n_user_angles,)")
            prepared_brdf = PreparedBrdf(
                brdf_f_0=np.zeros((angles.size, 2), dtype=float),
                brdf_f=np.array([float(brdf_f), 0.0], dtype=float),
                ubrdf_f=np.column_stack((ubrdf_f, np.zeros(angles.size, dtype=float))),
            )
        return PreparedInputs(
            source_mode="thermal",
            tau_arr=tau,
            omega_arr=omega,
            asymm_arr=asymm,
            d2s_scaling=d2s,
            height_grid=None if height_grid is None else np.asarray(height_grid, dtype=float),
            user_obsgeoms=None,
            stream_value=float(stream_value),
            flux_factor=float(flux_factor),
            albedo=float(albedo),
            earth_radius=float(earth_radius),
            geometry=geometry,
            brdf=prepared_brdf,
            surface_leaving=None,
            thermal=PreparedThermal(
                thermal_bb_input=thermal_bb,
                surfbb=float(surfbb),
                emissivity=float(emissivity),
            ),
        )

    if options.source_mode == "solar_lat":
        heights = np.asarray(height_grid, dtype=float)
        beams = _as_1d(
            "beam_szas",
            beam_szas,
            np.asarray(beam_szas, dtype=float).size if beam_szas is not None else 0,
        )
        angles = _as_1d(
            "user_angles",
            user_angles,
            np.asarray(user_angles, dtype=float).size if user_angles is not None else 0,
        )
        relazms = _as_1d(
            "user_relazms",
            user_relazms,
            np.asarray(user_relazms, dtype=float).size if user_relazms is not None else 0,
        )
        if beams.size == 0:
            raise ValueError("beam_szas must be provided for solar_lat mode")
        if not options.do_mvout_only and angles.size == 0:
            raise ValueError("user_angles must be provided for solar_lat mode")
        if not options.do_mvout_only and relazms.size == 0:
            raise ValueError("user_relazms must be provided for solar_lat mode")
        obs_rows = []
        if options.do_mvout_only:
            for sza in beams:
                obs_rows.append([float(sza), 0.0, 0.0])
        else:
            for sza in beams:
                for vza in angles:
                    for azm in relazms:
                        obs_rows.append([float(sza), float(vza), float(azm)])
        obsgeoms = np.asarray(obs_rows, dtype=float)
        do_postprocessing, normalized_earth_radius = _validate_inputs(
            options=options,
            tau_arr=tau,
            omega_arr=omega,
            asymm_arr=asymm,
            d2s_scaling=d2s,
            height_grid=heights,
            user_obsgeoms=obsgeoms,
            earth_radius=float(earth_radius),
        )
        geometry = _prepare_geometry(
            options=options,
            height_grid=heights,
            user_obsgeoms=obsgeoms,
            stream_value=float(stream_value),
            earth_radius=normalized_earth_radius,
            do_postprocessing=do_postprocessing,
        )
        prepared_brdf = _prepare_brdf(
            brdf=brdf,
            n_geoms=obsgeoms.shape[0],
            user_obsgeoms=obsgeoms,
            stream_value=float(stream_value),
        )
        prepared_surface_leaving = _prepare_surface_leaving(
            surface_leaving=surface_leaving,
            n_geoms=obsgeoms.shape[0],
        )
        if options.do_brdf_surface and prepared_brdf is None:
            raise ValueError("do_brdf_surface=True requires brdf coefficients")
        if options.do_surface_leaving and prepared_surface_leaving is None:
            raise ValueError("do_surface_leaving=True requires surface_leaving coefficients")
        return PreparedInputs(
            source_mode="solar_lat",
            tau_arr=tau,
            omega_arr=omega,
            asymm_arr=asymm,
            d2s_scaling=d2s,
            height_grid=heights,
            user_obsgeoms=obsgeoms,
            stream_value=float(stream_value),
            flux_factor=float(flux_factor),
            albedo=float(albedo),
            earth_radius=normalized_earth_radius,
            geometry=geometry,
            brdf=prepared_brdf,
            surface_leaving=prepared_surface_leaving,
            thermal=None,
            lattice_counts=(beams.size, max(angles.size, 1), max(relazms.size, 1)),
            lattice_axes=(
                np.array(beams, copy=True),
                np.array(angles, copy=True),
                np.array(relazms, copy=True),
            ),
        )

    heights = np.asarray(height_grid, dtype=float)
    obsgeoms = np.asarray(user_obsgeoms, dtype=float)

    do_postprocessing, normalized_earth_radius = _validate_inputs(
        options=options,
        tau_arr=tau,
        omega_arr=omega,
        asymm_arr=asymm,
        d2s_scaling=d2s,
        height_grid=heights,
        user_obsgeoms=obsgeoms,
        earth_radius=float(earth_radius),
    )

    geometry = _prepare_geometry(
        options=options,
        height_grid=heights,
        user_obsgeoms=obsgeoms,
        stream_value=float(stream_value),
        earth_radius=normalized_earth_radius,
        do_postprocessing=do_postprocessing,
    )
    prepared_brdf = _prepare_brdf(
        brdf=brdf,
        n_geoms=obsgeoms.shape[0],
        user_obsgeoms=obsgeoms,
        stream_value=float(stream_value),
    )
    prepared_surface_leaving = _prepare_surface_leaving(
        surface_leaving=surface_leaving,
        n_geoms=obsgeoms.shape[0],
    )
    if options.do_brdf_surface and prepared_brdf is None:
        raise ValueError("do_brdf_surface=True requires brdf coefficients")
    if options.do_surface_leaving and prepared_surface_leaving is None:
        raise ValueError("do_surface_leaving=True requires surface_leaving coefficients")

    return PreparedInputs(
        source_mode="solar_obs",
        tau_arr=tau,
        omega_arr=omega,
        asymm_arr=asymm,
        d2s_scaling=d2s,
        height_grid=heights,
        user_obsgeoms=obsgeoms,
        stream_value=float(stream_value),
        flux_factor=float(flux_factor),
        albedo=float(albedo),
        earth_radius=normalized_earth_radius,
        geometry=geometry,
        brdf=prepared_brdf,
        surface_leaving=prepared_surface_leaving,
        thermal=None,
        lattice_counts=None,
    )
