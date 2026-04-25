"""First-order thermal radiance solver.

This module ports the optimized Fortran FO thermal observation-geometry path.
It exposes endpoint quantities used by the public API and keeps full level
profiles internally available for parity checks and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fo_solar_obs import _fo_eps_geometry
from .preprocess import PreparedInputs


@dataclass(frozen=True)
class FoThermalResult:
    """Endpoint and optional level-profile outputs from the thermal FO solver.

    Attributes
    ----------
    intensity_atmos_up_toa, intensity_surface_toa, intensity_total_up_toa
        Upwelling TOA atmospheric, surface, and total FO thermal terms.
    intensity_atmos_dn_toa
        Downwelling atmospheric FO thermal term at TOA.
    intensity_atmos_up_boa, intensity_surface_boa, intensity_total_up_boa
        Upwelling BOA atmospheric, surface, and total FO thermal terms.
    intensity_atmos_dn_boa
        Downwelling atmospheric FO thermal term at BOA.
    mu1
        Viewing zenith cosines.
    intensity_*_profile
        Optional arrays with shape ``(n_geometries, n_layers + 1)``.
    """

    intensity_atmos_up_toa: np.ndarray
    intensity_surface_toa: np.ndarray
    intensity_total_up_toa: np.ndarray
    intensity_atmos_dn_toa: np.ndarray
    intensity_atmos_up_boa: np.ndarray
    intensity_surface_boa: np.ndarray
    intensity_total_up_boa: np.ndarray
    intensity_atmos_dn_boa: np.ndarray
    mu1: np.ndarray
    intensity_atmos_up_profile: np.ndarray | None = None
    intensity_surface_profile: np.ndarray | None = None
    intensity_total_up_profile: np.ndarray | None = None
    intensity_atmos_dn_profile: np.ndarray | None = None

    @property
    def radiance(self) -> np.ndarray:
        """Preferred public upwelling TOA thermal FO radiance output."""
        return self.intensity_total_up_toa

    @property
    def radiance_total(self) -> np.ndarray:
        """Total upwelling TOA thermal FO radiance."""
        return self.intensity_total_up_toa

    @property
    def radiance_surface(self) -> np.ndarray:
        """Surface contribution to upwelling TOA thermal FO radiance."""
        return self.intensity_surface_toa

    @property
    def radiance_atmosphere(self) -> np.ndarray:
        """Atmospheric contribution to upwelling TOA thermal FO radiance."""
        return self.intensity_atmos_up_toa

    @property
    def radiance_up_toa(self) -> np.ndarray:
        """Total upwelling radiance at TOA."""
        return self.intensity_total_up_toa

    @property
    def radiance_up_boa(self) -> np.ndarray:
        """Total upwelling radiance at BOA."""
        return self.intensity_total_up_boa

    @property
    def radiance_up_profile(self) -> np.ndarray | None:
        """Total upwelling radiance profile when level output is available."""
        return self.intensity_total_up_profile

    def toa_up_components(self) -> dict[str, Any]:
        """Returns grouped upwelling TOA thermal FO components."""
        return {
            "atmosphere": self.intensity_atmos_up_toa,
            "surface": self.intensity_surface_toa,
            "total": self.intensity_total_up_toa,
        }

    def boa_up_components(self) -> dict[str, Any]:
        """Returns grouped upwelling BOA thermal FO components."""
        return {
            "atmosphere": self.intensity_atmos_up_boa,
            "surface": self.intensity_surface_boa,
            "total": self.intensity_total_up_boa,
        }

    def endpoint_summary(self) -> dict[str, Any]:
        """Returns a compact dictionary of all thermal FO endpoint outputs."""
        return {
            "mu1": self.mu1,
            "toa_up": self.toa_up_components(),
            "toa_down_atmosphere": self.intensity_atmos_dn_toa,
            "boa_up": self.boa_up_components(),
            "boa_down_atmosphere": self.intensity_atmos_dn_boa,
        }


@dataclass
class _ThermalFoStorage:
    """Mutable storage arrays used while accumulating thermal FO profiles."""

    atmos_up_toa: np.ndarray
    surface_toa: np.ndarray
    total_up_toa: np.ndarray
    atmos_dn_toa: np.ndarray
    atmos_up_boa: np.ndarray
    surface_boa: np.ndarray
    total_up_boa: np.ndarray
    atmos_dn_boa: np.ndarray
    atmos_up_profile: np.ndarray
    surface_profile: np.ndarray
    total_up_profile: np.ndarray
    atmos_dn_profile: np.ndarray


def _thermal_coefficients(
    prepared: PreparedInputs,
    *,
    deltaus: np.ndarray,
    do_source_deltam_scaling: bool,
) -> np.ndarray:
    """Builds per-layer thermal source coefficients.

    Parameters
    ----------
    prepared
        Preprocessed thermal inputs.
    deltaus
        Optical depths used by the FO thermal integration path.

    Returns
    -------
    np.ndarray
        Array with shape ``(2, n_layers)``. The first row is the layer
        intercept term and the second row is the linear-in-optical-depth term.
    """
    if prepared.thermal is None:
        raise ValueError("thermal inputs are required for FO thermal")

    nlayers = prepared.tau_arr.size
    tcom = np.zeros((2, nlayers), dtype=float)
    bb_inputn1 = float(prepared.thermal.thermal_bb_input[0])

    for n in range(nlayers):
        single_scatter_scale = 1.0 - float(prepared.omega_arr[n])
        if do_source_deltam_scaling:
            single_scatter_scale /= 1.0 - float(prepared.omega_arr[n]) * float(
                prepared.d2s_scaling[n]
            )

        bb_inputn = float(prepared.thermal.thermal_bb_input[n + 1])
        thermal_slope = (bb_inputn - bb_inputn1) / float(deltaus[n])
        tcom[0, n] = bb_inputn1 * single_scatter_scale
        tcom[1, n] = thermal_slope * single_scatter_scale
        bb_inputn1 = bb_inputn

    return tcom


def _initialize_storage(n_geometries: int, n_layers: int) -> _ThermalFoStorage:
    """Allocates endpoint and level-profile storage for thermal FO."""

    def endpoint() -> np.ndarray:
        return np.zeros(n_geometries, dtype=float)

    def profile() -> np.ndarray:
        return np.zeros((n_geometries, n_layers + 1), dtype=float)

    return _ThermalFoStorage(
        atmos_up_toa=endpoint(),
        surface_toa=endpoint(),
        total_up_toa=endpoint(),
        atmos_dn_toa=endpoint(),
        atmos_up_boa=endpoint(),
        surface_boa=endpoint(),
        total_up_boa=endpoint(),
        atmos_dn_boa=endpoint(),
        atmos_up_profile=profile(),
        surface_profile=profile(),
        total_up_profile=profile(),
        atmos_dn_profile=profile(),
    )


def _prepare_eps_geometry(
    prepared: PreparedInputs, mu1: np.ndarray, *, nfine: int
) -> dict[str, Any]:
    """Creates the EPS geometry dictionary used by thermal FO."""
    if prepared.height_grid is None:
        raise ValueError("height_grid is required for EPS thermal FO")

    # The thermal FO geometry helper shares the solar EPS geometry machinery.
    # The solar-angle column is unused for direct thermal emission, so it is
    # filled with zeros while preserving the viewing zenith angle.
    dummy_obsgeoms = np.column_stack(
        (
            np.zeros(mu1.size, dtype=float),
            np.rad2deg(np.arccos(np.clip(mu1, -1.0, 1.0))),
            np.zeros(mu1.size, dtype=float),
        )
    )
    return _fo_eps_geometry(
        user_obsgeoms=dummy_obsgeoms,
        height_grid=prepared.height_grid,
        earth_radius=prepared.earth_radius,
        nfine=nfine,
        vsign=1.0,
    )


def _plane_parallel_up_sources(
    *,
    deltaus: np.ndarray,
    tcom: np.ndarray,
    mu1v: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes plane-parallel upwelling transmittances and layer sources."""
    nlayers = deltaus.size
    lostrans_up = np.zeros(nlayers, dtype=float)
    sources_up = np.zeros(nlayers, dtype=float)
    for n in range(nlayers):
        lostau = float(deltaus[n]) / mu1v
        lostrans = np.exp(-lostau) if lostau < cutoff else 0.0
        t_mult_up1 = tcom[0, n] + tcom[1, n] * mu1v
        t_mult_up0 = -t_mult_up1 - tcom[1, n] * float(deltaus[n])
        lostrans_up[n] = lostrans
        sources_up[n] = t_mult_up0 * lostrans + t_mult_up1
    return lostrans_up, sources_up


def _plane_parallel_down_sources(
    *,
    deltaus: np.ndarray,
    tcom: np.ndarray,
    mu1v: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes plane-parallel downwelling transmittances and layer sources."""
    nlayers = deltaus.size
    lostrans_dn = np.zeros(nlayers, dtype=float)
    sources_dn = np.zeros(nlayers, dtype=float)
    for n in range(nlayers - 1, -1, -1):
        lostau = float(deltaus[n]) / mu1v
        lostrans = np.exp(-lostau) if lostau < cutoff else 0.0
        t_mult_dn1 = tcom[0, n] - tcom[1, n] * mu1v
        lostrans_dn[n] = lostrans
        sources_dn[n] = -t_mult_dn1 * lostrans + t_mult_dn1 + tcom[1, n] * float(deltaus[n])
    return lostrans_dn, sources_dn


def _accumulate_up_profile(
    *,
    storage: _ThermalFoStorage,
    geometry_index: int,
    lostrans_up: np.ndarray,
    sources_up: np.ndarray,
    surface_source: float,
) -> None:
    """Accumulates upwelling atmospheric and surface profiles from BOA to TOA."""
    nlayers = lostrans_up.size
    cumsource_up = 0.0
    cumsource_surface = surface_source
    storage.atmos_up_profile[geometry_index, nlayers] = cumsource_up
    storage.surface_profile[geometry_index, nlayers] = cumsource_surface
    storage.total_up_profile[geometry_index, nlayers] = cumsource_up + cumsource_surface

    for n in range(nlayers - 1, -1, -1):
        lostrans_upn = lostrans_up[n]
        cumsource_surface = lostrans_upn * cumsource_surface
        cumsource_up = lostrans_upn * cumsource_up + sources_up[n]
        storage.atmos_up_profile[geometry_index, n] = cumsource_up
        storage.surface_profile[geometry_index, n] = cumsource_surface
        storage.total_up_profile[geometry_index, n] = cumsource_up + cumsource_surface

    storage.atmos_up_toa[geometry_index] = storage.atmos_up_profile[geometry_index, 0]
    storage.surface_toa[geometry_index] = storage.surface_profile[geometry_index, 0]
    storage.total_up_toa[geometry_index] = storage.total_up_profile[geometry_index, 0]
    storage.atmos_up_boa[geometry_index] = storage.atmos_up_profile[geometry_index, nlayers]
    storage.surface_boa[geometry_index] = storage.surface_profile[geometry_index, nlayers]
    storage.total_up_boa[geometry_index] = storage.total_up_profile[geometry_index, nlayers]


def _accumulate_down_profile(
    *,
    storage: _ThermalFoStorage,
    geometry_index: int,
    lostrans_dn: np.ndarray,
    sources_dn: np.ndarray,
) -> None:
    """Accumulates downwelling atmospheric profiles from TOA to BOA."""
    cumsource_dn = 0.0
    storage.atmos_dn_profile[geometry_index, 0] = cumsource_dn
    for n in range(lostrans_dn.size):
        cumsource_dn = sources_dn[n] + lostrans_dn[n] * cumsource_dn
        storage.atmos_dn_profile[geometry_index, n + 1] = cumsource_dn

    storage.atmos_dn_toa[geometry_index] = storage.atmos_dn_profile[geometry_index, 0]
    storage.atmos_dn_boa[geometry_index] = storage.atmos_dn_profile[geometry_index, -1]


def _eps_up_down_sources(
    *,
    geometry: dict[str, Any],
    geometry_index: int,
    deltaus: np.ndarray,
    extinction: np.ndarray,
    tcom: np.ndarray,
    height_grid: np.ndarray,
    earth_radius: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes EPS thermal up/down transmittances and layer sources.

    The formulas match the layer/fine-quadrature loops in the optimized
    Fortran-style path, but evaluate the per-layer quadrature terms with NumPy
    arrays to avoid Python loop overhead in scalar spectral runs.
    """
    do_nadir = bool(geometry["do_nadir"][geometry_index])
    kn = extinction
    tcom1 = tcom[0]
    tcom2 = tcom[1]

    if do_nadir:
        lostrans = np.where(deltaus < cutoff, np.exp(-deltaus), 0.0)
        xfine = geometry["xfine"][:, :, geometry_index]
        wfine = geometry["wfine"][:, :, geometry_index]
        xjkn = xfine * kn[np.newaxis, :]
        solution = tcom1[np.newaxis, :] + xjkn * tcom2[np.newaxis, :]
        sources_up = np.sum(solution * kn[np.newaxis, :] * np.exp(-xjkn) * wfine, axis=0)

        radii = earth_radius + height_grid
        rdiff = radii[:-1] - radii[1:]
        sources_dn = np.sum(
            solution
            * kn[np.newaxis, :]
            * np.exp(-kn[np.newaxis, :] * (rdiff[np.newaxis, :] - xfine))
            * wfine,
            axis=0,
        )
        return lostrans, sources_up, lostrans.copy(), sources_dn

    rayconv = float(geometry["raycon"][geometry_index])
    cota = geometry["cota"][:, geometry_index]
    cot_upper = cota[:-1]
    cot_lower = cota[1:]
    ke = rayconv * kn
    lostau = ke * (cot_upper - cot_lower)
    lostrans = np.where(lostau < cutoff, np.exp(-lostau), 0.0)

    xfine = geometry["xfine"][:, :, geometry_index]
    wfine = geometry["wfine"][:, :, geometry_index]
    cotfine = geometry["cotfine"][:, :, geometry_index]
    csqfine = geometry["csqfine"][:, :, geometry_index]
    xjkn = xfine * kn[np.newaxis, :]
    solution = tcom1[np.newaxis, :] + xjkn * tcom2[np.newaxis, :]
    weight = ke[np.newaxis, :] * csqfine * wfine
    sources_up = np.sum(
        solution * weight * np.exp(-ke[np.newaxis, :] * (cot_upper[np.newaxis, :] - cotfine)),
        axis=0,
    )
    sources_dn = np.sum(
        solution * weight * np.exp(-ke[np.newaxis, :] * (cotfine - cot_lower[np.newaxis, :])),
        axis=0,
    )
    return lostrans, sources_up, lostrans.copy(), sources_dn


def solve_fo_thermal(
    prepared: PreparedInputs,
    *,
    do_plane_parallel: bool,
    do_optical_deltam_scaling: bool = True,
    do_source_deltam_scaling: bool = False,
    nfine: int = 3,
) -> FoThermalResult:
    """Runs the optimized thermal FO observation-geometry solver.

    Parameters
    ----------
    prepared
        Validated and normalized input bundle.
    do_plane_parallel
        If true, use the plane-parallel thermal FO path. Otherwise use EPS
        line-of-sight geometry.
    do_optical_deltam_scaling
        If true, apply FO optical-depth delta-M scaling before integrating the
        thermal source.
    do_source_deltam_scaling
        If true, apply the Fortran FO thermal core source multiplier
        ``(1 - omega) / (1 - omega * truncfac)``. Drivers may disable this
        when reproducing wrapper conventions that scale optical depth but not
        the thermal source coefficients.
    nfine
        Number of fine-layer quadrature divisions used by the EPS geometry
        integration path.

    Returns
    -------
    FoThermalResult
        Thermal FO endpoint outputs plus level profiles.
    """
    if prepared.source_mode != "thermal":
        raise NotImplementedError("FO thermal is implemented for source_mode='thermal' only")
    if prepared.thermal is None:
        raise ValueError("thermal inputs are required for FO thermal")

    nlayers = prepared.tau_arr.size
    mu1 = prepared.geometry.user_streams
    if mu1.size == 0:
        raise ValueError("thermal FO requires at least one user angle")

    if do_optical_deltam_scaling:
        deltaus = prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling)
    else:
        deltaus = prepared.tau_arr.copy()
    surfbb = prepared.thermal.surfbb
    user_emissivity = np.full(mu1.size, prepared.thermal.emissivity, dtype=float)
    extinction = (
        deltaus / (prepared.height_grid[:-1] - prepared.height_grid[1:])
        if prepared.height_grid is not None
        else None
    )

    tcom = _thermal_coefficients(
        prepared,
        deltaus=deltaus,
        do_source_deltam_scaling=do_source_deltam_scaling,
    )
    storage = _initialize_storage(mu1.size, nlayers)

    cutoff = 88.0

    if do_plane_parallel:
        geometry = None
    else:
        geometry = _prepare_eps_geometry(prepared, mu1, nfine=nfine)

    for v, mu1v in enumerate(mu1):
        if do_plane_parallel:
            lostrans_up, sources_up = _plane_parallel_up_sources(
                deltaus=deltaus,
                tcom=tcom,
                mu1v=float(mu1v),
                cutoff=cutoff,
            )
        else:
            assert geometry is not None
            lostrans_up, sources_up, lostrans_dn, sources_dn = _eps_up_down_sources(
                geometry=geometry,
                geometry_index=v,
                deltaus=deltaus,
                extinction=extinction,
                tcom=tcom,
                height_grid=prepared.height_grid,
                earth_radius=prepared.earth_radius,
                cutoff=cutoff,
            )

        _accumulate_up_profile(
            storage=storage,
            geometry_index=v,
            lostrans_up=lostrans_up,
            sources_up=sources_up,
            surface_source=surfbb * user_emissivity[v],
        )

        if do_plane_parallel:
            lostrans_dn, sources_dn = _plane_parallel_down_sources(
                deltaus=deltaus,
                tcom=tcom,
                mu1v=float(mu1v),
                cutoff=cutoff,
            )

        _accumulate_down_profile(
            storage=storage,
            geometry_index=v,
            lostrans_dn=lostrans_dn,
            sources_dn=sources_dn,
        )

    return FoThermalResult(
        intensity_atmos_up_toa=storage.atmos_up_toa,
        intensity_surface_toa=storage.surface_toa,
        intensity_total_up_toa=storage.total_up_toa,
        intensity_atmos_dn_toa=storage.atmos_dn_toa,
        intensity_atmos_up_boa=storage.atmos_up_boa,
        intensity_surface_boa=storage.surface_boa,
        intensity_total_up_boa=storage.total_up_boa,
        intensity_atmos_dn_boa=storage.atmos_dn_boa,
        mu1=mu1.copy(),
        intensity_atmos_up_profile=storage.atmos_up_profile,
        intensity_surface_profile=storage.surface_profile,
        intensity_total_up_profile=storage.total_up_profile,
        intensity_atmos_dn_profile=storage.atmos_dn_profile,
    )
