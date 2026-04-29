"""Public API for the Python 2S-ESS package."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .rtsolver.backend import (
    TorchContext,
    detect_torch_context,
    has_torch,
    to_numpy,
    value_to_torch,
)
from .rtsolver.fo_solar_obs import (
    FoSolarObsResult,
    fo_scatter_term_henyey_greenstein,
    solve_fo_solar_obs,
)
from .rtsolver.fo_thermal import FoThermalResult, solve_fo_thermal
from .rtsolver.lattice_result import add_lattice_axes, lattice_shape, reshape_lattice_array
from .optical.delta_m import (
    default_delta_m_truncation_factor,
    validate_delta_m_truncation_factor,
)
from .rtsolver.preprocess import PreparedInputs, prepare_inputs
from .rtsolver.solver import solve_optimized_solar_obs

_MODE_TO_SOURCE_MODE = {
    "solar": "solar_obs",
    "solar_lattice": "solar_lat",
    "thermal": "thermal",
}

_GEOMETRY_TO_FO_MODE = {
    "pseudo_spherical": "eps",
    "enhanced_pseudo_spherical": "eps",
    "eps": "eps",
    "regular_pseudo_spherical": "rps",
    "rps": "rps",
}


@dataclass(frozen=True)
class _CoreOptions:
    """Internal option shape expected by the Fortran-style solver core."""

    n_layers: int
    source_mode: str
    do_upwelling: bool
    do_dnwelling: bool
    do_plane_parallel: bool
    do_delta_scaling: bool
    do_level_output: bool
    do_mvout_only: bool
    do_additional_mvout: bool
    do_surface_leaving: bool
    do_sl_isotropic: bool
    do_brdf_surface: bool
    bvp_solver: str
    thermal_tcutoff: float


@dataclass(frozen=True)
class TwoStreamEssOptions:
    """Configuration for a forward-model run.

    Parameters
    ----------
    nlyr
        Number of atmospheric layers in the problem.
    backend
        Execution backend. Use ``"numpy"`` for the reference CPU path or
        ``"torch"`` for the native tensor path where supported.
    mode
        Forward-model source mode. Supported values are ``"solar"``,
        ``"solar_lattice"``, and ``"thermal"``.
    upwelling, downwelling
        Flags controlling whether upwelling and/or downwelling outputs are
        computed.
    plane_parallel
        Whether to use the plane-parallel geometry approximation.
    delta_scaling
        Whether to apply delta-M scaling to the optical properties.
    output_levels
        Whether full level profiles should be returned.
    mvout_only, additional_mvout
        Flags controlling which flux outputs are produced.
    surface_leaving, sl_isotropic
        Solar surface-leaving configuration flags.
    brdf_surface
        Whether explicit BRDF inputs are enabled.
    bvp_solver
        Boundary-value-problem solver selection. Supported values are
        ``"scipy"``, ``"banded"``, and ``"pentadiag"``.
    thermal_tcutoff
        Thermal source cutoff used by the optimized thermal solver.
    torch_device
        Optional torch device string such as ``"cpu"``, ``"cuda"``, or
        ``"mps"`` used when the torch backend converts NumPy inputs to tensors.
        If omitted, the device is inferred from tensor inputs or defaults to CPU.
    torch_dtype
        Optional torch dtype name for converted NumPy inputs, such as
        ``"float64"`` or ``"float32"``. If omitted, CPU defaults to
        ``float64`` for parity; MPS defaults to ``float32`` because Apple MPS
        does not support float64 kernels.
    torch_enable_grad
        Whether torch forward calls should record autograd operations. Disable
        this for pure inference/comparison runs to avoid gradient bookkeeping.
    fo_optical_delta_m_scaling
        Optional override for FO optical-depth delta-M scaling. ``None``
        inherits ``delta_scaling``.
    fo_thermal_source_delta_m_scaling
        Optional override for the thermal FO source-side delta-M multiplier
        used by the raw Fortran FO thermal core. ``None`` disables source-side
        scaling, matching the current validated FO thermal path.
    """

    nlyr: int
    backend: str = "numpy"
    mode: str = "solar"
    upwelling: bool = True
    downwelling: bool = False
    plane_parallel: bool = False
    delta_scaling: bool = True
    output_levels: bool = False
    mvout_only: bool = False
    additional_mvout: bool = False
    surface_leaving: bool = False
    sl_isotropic: bool = True
    brdf_surface: bool = False
    bvp_solver: str = "scipy"
    thermal_tcutoff: float = 1.0e-8
    torch_device: str | None = None
    torch_dtype: str | None = None
    torch_enable_grad: bool = True
    fo_optical_delta_m_scaling: bool | None = None
    fo_thermal_source_delta_m_scaling: bool | None = None

    def __post_init__(self) -> None:
        if self.nlyr <= 0:
            raise ValueError("nlyr must be positive")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        if self.mode not in _MODE_TO_SOURCE_MODE:
            allowed = "', '".join(sorted(_MODE_TO_SOURCE_MODE))
            raise ValueError(f"mode must be one of: '{allowed}'")
        if self.backend == "torch" and not has_torch():
            raise ValueError("backend='torch' requires torch to be installed")
        if self.bvp_solver not in {"scipy", "banded", "pentadiag"}:
            raise ValueError("bvp_solver must be 'scipy', 'banded', or 'pentadiag'")
        if not math.isfinite(self.thermal_tcutoff) or self.thermal_tcutoff <= 0.0:
            raise ValueError("thermal_tcutoff must be positive and finite")

    @property
    def effective_fo_optical_deltam_scaling(self) -> bool:
        """Returns FO optical delta-M control after applying inheritance."""
        return (
            self.delta_scaling
            if self.fo_optical_delta_m_scaling is None
            else self.fo_optical_delta_m_scaling
        )

    @property
    def effective_fo_thermal_source_deltam_scaling(self) -> bool:
        """Returns thermal FO source delta-M control after applying inheritance."""
        if self.fo_thermal_source_delta_m_scaling is not None:
            return self.fo_thermal_source_delta_m_scaling
        return False


@dataclass(frozen=True)
class TwoStreamEssBatchResult:
    """Radiances from a batched public ``forward`` call.

    By default batched calls expose the fast wavelength/column endpoint path.
    When ``output_levels=True`` the result also includes upwelling radiance
    profiles ordered from TOA to BOA along the final level axis.
    """

    radiance_2s: Any
    radiance_total: Any
    radiance_fo: Any | None = None
    radiance_profile_2s: Any | None = None
    radiance_profile_fo: Any | None = None
    radiance_profile_total: Any | None = None
    batch_shape: tuple[int, ...] = ()
    geometry_shape: tuple[int, ...] = ()

    @property
    def radiance(self) -> Any:
        """Preferred public radiance output."""
        return self.radiance_total

    @property
    def radiance_profile(self) -> Any | None:
        """Preferred public level radiance profile when available."""
        return self.radiance_profile_total


@dataclass(frozen=True)
class TwoStreamEssResult:
    """Forward-model outputs plus convenience accessors.

    Attributes
    ----------
    intensity_toa, intensity_boa
        Main 2S radiance outputs at the top and bottom of atmosphere.
    fluxes_toa, fluxes_boa
        Regular and actinic flux outputs.
    radlevel_up, radlevel_dn
        Optional level-by-level upwelling and downwelling radiance profiles.
    combined_intensity_toa, combined_intensity_boa
        Combined 2S plus FO outputs when a validated combination rule exists.
        The BOA field is intentionally unset for currently unsupported cases.
    fo_*
        Optional first-order component outputs returned when ``include_fo=True``
        is used on the main forward API.
    lattice_counts, lattice_axes
        Original lattice-grid metadata for results produced from
        ``mode="solar_lattice"``.
    """

    intensity_toa: np.ndarray
    intensity_boa: np.ndarray
    fluxes_toa: np.ndarray
    fluxes_boa: np.ndarray
    radlevel_up: np.ndarray
    radlevel_dn: np.ndarray
    # `combined_intensity_toa` is the validated observation-geometry 2S-ESS
    # total radiance, assembled as 2S multiple scatter plus FO/ESS first order.
    # `combined_intensity_boa` stays unset until an equivalent BOA-level FO term
    # is exposed and validated in the Python API.
    combined_intensity_toa: np.ndarray | None = None
    combined_intensity_boa: np.ndarray | None = None
    fo_intensity_total: np.ndarray | None = None
    fo_intensity_ss: np.ndarray | None = None
    fo_intensity_db: np.ndarray | None = None
    fo_mu0: np.ndarray | None = None
    fo_mu1: np.ndarray | None = None
    fo_cosscat: np.ndarray | None = None
    fo_do_nadir: np.ndarray | None = None
    fo_thermal_atmos_up_toa: np.ndarray | None = None
    fo_thermal_surface_toa: np.ndarray | None = None
    fo_thermal_total_up_toa: np.ndarray | None = None
    fo_thermal_atmos_dn_toa: np.ndarray | None = None
    fo_thermal_atmos_up_boa: np.ndarray | None = None
    fo_thermal_surface_boa: np.ndarray | None = None
    fo_thermal_total_up_boa: np.ndarray | None = None
    fo_thermal_atmos_dn_boa: np.ndarray | None = None
    fo_intensity_total_profile: np.ndarray | None = None
    fo_thermal_total_up_profile: np.ndarray | None = None
    output_levels: bool = False
    lattice_counts: tuple[int, int, int] | None = None
    lattice_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def radiance(self) -> np.ndarray:
        """Preferred public radiance output."""
        return self.radiance_total

    @property
    def radiance_2s(self) -> np.ndarray:
        """Two-stream radiance at TOA."""
        return self.intensity_toa

    @property
    def radiance_fo(self) -> np.ndarray | None:
        """First-order radiance component when an FO solve was attached."""
        if self.fo_intensity_total is not None:
            return self.fo_intensity_total
        return self.fo_thermal_total_up_toa

    @property
    def radiance_total(self) -> np.ndarray:
        """Best available total radiance for the configured mode."""
        if self.combined_intensity_toa is not None:
            return self.combined_intensity_toa
        if self.fo_thermal_total_up_toa is not None:
            return self.intensity_toa + self.fo_thermal_total_up_toa
        return self.intensity_toa

    @property
    def radiance_profile_2s(self) -> np.ndarray | None:
        """Two-stream upwelling level radiance profile when requested."""
        return self.radlevel_up if self.output_levels else None

    @property
    def radiance_profile_fo(self) -> np.ndarray | None:
        """First-order upwelling level radiance profile when available."""
        if not self.output_levels:
            return None
        if self.fo_intensity_total_profile is not None:
            return self.fo_intensity_total_profile
        return self.fo_thermal_total_up_profile

    @property
    def radiance_profile_total(self) -> np.ndarray | None:
        """Best available total upwelling level radiance profile."""
        profile_2s = self.radiance_profile_2s
        if profile_2s is None:
            return None
        profile_fo = self.radiance_profile_fo
        if profile_fo is None and (
            self.fo_intensity_total is not None or self.fo_thermal_total_up_toa is not None
        ):
            return None
        return profile_2s if profile_fo is None else profile_2s + profile_fo

    @property
    def radiance_profile(self) -> np.ndarray | None:
        """Preferred public level radiance profile when available."""
        return self.radiance_profile_total

    def _lattice_shape(self) -> tuple[int, int, int]:
        """Returns the expected lattice shape for reshape helpers."""
        return lattice_shape(self.lattice_counts)

    def _reshape_lattice_array(self, values: np.ndarray) -> np.ndarray:
        """Reshapes a 1D geometry array back to lattice form."""
        return reshape_lattice_array(values, self.lattice_counts)

    def _reshape_lattice_radlevels(self, values: np.ndarray) -> np.ndarray:
        """Reshapes level outputs to lattice dimensions."""
        shape = self._lattice_shape()
        return values.reshape(shape + (values.shape[-1],))

    def _reshape_lattice_fluxes(self, values: np.ndarray) -> np.ndarray:
        """Reshapes flux arrays to lattice-compatible dimensions.

        Solar lattice flux outputs are beam-level quantities, so after the
        solver collapses them to one column per beam this helper returns a
        ``(2, n_beams)`` array instead of inventing user-angle and azimuth
        dimensions that are not physically present in the output.
        """
        shape = self._lattice_shape()
        if values.shape[1] == shape[0]:
            return values
        return values.reshape((values.shape[0],) + shape)

    def reshape_lattice(self) -> dict[str, Any]:
        """Returns all lattice-compatible outputs reshaped together."""
        reshaped: dict[str, Any] = {
            "intensity_toa": self._reshape_lattice_array(self.intensity_toa),
            "intensity_boa": self._reshape_lattice_array(self.intensity_boa),
            "radlevel_up": self._reshape_lattice_radlevels(self.radlevel_up),
            "radlevel_dn": self._reshape_lattice_radlevels(self.radlevel_dn),
            "fluxes_toa": self._reshape_lattice_fluxes(self.fluxes_toa),
            "fluxes_boa": self._reshape_lattice_fluxes(self.fluxes_boa),
        }
        if self.combined_intensity_toa is not None:
            reshaped["combined_intensity_toa"] = self._reshape_lattice_array(
                self.combined_intensity_toa
            )
        if self.combined_intensity_boa is not None:
            reshaped["combined_intensity_boa"] = self._reshape_lattice_array(
                self.combined_intensity_boa
            )
        if self.fo_intensity_total is not None:
            reshaped["fo_intensity_total"] = self._reshape_lattice_array(self.fo_intensity_total)
        if self.fo_intensity_ss is not None:
            reshaped["fo_intensity_ss"] = self._reshape_lattice_array(self.fo_intensity_ss)
        if self.fo_intensity_db is not None:
            reshaped["fo_intensity_db"] = self._reshape_lattice_array(self.fo_intensity_db)
        return add_lattice_axes(reshaped, self.lattice_axes)

    def solar_components(self) -> dict[str, Any]:
        """Returns the available solar-component breakdown as a flat mapping."""
        return {
            "twostream_toa": self.intensity_toa,
            "twostream_boa": self.intensity_boa,
            "fo_total": self.fo_intensity_total,
            "fo_ss": self.fo_intensity_ss,
            "fo_db": self.fo_intensity_db,
            "combined_toa": self.combined_intensity_toa,
        }

    def solar_components_lattice(self) -> dict[str, Any]:
        """Returns the solar-component breakdown reshaped to lattice form."""
        if self.lattice_counts is None:
            raise ValueError("lattice_counts are not available for this result")
        reshaped: dict[str, Any] = {
            "twostream_toa": self.intensity_toa_lattice(),
            "twostream_boa": self.intensity_boa_lattice(),
        }
        if self.fo_intensity_total is not None:
            reshaped["fo_total"] = self._reshape_lattice_array(self.fo_intensity_total)
        if self.fo_intensity_ss is not None:
            reshaped["fo_ss"] = self._reshape_lattice_array(self.fo_intensity_ss)
        if self.fo_intensity_db is not None:
            reshaped["fo_db"] = self._reshape_lattice_array(self.fo_intensity_db)
        if self.combined_intensity_toa is not None:
            reshaped["combined_toa"] = self.combined_intensity_toa_lattice()
        return add_lattice_axes(reshaped, self.lattice_axes)

    def thermal_components(self) -> dict[str, Any]:
        """Returns the available thermal-component breakdown as a flat mapping."""
        return {
            "twostream_toa": self.intensity_toa,
            "twostream_boa": self.intensity_boa,
            "fo_mu1": self.fo_mu1,
            "fo_toa_up": {
                "atmosphere": self.fo_thermal_atmos_up_toa,
                "surface": self.fo_thermal_surface_toa,
                "total": self.fo_thermal_total_up_toa,
            },
            "fo_toa_down_atmosphere": self.fo_thermal_atmos_dn_toa,
            "fo_boa_up": {
                "atmosphere": self.fo_thermal_atmos_up_boa,
                "surface": self.fo_thermal_surface_boa,
                "total": self.fo_thermal_total_up_boa,
            },
            "fo_boa_down_atmosphere": self.fo_thermal_atmos_dn_boa,
        }

    def intensity_toa_lattice(self) -> np.ndarray:
        """Returns TOA intensity reshaped to the original lattice grid."""
        return self._reshape_lattice_array(self.intensity_toa)

    def intensity_boa_lattice(self) -> np.ndarray:
        """Returns BOA intensity reshaped to the original lattice grid."""
        return self._reshape_lattice_array(self.intensity_boa)

    def combined_intensity_toa_lattice(self) -> np.ndarray:
        """Returns combined TOA intensity reshaped to the lattice grid."""
        if self.combined_intensity_toa is None:
            raise ValueError("combined_intensity_toa is not available for this result")
        return self._reshape_lattice_array(self.combined_intensity_toa)

    def fluxes_toa_lattice(self) -> np.ndarray:
        """Returns TOA fluxes reshaped to the original lattice grid."""
        return self._reshape_lattice_fluxes(self.fluxes_toa)

    def fluxes_boa_lattice(self) -> np.ndarray:
        """Returns BOA fluxes reshaped to the original lattice grid."""
        return self._reshape_lattice_fluxes(self.fluxes_boa)

    def radlevel_up_lattice(self) -> np.ndarray:
        """Returns upwelling level outputs reshaped to the lattice grid."""
        return self._reshape_lattice_radlevels(self.radlevel_up)

    def radlevel_dn_lattice(self) -> np.ndarray:
        """Returns downwelling level outputs reshaped to the lattice grid."""
        return self._reshape_lattice_radlevels(self.radlevel_dn)


class TwoStreamEss:
    """Primary solver entry point for the Python port.

    Parameters
    ----------
    options
        Forward-model configuration used for subsequent solver calls.
    """

    def __init__(self, options: TwoStreamEssOptions):
        self.options = options

    @property
    def _source_mode(self) -> str:
        """Returns the internal source-mode selector for the configured mode."""
        return _MODE_TO_SOURCE_MODE[self.options.mode]

    def _core_options(self) -> _CoreOptions:
        """Returns the private option adapter expected by core solver modules."""
        return _CoreOptions(
            n_layers=self.options.nlyr,
            source_mode=self._source_mode,
            do_upwelling=self.options.upwelling,
            do_dnwelling=self.options.downwelling,
            do_plane_parallel=self.options.plane_parallel,
            do_delta_scaling=self.options.delta_scaling,
            do_level_output=self.options.output_levels,
            do_mvout_only=self.options.mvout_only,
            do_additional_mvout=self.options.additional_mvout,
            do_surface_leaving=self.options.surface_leaving,
            do_sl_isotropic=self.options.sl_isotropic,
            do_brdf_surface=self.options.brdf_surface,
            bvp_solver=self.options.bvp_solver,
            thermal_tcutoff=self.options.thermal_tcutoff,
        )

    @staticmethod
    def _validate_fo_geometry_mode(fo_geometry_mode: str) -> None:
        """Validates the named FO solar geometry mode."""
        if fo_geometry_mode not in {"eps", "rps"}:
            raise ValueError("fo_geometry_mode must be 'eps' or 'rps'")

    @staticmethod
    def _normalize_fo_geometry(geometry: str) -> str:
        """Maps public FO geometry names to the internal EPS/RPS selector."""
        try:
            return _GEOMETRY_TO_FO_MODE[geometry]
        except KeyError as exc:
            allowed = "', '".join(sorted(_GEOMETRY_TO_FO_MODE))
            raise ValueError(f"geometry must be one of: '{allowed}'") from exc

    @staticmethod
    def _as_public_1d(name: str, value: Any | None) -> np.ndarray | None:
        """Normalizes scalar-or-1D public angle inputs."""
        if value is None:
            return None
        arr = np.asarray(to_numpy(value), dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1D array")
        return arr

    def _normalize_public_angles(self, angles: Any | None) -> np.ndarray | None:
        """Normalizes public ``angles`` before Fortran-style validation."""
        if angles is None:
            return None
        arr = np.asarray(to_numpy(angles), dtype=float)
        source_mode = self._source_mode
        if source_mode == "solar_obs":
            if arr.ndim == 1 and arr.size == 3:
                return arr.reshape(1, 3)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    "angles must have shape (3,) for one solar geometry or "
                    "(ngeom, 3) for multiple solar geometries; columns are "
                    "[sza, vza, raz] in degrees"
                )
            return arr
        if source_mode == "thermal":
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError("angles must be a scalar or 1D thermal view-zenith array")
            return arr
        if source_mode == "solar_lat":
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError("angles must be a scalar or 1D solar-lattice view-angle array")
            return arr
        return arr

    def _default_stream(self) -> float:
        """Returns the public API default two-stream quadrature cosine."""
        return 1.0 / math.sqrt(3.0)

    def _translate_public_forward_args(
        self,
        *,
        tau: Any,
        ssa: Any,
        g: Any,
        z: Any | None,
        angles: Any | None,
        stream: float | None,
        fbeam: Any,
        delta_m_truncation_factor: Any | None,
        view_angles: Any | None,
        beam_szas: Any | None,
        relazms: Any | None,
        planck: Any | None,
        surface_planck: Any,
        geometry: str,
    ) -> dict[str, Any]:
        """Translates friendly public names to the solver's internal names."""
        source_mode = self._source_mode
        user_obsgeoms = None
        user_angles = None
        user_relazms = None
        angles = self._normalize_public_angles(angles)
        view_angles = self._as_public_1d("view_angles", view_angles)
        beam_szas = self._as_public_1d("beam_szas", beam_szas)
        relazms = self._as_public_1d("relazms", relazms)

        if source_mode == "solar_obs":
            if angles is None:
                raise ValueError("angles are required for mode='solar'")
            user_obsgeoms = angles
        elif source_mode == "solar_lat":
            user_angles = view_angles if view_angles is not None else angles
            user_relazms = relazms
            if beam_szas is None:
                raise ValueError("beam_szas is required for mode='solar_lattice'")
            if user_angles is None:
                raise ValueError("angles or view_angles are required for mode='solar_lattice'")
            if user_relazms is None:
                raise ValueError("relazms is required for mode='solar_lattice'")
        elif source_mode == "thermal":
            user_angles = view_angles if view_angles is not None else angles
            if user_angles is None:
                raise ValueError("angles are required for mode='thermal'")
            if planck is None:
                raise ValueError("planck is required for mode='thermal'")

        stream_value = self._default_stream() if stream is None else float(stream)
        if not math.isfinite(stream_value) or stream_value <= 0.0 or stream_value > 1.0:
            raise ValueError("stream must satisfy 0 < stream <= 1")

        return {
            "tau_arr": tau,
            "omega_arr": ssa,
            "asymm_arr": g,
            "height_grid": z,
            "user_obsgeoms": user_obsgeoms,
            "user_angles": user_angles,
            "beam_szas": beam_szas,
            "user_relazms": user_relazms,
            "stream_value": stream_value,
            "flux_factor": fbeam,
            "d2s_scaling": delta_m_truncation_factor,
            "thermal_bb_input": planck,
            "surfbb": surface_planck,
            "fo_geometry_mode": self._normalize_fo_geometry(geometry),
        }

    def _public_forward_is_batched(self, tau_arr: Any) -> bool:
        """Returns true when ``tau`` has leading wavelength/column dimensions."""
        shape = getattr(tau_arr, "shape", None)
        if shape is None:
            shape = np.asarray(tau_arr).shape
        shape = tuple(int(dim) for dim in shape)
        return len(shape) > 1 and shape[-1] == self.options.nlyr

    @staticmethod
    def _require_finite(name: str, value: np.ndarray) -> None:
        """Raises a public error when a batched input contains non-finite values."""
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        if name == "tau" and np.any(value < 0.0):
            raise ValueError("tau must be nonnegative")
        if name == "g" and np.any((value <= -1.0) | (value >= 1.0)):
            raise ValueError("g must satisfy -1 < g < 1")

    @staticmethod
    def _broadcast_batch_layers(
        name: str,
        value: Any | None,
        *,
        batch_shape: tuple[int, ...],
        width: int,
        default: float | None = None,
    ) -> np.ndarray:
        """Broadcasts a layer-shaped public input to flattened batch rows."""
        target = batch_shape + (width,)
        if value is None:
            if default is None:
                raise ValueError(f"{name} is required")
            arr = np.full(target, float(default), dtype=float)
        else:
            arr = np.asarray(to_numpy(value), dtype=float)
            if arr.shape == (width,):
                arr = np.broadcast_to(arr, target)
            else:
                try:
                    arr = np.broadcast_to(arr, target)
                except ValueError as exc:
                    raise ValueError(
                        f"{name} must have shape {target} or ({width},) for batched forward"
                    ) from exc
        TwoStreamEss._require_finite(name, arr)
        return np.ascontiguousarray(arr.reshape(-1, width), dtype=float)

    @staticmethod
    def _broadcast_batch_scalar(
        name: str,
        value: Any,
        *,
        batch_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Broadcasts a scalar-or-batch public input to flattened batch rows."""
        arr = np.asarray(to_numpy(value), dtype=float)
        try:
            broadcast = np.broadcast_to(arr, batch_shape)
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar or broadcastable to {batch_shape}") from exc
        TwoStreamEss._require_finite(name, broadcast)
        return np.ascontiguousarray(broadcast.reshape(-1), dtype=float)

    @staticmethod
    def _thermal_angles(value: Any) -> np.ndarray:
        angles = np.asarray(to_numpy(value), dtype=float).reshape(-1)
        if angles.size == 0:
            raise ValueError("angles are required for mode='thermal'")
        if np.any((angles < 0.0) | (angles >= 90.0)):
            raise ValueError("thermal angles must satisfy 0 <= vza < 90")
        return angles

    @staticmethod
    def _require_finite_torch(name: str, value: Any) -> None:
        """Raises a public error when a torch batch input contains non-finite values."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError(f"{name} must be finite")
        if name == "tau" and bool(torch.any(value < 0.0).item()):
            raise ValueError("tau must be nonnegative")
        if name == "g" and not bool(torch.all((value > -1.0) & (value < 1.0)).item()):
            raise ValueError("g must satisfy -1 < g < 1")

    @staticmethod
    def _broadcast_batch_layers_torch(
        name: str,
        value: Any | None,
        *,
        context: TorchContext,
        batch_shape: tuple[int, ...],
        width: int,
        default: float | None = None,
    ):
        """Broadcasts a layer-shaped public input to flattened torch batch rows."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        target = batch_shape + (width,)
        if value is None:
            if default is None:
                raise ValueError(f"{name} is required")
            tensor = torch.full(target, float(default), dtype=context.dtype, device=context.device)
        else:
            tensor = value_to_torch(value, context)
            if tuple(tensor.shape) == (width,):
                tensor = torch.broadcast_to(tensor, target)
            else:
                try:
                    tensor = torch.broadcast_to(tensor, target)
                except RuntimeError as exc:
                    raise ValueError(
                        f"{name} must have shape {target} or ({width},) for batched forward"
                    ) from exc
        TwoStreamEss._require_finite_torch(name, tensor)
        return tensor.reshape(-1, width).contiguous()

    @staticmethod
    def _broadcast_batch_scalar_torch(
        name: str,
        value: Any,
        *,
        context: TorchContext,
        batch_shape: tuple[int, ...],
    ):
        """Broadcasts a scalar-or-batch public input to flattened torch rows."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        tensor = value_to_torch(value, context)
        try:
            tensor = torch.broadcast_to(tensor, batch_shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} must be scalar or broadcastable to {batch_shape}") from exc
        TwoStreamEss._require_finite_torch(name, tensor)
        return tensor.reshape(-1).contiguous()

    @staticmethod
    def _broadcast_truncation_factor(
        value: Any | None,
        *,
        asymm: np.ndarray,
        omega: np.ndarray,
        batch_shape: tuple[int, ...],
        width: int,
    ) -> np.ndarray:
        """Broadcasts or derives the public delta-M truncation factor."""
        if value is None:
            factor = default_delta_m_truncation_factor(asymm)
        else:
            factor = TwoStreamEss._broadcast_batch_layers(
                "delta_m_truncation_factor",
                value,
                batch_shape=batch_shape,
                width=width,
            )
        validate_delta_m_truncation_factor(factor, omega)
        return factor

    @staticmethod
    def _broadcast_truncation_factor_torch(
        value: Any | None,
        *,
        asymm,
        omega,
        context: TorchContext,
        batch_shape: tuple[int, ...],
        width: int,
    ):
        """Broadcasts or derives the public delta-M truncation factor for torch."""
        from .optical.delta_m_torch import (
            default_delta_m_truncation_factor_torch,
            validate_delta_m_truncation_factor_torch,
        )

        if value is None:
            factor = default_delta_m_truncation_factor_torch(asymm)
        else:
            factor = TwoStreamEss._broadcast_batch_layers_torch(
                "delta_m_truncation_factor",
                value,
                context=context,
                batch_shape=batch_shape,
                width=width,
            )
        validate_delta_m_truncation_factor_torch(factor, omega)
        return factor

    @staticmethod
    def _resolve_truncation_factor_torch(value: Any | None, *, asymm, omega, context: TorchContext):
        """Returns the scalar torch delta-M truncation factor."""
        from .optical.delta_m_torch import (
            default_delta_m_truncation_factor_torch,
            validate_delta_m_truncation_factor_torch,
        )

        if value is None:
            factor = default_delta_m_truncation_factor_torch(asymm)
        else:
            factor = value_to_torch(value, context)
            if tuple(int(dim) for dim in factor.shape) != tuple(int(dim) for dim in asymm.shape):
                raise ValueError(
                    "delta_m_truncation_factor must have the same shape as g for scalar forward"
                )
        validate_delta_m_truncation_factor_torch(factor, omega)
        return factor

    def _batch_bvp_engine(self) -> str:
        """Maps public BVP solver names to the optimized batch-kernel names."""
        if self.options.bvp_solver == "scipy":
            return "auto"
        if self.options.bvp_solver == "pentadiag":
            return "pentadiagonal"
        return self.options.bvp_solver

    def _torch_batch_bvp_engine(self) -> str:
        """Maps public BVP solver names to torch batch-kernel names."""
        engine = self._batch_bvp_engine()
        if engine == "banded":
            return "block"
        return engine

    @staticmethod
    def _solar_batch_chunk_size(n_rows: int, n_layers: int, *, backend: str) -> int:
        """Returns the same memory-friendly solar chunk size used by benchmarks."""
        if n_rows <= 0:
            return 1
        row_floats = (48 if backend == "torch" else 40) * max(int(n_layers), 1) + 64
        target_mib = 512 if backend == "torch" else 1400
        target_bytes = target_mib * 1024 * 1024
        granularity = 2000 if backend == "torch" else 1000
        chunk = max(granularity, int(target_bytes // (8 * row_floats)))
        chunk = min(n_rows, ((chunk + granularity - 1) // granularity) * granularity)
        return max(1, chunk)

    @staticmethod
    def _thermal_batch_chunk_size(n_rows: int, n_layers: int, *, backend: str) -> int:
        """Returns the same memory-friendly thermal chunk size used by benchmarks."""
        if n_rows <= 0:
            return 1
        row_floats = (6 if backend == "torch" else 4) * max(int(n_layers), 1) + 32
        target_bytes = 384 * 1024 * 1024
        granularity = 2000 if backend == "torch" else 1000
        chunk = max(granularity, int(target_bytes // (8 * row_floats)))
        chunk = min(n_rows, ((chunk + granularity - 1) // granularity) * granularity)
        return max(1, chunk)

    @staticmethod
    def _reshape_endpoint(
        values_by_geometry: list[np.ndarray],
        *,
        batch_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Reshapes flattened endpoint rows back to public batch/geometry shape."""
        if len(values_by_geometry) == 1:
            return values_by_geometry[0].reshape(batch_shape), ()
        stacked = np.stack(values_by_geometry, axis=-1)
        return stacked.reshape(batch_shape + (len(values_by_geometry),)), (len(values_by_geometry),)

    @staticmethod
    def _reshape_endpoint_torch(
        values_by_geometry: list[Any],
        *,
        batch_shape: tuple[int, ...],
    ) -> tuple[Any, tuple[int, ...]]:
        """Reshapes flattened torch endpoint rows back to public batch/geometry shape."""
        if len(values_by_geometry) == 1:
            return values_by_geometry[0].reshape(batch_shape), ()
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        stacked = torch.stack(values_by_geometry, dim=-1)
        return stacked.reshape(batch_shape + (len(values_by_geometry),)), (len(values_by_geometry),)

    @staticmethod
    def _reshape_profile(
        values_by_geometry: list[np.ndarray],
        *,
        batch_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Reshapes flattened profile rows back to public batch/geometry shape."""
        n_levels = values_by_geometry[0].shape[-1]
        if len(values_by_geometry) == 1:
            return values_by_geometry[0].reshape(batch_shape + (n_levels,)), ()
        stacked = np.stack(values_by_geometry, axis=-2)
        return (
            stacked.reshape(batch_shape + (len(values_by_geometry), n_levels)),
            (len(values_by_geometry),),
        )

    @staticmethod
    def _reshape_profile_torch(
        values_by_geometry: list[Any],
        *,
        batch_shape: tuple[int, ...],
    ) -> tuple[Any, tuple[int, ...]]:
        """Reshapes flattened torch profile rows back to public batch/geometry shape."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        n_levels = int(values_by_geometry[0].shape[-1])
        if len(values_by_geometry) == 1:
            return values_by_geometry[0].reshape(batch_shape + (n_levels,)), ()
        stacked = torch.stack(values_by_geometry, dim=-2)
        return (
            stacked.reshape(batch_shape + (len(values_by_geometry), n_levels)),
            (len(values_by_geometry),),
        )

    def _batched_solar_fo_scatter_torch(
        self,
        *,
        fo_scatter_term: Any,
        context: TorchContext,
        batch_shape: tuple[int, ...],
        geom_index: int,
        n_geometries: int,
    ):
        """Returns the FO scatter-term torch batch slice for one solar geometry."""
        n_layers = self.options.nlyr
        scatter = value_to_torch(fo_scatter_term, context)
        if n_geometries == 1:
            return self._broadcast_batch_layers_torch(
                "fo_scatter_term",
                scatter,
                context=context,
                batch_shape=batch_shape,
                width=n_layers,
            )
        leading = batch_shape
        shape_by_geom_layer = leading + (n_geometries, n_layers)
        shape_by_layer_geom = leading + (n_layers, n_geometries)
        if tuple(scatter.shape) == shape_by_geom_layer:
            selected = scatter.reshape((-1, n_geometries, n_layers))[:, geom_index, :]
        elif tuple(scatter.shape) == shape_by_layer_geom:
            selected = scatter.reshape((-1, n_layers, n_geometries))[:, :, geom_index]
        else:
            raise ValueError(
                "fo_scatter_term for multiple solar geometries must have shape "
                f"{shape_by_geom_layer} or {shape_by_layer_geom}"
            )
        self._require_finite_torch("fo_scatter_term", selected)
        return selected.contiguous()

    def _batched_solar_fo_scatter(
        self,
        *,
        fo_scatter_term: Any,
        batch_shape: tuple[int, ...],
        geom_index: int,
        n_geometries: int,
    ) -> np.ndarray:
        """Returns the FO scatter-term batch slice for one solar geometry."""
        n_layers = self.options.nlyr
        scatter = np.asarray(to_numpy(fo_scatter_term), dtype=float)
        if n_geometries == 1:
            return self._broadcast_batch_layers(
                "fo_scatter_term",
                scatter,
                batch_shape=batch_shape,
                width=n_layers,
            )
        leading = batch_shape
        shape_by_geom_layer = leading + (n_geometries, n_layers)
        shape_by_layer_geom = leading + (n_layers, n_geometries)
        if scatter.shape == shape_by_geom_layer:
            selected = scatter.reshape((-1, n_geometries, n_layers))[:, geom_index, :]
        elif scatter.shape == shape_by_layer_geom:
            selected = scatter.reshape((-1, n_layers, n_geometries))[:, :, geom_index]
        else:
            raise ValueError(
                "fo_scatter_term for multiple solar geometries must have shape "
                f"{shape_by_geom_layer} or {shape_by_layer_geom}"
            )
        self._require_finite("fo_scatter_term", selected)
        return np.ascontiguousarray(selected, dtype=float)

    def _forward_fo_solar_obs_batched_numpy(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        earth_radius: float,
        n_moments: int,
        nfine: int,
        fo_scatter_term: Any | None,
    ) -> FoSolarObsResult:
        """Runs the FO-only solar observation batch path."""
        from .rtsolver.fo_solar_obs_batch_numpy import (
            fo_solar_obs_batch_precompute,
            solve_fo_solar_obs_eps_batch_numpy,
        )

        if self.options.plane_parallel or mapped["fo_geometry_mode"] != "eps":
            raise ValueError("batched solar forward_fo currently supports pseudo_spherical only")
        n_layers = self.options.nlyr
        tau_arr = np.asarray(to_numpy(mapped["tau_arr"]), dtype=float)
        if tau_arr.ndim <= 1 or tau_arr.shape[-1] != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward_fo")
        if mapped["height_grid"] is None:
            raise ValueError("z is required for batched solar forward_fo")
        batch_shape = tuple(tau_arr.shape[:-1])
        tau = np.ascontiguousarray(tau_arr.reshape(-1, n_layers), dtype=float)
        omega = self._broadcast_batch_layers(
            "ssa", mapped["omega_arr"], batch_shape=batch_shape, width=n_layers
        )
        asymm = self._broadcast_batch_layers(
            "g", mapped["asymm_arr"], batch_shape=batch_shape, width=n_layers
        )
        scaling = self._broadcast_truncation_factor(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            batch_shape=batch_shape,
            width=n_layers,
        )
        albedo_rows = self._broadcast_batch_scalar("albedo", albedo, batch_shape=batch_shape)
        fbeam_rows = self._broadcast_batch_scalar(
            "fbeam", mapped["flux_factor"], batch_shape=batch_shape
        )
        self._require_finite("tau", tau)
        height_grid = np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        prepared = self._prepare_forward(
            tau_arr=np.zeros(n_layers, dtype=float),
            omega_arr=np.zeros(n_layers, dtype=float),
            asymm_arr=np.zeros(n_layers, dtype=float),
            height_grid=height_grid,
            user_obsgeoms=mapped["user_obsgeoms"],
            stream_value=mapped["stream_value"],
            flux_factor=1.0,
            albedo=0.0,
            d2s_scaling=np.zeros(n_layers, dtype=float),
            earth_radius=earth_radius,
        )
        want_profiles = self.options.output_levels
        total_by_geometry: list[np.ndarray] = []
        ss_by_geometry: list[np.ndarray] = []
        db_by_geometry: list[np.ndarray] = []
        total_profile_by_geometry: list[np.ndarray] = []
        ss_profile_by_geometry: list[np.ndarray] = []
        db_profile_by_geometry: list[np.ndarray] = []
        mu0_by_geometry: list[np.ndarray] = []
        mu1_by_geometry: list[np.ndarray] = []
        cosscat_by_geometry: list[np.ndarray] = []
        do_nadir_by_geometry: list[np.ndarray] = []
        n_rows = tau.shape[0]
        chunk_size = self._solar_batch_chunk_size(n_rows, n_layers, backend="numpy")
        scatter_input = fo_scatter_term
        if scatter_input is None:
            scatter_input = fo_scatter_term_henyey_greenstein(
                ssa=omega.reshape(batch_shape + (n_layers,)),
                g=asymm.reshape(batch_shape + (n_layers,)),
                angles=prepared.user_obsgeoms,
                delta_m_truncation_factor=scaling.reshape(batch_shape + (n_layers,)),
                n_moments=n_moments,
            )
        for geom_index, user_obsgeom in enumerate(prepared.user_obsgeoms):
            scatter = self._batched_solar_fo_scatter(
                fo_scatter_term=scatter_input,
                batch_shape=batch_shape,
                geom_index=geom_index,
                n_geometries=prepared.user_obsgeoms.shape[0],
            )
            precomputed = fo_solar_obs_batch_precompute(
                user_obsgeom=user_obsgeom,
                heights=height_grid,
                earth_radius=earth_radius,
                nfine=nfine,
            )
            total_rows = np.empty(n_rows, dtype=float)
            ss_rows = np.empty(n_rows, dtype=float)
            db_rows = np.empty(n_rows, dtype=float)
            total_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float) if want_profiles else None
            )
            ss_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float) if want_profiles else None
            )
            db_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float) if want_profiles else None
            )
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                row_slice = slice(start, stop)
                chunk = solve_fo_solar_obs_eps_batch_numpy(
                    tau=tau[row_slice],
                    omega=omega[row_slice],
                    scaling=scaling[row_slice],
                    albedo=albedo_rows[row_slice],
                    flux_factor=fbeam_rows[row_slice],
                    exact_scatter=scatter[row_slice],
                    precomputed=precomputed,
                    return_profile=want_profiles,
                    return_components=True,
                )
                total_rows[row_slice] = chunk.total
                ss_rows[row_slice] = chunk.single_scatter
                db_rows[row_slice] = chunk.direct_beam
                if want_profiles:
                    total_profile_rows[row_slice] = chunk.total_profile
                    ss_profile_rows[row_slice] = chunk.single_scatter_profile
                    db_profile_rows[row_slice] = chunk.direct_beam_profile
            sza = math.radians(float(user_obsgeom[0]))
            vza = math.radians(float(user_obsgeom[1]))
            azm = math.radians(float(user_obsgeom[2]))
            mu0 = math.cos(sza)
            mu1 = math.cos(vza)
            if math.isclose(float(user_obsgeom[0]), 0.0):
                cosscat = -mu1 if not math.isclose(mu1, 0.0) else 0.0
            else:
                cosscat = -(math.cos(vza) * math.cos(sza)) + math.sin(vza) * math.sin(
                    sza
                ) * math.cos(azm)
            total_by_geometry.append(total_rows)
            ss_by_geometry.append(ss_rows)
            db_by_geometry.append(db_rows)
            if want_profiles:
                total_profile_by_geometry.append(total_profile_rows)
                ss_profile_by_geometry.append(ss_profile_rows)
                db_profile_by_geometry.append(db_profile_rows)
            mu0_by_geometry.append(np.full(n_rows, mu0, dtype=float))
            mu1_by_geometry.append(np.full(n_rows, mu1, dtype=float))
            cosscat_by_geometry.append(np.full(n_rows, cosscat, dtype=float))
            do_nadir_by_geometry.append(
                np.full(n_rows, bool(math.isclose(float(user_obsgeom[1]), 0.0)), dtype=bool)
            )
        total, _geometry_shape = self._reshape_endpoint(total_by_geometry, batch_shape=batch_shape)
        ss, _ = self._reshape_endpoint(ss_by_geometry, batch_shape=batch_shape)
        db, _ = self._reshape_endpoint(db_by_geometry, batch_shape=batch_shape)
        mu0, _ = self._reshape_endpoint(mu0_by_geometry, batch_shape=batch_shape)
        mu1, _ = self._reshape_endpoint(mu1_by_geometry, batch_shape=batch_shape)
        cosscat, _ = self._reshape_endpoint(cosscat_by_geometry, batch_shape=batch_shape)
        do_nadir, _ = self._reshape_endpoint(do_nadir_by_geometry, batch_shape=batch_shape)
        if want_profiles:
            total_profile, _ = self._reshape_profile(
                total_profile_by_geometry,
                batch_shape=batch_shape,
            )
            ss_profile, _ = self._reshape_profile(ss_profile_by_geometry, batch_shape=batch_shape)
            db_profile, _ = self._reshape_profile(db_profile_by_geometry, batch_shape=batch_shape)
        else:
            total_profile = None
            ss_profile = None
            db_profile = None
        return FoSolarObsResult(
            intensity_total=total,
            intensity_ss=ss,
            intensity_db=db,
            mu0=mu0,
            mu1=mu1,
            cosscat=cosscat,
            do_nadir=do_nadir,
            intensity_total_profile=total_profile,
            intensity_ss_profile=ss_profile,
            intensity_db_profile=db_profile,
        )

    def _forward_fo_solar_obs_batched_torch(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        earth_radius: float,
        n_moments: int,
        nfine: int,
        fo_scatter_term: Any | None,
    ) -> FoSolarObsResult:
        """Runs the FO-only solar observation batch path on torch tensors."""
        from .rtsolver.backend import _load_torch
        from .rtsolver.fo_solar_obs_batch_numpy import fo_solar_obs_batch_precompute
        from .rtsolver.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch
        from .rtsolver.fo_solar_obs_torch import fo_scatter_term_henyey_greenstein_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        if self.options.plane_parallel or mapped["fo_geometry_mode"] != "eps":
            raise ValueError("batched solar forward_fo currently supports pseudo_spherical only")
        context = self._select_torch_context(
            detect_torch_context(
                mapped["tau_arr"],
                mapped["omega_arr"],
                mapped["asymm_arr"],
                mapped["d2s_scaling"],
                albedo,
                mapped["flux_factor"],
                fo_scatter_term,
            )
        )
        n_layers = self.options.nlyr
        tau_arr = value_to_torch(mapped["tau_arr"], context)
        if tau_arr.ndim <= 1 or int(tau_arr.shape[-1]) != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward_fo")
        if mapped["height_grid"] is None:
            raise ValueError("z is required for batched solar forward_fo")
        batch_shape = tuple(int(dim) for dim in tau_arr.shape[:-1])
        tau = tau_arr.reshape(-1, n_layers).contiguous()
        omega = self._broadcast_batch_layers_torch(
            "ssa",
            mapped["omega_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        asymm = self._broadcast_batch_layers_torch(
            "g",
            mapped["asymm_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        scaling = self._broadcast_truncation_factor_torch(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        albedo_rows = self._broadcast_batch_scalar_torch(
            "albedo", albedo, context=context, batch_shape=batch_shape
        )
        fbeam_rows = self._broadcast_batch_scalar_torch(
            "fbeam", mapped["flux_factor"], context=context, batch_shape=batch_shape
        )
        self._require_finite_torch("tau", tau)
        height_grid = np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        prepared = self._prepare_forward(
            tau_arr=np.zeros(n_layers, dtype=float),
            omega_arr=np.zeros(n_layers, dtype=float),
            asymm_arr=np.zeros(n_layers, dtype=float),
            height_grid=height_grid,
            user_obsgeoms=mapped["user_obsgeoms"],
            stream_value=mapped["stream_value"],
            flux_factor=1.0,
            albedo=0.0,
            d2s_scaling=np.zeros(n_layers, dtype=float),
            earth_radius=earth_radius,
        )
        want_profiles = self.options.output_levels
        total_by_geometry = []
        ss_by_geometry = []
        db_by_geometry = []
        total_profile_by_geometry = []
        ss_profile_by_geometry = []
        db_profile_by_geometry = []
        mu0_by_geometry = []
        mu1_by_geometry = []
        cosscat_by_geometry = []
        do_nadir_by_geometry = []
        n_rows = int(tau.shape[0])
        chunk_size = self._solar_batch_chunk_size(n_rows, n_layers, backend="torch")
        with self._torch_grad_context():
            scatter_input = fo_scatter_term
            if scatter_input is None:
                scatter_input = fo_scatter_term_henyey_greenstein_torch(
                    ssa=omega.reshape(batch_shape + (n_layers,)),
                    g=asymm.reshape(batch_shape + (n_layers,)),
                    angles=prepared.user_obsgeoms,
                    delta_m_truncation_factor=scaling.reshape(batch_shape + (n_layers,)),
                    n_moments=n_moments,
                    dtype=context.dtype,
                    device=context.device,
                )
            for geom_index, user_obsgeom in enumerate(prepared.user_obsgeoms):
                scatter = self._batched_solar_fo_scatter_torch(
                    fo_scatter_term=scatter_input,
                    context=context,
                    batch_shape=batch_shape,
                    geom_index=geom_index,
                    n_geometries=prepared.user_obsgeoms.shape[0],
                )
                precomputed = fo_solar_obs_batch_precompute(
                    user_obsgeom=user_obsgeom,
                    heights=height_grid,
                    earth_radius=earth_radius,
                    nfine=nfine,
                )
                total_chunks = []
                ss_chunks = []
                db_chunks = []
                total_profile_chunks = []
                ss_profile_chunks = []
                db_profile_chunks = []
                for start in range(0, n_rows, chunk_size):
                    stop = min(start + chunk_size, n_rows)
                    row_slice = slice(start, stop)
                    chunk = solve_fo_solar_obs_eps_batch_torch(
                        tau=tau[row_slice],
                        omega=omega[row_slice],
                        scaling=scaling[row_slice],
                        albedo=albedo_rows[row_slice],
                        flux_factor=fbeam_rows[row_slice],
                        exact_scatter=scatter[row_slice],
                        precomputed=precomputed,
                        dtype=context.dtype,
                        device=context.device,
                        return_profile=want_profiles,
                        return_components=True,
                    )
                    total_chunks.append(chunk.total)
                    ss_chunks.append(chunk.single_scatter)
                    db_chunks.append(chunk.direct_beam)
                    if want_profiles:
                        total_profile_chunks.append(chunk.total_profile)
                        ss_profile_chunks.append(chunk.single_scatter_profile)
                        db_profile_chunks.append(chunk.direct_beam_profile)
                sza = math.radians(float(user_obsgeom[0]))
                vza = math.radians(float(user_obsgeom[1]))
                azm = math.radians(float(user_obsgeom[2]))
                mu0_value = math.cos(sza)
                mu1_value = math.cos(vza)
                if math.isclose(float(user_obsgeom[0]), 0.0):
                    cosscat_value = -mu1_value if not math.isclose(mu1_value, 0.0) else 0.0
                else:
                    cosscat_value = -(math.cos(vza) * math.cos(sza)) + math.sin(vza) * math.sin(
                        sza
                    ) * math.cos(azm)
                total_by_geometry.append(torch.cat(total_chunks, dim=0))
                ss_by_geometry.append(torch.cat(ss_chunks, dim=0))
                db_by_geometry.append(torch.cat(db_chunks, dim=0))
                if want_profiles:
                    total_profile_by_geometry.append(torch.cat(total_profile_chunks, dim=0))
                    ss_profile_by_geometry.append(torch.cat(ss_profile_chunks, dim=0))
                    db_profile_by_geometry.append(torch.cat(db_profile_chunks, dim=0))
                mu0_by_geometry.append(
                    torch.full((n_rows,), mu0_value, dtype=context.dtype, device=context.device)
                )
                mu1_by_geometry.append(
                    torch.full((n_rows,), mu1_value, dtype=context.dtype, device=context.device)
                )
                cosscat_by_geometry.append(
                    torch.full((n_rows,), cosscat_value, dtype=context.dtype, device=context.device)
                )
                do_nadir_by_geometry.append(
                    torch.full(
                        (n_rows,),
                        bool(math.isclose(float(user_obsgeom[1]), 0.0)),
                        dtype=torch.bool,
                        device=context.device,
                    )
                )
            total, _geometry_shape = self._reshape_endpoint_torch(
                total_by_geometry,
                batch_shape=batch_shape,
            )
            ss, _ = self._reshape_endpoint_torch(ss_by_geometry, batch_shape=batch_shape)
            db, _ = self._reshape_endpoint_torch(db_by_geometry, batch_shape=batch_shape)
            mu0, _ = self._reshape_endpoint_torch(mu0_by_geometry, batch_shape=batch_shape)
            mu1, _ = self._reshape_endpoint_torch(mu1_by_geometry, batch_shape=batch_shape)
            cosscat, _ = self._reshape_endpoint_torch(
                cosscat_by_geometry,
                batch_shape=batch_shape,
            )
            do_nadir, _ = self._reshape_endpoint_torch(
                do_nadir_by_geometry,
                batch_shape=batch_shape,
            )
            if want_profiles:
                total_profile, _ = self._reshape_profile_torch(
                    total_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                ss_profile, _ = self._reshape_profile_torch(
                    ss_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                db_profile, _ = self._reshape_profile_torch(
                    db_profile_by_geometry,
                    batch_shape=batch_shape,
                )
            else:
                total_profile = None
                ss_profile = None
                db_profile = None
        return FoSolarObsResult(
            intensity_total=total,
            intensity_ss=ss,
            intensity_db=db,
            mu0=mu0,
            mu1=mu1,
            cosscat=cosscat,
            do_nadir=do_nadir,
            intensity_total_profile=total_profile,
            intensity_ss_profile=ss_profile,
            intensity_db_profile=db_profile,
        )

    def _forward_batched(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        brdf: Any | None,
        surface_leaving: Any | None,
        emissivity: Any,
        earth_radius: float,
        include_fo: bool,
        fo_n_moments: int,
        fo_nfine: int,
        fo_scatter_term: Any | None,
    ) -> TwoStreamEssBatchResult:
        """Runs the public batch path, keeping fast endpoint kernels as default."""
        if (
            not self.options.upwelling
            or self.options.downwelling
            or self.options.mvout_only
            or self.options.additional_mvout
        ):
            raise ValueError(
                "batched forward currently supports the default upwelling TOA endpoint only"
            )
        if brdf is not None or surface_leaving is not None:
            raise ValueError("batched forward does not support brdf or surface_leaving inputs")
        if self.options.brdf_surface or self.options.surface_leaving:
            raise ValueError(
                "batched forward does not support brdf_surface or surface_leaving options"
            )
        if self.options.backend == "torch":
            if self._source_mode == "solar_obs":
                return self._forward_solar_obs_batched_torch(
                    mapped=mapped,
                    albedo=albedo,
                    earth_radius=earth_radius,
                    include_fo=include_fo,
                    fo_n_moments=fo_n_moments,
                    fo_nfine=fo_nfine,
                    fo_scatter_term=fo_scatter_term,
                )
            if self._source_mode == "thermal":
                return self._forward_thermal_batched_torch(
                    mapped=mapped,
                    albedo=albedo,
                    emissivity=emissivity,
                    earth_radius=earth_radius,
                    include_fo=include_fo,
                    fo_nfine=fo_nfine,
                )
        elif self.options.backend != "numpy":
            raise NotImplementedError(
                "batched forward currently supports backend='numpy' or 'torch'"
            )
        if self._source_mode == "solar_obs":
            return self._forward_solar_obs_batched_numpy(
                mapped=mapped,
                albedo=albedo,
                earth_radius=earth_radius,
                include_fo=include_fo,
                fo_n_moments=fo_n_moments,
                fo_nfine=fo_nfine,
                fo_scatter_term=fo_scatter_term,
            )
        if self._source_mode == "thermal":
            return self._forward_thermal_batched_numpy(
                mapped=mapped,
                albedo=albedo,
                emissivity=emissivity,
                earth_radius=earth_radius,
                include_fo=include_fo,
                fo_nfine=fo_nfine,
            )
        raise NotImplementedError(
            "batched forward is currently implemented for mode='solar' and mode='thermal'"
        )

    def _forward_solar_obs_batched_numpy(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        earth_radius: float,
        include_fo: bool,
        fo_n_moments: int,
        fo_nfine: int,
        fo_scatter_term: Any | None,
    ) -> TwoStreamEssBatchResult:
        """Runs the solar observation-geometry public batch path."""
        from .rtsolver.fo_solar_obs_batch_numpy import (
            fo_solar_obs_batch_precompute,
            solve_fo_solar_obs_eps_batch_numpy,
        )
        from .rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy

        if include_fo and (self.options.plane_parallel or mapped["fo_geometry_mode"] != "eps"):
            raise ValueError(
                "batched solar include_fo=True currently supports pseudo_spherical geometry only"
            )
        n_layers = self.options.nlyr
        tau_arr = np.asarray(to_numpy(mapped["tau_arr"]), dtype=float)
        if tau_arr.ndim <= 1 or tau_arr.shape[-1] != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward")
        batch_shape = tuple(tau_arr.shape[:-1])
        tau = np.ascontiguousarray(tau_arr.reshape(-1, n_layers), dtype=float)
        omega = self._broadcast_batch_layers(
            "ssa", mapped["omega_arr"], batch_shape=batch_shape, width=n_layers
        )
        asymm = self._broadcast_batch_layers(
            "g", mapped["asymm_arr"], batch_shape=batch_shape, width=n_layers
        )
        scaling = self._broadcast_truncation_factor(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            batch_shape=batch_shape,
            width=n_layers,
        )
        albedo_rows = self._broadcast_batch_scalar("albedo", albedo, batch_shape=batch_shape)
        fbeam_rows = self._broadcast_batch_scalar(
            "fbeam", mapped["flux_factor"], batch_shape=batch_shape
        )
        self._require_finite("tau", tau)
        height_grid = np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        prepared = self._prepare_forward(
            tau_arr=np.zeros(n_layers, dtype=float),
            omega_arr=np.zeros(n_layers, dtype=float),
            asymm_arr=np.zeros(n_layers, dtype=float),
            height_grid=height_grid,
            user_obsgeoms=mapped["user_obsgeoms"],
            stream_value=mapped["stream_value"],
            flux_factor=1.0,
            albedo=0.0,
            d2s_scaling=np.zeros(n_layers, dtype=float),
            earth_radius=earth_radius,
        )
        geometry = prepared.geometry
        want_profiles = self.options.output_levels
        scatter_input = fo_scatter_term
        if include_fo and scatter_input is None:
            scatter_input = fo_scatter_term_henyey_greenstein(
                ssa=omega.reshape(batch_shape + (n_layers,)),
                g=asymm.reshape(batch_shape + (n_layers,)),
                angles=prepared.user_obsgeoms,
                delta_m_truncation_factor=scaling.reshape(batch_shape + (n_layers,)),
                n_moments=fo_n_moments,
            )
        two_stream_by_geometry: list[np.ndarray] = []
        fo_by_geometry: list[np.ndarray] = []
        two_stream_profile_by_geometry: list[np.ndarray] = []
        fo_profile_by_geometry: list[np.ndarray] = []
        for geom_index, user_obsgeom in enumerate(prepared.user_obsgeoms):
            if self.options.plane_parallel:
                secant = float(geometry.average_secant_pp[geom_index])
                chapman = np.triu(np.full((n_layers, n_layers), secant, dtype=float))
            else:
                chapman = geometry.chapman_factors[:, :, geom_index]
            n_rows = tau.shape[0]
            chunk_size = self._solar_batch_chunk_size(n_rows, n_layers, backend="numpy")
            two_stream_rows = np.empty(n_rows, dtype=float)
            fo_rows = np.empty(n_rows, dtype=float) if include_fo else None
            two_stream_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float) if want_profiles else None
            )
            fo_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float)
                if include_fo and want_profiles
                else None
            )
            scatter = None
            precomputed = None
            if include_fo:
                scatter = self._batched_solar_fo_scatter(
                    fo_scatter_term=scatter_input,
                    batch_shape=batch_shape,
                    geom_index=geom_index,
                    n_geometries=prepared.user_obsgeoms.shape[0],
                )
                precomputed = fo_solar_obs_batch_precompute(
                    user_obsgeom=user_obsgeom,
                    heights=height_grid,
                    earth_radius=earth_radius,
                    nfine=fo_nfine,
                )
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                row_slice = slice(start, stop)
                two_stream_chunk = solve_solar_obs_batch_numpy(
                    tau=tau[row_slice],
                    omega=omega[row_slice],
                    asymm=asymm[row_slice],
                    scaling=scaling[row_slice],
                    albedo=albedo_rows[row_slice],
                    flux_factor=fbeam_rows[row_slice],
                    stream_value=prepared.stream_value,
                    chapman=chapman,
                    x0=float(geometry.x0[geom_index]),
                    user_stream=float(geometry.user_streams[geom_index]),
                    user_secant=float(geometry.user_secants[geom_index]),
                    azmfac=float(geometry.azmfac[geom_index]),
                    px11=float(geometry.px11),
                    pxsq=geometry.pxsq,
                    px0x=geometry.px0x[geom_index],
                    ulp=float(geometry.ulp[geom_index]),
                    bvp_engine=self._batch_bvp_engine(),
                    return_profile=want_profiles,
                )
                if want_profiles:
                    two_stream_profile_rows[row_slice] = two_stream_chunk
                    two_stream_rows[row_slice] = two_stream_chunk[:, 0]
                else:
                    two_stream_rows[row_slice] = two_stream_chunk
                if include_fo:
                    fo_chunk = solve_fo_solar_obs_eps_batch_numpy(
                        tau=tau[row_slice],
                        omega=omega[row_slice],
                        scaling=scaling[row_slice],
                        albedo=albedo_rows[row_slice],
                        flux_factor=fbeam_rows[row_slice],
                        exact_scatter=scatter[row_slice],
                        precomputed=precomputed,
                        return_profile=want_profiles,
                    )
                    if want_profiles:
                        fo_profile_rows[row_slice] = fo_chunk
                        fo_rows[row_slice] = fo_chunk[:, 0]
                    else:
                        fo_rows[row_slice] = fo_chunk
            two_stream_by_geometry.append(two_stream_rows)
            if want_profiles:
                two_stream_profile_by_geometry.append(two_stream_profile_rows)
            if include_fo:
                fo_by_geometry.append(fo_rows)
                if want_profiles:
                    fo_profile_by_geometry.append(fo_profile_rows)
        radiance_2s, geometry_shape = self._reshape_endpoint(
            two_stream_by_geometry, batch_shape=batch_shape
        )
        if include_fo:
            radiance_fo, _ = self._reshape_endpoint(fo_by_geometry, batch_shape=batch_shape)
            radiance_total = radiance_2s + radiance_fo
        else:
            radiance_fo = None
            radiance_total = radiance_2s
        if want_profiles:
            radiance_profile_2s, profile_geometry_shape = self._reshape_profile(
                two_stream_profile_by_geometry,
                batch_shape=batch_shape,
            )
            if include_fo:
                radiance_profile_fo, _ = self._reshape_profile(
                    fo_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                radiance_profile_total = radiance_profile_2s + radiance_profile_fo
            else:
                radiance_profile_fo = None
                radiance_profile_total = radiance_profile_2s
            geometry_shape = profile_geometry_shape
        else:
            radiance_profile_2s = None
            radiance_profile_fo = None
            radiance_profile_total = None
        return TwoStreamEssBatchResult(
            radiance_2s=radiance_2s,
            radiance_fo=radiance_fo,
            radiance_total=radiance_total,
            radiance_profile_2s=radiance_profile_2s,
            radiance_profile_fo=radiance_profile_fo,
            radiance_profile_total=radiance_profile_total,
            batch_shape=batch_shape,
            geometry_shape=geometry_shape,
        )

    def _forward_solar_obs_batched_torch(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        earth_radius: float,
        include_fo: bool,
        fo_n_moments: int,
        fo_nfine: int,
        fo_scatter_term: Any | None,
    ) -> TwoStreamEssBatchResult:
        """Runs the solar observation-geometry public batch path on torch tensors."""
        from .rtsolver.backend import _load_torch
        from .rtsolver.fo_solar_obs_batch_numpy import fo_solar_obs_batch_precompute
        from .rtsolver.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch
        from .rtsolver.fo_solar_obs_torch import fo_scatter_term_henyey_greenstein_torch
        from .rtsolver.solar_obs_batch_torch import solve_solar_obs_batch_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        if include_fo and (self.options.plane_parallel or mapped["fo_geometry_mode"] != "eps"):
            raise ValueError(
                "batched solar include_fo=True currently supports pseudo_spherical geometry only"
            )
        context = self._select_torch_context(
            detect_torch_context(
                mapped["tau_arr"],
                mapped["omega_arr"],
                mapped["asymm_arr"],
                mapped["d2s_scaling"],
                albedo,
                mapped["flux_factor"],
                fo_scatter_term,
            )
        )
        n_layers = self.options.nlyr
        tau_arr = value_to_torch(mapped["tau_arr"], context)
        if tau_arr.ndim <= 1 or int(tau_arr.shape[-1]) != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward")
        batch_shape = tuple(int(dim) for dim in tau_arr.shape[:-1])
        tau = tau_arr.reshape(-1, n_layers).contiguous()
        omega = self._broadcast_batch_layers_torch(
            "ssa",
            mapped["omega_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        asymm = self._broadcast_batch_layers_torch(
            "g",
            mapped["asymm_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        scaling = self._broadcast_truncation_factor_torch(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        albedo_rows = self._broadcast_batch_scalar_torch(
            "albedo", albedo, context=context, batch_shape=batch_shape
        )
        fbeam_rows = self._broadcast_batch_scalar_torch(
            "fbeam", mapped["flux_factor"], context=context, batch_shape=batch_shape
        )
        self._require_finite_torch("tau", tau)
        height_grid = np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        prepared = self._prepare_forward(
            tau_arr=np.zeros(n_layers, dtype=float),
            omega_arr=np.zeros(n_layers, dtype=float),
            asymm_arr=np.zeros(n_layers, dtype=float),
            height_grid=height_grid,
            user_obsgeoms=mapped["user_obsgeoms"],
            stream_value=mapped["stream_value"],
            flux_factor=1.0,
            albedo=0.0,
            d2s_scaling=np.zeros(n_layers, dtype=float),
            earth_radius=earth_radius,
        )
        geometry = prepared.geometry
        want_profiles = self.options.output_levels
        two_stream_by_geometry = []
        fo_by_geometry = []
        two_stream_profile_by_geometry = []
        fo_profile_by_geometry = []
        with self._torch_grad_context():
            scatter_input = fo_scatter_term
            if include_fo and scatter_input is None:
                scatter_input = fo_scatter_term_henyey_greenstein_torch(
                    ssa=omega.reshape(batch_shape + (n_layers,)),
                    g=asymm.reshape(batch_shape + (n_layers,)),
                    angles=prepared.user_obsgeoms,
                    delta_m_truncation_factor=scaling.reshape(batch_shape + (n_layers,)),
                    n_moments=fo_n_moments,
                    dtype=context.dtype,
                    device=context.device,
                )
            for geom_index, user_obsgeom in enumerate(prepared.user_obsgeoms):
                if self.options.plane_parallel:
                    secant = float(geometry.average_secant_pp[geom_index])
                    chapman = np.triu(np.full((n_layers, n_layers), secant, dtype=float))
                else:
                    chapman = geometry.chapman_factors[:, :, geom_index]
                n_rows = int(tau.shape[0])
                chunk_size = self._solar_batch_chunk_size(n_rows, n_layers, backend="torch")
                two_chunks = []
                fo_chunks = []
                two_profile_chunks = []
                fo_profile_chunks = []
                scatter = None
                precomputed = None
                if include_fo:
                    scatter = self._batched_solar_fo_scatter_torch(
                        fo_scatter_term=scatter_input,
                        context=context,
                        batch_shape=batch_shape,
                        geom_index=geom_index,
                        n_geometries=prepared.user_obsgeoms.shape[0],
                    )
                    precomputed = fo_solar_obs_batch_precompute(
                        user_obsgeom=user_obsgeom,
                        heights=height_grid,
                        earth_radius=earth_radius,
                        nfine=fo_nfine,
                    )
                for start in range(0, n_rows, chunk_size):
                    stop = min(start + chunk_size, n_rows)
                    row_slice = slice(start, stop)
                    two_chunk = solve_solar_obs_batch_torch(
                        tau=tau[row_slice],
                        omega=omega[row_slice],
                        asymm=asymm[row_slice],
                        scaling=scaling[row_slice],
                        albedo=albedo_rows[row_slice],
                        flux_factor=fbeam_rows[row_slice],
                        stream_value=prepared.stream_value,
                        chapman=chapman,
                        x0=float(geometry.x0[geom_index]),
                        user_stream=float(geometry.user_streams[geom_index]),
                        user_secant=float(geometry.user_secants[geom_index]),
                        azmfac=float(geometry.azmfac[geom_index]),
                        px11=float(geometry.px11),
                        pxsq=geometry.pxsq,
                        px0x=geometry.px0x[geom_index],
                        ulp=float(geometry.ulp[geom_index]),
                        dtype=context.dtype,
                        device=context.device,
                        bvp_engine=self._torch_batch_bvp_engine(),
                        return_profile=want_profiles,
                    )
                    if want_profiles:
                        two_profile_chunks.append(two_chunk)
                        two_chunks.append(two_chunk[:, 0])
                    else:
                        two_chunks.append(two_chunk)
                    if include_fo:
                        fo_chunk = solve_fo_solar_obs_eps_batch_torch(
                            tau=tau[row_slice],
                            omega=omega[row_slice],
                            scaling=scaling[row_slice],
                            albedo=albedo_rows[row_slice],
                            flux_factor=fbeam_rows[row_slice],
                            exact_scatter=scatter[row_slice],
                            precomputed=precomputed,
                            dtype=context.dtype,
                            device=context.device,
                            return_profile=want_profiles,
                        )
                        if want_profiles:
                            fo_profile_chunks.append(fo_chunk)
                            fo_chunks.append(fo_chunk[:, 0])
                        else:
                            fo_chunks.append(fo_chunk)
                two_stream_by_geometry.append(torch.cat(two_chunks, dim=0))
                if want_profiles:
                    two_stream_profile_by_geometry.append(torch.cat(two_profile_chunks, dim=0))
                if include_fo:
                    fo_by_geometry.append(torch.cat(fo_chunks, dim=0))
                    if want_profiles:
                        fo_profile_by_geometry.append(torch.cat(fo_profile_chunks, dim=0))
            radiance_2s, geometry_shape = self._reshape_endpoint_torch(
                two_stream_by_geometry, batch_shape=batch_shape
            )
            if include_fo:
                radiance_fo, _ = self._reshape_endpoint_torch(
                    fo_by_geometry, batch_shape=batch_shape
                )
                radiance_total = radiance_2s + radiance_fo
            else:
                radiance_fo = None
                radiance_total = radiance_2s
            if want_profiles:
                radiance_profile_2s, profile_geometry_shape = self._reshape_profile_torch(
                    two_stream_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                if include_fo:
                    radiance_profile_fo, _ = self._reshape_profile_torch(
                        fo_profile_by_geometry,
                        batch_shape=batch_shape,
                    )
                    radiance_profile_total = radiance_profile_2s + radiance_profile_fo
                else:
                    radiance_profile_fo = None
                    radiance_profile_total = radiance_profile_2s
                geometry_shape = profile_geometry_shape
            else:
                radiance_profile_2s = None
                radiance_profile_fo = None
                radiance_profile_total = None
        return TwoStreamEssBatchResult(
            radiance_2s=radiance_2s,
            radiance_fo=radiance_fo,
            radiance_total=radiance_total,
            radiance_profile_2s=radiance_profile_2s,
            radiance_profile_fo=radiance_profile_fo,
            radiance_profile_total=radiance_profile_total,
            batch_shape=batch_shape,
            geometry_shape=geometry_shape,
        )

    def _forward_thermal_batched_numpy(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        emissivity: Any,
        earth_radius: float,
        include_fo: bool,
        fo_nfine: int,
    ) -> TwoStreamEssBatchResult:
        """Runs the thermal observation-geometry public batch path."""
        from .rtsolver.thermal_batch_numpy import (
            _fo_thermal_toa,
            _two_stream_thermal_toa,
            precompute_fo_thermal_geometry_numpy,
        )

        n_layers = self.options.nlyr
        tau_arr = np.asarray(to_numpy(mapped["tau_arr"]), dtype=float)
        if tau_arr.ndim <= 1 or tau_arr.shape[-1] != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward")
        batch_shape = tuple(tau_arr.shape[:-1])
        tau = np.ascontiguousarray(tau_arr.reshape(-1, n_layers), dtype=float)
        omega = self._broadcast_batch_layers(
            "ssa", mapped["omega_arr"], batch_shape=batch_shape, width=n_layers
        )
        asymm = self._broadcast_batch_layers(
            "g", mapped["asymm_arr"], batch_shape=batch_shape, width=n_layers
        )
        scaling = self._broadcast_truncation_factor(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            batch_shape=batch_shape,
            width=n_layers,
        )
        planck = self._broadcast_batch_layers(
            "planck",
            mapped["thermal_bb_input"],
            batch_shape=batch_shape,
            width=n_layers + 1,
        )
        surfbb = self._broadcast_batch_scalar(
            "surface_planck", mapped["surfbb"], batch_shape=batch_shape
        )
        albedo_rows = self._broadcast_batch_scalar("albedo", albedo, batch_shape=batch_shape)
        emissivity_rows = self._broadcast_batch_scalar(
            "emissivity", emissivity, batch_shape=batch_shape
        )
        self._require_finite("tau", tau)
        angles = self._thermal_angles(mapped["user_angles"])
        want_profiles = self.options.output_levels
        two_stream_by_geometry: list[np.ndarray] = []
        fo_by_geometry: list[np.ndarray] = []
        two_stream_profile_by_geometry: list[np.ndarray] = []
        fo_profile_by_geometry: list[np.ndarray] = []
        n_rows = tau.shape[0]
        chunk_size = self._thermal_batch_chunk_size(n_rows, n_layers, backend="numpy")
        height_grid = None
        if include_fo:
            if mapped["height_grid"] is None:
                raise ValueError("z is required for batched thermal include_fo=True")
            height_grid = np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        for angle in angles:
            user_stream = float(np.cos(np.deg2rad(float(angle))))
            two_stream_rows = np.empty(n_rows, dtype=float)
            fo_rows = np.empty(n_rows, dtype=float) if include_fo else None
            two_stream_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float) if want_profiles else None
            )
            fo_profile_rows = (
                np.empty((n_rows, n_layers + 1), dtype=float)
                if include_fo and want_profiles
                else None
            )
            fo_geometry = None
            if include_fo:
                fo_geometry = precompute_fo_thermal_geometry_numpy(
                    heights=height_grid,
                    user_angle_degrees=float(angle),
                    earth_radius=earth_radius,
                    nfine=fo_nfine,
                )
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                row_slice = slice(start, stop)
                two_stream = _two_stream_thermal_toa(
                    tau=tau[row_slice],
                    omega=omega[row_slice],
                    asymm=asymm[row_slice],
                    scaling=scaling[row_slice],
                    thermal_bb_input=planck[row_slice],
                    surfbb=surfbb[row_slice],
                    emissivity=emissivity_rows[row_slice],
                    albedo=albedo_rows[row_slice],
                    stream_value=mapped["stream_value"],
                    user_stream=user_stream,
                    thermal_tcutoff=self.options.thermal_tcutoff,
                    bvp_engine=self._batch_bvp_engine(),
                    return_profile=want_profiles,
                )
                if want_profiles:
                    two_stream_profile_rows[row_slice] = two_stream
                    two_stream_rows[row_slice] = two_stream[:, 0]
                else:
                    two_stream_rows[row_slice] = two_stream
                if include_fo:
                    fo = _fo_thermal_toa(
                        tau=tau[row_slice],
                        omega=omega[row_slice],
                        scaling=scaling[row_slice],
                        thermal_bb_input=planck[row_slice],
                        surfbb=surfbb[row_slice],
                        emissivity=emissivity_rows[row_slice],
                        heights=height_grid,
                        user_angle_degrees=float(angle),
                        earth_radius=earth_radius,
                        nfine=fo_nfine,
                        geometry=fo_geometry,
                        return_profile=want_profiles,
                        do_optical_deltam_scaling=(
                            self.options.effective_fo_optical_deltam_scaling
                        ),
                        do_source_deltam_scaling=(
                            self.options.effective_fo_thermal_source_deltam_scaling
                        ),
                    )
                    if want_profiles:
                        fo_profile_rows[row_slice] = fo
                        fo_rows[row_slice] = fo[:, 0]
                    else:
                        fo_rows[row_slice] = fo
            two_stream_by_geometry.append(two_stream_rows)
            if want_profiles:
                two_stream_profile_by_geometry.append(two_stream_profile_rows)
            if include_fo:
                fo_by_geometry.append(fo_rows)
                if want_profiles:
                    fo_profile_by_geometry.append(fo_profile_rows)
        radiance_2s, geometry_shape = self._reshape_endpoint(
            two_stream_by_geometry, batch_shape=batch_shape
        )
        if include_fo:
            radiance_fo, _ = self._reshape_endpoint(fo_by_geometry, batch_shape=batch_shape)
            radiance_total = radiance_2s + radiance_fo
        else:
            radiance_fo = None
            radiance_total = radiance_2s
        if want_profiles:
            radiance_profile_2s, profile_geometry_shape = self._reshape_profile(
                two_stream_profile_by_geometry,
                batch_shape=batch_shape,
            )
            if include_fo:
                radiance_profile_fo, _ = self._reshape_profile(
                    fo_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                radiance_profile_total = radiance_profile_2s + radiance_profile_fo
            else:
                radiance_profile_fo = None
                radiance_profile_total = radiance_profile_2s
            geometry_shape = profile_geometry_shape
        else:
            radiance_profile_2s = None
            radiance_profile_fo = None
            radiance_profile_total = None
        return TwoStreamEssBatchResult(
            radiance_2s=radiance_2s,
            radiance_fo=radiance_fo,
            radiance_total=radiance_total,
            radiance_profile_2s=radiance_profile_2s,
            radiance_profile_fo=radiance_profile_fo,
            radiance_profile_total=radiance_profile_total,
            batch_shape=batch_shape,
            geometry_shape=geometry_shape,
        )

    def _forward_thermal_batched_torch(
        self,
        *,
        mapped: dict[str, Any],
        albedo: Any,
        emissivity: Any,
        earth_radius: float,
        include_fo: bool,
        fo_nfine: int,
    ) -> TwoStreamEssBatchResult:
        """Runs the thermal observation-geometry public batch path on torch tensors."""
        from .rtsolver.backend import _load_torch
        from .rtsolver.thermal_batch_numpy import precompute_fo_thermal_geometry_numpy
        from .rtsolver.thermal_batch_torch import (
            _fo_thermal_toa_batch,
            _two_stream_thermal_toa_batch,
        )

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        context = self._select_torch_context(
            detect_torch_context(
                mapped["tau_arr"],
                mapped["omega_arr"],
                mapped["asymm_arr"],
                mapped["d2s_scaling"],
                mapped["thermal_bb_input"],
                mapped["surfbb"],
                albedo,
                emissivity,
            )
        )
        n_layers = self.options.nlyr
        tau_arr = value_to_torch(mapped["tau_arr"], context)
        if tau_arr.ndim <= 1 or int(tau_arr.shape[-1]) != n_layers:
            raise ValueError(f"tau must have shape (..., {n_layers}) for batched forward")
        batch_shape = tuple(int(dim) for dim in tau_arr.shape[:-1])
        tau = tau_arr.reshape(-1, n_layers).contiguous()
        omega = self._broadcast_batch_layers_torch(
            "ssa",
            mapped["omega_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        asymm = self._broadcast_batch_layers_torch(
            "g",
            mapped["asymm_arr"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        scaling = self._broadcast_truncation_factor_torch(
            mapped["d2s_scaling"],
            asymm=asymm,
            omega=omega,
            context=context,
            batch_shape=batch_shape,
            width=n_layers,
        )
        planck = self._broadcast_batch_layers_torch(
            "planck",
            mapped["thermal_bb_input"],
            context=context,
            batch_shape=batch_shape,
            width=n_layers + 1,
        )
        surfbb = self._broadcast_batch_scalar_torch(
            "surface_planck",
            mapped["surfbb"],
            context=context,
            batch_shape=batch_shape,
        )
        albedo_rows = self._broadcast_batch_scalar_torch(
            "albedo",
            albedo,
            context=context,
            batch_shape=batch_shape,
        )
        emissivity_rows = self._broadcast_batch_scalar_torch(
            "emissivity",
            emissivity,
            context=context,
            batch_shape=batch_shape,
        )
        self._require_finite_torch("tau", tau)
        angles = self._thermal_angles(mapped["user_angles"])
        if include_fo and mapped["height_grid"] is None:
            raise ValueError("z is required for batched thermal include_fo=True")
        height_grid = (
            None
            if mapped["height_grid"] is None
            else np.asarray(to_numpy(mapped["height_grid"]), dtype=float)
        )
        want_profiles = self.options.output_levels
        two_stream_by_geometry = []
        fo_by_geometry = []
        two_stream_profile_by_geometry = []
        fo_profile_by_geometry = []
        n_rows = int(tau.shape[0])
        chunk_size = self._thermal_batch_chunk_size(n_rows, n_layers, backend="torch")
        keep_graph = self.options.torch_enable_grad
        with self._torch_grad_context():
            for angle in angles:
                user_stream = float(np.cos(np.deg2rad(float(angle))))
                two_stream_chunks = []
                two_stream_profile_chunks = []
                fo_chunks = []
                fo_profile_chunks = []
                if keep_graph:
                    two_stream_rows = None
                    two_stream_profile_rows = None
                    fo_rows = None
                    fo_profile_rows = None
                else:
                    two_stream_rows = torch.empty(n_rows, dtype=tau.dtype, device=tau.device)
                    two_stream_profile_rows = (
                        torch.empty(
                            (n_rows, n_layers + 1),
                            dtype=tau.dtype,
                            device=tau.device,
                        )
                        if want_profiles
                        else None
                    )
                    fo_rows = (
                        torch.empty(n_rows, dtype=tau.dtype, device=tau.device)
                        if include_fo
                        else None
                    )
                    fo_profile_rows = (
                        torch.empty(
                            (n_rows, n_layers + 1),
                            dtype=tau.dtype,
                            device=tau.device,
                        )
                        if include_fo and want_profiles
                        else None
                    )
                fo_geometry = None
                if include_fo:
                    fo_geometry = precompute_fo_thermal_geometry_numpy(
                        heights=height_grid,
                        user_angle_degrees=float(angle),
                        earth_radius=earth_radius,
                        nfine=fo_nfine,
                    )
                for start in range(0, n_rows, chunk_size):
                    stop = min(start + chunk_size, n_rows)
                    row_slice = slice(start, stop)
                    two_stream = _two_stream_thermal_toa_batch(
                        tau=tau[row_slice],
                        omega=omega[row_slice],
                        asymm=asymm[row_slice],
                        scaling=scaling[row_slice],
                        thermal_bb_input=planck[row_slice],
                        surfbb=surfbb[row_slice],
                        emissivity=emissivity_rows[row_slice],
                        albedo=albedo_rows[row_slice],
                        stream_value=mapped["stream_value"],
                        user_stream=user_stream,
                        pxsq=mapped["stream_value"] * mapped["stream_value"],
                        thermal_tcutoff=self.options.thermal_tcutoff,
                        bvp_engine=self._torch_batch_bvp_engine(),
                        return_profile=want_profiles,
                    )
                    if want_profiles:
                        if keep_graph:
                            two_stream_profile_chunks.append(two_stream)
                            two_stream_chunks.append(two_stream[:, 0])
                        else:
                            two_stream_profile_rows[row_slice] = two_stream
                            two_stream_rows[row_slice] = two_stream[:, 0]
                    else:
                        if keep_graph:
                            two_stream_chunks.append(two_stream)
                        else:
                            two_stream_rows[row_slice] = two_stream
                    if include_fo:
                        fo = _fo_thermal_toa_batch(
                            tau=tau[row_slice],
                            omega=omega[row_slice],
                            scaling=scaling[row_slice],
                            thermal_bb_input=planck[row_slice],
                            surfbb=surfbb[row_slice],
                            emissivity=emissivity_rows[row_slice],
                            heights=height_grid,
                            user_angle_degrees=float(angle),
                            earth_radius=earth_radius,
                            nfine=fo_nfine,
                            fo_geometry=fo_geometry,
                            return_profile=want_profiles,
                            do_optical_deltam_scaling=(
                                self.options.effective_fo_optical_deltam_scaling
                            ),
                            do_source_deltam_scaling=(
                                self.options.effective_fo_thermal_source_deltam_scaling
                            ),
                        )
                        if want_profiles:
                            if keep_graph:
                                fo_profile_chunks.append(fo)
                                fo_chunks.append(fo[:, 0])
                            else:
                                fo_profile_rows[row_slice] = fo
                                fo_rows[row_slice] = fo[:, 0]
                        else:
                            if keep_graph:
                                fo_chunks.append(fo)
                            else:
                                fo_rows[row_slice] = fo
                if keep_graph:
                    two_stream_by_geometry.append(torch.cat(two_stream_chunks, dim=0))
                else:
                    two_stream_by_geometry.append(two_stream_rows)
                if want_profiles:
                    if keep_graph:
                        two_stream_profile_by_geometry.append(
                            torch.cat(two_stream_profile_chunks, dim=0)
                        )
                    else:
                        two_stream_profile_by_geometry.append(two_stream_profile_rows)
                if include_fo:
                    if keep_graph:
                        fo_by_geometry.append(torch.cat(fo_chunks, dim=0))
                    else:
                        fo_by_geometry.append(fo_rows)
                    if want_profiles:
                        if keep_graph:
                            fo_profile_by_geometry.append(torch.cat(fo_profile_chunks, dim=0))
                        else:
                            fo_profile_by_geometry.append(fo_profile_rows)
            radiance_2s, geometry_shape = self._reshape_endpoint_torch(
                two_stream_by_geometry,
                batch_shape=batch_shape,
            )
            if include_fo:
                radiance_fo, _ = self._reshape_endpoint_torch(
                    fo_by_geometry,
                    batch_shape=batch_shape,
                )
                radiance_total = radiance_2s + radiance_fo
            else:
                radiance_fo = None
                radiance_total = radiance_2s
            if want_profiles:
                radiance_profile_2s, profile_geometry_shape = self._reshape_profile_torch(
                    two_stream_profile_by_geometry,
                    batch_shape=batch_shape,
                )
                if include_fo:
                    radiance_profile_fo, _ = self._reshape_profile_torch(
                        fo_profile_by_geometry,
                        batch_shape=batch_shape,
                    )
                    radiance_profile_total = radiance_profile_2s + radiance_profile_fo
                else:
                    radiance_profile_fo = None
                    radiance_profile_total = radiance_profile_2s
                geometry_shape = profile_geometry_shape
            else:
                radiance_profile_2s = None
                radiance_profile_fo = None
                radiance_profile_total = None
        return TwoStreamEssBatchResult(
            radiance_2s=radiance_2s,
            radiance_fo=radiance_fo,
            radiance_total=radiance_total,
            radiance_profile_2s=radiance_profile_2s,
            radiance_profile_fo=radiance_profile_fo,
            radiance_profile_total=radiance_profile_total,
            batch_shape=batch_shape,
            geometry_shape=geometry_shape,
        )

    def _default_torch_context(self) -> TorchContext:
        """Returns the default torch context for converted inputs.

        Returns
        -------
        TorchContext
            ``float64`` context used when the caller asks for the torch backend
            but passes NumPy inputs.
        """
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        device = self._resolve_torch_device()
        return TorchContext(dtype=self._resolve_torch_dtype(None, device), device=device)

    def _resolve_torch_device(self):
        """Resolves and validates the configured torch device."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        device = torch.device(self.options.torch_device or "cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("torch_device='cuda' requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "torch_device='mps' requested but MPS is not available in this process. "
                "Use torch_device='cpu' or run in an environment where PyTorch exposes MPS."
            )
        return device

    def _resolve_torch_dtype(self, detected: TorchContext | None, device):
        """Resolves and validates the configured torch dtype."""
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        dtype_names = {
            "float64": torch.float64,
            "double": torch.float64,
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.options.torch_dtype is not None:
            try:
                dtype = dtype_names[self.options.torch_dtype.lower()]
            except KeyError as exc:
                allowed = ", ".join(sorted(dtype_names))
                raise ValueError(f"torch_dtype must be one of: {allowed}") from exc
        elif detected is not None:
            dtype = detected.dtype
        elif device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float64
        if device.type == "mps" and dtype == torch.float64:
            raise ValueError(
                "torch_device='mps' requires torch_dtype='float32' or lower precision; "
                "MPS does not support float64"
            )
        return dtype

    def _select_torch_context(self, detected: TorchContext | None) -> TorchContext:
        """Returns the torch dtype/device context for this solver call."""
        if self.options.torch_device is None and self.options.torch_dtype is None:
            return detected if detected is not None else self._default_torch_context()
        device = (
            self._resolve_torch_device()
            if self.options.torch_device is not None
            else (detected.device if detected is not None else self._resolve_torch_device())
        )
        return TorchContext(dtype=self._resolve_torch_dtype(detected, device), device=device)

    def _torch_grad_context(self):
        """Returns the autograd context for the torch backend."""
        if self.options.backend != "torch":
            return nullcontext()
        from .rtsolver.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        return torch.set_grad_enabled(self.options.torch_enable_grad)

    def _fo_result_kwargs(self, fo_result) -> dict[str, Any]:
        """Builds result fields derived from the FO solver output.

        Parameters
        ----------
        fo_result
            FO solver result object or ``None``.

        Returns
        -------
        dict
            Keyword arguments used to populate FO-related fields on
            :class:`TwoStreamEssResult`.
        """
        return {
            "fo_intensity_total": None
            if not isinstance(fo_result, FoSolarObsResult)
            else fo_result.intensity_total,
            "fo_intensity_ss": None
            if not isinstance(fo_result, FoSolarObsResult)
            else fo_result.intensity_ss,
            "fo_intensity_db": None
            if not isinstance(fo_result, FoSolarObsResult)
            else fo_result.intensity_db,
            "fo_intensity_total_profile": (
                None
                if not isinstance(fo_result, FoSolarObsResult)
                else fo_result.intensity_total_profile
            ),
            "fo_mu0": None if not isinstance(fo_result, FoSolarObsResult) else fo_result.mu0,
            "fo_mu1": None if fo_result is None else fo_result.mu1,
            "fo_cosscat": None
            if not isinstance(fo_result, FoSolarObsResult)
            else fo_result.cosscat,
            "fo_do_nadir": None
            if not isinstance(fo_result, FoSolarObsResult)
            else fo_result.do_nadir,
            "fo_thermal_atmos_up_toa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_atmos_up_toa
            ),
            "fo_thermal_surface_toa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_surface_toa
            ),
            "fo_thermal_total_up_toa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_total_up_toa
            ),
            "fo_thermal_atmos_dn_toa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_atmos_dn_toa
            ),
            "fo_thermal_atmos_up_boa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_atmos_up_boa
            ),
            "fo_thermal_surface_boa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_surface_boa
            ),
            "fo_thermal_total_up_boa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_total_up_boa
            ),
            "fo_thermal_total_up_profile": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_total_up_profile
            ),
            "fo_thermal_atmos_dn_boa": (
                None
                if not isinstance(fo_result, FoThermalResult)
                else fo_result.intensity_atmos_dn_boa
            ),
        }

    def _prepare_forward_with_context(
        self,
        *,
        tau_arr: Any,
        omega_arr: Any,
        asymm_arr: Any,
        height_grid: Any | None,
        user_obsgeoms: Any | None,
        stream_value: float,
        flux_factor: float,
        albedo: float,
        d2s_scaling: Any | None = None,
        brdf: Any | None = None,
        surface_leaving: Any | None = None,
        user_angles: Any | None = None,
        beam_szas: Any | None = None,
        user_relazms: Any | None = None,
        thermal_bb_input: Any | None = None,
        surfbb: float = 0.0,
        emissivity: float = 0.0,
        earth_radius: float = 6371.0,
    ) -> tuple[PreparedInputs, Any | None]:
        """Normalizes inputs and detects the active torch context.

        Returns
        -------
        tuple
            Prepared solver inputs and an optional detected torch context.
        """
        torch_context = detect_torch_context(
            tau_arr,
            omega_arr,
            asymm_arr,
            height_grid,
            user_obsgeoms,
            d2s_scaling,
            user_angles,
            thermal_bb_input,
            flux_factor,
            albedo,
            surfbb,
            emissivity,
        )
        return prepare_inputs(
            options=self._core_options(),
            tau_arr=to_numpy(tau_arr),
            omega_arr=to_numpy(omega_arr),
            asymm_arr=to_numpy(asymm_arr),
            height_grid=to_numpy(height_grid),
            user_obsgeoms=to_numpy(user_obsgeoms),
            stream_value=stream_value,
            flux_factor=flux_factor,
            albedo=albedo,
            d2s_scaling=to_numpy(d2s_scaling),
            brdf=brdf,
            surface_leaving=surface_leaving,
            user_angles=to_numpy(user_angles),
            beam_szas=to_numpy(beam_szas),
            user_relazms=to_numpy(user_relazms),
            thermal_bb_input=to_numpy(thermal_bb_input),
            surfbb=surfbb,
            emissivity=emissivity,
            earth_radius=earth_radius,
        ), torch_context

    def _prepare_forward(
        self,
        *,
        tau_arr: Any,
        omega_arr: Any,
        asymm_arr: Any,
        height_grid: Any | None,
        user_obsgeoms: Any | None,
        stream_value: float,
        flux_factor: float,
        albedo: float,
        d2s_scaling: Any | None = None,
        brdf: Any | None = None,
        surface_leaving: Any | None = None,
        user_angles: Any | None = None,
        beam_szas: Any | None = None,
        user_relazms: Any | None = None,
        thermal_bb_input: Any | None = None,
        surfbb: float = 0.0,
        emissivity: float = 0.0,
        earth_radius: float = 6371.0,
    ) -> PreparedInputs:
        """Normalizes inputs and returns prepared solver inputs."""
        prepared, _ = self._prepare_forward_with_context(
            tau_arr=tau_arr,
            omega_arr=omega_arr,
            asymm_arr=asymm_arr,
            height_grid=height_grid,
            user_obsgeoms=user_obsgeoms,
            stream_value=stream_value,
            flux_factor=flux_factor,
            albedo=albedo,
            d2s_scaling=d2s_scaling,
            brdf=brdf,
            surface_leaving=surface_leaving,
            user_angles=user_angles,
            beam_szas=beam_szas,
            user_relazms=user_relazms,
            thermal_bb_input=thermal_bb_input,
            surfbb=surfbb,
            emissivity=emissivity,
            earth_radius=earth_radius,
        )
        return prepared

    def _solve_forward_core(
        self,
        *,
        prepared: PreparedInputs,
        torch_context,
        tau_arr,
        omega_arr,
        asymm_arr,
        d2s_scaling,
        flux_factor=None,
        albedo=None,
        thermal_bb_input=None,
        surfbb=None,
        emissivity=None,
    ):
        """Dispatches the main forward solve to NumPy or torch.

        Returns
        -------
        dict
            Raw solver outputs from the active backend.
        """
        if (
            self.options.backend == "torch"
            and self._source_mode in {"solar_obs", "solar_lat"}
            and torch_context is not None
        ):
            from .rtsolver.solver_torch import solve_optimized_solar_obs_torch

            return solve_optimized_solar_obs_torch(
                prepared,
                self._core_options(),
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                d2s_scaling=d2s_scaling,
                flux_factor=flux_factor,
                albedo=albedo,
            )
        if (
            self.options.backend == "torch"
            and self._source_mode == "thermal"
            and torch_context is not None
        ):
            from .rtsolver.solver_torch import solve_optimized_thermal_torch

            return solve_optimized_thermal_torch(
                prepared,
                self._core_options(),
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                d2s_scaling=d2s_scaling,
                thermal_bb_input=thermal_bb_input,
                surfbb=surfbb,
                emissivity=emissivity,
                albedo=albedo,
            )
        return solve_optimized_solar_obs(prepared, self._core_options())

    def _solve_fo_forward(
        self,
        *,
        prepared: PreparedInputs,
        torch_context,
        tau_arr,
        omega_arr,
        asymm_arr,
        height_grid,
        user_angles,
        thermal_bb_input,
        d2s_scaling,
        user_obsgeoms,
        albedo: Any,
        flux_factor: Any,
        earth_radius: float,
        surfbb: Any,
        emissivity: Any,
        fo_geometry_mode: str,
        fo_n_moments: int,
        fo_nfine: int,
        fo_scatter_term,
    ):
        """Dispatches the FO solve to NumPy or torch.

        Returns
        -------
        FoSolarObsResult or FoThermalResult
            First-order solver output for the requested source mode.
        """
        if self._source_mode in {"solar_obs", "solar_lat"}:
            self._validate_fo_geometry_mode(fo_geometry_mode)
        fo_tau_arr = tau_arr
        fo_prepared = prepared
        if self.options.effective_fo_optical_deltam_scaling:
            if self.options.backend == "torch":
                fo_tau_arr = tau_arr * (1.0 - omega_arr * d2s_scaling)
            else:
                fo_tau_np = prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling)
                fo_prepared = replace(prepared, tau_arr=fo_tau_np)
        if self.options.backend == "torch":
            from .rtsolver.fo_solar_obs_torch import (
                solve_fo_solar_obs_eps_torch,
                solve_fo_solar_obs_plane_parallel_torch,
                solve_fo_solar_obs_rps_torch,
            )
            from .rtsolver.fo_thermal_torch import solve_fo_thermal_torch

            fo_scatter_term_t = value_to_torch(fo_scatter_term, torch_context)
            if self._source_mode in {"solar_obs", "solar_lat"}:
                if self.options.plane_parallel:
                    return solve_fo_solar_obs_plane_parallel_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=fo_n_moments,
                        exact_scatter=fo_scatter_term_t,
                    )
                if fo_geometry_mode == "rps":
                    return solve_fo_solar_obs_rps_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        height_grid=height_grid,
                        earth_radius=earth_radius,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=fo_n_moments,
                        exact_scatter=fo_scatter_term_t,
                    )
                return solve_fo_solar_obs_eps_torch(
                    tau_arr=fo_tau_arr,
                    omega_arr=omega_arr,
                    asymm_arr=asymm_arr,
                    user_obsgeoms=user_obsgeoms,
                    d2s_scaling=d2s_scaling,
                    height_grid=height_grid,
                    earth_radius=earth_radius,
                    albedo=albedo,
                    flux_factor=flux_factor,
                    n_moments=fo_n_moments,
                    nfine=fo_nfine,
                    exact_scatter=fo_scatter_term_t,
                )
            if self._source_mode == "thermal":
                if user_angles is None:
                    raise ValueError(
                        "backend='torch' thermal include_fo requires user_angles tensor input"
                    )
                if thermal_bb_input is None:
                    raise ValueError(
                        "backend='torch' thermal include_fo requires thermal_bb_input tensor input"
                    )
                return solve_fo_thermal_torch(
                    tau_arr=tau_arr,
                    omega_arr=omega_arr,
                    height_grid=height_grid,
                    user_angles=user_angles,
                    d2s_scaling=d2s_scaling,
                    thermal_bb_input=thermal_bb_input,
                    surfbb=surfbb,
                    emissivity=emissivity,
                    earth_radius=earth_radius,
                    do_plane_parallel=self.options.plane_parallel,
                    do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                    do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                    nfine=fo_nfine,
                )
            raise NotImplementedError(
                "include_fo with backend='torch' is currently implemented for "
                "mode='solar', 'solar_lattice', and 'thermal' only"
            )
        if self._source_mode in {"solar_obs", "solar_lat"}:
            return solve_fo_solar_obs(
                fo_prepared,
                do_plane_parallel=self.options.plane_parallel,
                geometry_mode=fo_geometry_mode,
                n_moments=fo_n_moments,
                nfine=fo_nfine,
                exact_scatter=None if fo_scatter_term is None else to_numpy(fo_scatter_term),
            )
        if self._source_mode == "thermal":
            return solve_fo_thermal(
                prepared,
                do_plane_parallel=self.options.plane_parallel,
                do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                nfine=fo_nfine,
            )
        raise NotImplementedError(
            "include_fo is implemented for mode='solar', 'solar_lattice', and 'thermal' only"
        )

    def _build_forward_result(
        self, *, solved, fo_result, prepared: PreparedInputs
    ) -> TwoStreamEssResult:
        """Builds the public result object from solver outputs.

        Returns
        -------
        TwoStreamEssResult
            Public-facing result wrapper with convenience helpers.
        """
        combined_intensity_toa = None
        combined_intensity_boa = None
        if isinstance(fo_result, FoSolarObsResult):
            combined_intensity_toa = solved["intensity_toa"] + fo_result.intensity_total
        return TwoStreamEssResult(
            intensity_toa=solved["intensity_toa"],
            intensity_boa=solved["intensity_boa"],
            combined_intensity_toa=combined_intensity_toa,
            combined_intensity_boa=combined_intensity_boa,
            fluxes_toa=solved["fluxes_toa"],
            fluxes_boa=solved["fluxes_boa"],
            radlevel_up=solved["radlevel_up"],
            radlevel_dn=solved["radlevel_dn"],
            output_levels=self.options.output_levels,
            lattice_counts=prepared.lattice_counts,
            lattice_axes=prepared.lattice_axes,
            **self._fo_result_kwargs(fo_result),
        )

    def forward(
        self,
        *,
        tau: Any,
        ssa: Any,
        g: Any,
        z: Any | None = None,
        angles: Any | None = None,
        stream: float | None = None,
        fbeam: Any = 1.0,
        albedo: Any = 0.0,
        delta_m_truncation_factor: Any | None = None,
        brdf: Any | None = None,
        surface_leaving: Any | None = None,
        view_angles: Any | None = None,
        beam_szas: Any | None = None,
        relazms: Any | None = None,
        planck: Any | None = None,
        surface_planck: Any = 0.0,
        emissivity: Any = 0.0,
        earth_radius: float = 6371.0,
        include_fo: bool = False,
        geometry: str = "pseudo_spherical",
        fo_n_moments: int = 5000,
        fo_nfine: int = 3,
        fo_scatter_term: Any | None = None,
    ) -> TwoStreamEssResult | TwoStreamEssBatchResult:
        """Runs the main 2S-ESS forward model.

        Parameters
        ----------
        tau, ssa, g
            Layer optical thickness, single-scattering albedo, and asymmetry
            inputs.
        z
            Height grid in descending order.
        angles
            Solar observation geometry in degrees, either ``[sza, vza, raz]``
            for one geometry or ``(ngeom, 3)`` for many geometries. For
            ``mode="thermal"``, this is scalar or 1D viewing zenith angle(s).
        stream
            Optional two-stream quadrature cosine. Defaults to ``1/sqrt(3)``.
        fbeam
            Direct solar beam/source normalization.
        albedo
            Lambertian surface albedo.
        delta_m_truncation_factor
            Optional per-layer delta-M truncation factor. When omitted, py2sess
            derives the Henyey-Greenstein fallback ``g**2``.
        brdf, surface_leaving
            Optional solar or thermal surface supplements.
        view_angles, beam_szas, relazms
            Advanced lattice-style or thermal viewing-angle inputs.
        planck, surface_planck, emissivity
            Thermal source and surface-emission inputs.
        earth_radius
            Planetary radius used by spherical geometry paths.
        include_fo
            Whether to also run and attach first-order outputs.
        geometry
            FO solar geometry mode, typically ``"pseudo_spherical"`` or
            ``"regular_pseudo_spherical"``.
        fo_n_moments
            Number of phase-function moments used by the FO solver.
        fo_nfine
            Number of fine-layer quadrature divisions used by the FO EPS path.
        fo_scatter_term
            Optional precomputed solar FO scatter term for the attached FO
            solve. This is the phase-function value multiplied by the
            single-scattering and delta-M source-scaling factor.

        Returns
        -------
        TwoStreamEssResult or TwoStreamEssBatchResult
            Public forward-model output object. Batched inputs return endpoint
            radiances with the same leading batch dimensions as ``tau``.
        """
        if include_fo:
            if fo_n_moments < 0:
                raise ValueError("fo_n_moments must be non-negative")
            if fo_nfine <= 0:
                raise ValueError("fo_nfine must be positive")
        mapped = self._translate_public_forward_args(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            stream=stream,
            fbeam=fbeam,
            delta_m_truncation_factor=delta_m_truncation_factor,
            view_angles=view_angles,
            beam_szas=beam_szas,
            relazms=relazms,
            planck=planck,
            surface_planck=surface_planck,
            geometry=geometry,
        )
        tau_arr = mapped["tau_arr"]
        omega_arr = mapped["omega_arr"]
        asymm_arr = mapped["asymm_arr"]
        height_grid = mapped["height_grid"]
        user_obsgeoms = mapped["user_obsgeoms"]
        user_angles = mapped["user_angles"]
        user_relazms = mapped["user_relazms"]
        stream_value = mapped["stream_value"]
        flux_factor = mapped["flux_factor"]
        d2s_scaling = mapped["d2s_scaling"]
        thermal_bb_input = mapped["thermal_bb_input"]
        surfbb = mapped["surfbb"]
        fo_geometry_mode = mapped["fo_geometry_mode"]
        if self._public_forward_is_batched(tau_arr):
            return self._forward_batched(
                mapped=mapped,
                albedo=albedo,
                brdf=brdf,
                surface_leaving=surface_leaving,
                emissivity=emissivity,
                earth_radius=earth_radius,
                include_fo=include_fo,
                fo_n_moments=fo_n_moments,
                fo_nfine=fo_nfine,
                fo_scatter_term=fo_scatter_term,
            )
        prepared, torch_context = self._prepare_forward_with_context(
            tau_arr=tau_arr,
            omega_arr=omega_arr,
            asymm_arr=asymm_arr,
            height_grid=height_grid,
            user_obsgeoms=user_obsgeoms,
            stream_value=stream_value,
            flux_factor=flux_factor,
            albedo=albedo,
            d2s_scaling=d2s_scaling,
            brdf=brdf,
            surface_leaving=surface_leaving,
            user_angles=user_angles,
            beam_szas=beam_szas,
            user_relazms=user_relazms,
            thermal_bb_input=thermal_bb_input,
            surfbb=surfbb,
            emissivity=emissivity,
            earth_radius=earth_radius,
        )
        if height_grid is None:
            height_grid = prepared.height_grid
        if self.options.backend == "torch":
            torch_context = self._select_torch_context(torch_context)
        if self.options.backend == "torch":
            tau_arr = value_to_torch(tau_arr, torch_context)
            omega_arr = value_to_torch(omega_arr, torch_context)
            asymm_arr = value_to_torch(asymm_arr, torch_context)
            height_grid = value_to_torch(height_grid, torch_context)
            user_obsgeoms = value_to_torch(user_obsgeoms, torch_context)
            user_angles = value_to_torch(user_angles, torch_context)
            thermal_bb_input = value_to_torch(thermal_bb_input, torch_context)
            flux_factor = value_to_torch(flux_factor, torch_context)
            albedo = value_to_torch(albedo, torch_context)
            surfbb = value_to_torch(surfbb, torch_context)
            emissivity = value_to_torch(emissivity, torch_context)
            with self._torch_grad_context():
                d2s_scaling = self._resolve_truncation_factor_torch(
                    d2s_scaling,
                    asymm=asymm_arr,
                    omega=omega_arr,
                    context=torch_context,
                )
        elif d2s_scaling is None:
            d2s_scaling = prepared.d2s_scaling
        fo_user_obsgeoms = user_obsgeoms
        if (
            self.options.backend == "torch"
            and fo_user_obsgeoms is None
            and prepared.user_obsgeoms is not None
            and self._source_mode in {"solar_obs", "solar_lat"}
        ):
            fo_user_obsgeoms = value_to_torch(prepared.user_obsgeoms, torch_context)
        with self._torch_grad_context():
            solved = self._solve_forward_core(
                prepared=prepared,
                torch_context=torch_context,
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                d2s_scaling=d2s_scaling,
                flux_factor=flux_factor,
                albedo=albedo,
                thermal_bb_input=thermal_bb_input,
                surfbb=surfbb,
                emissivity=emissivity,
            )
            fo_result = None
            if include_fo:
                fo_result = self._solve_fo_forward(
                    prepared=prepared,
                    torch_context=torch_context,
                    tau_arr=tau_arr,
                    omega_arr=omega_arr,
                    asymm_arr=asymm_arr,
                    height_grid=height_grid,
                    user_angles=user_angles,
                    thermal_bb_input=thermal_bb_input,
                    d2s_scaling=d2s_scaling,
                    user_obsgeoms=fo_user_obsgeoms,
                    albedo=albedo,
                    flux_factor=flux_factor,
                    earth_radius=earth_radius,
                    surfbb=surfbb,
                    emissivity=emissivity,
                    fo_geometry_mode=fo_geometry_mode,
                    fo_n_moments=fo_n_moments,
                    fo_nfine=fo_nfine,
                    fo_scatter_term=fo_scatter_term,
                )
        return self._build_forward_result(
            solved=solved,
            fo_result=fo_result,
            prepared=prepared,
        )

    def forward_fo(
        self,
        *,
        tau: Any,
        ssa: Any,
        g: Any,
        z: Any | None = None,
        angles: Any | None = None,
        stream: float | None = None,
        fbeam: Any = 1.0,
        albedo: Any = 0.0,
        delta_m_truncation_factor: Any | None = None,
        view_angles: Any | None = None,
        beam_szas: Any | None = None,
        relazms: Any | None = None,
        brdf: Any | None = None,
        surface_leaving: Any | None = None,
        earth_radius: float = 6371.0,
        planck: Any | None = None,
        surface_planck: Any = 0.0,
        emissivity: Any = 0.0,
        geometry: str = "pseudo_spherical",
        n_moments: int = 5000,
        nfine: int = 3,
        fo_scatter_term: Any | None = None,
    ) -> FoSolarObsResult | FoThermalResult:
        """Runs the FO-only solver path.

        Parameters
        ----------
        tau, ssa, g
            Layer optical inputs.
        z
            Height grid in descending order.
        angles
            Solar observation geometry in degrees, either ``[sza, vza, raz]``
            for one geometry or ``(ngeom, 3)`` for many geometries. For
            ``mode="thermal"``, this is scalar or 1D viewing zenith angle(s).
        stream, fbeam, albedo
            Optional two-stream quadrature, beam normalization, and surface
            albedo inputs reused by the FO path.
        delta_m_truncation_factor
            Optional per-layer delta-M truncation factor. When omitted, py2sess
            derives the Henyey-Greenstein fallback ``g**2``.
        view_angles, beam_szas, relazms
            Advanced lattice-style or thermal viewing-angle inputs.
        brdf, surface_leaving
            Optional BRDF and surface-leaving inputs passed to the solver core.
        earth_radius
            Planetary radius used by spherical geometry paths.
        planck, surface_planck, emissivity
            Thermal source and surface-emission inputs.
        geometry
            Solar FO geometry mode.
        n_moments
            Number of phase-function moments used by the FO solver.
        nfine
            Number of fine-layer quadrature divisions used by the FO EPS path.
        fo_scatter_term
            Optional precomputed solar FO scatter term with shape
            ``(n_layers,)`` or ``(n_layers, n_geometries)``.

        Returns
        -------
        FoSolarObsResult or FoThermalResult
            FO-only result object for the requested source mode.
        """
        if n_moments < 0:
            raise ValueError("n_moments must be non-negative")
        if nfine <= 0:
            raise ValueError("nfine must be positive")
        mapped = self._translate_public_forward_args(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            stream=stream,
            fbeam=fbeam,
            delta_m_truncation_factor=delta_m_truncation_factor,
            view_angles=view_angles,
            beam_szas=beam_szas,
            relazms=relazms,
            planck=planck,
            surface_planck=surface_planck,
            geometry=geometry,
        )
        tau_arr = mapped["tau_arr"]
        omega_arr = mapped["omega_arr"]
        asymm_arr = mapped["asymm_arr"]
        height_grid = mapped["height_grid"]
        user_obsgeoms = mapped["user_obsgeoms"]
        user_angles = mapped["user_angles"]
        user_relazms = mapped["user_relazms"]
        stream_value = mapped["stream_value"]
        flux_factor = mapped["flux_factor"]
        d2s_scaling = mapped["d2s_scaling"]
        thermal_bb_input = mapped["thermal_bb_input"]
        surfbb = mapped["surfbb"]
        fo_geometry_mode = mapped["fo_geometry_mode"]
        if self._source_mode in {"solar_obs", "solar_lat"}:
            self._validate_fo_geometry_mode(fo_geometry_mode)
        if self._public_forward_is_batched(tau_arr):
            if brdf is not None or surface_leaving is not None:
                raise ValueError("batched forward_fo does not support brdf or surface_leaving")
            if self.options.brdf_surface or self.options.surface_leaving:
                raise ValueError(
                    "batched forward_fo does not support brdf_surface or surface_leaving options"
                )
            if self._source_mode != "solar_obs":
                raise ValueError("batched forward_fo currently supports mode='solar' only")
            if self.options.backend == "torch":
                return self._forward_fo_solar_obs_batched_torch(
                    mapped=mapped,
                    albedo=albedo,
                    earth_radius=earth_radius,
                    n_moments=n_moments,
                    nfine=nfine,
                    fo_scatter_term=fo_scatter_term,
                )
            return self._forward_fo_solar_obs_batched_numpy(
                mapped=mapped,
                albedo=albedo,
                earth_radius=earth_radius,
                n_moments=n_moments,
                nfine=nfine,
                fo_scatter_term=fo_scatter_term,
            )
        if self._source_mode == "solar_lat":
            prepared, torch_context = self._prepare_forward_with_context(
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                height_grid=height_grid,
                user_obsgeoms=None,
                stream_value=stream_value,
                flux_factor=flux_factor,
                albedo=albedo,
                d2s_scaling=d2s_scaling,
                brdf=brdf,
                surface_leaving=surface_leaving,
                user_angles=user_angles,
                beam_szas=beam_szas,
                user_relazms=user_relazms,
                thermal_bb_input=thermal_bb_input,
                surfbb=surfbb,
                emissivity=emissivity,
                earth_radius=earth_radius,
            )
            if height_grid is None:
                height_grid = prepared.height_grid
            if self.options.backend == "torch":
                from .rtsolver.fo_solar_obs_torch import (
                    solve_fo_solar_obs_eps_torch,
                    solve_fo_solar_obs_plane_parallel_torch,
                    solve_fo_solar_obs_rps_torch,
                )

                torch_context = self._select_torch_context(torch_context)
                with self._torch_grad_context():
                    tau_arr = value_to_torch(tau_arr, torch_context)
                    omega_arr = value_to_torch(omega_arr, torch_context)
                    asymm_arr = value_to_torch(asymm_arr, torch_context)
                    d2s_scaling = self._resolve_truncation_factor_torch(
                        d2s_scaling,
                        asymm=asymm_arr,
                        omega=omega_arr,
                        context=torch_context,
                    )
                    height_grid = value_to_torch(height_grid, torch_context)
                    user_obsgeoms = value_to_torch(prepared.user_obsgeoms, torch_context)
                    flux_factor = value_to_torch(flux_factor, torch_context)
                    albedo = value_to_torch(albedo, torch_context)
                    fo_scatter_term_t = value_to_torch(fo_scatter_term, torch_context)
                    fo_tau_arr = tau_arr
                    if self.options.effective_fo_optical_deltam_scaling:
                        fo_tau_arr = tau_arr * (1.0 - omega_arr * d2s_scaling)
                    if self.options.plane_parallel:
                        result = solve_fo_solar_obs_plane_parallel_torch(
                            tau_arr=fo_tau_arr,
                            omega_arr=omega_arr,
                            asymm_arr=asymm_arr,
                            user_obsgeoms=user_obsgeoms,
                            d2s_scaling=d2s_scaling,
                            albedo=albedo,
                            flux_factor=flux_factor,
                            n_moments=n_moments,
                            exact_scatter=fo_scatter_term_t,
                        )
                    elif fo_geometry_mode == "rps":
                        result = solve_fo_solar_obs_rps_torch(
                            tau_arr=fo_tau_arr,
                            omega_arr=omega_arr,
                            asymm_arr=asymm_arr,
                            user_obsgeoms=user_obsgeoms,
                            d2s_scaling=d2s_scaling,
                            height_grid=height_grid,
                            earth_radius=earth_radius,
                            albedo=albedo,
                            flux_factor=flux_factor,
                            n_moments=n_moments,
                            exact_scatter=fo_scatter_term_t,
                        )
                    else:
                        result = solve_fo_solar_obs_eps_torch(
                            tau_arr=fo_tau_arr,
                            omega_arr=omega_arr,
                            asymm_arr=asymm_arr,
                            user_obsgeoms=user_obsgeoms,
                            d2s_scaling=d2s_scaling,
                            height_grid=height_grid,
                            earth_radius=earth_radius,
                            albedo=albedo,
                            flux_factor=flux_factor,
                            n_moments=n_moments,
                            nfine=nfine,
                            exact_scatter=fo_scatter_term_t,
                        )
                return replace(
                    result,
                    lattice_counts=prepared.lattice_counts,
                    lattice_axes=prepared.lattice_axes,
                )
            if self.options.effective_fo_optical_deltam_scaling:
                prepared = replace(
                    prepared,
                    tau_arr=prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling),
                )
            result = solve_fo_solar_obs(
                prepared,
                do_plane_parallel=self.options.plane_parallel,
                geometry_mode=fo_geometry_mode,
                n_moments=n_moments,
                nfine=nfine,
                exact_scatter=None if fo_scatter_term is None else to_numpy(fo_scatter_term),
            )
            return replace(
                result, lattice_counts=prepared.lattice_counts, lattice_axes=prepared.lattice_axes
            )
        if self.options.backend == "torch":
            from .rtsolver.fo_solar_obs_torch import (
                solve_fo_solar_obs_eps_torch,
                solve_fo_solar_obs_plane_parallel_torch,
                solve_fo_solar_obs_rps_torch,
            )
            from .rtsolver.fo_thermal_torch import solve_fo_thermal_torch

            torch_context = detect_torch_context(
                tau_arr,
                omega_arr,
                asymm_arr,
                user_obsgeoms,
                d2s_scaling,
                height_grid,
                thermal_bb_input,
                flux_factor,
                albedo,
                surfbb,
                emissivity,
            )
            torch_context = self._select_torch_context(torch_context)
            with self._torch_grad_context():
                tau_arr = value_to_torch(tau_arr, torch_context)
                omega_arr = value_to_torch(omega_arr, torch_context)
                asymm_arr = value_to_torch(asymm_arr, torch_context)
                user_obsgeoms = value_to_torch(user_obsgeoms, torch_context)
                d2s_scaling = self._resolve_truncation_factor_torch(
                    d2s_scaling,
                    asymm=asymm_arr,
                    omega=omega_arr,
                    context=torch_context,
                )
                height_grid = value_to_torch(height_grid, torch_context)
                user_angles = value_to_torch(user_angles, torch_context)
                thermal_bb_input = value_to_torch(thermal_bb_input, torch_context)
                flux_factor = value_to_torch(flux_factor, torch_context)
                albedo = value_to_torch(albedo, torch_context)
                surfbb = value_to_torch(surfbb, torch_context)
                emissivity = value_to_torch(emissivity, torch_context)
                fo_scatter_term_t = value_to_torch(fo_scatter_term, torch_context)
                fo_tau_arr = tau_arr
                if self.options.effective_fo_optical_deltam_scaling:
                    fo_tau_arr = tau_arr * (1.0 - omega_arr * d2s_scaling)
                if self._source_mode == "solar_obs" and self.options.plane_parallel:
                    return solve_fo_solar_obs_plane_parallel_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=n_moments,
                        exact_scatter=fo_scatter_term_t,
                    )
                if self._source_mode == "solar_obs" and fo_geometry_mode == "rps":
                    return solve_fo_solar_obs_rps_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        height_grid=height_grid,
                        earth_radius=earth_radius,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=n_moments,
                        exact_scatter=fo_scatter_term_t,
                    )
                if self._source_mode == "solar_obs" and fo_geometry_mode == "eps":
                    return solve_fo_solar_obs_eps_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        height_grid=height_grid,
                        earth_radius=earth_radius,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=n_moments,
                        nfine=nfine,
                        exact_scatter=fo_scatter_term_t,
                    )
                if self._source_mode == "thermal":
                    if user_angles is None:
                        raise ValueError(
                            "backend='torch' thermal forward_fo requires user_angles tensor input"
                        )
                    if thermal_bb_input is None:
                        raise ValueError(
                            "backend='torch' thermal forward_fo requires thermal_bb_input tensor input"
                        )
                    return solve_fo_thermal_torch(
                        tau_arr=tau_arr,
                        omega_arr=omega_arr,
                        d2s_scaling=d2s_scaling,
                        user_angles=user_angles,
                        thermal_bb_input=thermal_bb_input,
                        surfbb=surfbb,
                        emissivity=emissivity,
                        do_plane_parallel=self.options.plane_parallel,
                        height_grid=height_grid,
                        earth_radius=earth_radius,
                        do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                        do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                        nfine=nfine,
                    )
            raise NotImplementedError(
                "backend='torch' forward_fo is currently implemented for "
                "mode='solar', 'solar_lattice', and 'thermal' only"
            )
        prepared = self._prepare_forward(
            tau_arr=tau_arr,
            omega_arr=omega_arr,
            asymm_arr=asymm_arr,
            height_grid=height_grid,
            user_obsgeoms=user_obsgeoms,
            stream_value=stream_value,
            flux_factor=flux_factor,
            albedo=albedo,
            d2s_scaling=d2s_scaling,
            brdf=brdf,
            surface_leaving=surface_leaving,
            user_angles=user_angles,
            thermal_bb_input=thermal_bb_input,
            surfbb=surfbb,
            emissivity=emissivity,
            earth_radius=earth_radius,
        )
        if self._source_mode == "thermal":
            return solve_fo_thermal(
                prepared,
                do_plane_parallel=self.options.plane_parallel,
                do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                nfine=nfine,
            )
        if self.options.effective_fo_optical_deltam_scaling:
            prepared = replace(
                prepared,
                tau_arr=prepared.tau_arr * (1.0 - prepared.omega_arr * prepared.d2s_scaling),
            )
        result = solve_fo_solar_obs(
            prepared,
            do_plane_parallel=self.options.plane_parallel,
            geometry_mode=fo_geometry_mode,
            n_moments=n_moments,
            nfine=nfine,
            exact_scatter=None if fo_scatter_term is None else to_numpy(fo_scatter_term),
        )
        if prepared.lattice_counts is not None:
            return replace(
                result, lattice_counts=prepared.lattice_counts, lattice_axes=prepared.lattice_axes
            )
        return result
