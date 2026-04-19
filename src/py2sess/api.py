"""Public API for the Python 2S-ESS package."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .core.backend import TorchContext, detect_torch_context, has_torch, to_numpy, value_to_torch
from .core.fo_solar_obs import FoSolarObsResult, solve_fo_solar_obs
from .core.fo_thermal import FoThermalResult, solve_fo_thermal
from .core.lattice_result import add_lattice_axes, lattice_shape, reshape_lattice_array
from .core.preprocess import PreparedInputs, prepare_inputs
from .core.result_components import build_solar_components, build_thermal_components
from .core.solver import solve_optimized_solar_obs


@dataclass(frozen=True)
class TwoStreamEssOptions:
    """Configuration for a forward-model run.

    Parameters
    ----------
    n_layers
        Number of atmospheric layers in the problem.
    backend
        Execution backend. Use ``"numpy"`` for the reference CPU path or
        ``"torch"`` for the native tensor path where supported.
    source_mode
        Forward-model source mode. Supported values are ``"solar_obs"``,
        ``"solar_lat"``, and ``"thermal"``.
    do_upwelling, do_dnwelling
        Flags controlling whether upwelling and/or downwelling outputs are
        computed.
    do_plane_parallel
        Whether to use the plane-parallel geometry approximation.
    do_delta_scaling
        Whether to apply delta-M scaling to the optical properties.
    do_level_output
        Whether full level profiles should be returned.
    do_mvout_only, do_additional_mvout
        Flags controlling which flux outputs are produced.
    do_surface_leaving, do_sl_isotropic
        Solar surface-leaving configuration flags.
    do_brdf_surface
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
    fo_optical_deltam_scaling
        Optional override for FO optical-depth delta-M scaling. ``None``
        inherits ``do_delta_scaling``.
    fo_thermal_source_deltam_scaling
        Optional override for the thermal FO source-side delta-M multiplier
        used by the raw Fortran FO thermal core. ``None`` disables source-side
        scaling, matching the corrected FO thermal handling described by the
        code author.
    """

    n_layers: int
    backend: str = "numpy"
    source_mode: str = "solar_obs"
    do_upwelling: bool = True
    do_dnwelling: bool = False
    do_plane_parallel: bool = False
    do_delta_scaling: bool = True
    do_level_output: bool = False
    do_mvout_only: bool = False
    do_additional_mvout: bool = False
    do_surface_leaving: bool = False
    do_sl_isotropic: bool = True
    do_brdf_surface: bool = False
    bvp_solver: str = "scipy"
    thermal_tcutoff: float = 1.0e-8
    torch_device: str | None = None
    torch_dtype: str | None = None
    torch_enable_grad: bool = True
    fo_optical_deltam_scaling: bool | None = None
    fo_thermal_source_deltam_scaling: bool | None = None

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        if self.source_mode not in {"solar_obs", "solar_lat", "thermal"}:
            raise ValueError("source_mode must be 'solar_obs', 'solar_lat', or 'thermal'")
        if self.backend == "torch" and not has_torch():
            raise ValueError("backend='torch' requires torch to be installed")
        if self.bvp_solver not in {"scipy", "banded", "pentadiag"}:
            raise ValueError("bvp_solver must be 'scipy', 'banded', or 'pentadiag'")

    @property
    def effective_fo_optical_deltam_scaling(self) -> bool:
        """Returns FO optical delta-M control after applying inheritance."""
        return (
            self.do_delta_scaling
            if self.fo_optical_deltam_scaling is None
            else self.fo_optical_deltam_scaling
        )

    @property
    def effective_fo_thermal_source_deltam_scaling(self) -> bool:
        """Returns thermal FO source delta-M control after applying inheritance."""
        if self.fo_thermal_source_deltam_scaling is not None:
            return self.fo_thermal_source_deltam_scaling
        return False


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
        ``source_mode="solar_lat"``.
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
    lattice_counts: tuple[int, int, int] | None = None
    lattice_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

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
        return build_solar_components(
            intensity_toa=self.intensity_toa,
            intensity_boa=self.intensity_boa,
            fo_intensity_total=self.fo_intensity_total,
            fo_intensity_ss=self.fo_intensity_ss,
            fo_intensity_db=self.fo_intensity_db,
            combined_intensity_toa=self.combined_intensity_toa,
        )

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
        return build_thermal_components(
            intensity_toa=self.intensity_toa,
            intensity_boa=self.intensity_boa,
            fo_mu1=self.fo_mu1,
            fo_thermal_atmos_up_toa=self.fo_thermal_atmos_up_toa,
            fo_thermal_surface_toa=self.fo_thermal_surface_toa,
            fo_thermal_total_up_toa=self.fo_thermal_total_up_toa,
            fo_thermal_atmos_dn_toa=self.fo_thermal_atmos_dn_toa,
            fo_thermal_atmos_up_boa=self.fo_thermal_atmos_up_boa,
            fo_thermal_surface_boa=self.fo_thermal_surface_boa,
            fo_thermal_total_up_boa=self.fo_thermal_total_up_boa,
            fo_thermal_atmos_dn_boa=self.fo_thermal_atmos_dn_boa,
        )

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

    @staticmethod
    def _validate_fo_geometry_mode(fo_geometry_mode: str) -> None:
        """Validates the named FO solar geometry mode."""
        if fo_geometry_mode not in {"eps", "rps"}:
            raise ValueError("fo_geometry_mode must be 'eps' or 'rps'")

    def _default_torch_context(self) -> TorchContext:
        """Returns the default torch context for converted inputs.

        Returns
        -------
        TorchContext
            ``float64`` context used when the caller asks for the torch backend
            but passes NumPy inputs.
        """
        from .core.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        device = self._resolve_torch_device()
        return TorchContext(dtype=self._resolve_torch_dtype(None, device), device=device)

    def _resolve_torch_device(self):
        """Resolves and validates the configured torch device."""
        from .core.backend import _load_torch

        torch = _load_torch()
        if torch is None:  # pragma: no cover
            raise RuntimeError("backend='torch' requires torch to be installed")
        device = torch.device(self.options.torch_device or "cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("torch_device='cuda' requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "torch_device='mps' requested but MPS is not available in this process. "
                "On macOS this can happen inside the Codex sandbox even when the same "
                "Python interpreter sees MPS outside the sandbox; rerun the command outside "
                "the sandbox/escalated environment, or use torch_device='cpu'."
            )
        return device

    def _resolve_torch_dtype(self, detected: TorchContext | None, device):
        """Resolves and validates the configured torch dtype."""
        from .core.backend import _load_torch

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
        from .core.backend import _load_torch

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
        )
        return prepare_inputs(
            options=self.options,
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
    ):
        """Dispatches the main forward solve to NumPy or torch.

        Returns
        -------
        dict
            Raw solver outputs from the active backend.
        """
        if (
            self.options.backend == "torch"
            and self.options.source_mode in {"solar_obs", "solar_lat"}
            and torch_context is not None
        ):
            from .core.solver_torch import solve_optimized_solar_obs_torch

            return solve_optimized_solar_obs_torch(
                prepared,
                self.options,
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                d2s_scaling=d2s_scaling,
            )
        if (
            self.options.backend == "torch"
            and self.options.source_mode == "thermal"
            and torch_context is not None
        ):
            from .core.solver_torch import solve_optimized_thermal_torch

            return solve_optimized_thermal_torch(
                prepared,
                self.options,
                tau_arr=tau_arr,
                omega_arr=omega_arr,
                asymm_arr=asymm_arr,
                d2s_scaling=d2s_scaling,
            )
        return solve_optimized_solar_obs(prepared, self.options)

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
        albedo: float,
        earth_radius: float,
        surfbb: float,
        emissivity: float,
        fo_geometry_mode: str,
        fo_n_moments: int,
        fo_nfine: int,
        fo_exact_scatter,
    ):
        """Dispatches the FO solve to NumPy or torch.

        Returns
        -------
        FoSolarObsResult or FoThermalResult
            First-order solver output for the requested source mode.
        """
        if self.options.source_mode in {"solar_obs", "solar_lat"}:
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
            from .core.fo_solar_obs_torch import (
                solve_fo_solar_obs_eps_torch,
                solve_fo_solar_obs_plane_parallel_torch,
                solve_fo_solar_obs_rps_torch,
            )
            from .core.fo_thermal_torch import solve_fo_thermal_torch

            fo_exact_scatter_t = value_to_torch(fo_exact_scatter, torch_context)
            if self.options.source_mode in {"solar_obs", "solar_lat"}:
                if self.options.do_plane_parallel:
                    return solve_fo_solar_obs_plane_parallel_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        albedo=albedo,
                        flux_factor=prepared.flux_factor,
                        n_moments=fo_n_moments,
                        exact_scatter=fo_exact_scatter_t,
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
                        flux_factor=prepared.flux_factor,
                        n_moments=fo_n_moments,
                        exact_scatter=fo_exact_scatter_t,
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
                    flux_factor=prepared.flux_factor,
                    n_moments=fo_n_moments,
                    nfine=fo_nfine,
                    exact_scatter=fo_exact_scatter_t,
                )
            if self.options.source_mode == "thermal":
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
                    do_plane_parallel=self.options.do_plane_parallel,
                    do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                    do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                    nfine=fo_nfine,
                )
            raise NotImplementedError(
                "include_fo with backend='torch' is currently implemented for "
                "source_mode='solar_obs', 'solar_lat', and 'thermal' only"
            )
        if self.options.source_mode in {"solar_obs", "solar_lat"}:
            return solve_fo_solar_obs(
                fo_prepared,
                do_plane_parallel=self.options.do_plane_parallel,
                geometry_mode=fo_geometry_mode,
                n_moments=fo_n_moments,
                nfine=fo_nfine,
                exact_scatter=None if fo_exact_scatter is None else to_numpy(fo_exact_scatter),
            )
        if self.options.source_mode == "thermal":
            return solve_fo_thermal(
                prepared,
                do_plane_parallel=self.options.do_plane_parallel,
                do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                nfine=fo_nfine,
            )
        raise NotImplementedError(
            "include_fo is implemented for source_mode='solar_obs', 'solar_lat', and 'thermal' only"
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
            lattice_counts=prepared.lattice_counts,
            lattice_axes=prepared.lattice_axes,
            **self._fo_result_kwargs(fo_result),
        )

    def forward(
        self,
        *,
        tau_arr: Any,
        omega_arr: Any,
        asymm_arr: Any,
        height_grid: Any | None = None,
        user_obsgeoms: Any | None = None,
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
        include_fo: bool = False,
        fo_geometry_mode: str = "eps",
        fo_n_moments: int = 5000,
        fo_nfine: int = 3,
        fo_exact_scatter: Any | None = None,
    ) -> TwoStreamEssResult:
        """Runs the main 2S-ESS forward model.

        Parameters
        ----------
        tau_arr, omega_arr, asymm_arr
            Layer optical thickness, single-scattering albedo, and asymmetry
            inputs.
        height_grid
            Height grid in descending order.
        user_obsgeoms
            Observation-geometry array for solar observation mode.
        stream_value
            Two-stream quadrature value.
        flux_factor
            Flux normalization factor.
        albedo
            Lambertian surface albedo.
        d2s_scaling
            Optional delta-M scaling factors.
        brdf, surface_leaving
            Optional solar or thermal surface supplements.
        user_angles, beam_szas, user_relazms
            Lattice-style or thermal viewing-angle inputs where applicable.
        thermal_bb_input, surfbb, emissivity
            Thermal source and surface-emission inputs.
        earth_radius
            Planetary radius used by spherical geometry paths.
        include_fo
            Whether to also run and attach first-order outputs.
        fo_geometry_mode
            FO solar geometry mode, typically ``"eps"`` or ``"rps"``.
        fo_n_moments
            Number of phase-function moments used by the FO solver.
        fo_nfine
            Number of fine-layer quadrature divisions used by the FO EPS path.
        fo_exact_scatter
            Optional precomputed solar single-scatter phase term for the
            attached FO solve.

        Returns
        -------
        TwoStreamEssResult
            Public forward-model output object.
        """
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
        if self.options.backend == "torch":
            torch_context = self._select_torch_context(torch_context)
        if self.options.backend == "torch":
            tau_arr = value_to_torch(tau_arr, torch_context)
            omega_arr = value_to_torch(omega_arr, torch_context)
            asymm_arr = value_to_torch(asymm_arr, torch_context)
            height_grid = value_to_torch(height_grid, torch_context)
            user_obsgeoms = value_to_torch(user_obsgeoms, torch_context)
            d2s_scaling = value_to_torch(d2s_scaling, torch_context)
            user_angles = value_to_torch(user_angles, torch_context)
            thermal_bb_input = value_to_torch(thermal_bb_input, torch_context)
        fo_user_obsgeoms = user_obsgeoms
        if (
            self.options.backend == "torch"
            and fo_user_obsgeoms is None
            and prepared.user_obsgeoms is not None
            and self.options.source_mode in {"solar_obs", "solar_lat"}
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
                    earth_radius=earth_radius,
                    surfbb=surfbb,
                    emissivity=emissivity,
                    fo_geometry_mode=fo_geometry_mode,
                    fo_n_moments=fo_n_moments,
                    fo_nfine=fo_nfine,
                    fo_exact_scatter=fo_exact_scatter,
                )
        return self._build_forward_result(
            solved=solved,
            fo_result=fo_result,
            prepared=prepared,
        )

    def forward_fo(
        self,
        *,
        tau_arr: Any,
        omega_arr: Any,
        asymm_arr: Any,
        height_grid: Any | None = None,
        user_obsgeoms: Any | None = None,
        user_angles: Any | None = None,
        beam_szas: Any | None = None,
        user_relazms: Any | None = None,
        stream_value: float,
        flux_factor: float,
        albedo: float,
        d2s_scaling: Any | None = None,
        brdf: Any | None = None,
        surface_leaving: Any | None = None,
        earth_radius: float = 6371.0,
        thermal_bb_input: Any | None = None,
        surfbb: float = 0.0,
        emissivity: float = 0.0,
        fo_geometry_mode: str = "eps",
        n_moments: int = 5000,
        nfine: int = 3,
        fo_exact_scatter: Any | None = None,
    ) -> FoSolarObsResult | FoThermalResult:
        """Runs the FO-only solver path.

        Parameters
        ----------
        tau_arr, omega_arr, asymm_arr
            Layer optical inputs.
        height_grid
            Height grid in descending order.
        user_obsgeoms
            Observation-geometry array for solar observation mode.
        user_angles, beam_szas, user_relazms
            Thermal or lattice-style viewing-angle inputs where applicable.
        stream_value, flux_factor, albedo
            Standard forward-model scalar inputs reused by the FO path.
        d2s_scaling
            Optional delta-M scaling factors.
        brdf, surface_leaving
            Optional surface supplements for preprocessing compatibility.
        earth_radius
            Planetary radius used by spherical geometry paths.
        thermal_bb_input, surfbb, emissivity
            Thermal source and surface-emission inputs.
        fo_geometry_mode
            Solar FO geometry mode.
        n_moments
            Number of phase-function moments used by the FO solver.
        nfine
            Number of fine-layer quadrature divisions used by the FO EPS path.
        fo_exact_scatter
            Optional precomputed solar single-scatter phase term with shape
            ``(n_layers,)`` or ``(n_layers, n_geometries)``.

        Returns
        -------
        FoSolarObsResult or FoThermalResult
            FO-only result object for the requested source mode.
        """
        if self.options.source_mode in {"solar_obs", "solar_lat"}:
            self._validate_fo_geometry_mode(fo_geometry_mode)
        if self.options.source_mode == "solar_lat":
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
            if self.options.backend == "torch":
                from .core.fo_solar_obs_torch import (
                    solve_fo_solar_obs_eps_torch,
                    solve_fo_solar_obs_plane_parallel_torch,
                    solve_fo_solar_obs_rps_torch,
                )

                torch_context = self._select_torch_context(torch_context)
                with self._torch_grad_context():
                    tau_arr = value_to_torch(tau_arr, torch_context)
                    omega_arr = value_to_torch(omega_arr, torch_context)
                    asymm_arr = value_to_torch(asymm_arr, torch_context)
                    d2s_scaling = value_to_torch(d2s_scaling, torch_context)
                    height_grid = value_to_torch(height_grid, torch_context)
                    user_obsgeoms = value_to_torch(prepared.user_obsgeoms, torch_context)
                    fo_exact_scatter_t = value_to_torch(fo_exact_scatter, torch_context)
                    fo_tau_arr = tau_arr
                    if self.options.effective_fo_optical_deltam_scaling:
                        fo_tau_arr = tau_arr * (1.0 - omega_arr * d2s_scaling)
                    if self.options.do_plane_parallel:
                        result = solve_fo_solar_obs_plane_parallel_torch(
                            tau_arr=fo_tau_arr,
                            omega_arr=omega_arr,
                            asymm_arr=asymm_arr,
                            user_obsgeoms=user_obsgeoms,
                            d2s_scaling=d2s_scaling,
                            albedo=albedo,
                            flux_factor=prepared.flux_factor,
                            n_moments=n_moments,
                            exact_scatter=fo_exact_scatter_t,
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
                            flux_factor=prepared.flux_factor,
                            n_moments=n_moments,
                            exact_scatter=fo_exact_scatter_t,
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
                            flux_factor=prepared.flux_factor,
                            n_moments=n_moments,
                            nfine=nfine,
                            exact_scatter=fo_exact_scatter_t,
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
                do_plane_parallel=self.options.do_plane_parallel,
                geometry_mode=fo_geometry_mode,
                n_moments=n_moments,
                nfine=nfine,
                exact_scatter=None if fo_exact_scatter is None else to_numpy(fo_exact_scatter),
            )
            return replace(
                result, lattice_counts=prepared.lattice_counts, lattice_axes=prepared.lattice_axes
            )
        if self.options.backend == "torch":
            from .core.fo_solar_obs_torch import (
                solve_fo_solar_obs_eps_torch,
                solve_fo_solar_obs_plane_parallel_torch,
                solve_fo_solar_obs_rps_torch,
            )
            from .core.fo_thermal_torch import solve_fo_thermal_torch

            torch_context = detect_torch_context(
                tau_arr,
                omega_arr,
                asymm_arr,
                user_obsgeoms,
                d2s_scaling,
                height_grid,
            )
            torch_context = self._select_torch_context(torch_context)
            with self._torch_grad_context():
                tau_arr = value_to_torch(tau_arr, torch_context)
                omega_arr = value_to_torch(omega_arr, torch_context)
                asymm_arr = value_to_torch(asymm_arr, torch_context)
                user_obsgeoms = value_to_torch(user_obsgeoms, torch_context)
                d2s_scaling = value_to_torch(d2s_scaling, torch_context)
                height_grid = value_to_torch(height_grid, torch_context)
                user_angles = value_to_torch(user_angles, torch_context)
                thermal_bb_input = value_to_torch(thermal_bb_input, torch_context)
                fo_exact_scatter_t = value_to_torch(fo_exact_scatter, torch_context)
                fo_tau_arr = tau_arr
                if self.options.effective_fo_optical_deltam_scaling:
                    fo_tau_arr = tau_arr * (1.0 - omega_arr * d2s_scaling)
                if self.options.source_mode == "solar_obs" and self.options.do_plane_parallel:
                    return solve_fo_solar_obs_plane_parallel_torch(
                        tau_arr=fo_tau_arr,
                        omega_arr=omega_arr,
                        asymm_arr=asymm_arr,
                        user_obsgeoms=user_obsgeoms,
                        d2s_scaling=d2s_scaling,
                        albedo=albedo,
                        flux_factor=flux_factor,
                        n_moments=n_moments,
                        exact_scatter=fo_exact_scatter_t,
                    )
                if self.options.source_mode == "solar_obs" and fo_geometry_mode == "rps":
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
                        exact_scatter=fo_exact_scatter_t,
                    )
                if self.options.source_mode == "solar_obs" and fo_geometry_mode == "eps":
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
                        exact_scatter=fo_exact_scatter_t,
                    )
                if self.options.source_mode == "thermal":
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
                        do_plane_parallel=self.options.do_plane_parallel,
                        height_grid=height_grid,
                        earth_radius=earth_radius,
                        do_optical_deltam_scaling=self.options.effective_fo_optical_deltam_scaling,
                        do_source_deltam_scaling=self.options.effective_fo_thermal_source_deltam_scaling,
                        nfine=nfine,
                    )
            raise NotImplementedError(
                "backend='torch' forward_fo is currently implemented for "
                "source_mode='solar_obs', 'solar_lat', and 'thermal' only"
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
        if self.options.source_mode == "thermal":
            return solve_fo_thermal(
                prepared,
                do_plane_parallel=self.options.do_plane_parallel,
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
            do_plane_parallel=self.options.do_plane_parallel,
            geometry_mode=fo_geometry_mode,
            n_moments=n_moments,
            nfine=nfine,
            exact_scatter=None if fo_exact_scatter is None else to_numpy(fo_exact_scatter),
        )
        if prepared.lattice_counts is not None:
            return replace(
                result, lattice_counts=prepared.lattice_counts, lattice_axes=prepared.lattice_axes
            )
        return result
