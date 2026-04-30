"""High-level scene runner built on top of the direct RT-array solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .api import TwoStreamEss, TwoStreamEssOptions
from .optical.phase import (
    aerosol_interp_fraction,
    build_solar_fo_scatter_term,
    build_solar_phase_inputs_from_scattering_tau,
    build_two_stream_phase_inputs,
    build_two_stream_phase_inputs_from_scattering_tau,
)
from .optical.planck import thermal_source_from_temperature_profile
from .optical.scene import (
    AtmosphericProfile,
    atmospheric_profile_from_levels,
    build_scene_layer_optical_properties,
    build_scene_layer_optical_properties_from_gas_tau,
)
from .optical.scene_io import build_benchmark_scene_inputs, load_scene_yaml


@dataclass(frozen=True)
class SceneForwardInputs:
    """Public ``TwoStreamEss.forward`` inputs generated from a scene."""

    mode: str
    kwargs: dict[str, Any]
    wavelengths: np.ndarray | None = None
    reference_total: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)
    timing_modes: dict[str, str] = field(default_factory=dict)


@dataclass
class SceneRun:
    """Scene/profile forward model runner.

    Use ``load_scene(profile=..., config=...)`` for YAML scenes. Direct Python
    construction accepts mappings for ``profile``, ``spectral``, ``geometry``,
    ``surface``, ``opacity``, and ``rt``.
    """

    mode: str
    profile: Any | None = None
    spectral: Mapping[str, Any] | None = None
    geometry: Mapping[str, Any] | None = None
    surface: Mapping[str, Any] | None = None
    opacity: Mapping[str, Any] | None = None
    rt: Mapping[str, Any] | None = None
    reference: Mapping[str, Any] | None = None
    _bundle: dict[str, Any] | None = None
    _prepared: SceneForwardInputs | None = None

    def __post_init__(self) -> None:
        self.mode = _normalize_scene_mode(self.mode)

    @classmethod
    def from_bundle(cls, *, mode: str, bundle: Mapping[str, Any]) -> "SceneRun":
        """Build a scene from already-loaded scene inputs."""
        return cls(mode=mode, _bundle=dict(bundle))

    def to_forward_inputs(self) -> SceneForwardInputs:
        """Return cached public ``TwoStreamEss.forward`` keyword arguments."""
        if self._prepared is None:
            bundle = dict(self._bundle) if self._bundle is not None else self._bundle_from_objects()
            self._prepared = _prepare_forward_inputs(self.mode, bundle)
        return self._prepared

    def forward(
        self,
        *,
        backend: str = "numpy",
        include_fo: bool = True,
        options: TwoStreamEssOptions | None = None,
        **option_overrides: Any,
    ):
        """Run the scene through ``TwoStreamEss.forward``."""
        if options is not None:
            if options.mode != self.mode:
                raise ValueError(
                    f"scene mode {self.mode!r} does not match options mode {options.mode!r}"
                )
            if option_overrides:
                raise ValueError("pass either options or option overrides, not both")
        inputs = self.to_forward_inputs()
        nlyr = int(np.asarray(inputs.kwargs["tau"]).shape[-1])
        if options is None:
            options = TwoStreamEssOptions(
                nlyr=nlyr,
                mode=self.mode,
                backend=backend,
                **option_overrides,
            )
        return TwoStreamEss(options).forward(**inputs.kwargs, include_fo=include_fo)

    def _bundle_from_objects(self) -> dict[str, Any]:
        profile = _profile_from_object(self.profile)
        spectral = dict(self.spectral or {})
        geometry = dict(self.geometry or {})
        surface = dict(self.surface or {})
        opacity = dict(self.opacity or {})
        rt = dict(self.rt or {})
        wavelengths = _wavelengths_from_spectral(spectral)
        bundle: dict[str, Any] = {
            "wavelengths": wavelengths,
            "pressure_hpa": profile.pressure_hpa,
            "temperature_k": profile.temperature_k,
            "gas_vmr": profile.gas_density_per_km / profile.air_density_per_km[:, np.newaxis]
            if profile.gas_density_per_km.shape[-1]
            else np.zeros((profile.pressure_hpa.size, 0), dtype=float),
            "heights": profile.heights_km,
            "surface_altitude_m": np.array(0.0, dtype=float),
        }
        bundle.update({key: value for key, value in spectral.items() if key != "wavelengths_nm"})
        if "gas_absorption_tau" in opacity:
            bundle["gas_absorption_tau"] = opacity["gas_absorption_tau"]
        elif "gas_cross_sections" in opacity:
            bundle["gas_cross_sections"] = opacity["gas_cross_sections"]
        else:
            bundle["gas_absorption_tau"] = np.zeros(
                (wavelengths.shape[0], profile.air_columns.shape[0]), dtype=float
            )
        for key in (
            "aerosol_loadings",
            "aerosol_wavelengths_microns",
            "aerosol_bulk_iops",
            "aerosol_extinction_per_loading",
            "aerosol_scattering_per_loading",
            "aerosol_moments",
            "aerosol_interp_fraction",
            "co2_ppmv",
        ):
            if key in opacity:
                bundle[key] = opacity[key]
        if self.mode == "solar":
            bundle["user_obsgeom"] = np.asarray(geometry["angles"], dtype=float)
            bundle["albedo"] = _spectral_value(surface.get("albedo", 0.0), wavelengths.shape[0])
            bundle["flux_factor"] = _spectral_value(
                surface.get("fbeam", rt.get("fbeam", 1.0)), wavelengths.shape[0]
            )
        else:
            bundle["user_angle"] = np.array([float(geometry["view_angle"])], dtype=float)
            bundle["albedo"] = _spectral_value(surface.get("albedo", 0.0), wavelengths.shape[0])
            if "emissivity" in surface:
                bundle["emissivity"] = _spectral_value(surface["emissivity"], wavelengths.shape[0])
            bundle["level_temperature_k"] = surface.get(
                "level_temperature_k", profile.temperature_k
            )
            bundle["surface_temperature_k"] = np.array(
                [surface.get("surface_temperature_k", profile.temperature_k[-1])],
                dtype=float,
            )
        if "stream" in rt:
            bundle["stream_value"] = np.asarray(rt["stream"], dtype=float)
        for key, value in dict(self.reference or {}).items():
            bundle[key] = value
        return bundle


def load_scene(
    *,
    profile: str | Path,
    config: str | Path,
    spectral_limit: int | None = None,
    strict_runtime_inputs: bool = False,
) -> SceneRun:
    """Load a profile text file and scene YAML into a runnable scene."""
    scene_path = Path(config)
    mode = _normalize_scene_mode(load_scene_yaml(scene_path).get("mode", "solar"))
    bundle = build_benchmark_scene_inputs(
        profile_path=profile,
        scene_path=scene_path,
        kind="uv" if mode == "solar" else "tir",
        spectral_limit=spectral_limit,
        strict_runtime_inputs=strict_runtime_inputs,
    )
    return SceneRun.from_bundle(mode=mode, bundle=bundle)


def _prepare_forward_inputs(mode: str, bundle: dict[str, Any]) -> SceneForwardInputs:
    timings: dict[str, float] = {}
    timing_modes: dict[str, str] = {}
    start = time.perf_counter()
    bundle, timing_modes["layer optical properties"] = _prepare_layer_properties(mode, bundle)
    timings["layer optical properties"] = time.perf_counter() - start

    start = time.perf_counter()
    bundle, timing_modes["optical preprocessing"] = _prepare_phase_inputs(mode, bundle)
    timings["optical preprocessing"] = time.perf_counter() - start

    if mode == "thermal":
        start = time.perf_counter()
        bundle, timing_modes["thermal source"] = _prepare_thermal_source(bundle)
        timings["thermal source"] = time.perf_counter() - start

    kwargs = _forward_kwargs(mode, bundle)
    return SceneForwardInputs(
        mode=mode,
        kwargs=kwargs,
        wavelengths=np.asarray(bundle["wavelengths"], dtype=float),
        reference_total=(
            np.asarray(bundle["ref_total"], dtype=float) if "ref_total" in bundle else None
        ),
        timings=timings,
        timing_modes=timing_modes,
    )


def _prepare_layer_properties(mode: str, bundle: dict[str, Any]) -> tuple[dict[str, Any], str]:
    total_key = "tau" if mode == "solar" else "tau_arr"
    ssa_key = "omega" if mode == "solar" else "omega_arr"
    if total_key in bundle and ssa_key in bundle:
        prepared = dict(bundle)
        prepared["tau"] = prepared[total_key]
        prepared["ssa"] = prepared[ssa_key]
        return prepared, "direct input"

    profile = atmospheric_profile_from_levels(
        pressure_hpa=bundle["pressure_hpa"],
        temperature_k=bundle["temperature_k"],
        gas_vmr=bundle.get("gas_vmr"),
        heights_km=bundle.get("heights"),
        surface_altitude_m=_scalar(bundle.get("surface_altitude_m", 0.0)),
    )
    aerosol_kwargs = {}
    if "aerosol_loadings" in bundle:
        aerosol_kwargs = {
            "aerosol_loadings": bundle["aerosol_loadings"],
            "aerosol_wavelengths_microns": bundle.get("aerosol_wavelengths_microns"),
            "aerosol_bulk_iops": bundle.get("aerosol_bulk_iops"),
            "aerosol_extinction_per_loading": bundle.get("aerosol_extinction_per_loading"),
            "aerosol_scattering_per_loading": bundle.get("aerosol_scattering_per_loading"),
            "aerosol_select_wavelength_microns": _scalar(
                bundle.get("aerosol_select_wavelength_microns", 0.4)
            ),
        }
    scene_kwargs = {
        "wavelengths_nm": bundle.get("opacity_wavelengths", bundle["wavelengths"]),
        "profile": profile,
        "co2_ppmv": _scalar(bundle.get("co2_ppmv", 385.0)),
        **aerosol_kwargs,
    }
    if "gas_absorption_tau" in bundle:
        scene = build_scene_layer_optical_properties_from_gas_tau(
            gas_absorption_tau=bundle["gas_absorption_tau"],
            **scene_kwargs,
        )
    else:
        scene = build_scene_layer_optical_properties(
            gas_cross_sections=bundle["gas_cross_sections"],
            **scene_kwargs,
        )
    prepared = {
        **bundle,
        "heights": profile.heights_km,
        "tau": scene.layer.tau,
        "ssa": scene.layer.ssa,
        "rayleigh_scattering_tau": scene.rayleigh_scattering_tau,
        "aerosol_scattering_tau": scene.aerosol_scattering_tau,
        "rayleigh_fraction": scene.layer.rayleigh_fraction,
        "aerosol_fraction": scene.layer.aerosol_fraction,
        "depol": scene.depol,
    }
    prepared[total_key] = scene.layer.tau
    prepared[ssa_key] = scene.layer.ssa
    return prepared, "python-generated from scene/profile inputs"


def _prepare_phase_inputs(mode: str, bundle: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if "g" in bundle and "delta_m_truncation_factor" in bundle:
        if mode == "solar" and "fo_scatter_term" not in bundle:
            raise ValueError("solar direct phase input requires fo_scatter_term")
        return bundle, "direct input"

    has_component_phase = all(
        key in bundle
        for key in ("depol", "rayleigh_scattering_tau", "aerosol_scattering_tau", "aerosol_moments")
    )
    has_fraction_phase = all(
        key in bundle
        for key in ("depol", "rayleigh_fraction", "aerosol_fraction", "aerosol_moments")
    )
    if not (has_component_phase or has_fraction_phase):
        raise ValueError("scene requires physical phase inputs")
    fac, mode_text = _aerosol_interp(mode, bundle)
    if has_component_phase and mode == "solar":
        optics = build_solar_phase_inputs_from_scattering_tau(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_scattering_tau=bundle["rayleigh_scattering_tau"],
            aerosol_scattering_tau=bundle["aerosol_scattering_tau"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            angles=bundle["user_obsgeom"],
            validate_inputs=False,
        )
        fo_scatter = optics.fo_scatter_term
    elif has_component_phase:
        optics = build_two_stream_phase_inputs_from_scattering_tau(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_scattering_tau=bundle["rayleigh_scattering_tau"],
            aerosol_scattering_tau=bundle["aerosol_scattering_tau"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            validate_inputs=False,
        )
        fo_scatter = None
    else:
        optics = build_two_stream_phase_inputs(
            ssa=bundle["ssa"],
            depol=bundle["depol"],
            rayleigh_fraction=bundle["rayleigh_fraction"],
            aerosol_fraction=bundle["aerosol_fraction"],
            aerosol_moments=bundle["aerosol_moments"],
            aerosol_interp_fraction=fac,
            validate_inputs=False,
        )
        fo_scatter = None
        if mode == "solar":
            fo_scatter = build_solar_fo_scatter_term(
                ssa=bundle["ssa"],
                depol=bundle["depol"],
                rayleigh_fraction=bundle["rayleigh_fraction"],
                aerosol_fraction=bundle["aerosol_fraction"],
                aerosol_moments=bundle["aerosol_moments"],
                aerosol_interp_fraction=fac,
                angles=bundle["user_obsgeom"],
                delta_m_truncation_factor=optics.delta_m_truncation_factor,
                validate_inputs=False,
            )
    prepared = dict(bundle)
    prepared["aerosol_interp_fraction"] = fac
    prepared["g"] = optics.g
    prepared["delta_m_truncation_factor"] = optics.delta_m_truncation_factor
    if fo_scatter is not None:
        prepared["fo_scatter_term"] = fo_scatter
    return prepared, mode_text


def _prepare_thermal_source(bundle: dict[str, Any]) -> tuple[dict[str, Any], str]:
    coordinate_name = next(
        (
            key
            for key in ("wavenumber_band_cm_inv", "wavenumber_cm_inv", "wavelength_microns")
            if key in bundle
        ),
        None,
    )
    if coordinate_name is None:
        raise ValueError("thermal scene requires a spectral coordinate for Planck source")
    source = thermal_source_from_temperature_profile(
        bundle["level_temperature_k"],
        bundle["surface_temperature_k"],
        **{coordinate_name: bundle[coordinate_name]},
    )
    prepared = dict(bundle)
    prepared["planck"] = np.asarray(source.planck, dtype=float)
    prepared["surface_planck"] = np.asarray(source.surface_planck, dtype=float)
    if "emissivity" not in prepared:
        prepared["emissivity"] = 1.0 - np.asarray(prepared["albedo"], dtype=float)
    return prepared, f"temperature ({coordinate_name})"


def _forward_kwargs(mode: str, bundle: dict[str, Any]) -> dict[str, Any]:
    stream = bundle.get("stream_value")
    if stream is None and mode == "thermal":
        stream = 0.5
    common = {
        "tau": bundle["tau"],
        "ssa": bundle["ssa"],
        "g": bundle["g"],
        "z": bundle["heights"],
        "albedo": bundle["albedo"],
        "delta_m_truncation_factor": bundle["delta_m_truncation_factor"],
    }
    if stream is not None:
        common["stream"] = _scalar(stream)
    if mode == "solar":
        return {
            **common,
            "angles": bundle["user_obsgeom"],
            "fbeam": bundle["flux_factor"],
            "fo_scatter_term": bundle["fo_scatter_term"],
        }
    return {
        **common,
        "angles": _scalar(bundle["user_angle"]),
        "planck": bundle["planck"],
        "surface_planck": bundle["surface_planck"],
        "emissivity": bundle["emissivity"],
    }


def _aerosol_interp(mode: str, bundle: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    if "aerosol_interp_fraction" in bundle:
        return np.asarray(bundle["aerosol_interp_fraction"], dtype=float), "python-generated"
    wavelengths = np.asarray(bundle["wavelengths"], dtype=float)
    if _looks_like_row_index(wavelengths):
        if "wavelength_microns" in bundle:
            wavelengths = np.asarray(bundle["wavelength_microns"], dtype=float) * 1000.0
        elif "wavenumber_cm_inv" in bundle:
            wavelengths = 1.0e7 / np.asarray(bundle["wavenumber_cm_inv"], dtype=float)
    if _looks_like_row_index(wavelengths):
        raise ValueError("scene requires physical wavelengths or aerosol_interp_fraction")
    return aerosol_interp_fraction(wavelengths, reverse=True), (
        "python-generated (aerosol interpolation from wavelengths)"
        if mode == "solar"
        else "python-generated (aerosol interpolation from spectral coordinate)"
    )


def _profile_from_object(profile: Any) -> AtmosphericProfile:
    if isinstance(profile, AtmosphericProfile):
        return profile
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be an AtmosphericProfile or mapping")
    return atmospheric_profile_from_levels(
        pressure_hpa=profile["pressure_hpa"],
        temperature_k=profile["temperature_k"],
        gas_vmr=profile.get("gas_vmr"),
        heights_km=profile.get("heights_km", profile.get("heights")),
        surface_altitude_m=_scalar(profile.get("surface_altitude_m", 0.0)),
    )


def _wavelengths_from_spectral(spectral: Mapping[str, Any]) -> np.ndarray:
    if "wavelengths_nm" in spectral:
        return np.asarray(spectral["wavelengths_nm"], dtype=float)
    if "wavelengths" in spectral:
        return np.asarray(spectral["wavelengths"], dtype=float)
    if "wavelength_microns" in spectral:
        return 1000.0 * np.asarray(spectral["wavelength_microns"], dtype=float)
    if "wavenumber_cm_inv" in spectral:
        wavenumber = np.asarray(spectral["wavenumber_cm_inv"], dtype=float)
        return 1.0e7 / wavenumber
    raise ValueError("spectral must define wavelengths_nm, wavelengths, or wavenumber_cm_inv")


def _spectral_value(value: Any, nspec: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(nspec, float(arr), dtype=float)
    if arr.shape != (nspec,):
        raise ValueError(f"spectral value must be scalar or shape ({nspec},)")
    return arr


def _normalize_scene_mode(mode: Any) -> str:
    normalized = {"uv": "solar", "solar": "solar", "tir": "thermal", "thermal": "thermal"}.get(
        str(mode).lower()
    )
    if normalized is None:
        raise ValueError("scene mode must be solar/uv or thermal/tir")
    return normalized


def _looks_like_row_index(values: np.ndarray) -> bool:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        return False
    return np.allclose(grid, np.arange(1, grid.size + 1, dtype=float), rtol=0.0, atol=1.0e-12)


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])
