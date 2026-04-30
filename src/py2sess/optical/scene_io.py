"""Profile and YAML scene loaders for benchmark scene inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .geocape import (
    gas_cross_sections_from_tables,
    geocape_select_wavelength_microns,
    load_geocape_aerosol_loadings,
    load_geocape_aerosol_tables,
    load_geocape_solar_flux,
    load_geocape_surface_albedo,
)
from .hitran import (
    hitran_cross_sections,
    load_hitran_partition_functions,
    read_hitran_lines,
)
from .opacity_table import gas_cross_sections_from_table3d
from .phase import aerosol_interp_fraction as phase_aerosol_interp_fraction
from .scene import (
    AtmosphericProfile,
    atmospheric_profile_from_levels,
    gas_absorption_tau_from_cross_sections,
)


@dataclass(frozen=True)
class ProfileTextData:
    """Parsed atmospheric profile text data."""

    pressure_hpa: np.ndarray
    temperature_k: np.ndarray
    gas_names: tuple[str, ...]
    gas_vmr: np.ndarray
    heights_km: np.ndarray | None
    surface_temperature_k: float | None
    surface_altitude_m: float
    metadata: dict[str, str]
    columns: dict[str, np.ndarray]


def load_profile_text(path: str | Path, *, gas_species: tuple[str, ...] = ()) -> ProfileTextData:
    """Load a GEOCAPE-style or simple column profile text file."""
    profile_path = Path(path)
    lines = profile_path.read_text(encoding="utf-8").splitlines()
    metadata = _parse_metadata(lines)
    table_header_index = _find_table_header(lines)
    if table_header_index is None:
        return _load_simple_profile(profile_path, gas_species=gas_species)
    return _load_geocape_profile(
        lines,
        table_header_index=table_header_index,
        metadata=metadata,
        gas_species=gas_species,
    )


def load_scene_yaml(path: str | Path) -> dict[str, Any]:
    """Load a benchmark scene YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to read scene YAML files") from exc

    scene_path = Path(path)
    with scene_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("scene YAML must contain a mapping")
    return data


def build_benchmark_scene_inputs(
    *,
    profile_path: str | Path,
    scene_path: str | Path,
    kind: str,
    spectral_limit: int | None = None,
    strict_runtime_inputs: bool = False,
) -> dict[str, np.ndarray]:
    """Build benchmark runtime inputs from profile text and scene YAML."""
    scene_file = Path(scene_path)
    scene = load_scene_yaml(scene_file)
    if strict_runtime_inputs:
        _validate_strict_runtime_scene(scene)
    mode = _scene_mode(scene, kind)
    _reject_opacity_provider(scene)
    spectral = _spectral_arrays(scene, scene_file.parent)
    aerosol_moment_wavelengths = spectral["wavelengths"]
    full_row_count = spectral["wavelengths"].shape[0]
    if spectral_limit is not None:
        if spectral_limit <= 0:
            raise ValueError("spectral_limit must be positive")
        stop = min(int(spectral_limit), full_row_count)
        spectral = _slice_matching_rows(spectral, full_row_count, stop)
    gases = _scene_gases(scene)
    gas_defaults = _scene_gas_vmr_defaults(scene, scene_file.parent)
    profile_gases = tuple(gas for gas in gases if gas.upper() not in gas_defaults)
    profile = load_profile_text(profile_path, gas_species=profile_gases)
    profile = _profile_with_scene_heights(profile, scene=scene, base_dir=scene_file.parent)
    if not gases:
        profile = replace(
            profile,
            gas_names=(),
            gas_vmr=np.zeros((profile.pressure_hpa.shape[0], 0), dtype=float),
        )
    else:
        profile = _profile_with_scene_gases(profile, gases=gases, gas_defaults=gas_defaults)
    surface_altitude = _surface_altitude(scene, profile)
    opacity_profile = atmospheric_profile_from_levels(
        pressure_hpa=profile.pressure_hpa,
        temperature_k=profile.temperature_k,
        gas_vmr=profile.gas_vmr,
        heights_km=profile.heights_km,
        surface_altitude_m=surface_altitude,
    )
    bundle: dict[str, np.ndarray] = {
        "wavelengths": spectral["wavelengths"],
        "pressure_hpa": profile.pressure_hpa,
        "temperature_k": profile.temperature_k,
        "gas_vmr": profile.gas_vmr,
        "heights": opacity_profile.heights_km,
        "surface_altitude_m": np.array(surface_altitude, dtype=float),
    }
    bundle["gas_absorption_tau"] = _gas_absorption_tau(
        scene,
        scene_file.parent,
        spectral,
        opacity_profile,
    )
    for key in (
        "wavenumber_cm_inv",
        "wavenumber_band_cm_inv",
        "wavelength_microns",
        "opacity_wavelengths",
        "aerosol_interp_fraction",
    ):
        if key in spectral:
            bundle[key] = spectral[key]

    _add_surface_arrays(
        bundle,
        scene,
        scene_file.parent,
        spectral=spectral,
        mode=mode,
        nspec=spectral["wavelengths"].shape[0],
        source_nspec=full_row_count,
    )
    _add_aerosol_arrays(
        bundle,
        scene,
        scene_file.parent,
        profile=profile,
        n_layers=profile.pressure_hpa.size - 1,
        moment_wavelengths_nm=aerosol_moment_wavelengths,
    )
    _add_reference_arrays(bundle, scene, scene_file.parent, spectral=spectral)

    rt_config = _section(scene, "rt")
    if "stream" in rt_config:
        bundle["stream_value"] = np.asarray(rt_config["stream"], dtype=float)

    if mode == "solar":
        if "user_obsgeom" not in bundle or "angles" in _section(scene, "geometry"):
            bundle["user_obsgeom"] = _solar_geometry(scene)
        if "flux_factor" not in bundle or "flux_factor" in _section(scene, "solar"):
            bundle["flux_factor"] = _solar_flux_factor(
                _section(scene, "solar").get("flux_factor", 1.0),
                scene_file.parent,
                spectral=spectral,
                nspec=spectral["wavelengths"].shape[0],
                source_nspec=full_row_count,
            )
    else:
        if "user_angle" not in bundle or "view_angle" in _section(scene, "geometry"):
            bundle["user_angle"] = np.array([_thermal_view_angle(scene)], dtype=float)
        surface_temperature = _surface_temperature(scene, profile)
        if "level_temperature_k" not in bundle:
            bundle["level_temperature_k"] = profile.temperature_k
        if "surface_temperature_k" not in bundle or "temperature_k" in _section(scene, "surface"):
            bundle["surface_temperature_k"] = np.array([surface_temperature], dtype=float)

    if spectral_limit is not None:
        bundle = _slice_matching_rows(bundle, full_row_count, spectral["wavelengths"].shape[0])
    return bundle


def _parse_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("End_of_Header"):
            break
        if "=" in stripped:
            key, value = stripped.split("=", maxsplit=1)
            metadata[key.strip()] = value.strip()
    return metadata


def _find_table_header(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        values = line.split()
        if len(values) >= 3 and values[0].lower() == "level":
            return index
    return None


def _load_geocape_profile(
    lines: list[str],
    *,
    table_header_index: int,
    metadata: dict[str, str],
    gas_species: tuple[str, ...],
) -> ProfileTextData:
    columns = tuple(lines[table_header_index].split())
    rows: list[list[float]] = []
    for line in lines[table_header_index + 1 :]:
        values = line.split()
        if values and _is_int_token(values[0]):
            rows.append([float(value) for value in values])
    if not rows:
        raise ValueError("profile contains no numeric level rows")

    data = np.asarray(rows, dtype=float)
    pressure = data[:, columns.index("Pressure")]
    temperature = data[:, columns.index("TATM")]
    gas_names = columns[3:]
    gas = _select_gas_vmr(data[:, 3:], gas_names=gas_names, gas_species=gas_species)
    pressure, temperature, gas, heights, _ = _top_to_bottom(
        pressure=pressure,
        temperature=temperature,
        gas_vmr=gas,
        heights=None,
        columns={},
    )
    return ProfileTextData(
        pressure_hpa=pressure,
        temperature_k=temperature,
        gas_names=gas_species or gas_names,
        gas_vmr=gas,
        heights_km=heights,
        surface_temperature_k=_metadata_float(metadata, "surfaceTemperature(K)"),
        surface_altitude_m=_metadata_float(metadata, "ZSUR(m)", default=0.0),
        metadata=metadata,
        columns={},
    )


def _load_simple_profile(profile_path: Path, *, gas_species: tuple[str, ...]) -> ProfileTextData:
    first_data_line = _first_noncomment_line(profile_path)
    delimiter = "," if "," in first_data_line else None
    data = np.atleast_1d(np.genfromtxt(profile_path, names=True, comments="#", delimiter=delimiter))
    names = tuple(data.dtype.names or ())
    if not names:
        raise ValueError("simple profile must have a header row")

    pressure_name = _first_existing(names, ("pressure_hpa", "pressure", "p_hpa"))
    temperature_name = _first_existing(names, ("temperature_k", "temperature", "tatm"))
    height_name = _first_existing(names, ("height_km", "height", "z_km", "z"), required=False)
    if pressure_name is None or temperature_name is None:  # pragma: no cover
        raise ValueError("simple profile requires pressure_hpa and temperature_k columns")

    pressure = np.asarray(data[pressure_name], dtype=float)
    temperature = np.asarray(data[temperature_name], dtype=float)
    heights = None if height_name is None else np.asarray(data[height_name], dtype=float)
    gas_names = tuple(
        name for name in names if name not in {pressure_name, temperature_name, height_name}
    )
    gas_columns = _gas_columns(data, gas_names)
    raw_columns = {name: np.asarray(data[name], dtype=float) for name in names}
    gas = _select_gas_vmr(gas_columns, gas_names=gas_names, gas_species=gas_species)
    pressure, temperature, gas, heights, columns = _top_to_bottom(
        pressure=pressure,
        temperature=temperature,
        gas_vmr=gas,
        heights=heights,
        columns=raw_columns,
    )
    return ProfileTextData(
        pressure_hpa=pressure,
        temperature_k=temperature,
        gas_names=gas_species or gas_names,
        gas_vmr=gas,
        heights_km=heights,
        surface_temperature_k=None,
        surface_altitude_m=0.0,
        metadata={},
        columns=columns,
    )


def _scene_mode(scene: dict[str, Any], kind: str) -> str:
    mode = {"uv": "solar", "solar": "solar", "tir": "thermal", "thermal": "thermal"}.get(
        str(scene.get("mode", kind)).lower()
    )
    if mode is None:
        raise ValueError("scene mode must be solar/uv or thermal/tir")
    expected = "solar" if kind == "uv" else "thermal"
    if mode != expected:
        raise ValueError(f"{kind} benchmark requires scene mode {expected}")
    return mode


def _scene_gases(scene: dict[str, Any]) -> tuple[str, ...]:
    gases = scene.get("gases", ())
    if isinstance(gases, str):
        return (gases,)
    return tuple(str(name) for name in gases)


def _validate_strict_runtime_scene(scene: dict[str, Any]) -> None:
    opacity = _section(scene, "opacity")
    _reject_opacity_provider(scene)
    gas_cfg = opacity.get("gas_cross_sections")
    if isinstance(gas_cfg, dict) and "hitran" in gas_cfg:
        raise ValueError(
            "strict scene mode must not run direct HITRAN; create a NetCDF "
            "gas table and use opacity.gas_cross_sections.table3d"
        )
    if _scene_gases(scene) and not (isinstance(gas_cfg, dict) and "table3d" in gas_cfg):
        raise ValueError("strict scene mode requires opacity.gas_cross_sections.table3d")


def _reject_opacity_provider(scene: dict[str, Any]) -> None:
    if _section(scene, "opacity").get("provider") is not None:
        raise ValueError(
            "opacity.provider is no longer supported; use profile, "
            "opacity.gas_cross_sections.table3d, and aerosol table specs"
        )


def _scene_gas_vmr_defaults(scene: dict[str, Any], base_dir: Path) -> dict[str, np.ndarray]:
    defaults = scene.get("gas_vmr", {})
    if defaults is None:
        return {}
    if not isinstance(defaults, dict):
        raise ValueError("gas_vmr must be a mapping from gas name to scalar or level array")
    return {
        str(name).upper(): _load_array(value, base_dir, f"gas_vmr.{name}")
        for name, value in defaults.items()
    }


def _profile_with_scene_gases(
    profile: ProfileTextData,
    *,
    gases: tuple[str, ...],
    gas_defaults: dict[str, np.ndarray],
) -> ProfileTextData:
    profile_columns = {name.upper(): index for index, name in enumerate(profile.gas_names)}
    columns = []
    for gas in gases:
        key = gas.upper()
        if key in profile_columns:
            columns.append(profile.gas_vmr[:, profile_columns[key]])
            continue
        if key not in gas_defaults:
            raise ValueError(f"profile is missing requested gas species {gas}")
        columns.append(_gas_default_column(gas_defaults[key], profile.pressure_hpa.size, gas))
    return replace(profile, gas_names=gases, gas_vmr=np.column_stack(columns))


def _profile_with_scene_heights(
    profile: ProfileTextData,
    *,
    scene: dict[str, Any],
    base_dir: Path,
) -> ProfileTextData:
    atmosphere = _section(scene, "atmosphere")
    if "heights_km" not in atmosphere:
        return profile
    heights = _load_array(atmosphere["heights_km"], base_dir, "atmosphere.heights_km")
    return replace(profile, heights_km=np.asarray(heights, dtype=float))


def _gas_default_column(value: np.ndarray, nlevel: int, gas: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(nlevel, float(arr), dtype=float)
    if arr.shape == (nlevel,):
        return arr
    raise ValueError(f"gas_vmr.{gas} must be scalar or have shape ({nlevel},)")


def _spectral_arrays(
    scene: dict[str, Any],
    base_dir: Path,
) -> dict[str, np.ndarray]:
    spectral = _section(scene, "spectral")
    if "wavelengths_nm" in spectral:
        wavelengths = _as_1d_float(
            _spectral_1d(spectral["wavelengths_nm"], base_dir, "spectral.wavelengths_nm"),
            "spectral.wavelengths_nm",
        )
        return {"wavelengths": wavelengths}
    if "wavelength_microns" in spectral:
        microns = _as_1d_float(
            _spectral_1d(spectral["wavelength_microns"], base_dir, "spectral.wavelength_microns"),
            "spectral.wavelength_microns",
        )
        return {"wavelengths": microns * 1000.0, "wavelength_microns": microns}
    if "wavenumber_cm_inv" in spectral:
        wavenumber = _as_1d_float(
            _spectral_1d(spectral["wavenumber_cm_inv"], base_dir, "spectral.wavenumber_cm_inv"),
            "spectral.wavenumber_cm_inv",
        )
        wavelength_from_wavenumber = 1.0e7 / wavenumber
        arrays = {
            "wavelengths": wavelength_from_wavenumber,
            "wavenumber_cm_inv": wavenumber,
        }
        if str(spectral.get("wavelength_order", "")).lower() == "reverse_from_wavenumber":
            arrays["wavelengths"] = arrays["wavelengths"][::-1]
            arrays["opacity_wavelengths"] = wavelength_from_wavenumber
            arrays["aerosol_interp_fraction"] = phase_aerosol_interp_fraction(
                arrays["wavelengths"], reverse=True
            )
        if "wavenumber_band_width_cm_inv" in spectral:
            half_width = 0.5 * float(spectral["wavenumber_band_width_cm_inv"])
            arrays["wavenumber_band_cm_inv"] = np.column_stack(
                (wavenumber - half_width, wavenumber + half_width)
            )
        return arrays
    if "wavenumber_band_cm_inv" in spectral:
        bands = _load_array(
            spectral["wavenumber_band_cm_inv"], base_dir, "spectral.wavenumber_band_cm_inv"
        )
        if bands.ndim != 2 or bands.shape[1] != 2:
            raise ValueError("spectral.wavenumber_band_cm_inv must have shape (n, 2)")
        center = 0.5 * (bands[:, 0] + bands[:, 1])
        return {
            "wavelengths": 1.0e7 / center,
            "wavenumber_band_cm_inv": bands,
            "wavenumber_cm_inv": center,
        }
    raise ValueError("scene spectral section must define wavelengths or wavenumbers")


def _gas_absorption_tau(
    scene: dict[str, Any],
    base_dir: Path,
    spectral: dict[str, np.ndarray],
    profile: AtmosphericProfile,
) -> np.ndarray:
    opacity = _section(scene, "opacity")
    gas_cfg = opacity.get("gas_cross_sections")
    gases = _scene_gases(scene)
    if gas_cfg is None:
        if gases:
            raise ValueError("scene gases require opacity.gas_cross_sections")
        return np.zeros(
            (spectral["wavelengths"].shape[0], profile.heights_km.size - 1), dtype=float
        )
    if isinstance(gas_cfg, dict) and "hitran" in gas_cfg:
        return _hitran_gas_absorption_tau(gas_cfg["hitran"], base_dir, spectral, profile, gases)
    if isinstance(gas_cfg, dict) and "table3d" in gas_cfg:
        xsec = _table3d_gas_cross_sections(gas_cfg["table3d"], base_dir, spectral, profile, gases)
        return gas_absorption_tau_from_cross_sections(
            heights_km=profile.heights_km,
            gas_density_per_km=profile.gas_density_per_km,
            cross_sections=xsec,
        )
    if isinstance(gas_cfg, dict) and "tables" in gas_cfg:
        xsec = gas_cross_sections_from_tables(
            wavelengths_nm=spectral["wavelengths"],
            gas_names=gases,
            tables=_resolve_table_specs(gas_cfg["tables"], base_dir),
        )
    else:
        xsec = _load_array(gas_cfg, base_dir, "gas_cross_sections")
    if xsec.ndim == 1 and len(gases) == 1:
        xsec = xsec[:, np.newaxis]
    if xsec.shape[0] != spectral["wavelengths"].shape[0]:
        raise ValueError("gas_cross_sections first axis must match spectral grid")
    if gases and xsec.shape[-1] != len(gases):
        raise ValueError("gas_cross_sections gas axis must match scene gases")
    return gas_absorption_tau_from_cross_sections(
        heights_km=profile.heights_km,
        gas_density_per_km=profile.gas_density_per_km,
        cross_sections=xsec,
    )


def _table3d_gas_cross_sections(
    spec: Any,
    base_dir: Path,
    spectral: dict[str, np.ndarray],
    profile: AtmosphericProfile,
    gases: tuple[str, ...],
) -> np.ndarray:
    if not gases:
        return np.zeros((spectral["wavelengths"].shape[0], profile.heights_km.size, 0), dtype=float)
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("gas_cross_sections.table3d requires path")
    return gas_cross_sections_from_table3d(
        path=_resolve_path(spec["path"], base_dir),
        gas_names=gases,
        pressure_hpa=profile.pressure_hpa,
        temperature_k=profile.temperature_k,
        spectral=spectral,
    )


def _slice_matching_rows(
    arrays: dict[str, np.ndarray],
    row_count: int,
    stop: int,
) -> dict[str, np.ndarray]:
    return {
        key: value[:stop] if np.asarray(value).shape[:1] == (row_count,) else value
        for key, value in arrays.items()
    }


def _hitran_gas_absorption_tau(
    spec: Any,
    base_dir: Path,
    spectral: dict[str, np.ndarray],
    profile: AtmosphericProfile,
    gases: tuple[str, ...],
) -> np.ndarray:
    if not gases:
        return np.zeros(
            (spectral["wavelengths"].shape[0], profile.heights_km.size - 1), dtype=float
        )
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("gas_cross_sections.hitran requires path")
    if "wavenumber_cm_inv" in spectral:
        grid = spectral["wavenumber_cm_inv"]
        is_wavenumber = True
    else:
        grid = spectral["wavelengths"]
        is_wavenumber = False
    hitran_dir = _resolve_path(spec["path"], base_dir)
    partition = load_hitran_partition_functions(hitran_dir)
    gas_tau = np.zeros((grid.shape[0], profile.heights_km.size - 1), dtype=float)
    for gas_index, gas in enumerate(gases):
        lines = read_hitran_lines(hitran_dir, gas, grid if is_wavenumber else 1.0e7 / grid)
        xsec = hitran_cross_sections(
            hitran_dir=hitran_dir,
            molecule=gas,
            spectral_grid=grid,
            pressure_atm=profile.pressure_hpa / 1013.25,
            temperature_k=profile.temperature_k,
            is_wavenumber=is_wavenumber,
            fwhm=float(spec.get("fwhm", 0.0)),
            partition_functions=partition,
            lines=lines,
        )
        layer_tau = gas_absorption_tau_from_cross_sections(
            heights_km=profile.heights_km,
            gas_density_per_km=profile.gas_density_per_km[:, gas_index : gas_index + 1],
            cross_sections=xsec[:, :, np.newaxis],
        )
        gas_tau += layer_tau
    return gas_tau


def _add_surface_arrays(
    bundle: dict[str, np.ndarray],
    scene: dict[str, Any],
    base_dir: Path,
    *,
    spectral: dict[str, np.ndarray],
    mode: str,
    nspec: int,
    source_nspec: int,
) -> None:
    surface = _section(scene, "surface")
    if "albedo" in surface or "albedo" not in bundle:
        bundle["albedo"] = _surface_albedo(
            surface.get("albedo", 0.0),
            base_dir,
            spectral=spectral,
            nspec=nspec,
            source_nspec=source_nspec,
        )
    if mode == "thermal" and "emissivity" in surface:
        bundle["emissivity"] = _broadcast_spectral(
            surface["emissivity"],
            nspec,
            "emissivity",
            base_dir,
            source_nspec=source_nspec,
        )


def _surface_albedo(
    value: Any,
    base_dir: Path,
    *,
    spectral: dict[str, np.ndarray],
    nspec: int,
    source_nspec: int,
) -> np.ndarray:
    if isinstance(value, dict) and "geocape_emissivity" in value:
        if "wavenumber_cm_inv" not in spectral:
            raise ValueError("surface.albedo.geocape_emissivity requires wavenumber_cm_inv")
        spec = value["geocape_emissivity"]
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError("surface.albedo.geocape_emissivity requires path")
        return load_geocape_surface_albedo(
            _resolve_path(spec["path"], base_dir),
            spectral["wavenumber_cm_inv"],
        )[:nspec]
    return _broadcast_spectral(value, nspec, "albedo", base_dir, source_nspec=source_nspec)


def _solar_flux_factor(
    value: Any,
    base_dir: Path,
    *,
    spectral: dict[str, np.ndarray],
    nspec: int,
    source_nspec: int,
) -> np.ndarray:
    if isinstance(value, dict) and "geocape_solar_spectrum" in value:
        if "wavenumber_cm_inv" not in spectral:
            raise ValueError("solar.flux_factor.geocape_solar_spectrum requires wavenumber_cm_inv")
        spec = value["geocape_solar_spectrum"]
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError("solar.flux_factor.geocape_solar_spectrum requires path")
        return load_geocape_solar_flux(
            _resolve_path(spec["path"], base_dir),
            spectral["wavenumber_cm_inv"],
            scale=float(spec.get("scale", 1.0e4)),
        )[:nspec]
    return _broadcast_spectral(value, nspec, "flux_factor", base_dir, source_nspec=source_nspec)


def _add_aerosol_arrays(
    bundle: dict[str, np.ndarray],
    scene: dict[str, Any],
    base_dir: Path,
    *,
    profile: ProfileTextData,
    n_layers: int,
    moment_wavelengths_nm: np.ndarray,
) -> None:
    opacity = _section(scene, "opacity")
    aerosol = opacity.get("aerosol", {})
    if aerosol is None:
        aerosol = {}
    if not isinstance(aerosol, dict):
        raise ValueError("aerosol section must be a mapping")
    moments = aerosol.get("moments")
    loadings = aerosol.get("loadings")
    ssprops = aerosol.get("ssprops")
    properties = aerosol.get("properties")
    if properties is not None:
        table = _load_aerosol_properties(
            properties,
            base_dir,
            wavelengths_nm=np.asarray(moment_wavelengths_nm, dtype=float),
        )
        bundle["aerosol_extinction_per_loading"] = table["bulk_extinction"]
        bundle["aerosol_scattering_per_loading"] = table["bulk_scattering"]
        bundle["aerosol_moments"] = table["phase_moments"]
        if loadings is None:
            loadings = aerosol.get("loading_columns")
        if loadings is None:
            raise ValueError("aerosol.properties requires aerosol.loading_columns")
    if ssprops is not None:
        tables = _load_geocape_ssprops(
            ssprops,
            base_dir,
            wavelengths_nm=np.asarray(moment_wavelengths_nm, dtype=float),
        )
        bundle["aerosol_wavelengths_microns"] = tables.wavelengths_microns
        bundle["aerosol_bulk_iops"] = tables.bulk_iops
        if moments is None:
            bundle["aerosol_moments"] = tables.moments

    if (
        loadings is not None
        and moments is None
        and ssprops is None
        and "aerosol_moments" not in bundle
    ):
        raise ValueError("aerosol loadings require aerosol moments")
    if moments is None and "aerosol_moments" not in bundle:
        bundle["aerosol_moments"] = np.zeros((2, 3, 0), dtype=float)
    elif moments is not None:
        bundle["aerosol_moments"] = _load_array(moments, base_dir, "aerosol_moments")
    if loadings is None:
        return
    if properties is not None and isinstance(loadings, dict):
        bundle["aerosol_loadings"] = _aerosol_loadings_from_profile(
            profile,
            loadings,
            n_layers=n_layers,
            aerosol_names=table["aerosol_names"],
        )
    elif isinstance(loadings, dict) and "columns" in loadings:
        bundle["aerosol_loadings"] = _aerosol_loadings_from_profile(
            profile,
            loadings["columns"],
            n_layers=n_layers,
        )
    if isinstance(loadings, dict) and str(loadings.get("kind", "")).lower() == "geocape_files":
        select_index = int(loadings.get("select_index", aerosol.get("select_index", 2)))
        files = [_resolve_path(path, base_dir) for path in loadings.get("paths", ())]
        if not files:
            raise ValueError("aerosol loadings geocape_files requires paths")
        bundle["aerosol_loadings"] = load_geocape_aerosol_loadings(
            files,
            n_layers=n_layers,
            select_index=select_index,
            active_layers=int(loadings.get("active_layers", 50)),
        )
        bundle.setdefault(
            "aerosol_select_wavelength_microns",
            np.asarray(geocape_select_wavelength_microns(select_index), dtype=float),
        )
    elif "aerosol_loadings" not in bundle:
        bundle["aerosol_loadings"] = _load_array(loadings, base_dir, "aerosol_loadings")
    uses_unit_loading_table = "aerosol_extinction_per_loading" in bundle
    if not uses_unit_loading_table and "aerosol_wavelengths_microns" not in bundle:
        bundle["aerosol_wavelengths_microns"] = _load_array(
            aerosol["wavelengths_microns"],
            base_dir,
            "aerosol_wavelengths_microns",
        )
    if not uses_unit_loading_table and "aerosol_bulk_iops" not in bundle:
        bundle["aerosol_bulk_iops"] = _load_array(
            aerosol["bulk_iops"], base_dir, "aerosol_bulk_iops"
        )
    if "select_wavelength_microns" in aerosol:
        bundle["aerosol_select_wavelength_microns"] = np.asarray(
            aerosol["select_wavelength_microns"],
            dtype=float,
        )


def _load_aerosol_properties(
    spec: Any,
    base_dir: Path,
    *,
    wavelengths_nm: np.ndarray,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("aerosol.properties requires path")
    path = _resolve_path(spec["path"], base_dir)
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required to read aerosol NetCDF properties") from exc

    with netcdf_file(path, "r", mmap=False) as data:
        names = _netcdf_string_array(data.variables.get("aerosol_name"))
        if "wavelength_nm" in data.variables:
            table_wavelengths = np.asarray(data.variables["wavelength_nm"].data, dtype=float)
        elif "wavelength_microns" in data.variables:
            table_wavelengths = 1000.0 * np.asarray(
                data.variables["wavelength_microns"].data, dtype=float
            )
        else:
            raise ValueError("aerosol properties NetCDF requires wavelength_nm")
        bulk_ext = np.asarray(data.variables["bulk_extinction"].data, dtype=float)
        bulk_scat = np.asarray(data.variables["bulk_scattering"].data, dtype=float)
        phase = np.asarray(data.variables["phase_moments"].data, dtype=float)

    if table_wavelengths.ndim != 1 or np.any(np.diff(table_wavelengths) <= 0.0):
        raise ValueError("aerosol property wavelengths must be one-dimensional and increasing")
    if bulk_ext.shape != bulk_scat.shape or bulk_ext.shape[0] != table_wavelengths.size:
        raise ValueError("bulk_extinction and bulk_scattering must have shape (nwave, naerosol)")
    if not np.all(np.isfinite(bulk_ext)) or not np.all(np.isfinite(bulk_scat)):
        raise ValueError("aerosol bulk tables must be finite")
    if np.any(bulk_ext < 0.0) or np.any(bulk_scat < 0.0) or np.any(bulk_scat > bulk_ext + 1e-14):
        raise ValueError("aerosol bulk scattering must satisfy 0 <= scattering <= extinction")
    return {
        "aerosol_names": names,
        "bulk_extinction": _interp_columns(wavelengths_nm, table_wavelengths, bulk_ext),
        "bulk_scattering": _interp_columns(wavelengths_nm, table_wavelengths, bulk_scat),
        "phase_moments": _phase_moment_endpoints(wavelengths_nm, table_wavelengths, phase),
    }


def _aerosol_loadings_from_profile(
    profile: ProfileTextData,
    spec: Any,
    *,
    n_layers: int,
    aerosol_names: tuple[str, ...] = (),
) -> np.ndarray:
    if not isinstance(spec, dict):
        raise ValueError("aerosol loading_columns must map profile columns to aerosol names")
    if aerosol_names:
        column_by_aerosol = {str(aerosol).lower(): str(column) for column, aerosol in spec.items()}
        columns = [column_by_aerosol.get(name.lower(), name) for name in aerosol_names]
    else:
        columns = [str(column) for column in spec]
    normalized = {name.lower(): values for name, values in profile.columns.items()}
    loadings = []
    for column in columns:
        values = normalized.get(column.lower())
        if values is None:
            raise ValueError(f"profile is missing aerosol loading column {column}")
        arr = np.asarray(values, dtype=float)
        if arr.shape == (n_layers + 1,):
            arr = 0.5 * (arr[:-1] + arr[1:])
        elif arr.shape != (n_layers,):
            raise ValueError(
                f"aerosol loading column {column} must have {n_layers} layer values "
                f"or {n_layers + 1} level values"
            )
        loadings.append(arr)
    out = np.column_stack(loadings) if loadings else np.zeros((n_layers, 0), dtype=float)
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("aerosol loading columns must be finite and nonnegative")
    return out


def _interp_columns(x: np.ndarray, xp: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(x, dtype=float)
    if grid.min() < xp[0] or grid.max() > xp[-1]:
        raise ValueError("aerosol property table does not cover the scene spectral grid")
    out = np.empty((grid.size, values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(grid, xp, values[:, column])
    return out


def _phase_moment_endpoints(
    wavelengths_nm: np.ndarray,
    table_wavelengths: np.ndarray,
    phase_moments: np.ndarray,
) -> np.ndarray:
    if phase_moments.ndim == 3 and phase_moments.shape[0] == 2:
        return np.asarray(phase_moments, dtype=float)
    if phase_moments.ndim != 3 or phase_moments.shape[0] != table_wavelengths.size:
        raise ValueError("phase_moments must have shape (nwave, nmoment, naerosol)")
    endpoints = (float(wavelengths_nm[0]), float(wavelengths_nm[-1]))
    out = np.empty((2, phase_moments.shape[1], phase_moments.shape[2]), dtype=float)
    flat = phase_moments.reshape(phase_moments.shape[0], -1)
    for endpoint_index, endpoint in enumerate(endpoints):
        out[endpoint_index] = _interp_columns(
            np.array([endpoint], dtype=float), table_wavelengths, flat
        )[0].reshape(phase_moments.shape[1], phase_moments.shape[2])
    return out


def _netcdf_string_array(variable: Any) -> tuple[str, ...]:
    if variable is None:
        return ()
    raw = np.asarray(variable.data)
    if raw.ndim == 1:
        return tuple(
            value.decode("utf-8").strip() if isinstance(value, bytes) else str(value).strip()
            for value in raw
        )
    return tuple(bytes(row).decode("utf-8").strip() for row in raw)


def _add_reference_arrays(
    bundle: dict[str, np.ndarray],
    scene: dict[str, Any],
    base_dir: Path,
    *,
    spectral: dict[str, np.ndarray],
) -> None:
    reference = scene.get("reference")
    if reference is None:
        return
    if not isinstance(reference, dict) or "path" not in reference:
        raise ValueError("reference must be a mapping with path")
    path = _resolve_path(reference["path"], base_dir)
    key = str(reference.get("total", "ref_total"))
    with np.load(path) as data:
        _validate_reference_spectral_grid(data, spectral)
        if key not in data:
            raise KeyError(f"reference file is missing {key}")
        bundle["ref_total"] = np.asarray(data[key], dtype=float)


def _validate_reference_spectral_grid(data: Any, spectral: dict[str, np.ndarray]) -> None:
    if "wavelength_nm" not in data:
        raise KeyError("reference file is missing wavelength_nm")
    _assert_reference_grid_matches(
        "wavelength_nm",
        np.asarray(data["wavelength_nm"], dtype=float),
        np.asarray(spectral["wavelengths"], dtype=float),
    )
    for key in ("wavenumber_cm_inv", "wavenumber_band_cm_inv"):
        if key in data and key in spectral:
            if key == "wavenumber_cm_inv":
                candidate = np.asarray(data[key], dtype=float)
                paired = 1.0e7 / np.asarray(spectral["wavelengths"], dtype=float)
                if candidate.shape == paired.shape and np.allclose(
                    candidate, paired, rtol=0.0, atol=1.0e-10
                ):
                    continue
            _assert_reference_grid_matches(
                key,
                np.asarray(data[key], dtype=float),
                np.asarray(spectral[key], dtype=float),
            )


def _assert_reference_grid_matches(name: str, value: np.ndarray, expected: np.ndarray) -> None:
    if (
        value.ndim == expected.ndim
        and value.shape[0] >= expected.shape[0]
        and value.shape[1:] == expected.shape[1:]
    ):
        value = value[: expected.shape[0]]
    if value.shape != expected.shape or not np.allclose(value, expected, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"reference {name} does not match scene spectral grid")


def _solar_geometry(scene: dict[str, Any]) -> np.ndarray:
    geometry = _section(scene, "geometry")
    angles = geometry.get("angles")
    if angles is None:
        raise ValueError("solar scene requires geometry.angles = [sza, vza, raz]")
    arr = np.asarray(angles, dtype=float)
    if arr.ndim == 1 and arr.size == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError("geometry.angles must have shape (3,) or (n_geometry, 3)")


def _thermal_view_angle(scene: dict[str, Any]) -> float:
    geometry = _section(scene, "geometry")
    if "view_angle" in geometry:
        return float(geometry["view_angle"])
    raise ValueError("thermal scene requires geometry.view_angle")


def _surface_temperature(scene: dict[str, Any], profile: ProfileTextData) -> float:
    surface = _section(scene, "surface")
    if "temperature_k" in surface:
        return float(surface["temperature_k"])
    if profile.surface_temperature_k is not None:
        return float(profile.surface_temperature_k)
    return float(profile.temperature_k[-1])


def _surface_altitude(scene: dict[str, Any], profile: ProfileTextData) -> float:
    surface = _section(scene, "surface")
    if "altitude_m" in surface:
        return float(surface["altitude_m"])
    return float(profile.surface_altitude_m)


def _load_array(spec: Any, base_dir: Path, name: str) -> np.ndarray:
    if isinstance(spec, dict):
        if "value" in spec:
            array = np.asarray(spec["value"], dtype=float)
        elif "path" in spec:
            path = _resolve_path(spec["path"], base_dir)
            if path.suffix == ".npy":
                array = np.load(path)
            else:
                array = np.loadtxt(path, dtype=float, delimiter=spec.get("delimiter"))
        else:
            raise ValueError(f"{name} array spec requires value or path")
        if "shape" in spec:
            array = np.asarray(array, dtype=float).reshape(tuple(spec["shape"]))
        return np.asarray(array, dtype=float)
    return np.asarray(spec, dtype=float)


def _resolve_table_specs(value: Any, base_dir: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("gas_cross_sections.tables must be a mapping")
    tables = {}
    for gas, spec in value.items():
        if not isinstance(spec, dict):
            raise ValueError(f"gas_cross_sections.tables.{gas} must be a mapping")
        resolved = dict(spec)
        if "path" in resolved:
            resolved["path"] = _resolve_path(resolved["path"], base_dir)
        tables[str(gas)] = resolved
    return tables


def _load_geocape_ssprops(spec: Any, base_dir: Path, *, wavelengths_nm: np.ndarray):
    if not isinstance(spec, dict) or "path" not in spec:
        raise ValueError("aerosol.ssprops requires path")
    aggregates = tuple(str(name) for name in spec.get("aggregates", ()))
    kwargs = {}
    if aggregates:
        kwargs["aggregates"] = aggregates
    if "moment_cutoff" in spec:
        kwargs["moment_cutoff"] = float(spec["moment_cutoff"])
    if "max_moments" in spec:
        kwargs["max_moments"] = int(spec["max_moments"])
    wavelengths_microns = np.asarray(wavelengths_nm, dtype=float) / 1000.0
    return load_geocape_aerosol_tables(
        _resolve_path(spec["path"], base_dir),
        first_wavelength_microns=float(wavelengths_microns[0]),
        last_wavelength_microns=float(wavelengths_microns[-1]),
        **kwargs,
    )


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _spectral_1d(value: Any, base_dir: Path, name: str) -> np.ndarray:
    if isinstance(value, dict) and {"start", "step", "count"}.issubset(value):
        count = int(value["count"])
        if count <= 0:
            raise ValueError(f"{name}.count must be positive")
        return float(value["start"]) + float(value["step"]) * np.arange(count, dtype=float)
    return _load_array(value, base_dir, name)


def _broadcast_spectral(
    value: Any,
    nspec: int,
    name: str,
    base_dir: Path,
    *,
    source_nspec: int | None = None,
) -> np.ndarray:
    arr = _load_array(value, base_dir, name)
    if arr.ndim == 0:
        return np.full(nspec, float(arr), dtype=float)
    if arr.shape == (nspec,):
        return arr
    if source_nspec is not None and arr.shape == (source_nspec,):
        return arr[:nspec]
    raise ValueError(f"{name} must be scalar or have shape ({nspec},)")


def _as_1d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return arr


def _top_to_bottom(
    *,
    pressure: np.ndarray,
    temperature: np.ndarray,
    gas_vmr: np.ndarray,
    heights: np.ndarray | None,
    columns: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    pressure_step = np.diff(pressure)
    output_columns = {
        name: np.asarray(value, dtype=float) for name, value in (columns or {}).items()
    }
    if np.all(pressure_step < 0.0):
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        gas_vmr = gas_vmr[::-1]
        heights = None if heights is None else heights[::-1]
        output_columns = {name: value[::-1] for name, value in output_columns.items()}
    elif not np.all(pressure_step > 0.0):
        raise ValueError("profile pressure must be strictly monotonic")
    return pressure, temperature, gas_vmr, heights, output_columns


def _select_gas_vmr(
    gas_data: np.ndarray,
    *,
    gas_names: tuple[str, ...],
    gas_species: tuple[str, ...],
) -> np.ndarray:
    if not gas_species:
        return np.asarray(gas_data, dtype=float)
    indices = []
    normalized = {name.upper(): index for index, name in enumerate(gas_names)}
    for species in gas_species:
        key = species.upper()
        if key not in normalized:
            raise ValueError(f"profile is missing requested gas species {species}")
        indices.append(normalized[key])
    return np.asarray(gas_data[:, indices], dtype=float)


def _gas_columns(data, gas_names: tuple[str, ...]) -> np.ndarray:
    if not gas_names:
        first_name = data.dtype.names[0]
        return np.zeros((np.asarray(data[first_name]).shape[0], 0), dtype=float)
    return np.column_stack([np.asarray(data[name], dtype=float) for name in gas_names])


def _first_existing(
    names: tuple[str, ...],
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    normalized = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if required:
        raise ValueError(f"profile is missing one of: {', '.join(candidates)}")
    return None


def _first_noncomment_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise ValueError("profile file is empty")


def _is_int_token(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _metadata_float(
    metadata: dict[str, str],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    if key not in metadata:
        return default
    return float(metadata[key])


def _section(scene: dict[str, Any], name: str) -> dict[str, Any]:
    value = scene.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} section must be a mapping")
    return value
