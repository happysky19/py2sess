"""Profile and YAML scene loaders for benchmark scene inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .createprops import load_createprops_provider


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


def load_profile_text(path: str | Path, *, gas_species: tuple[str, ...] = ()) -> ProfileTextData:
    """Load a GEOCAPE-style or simple column profile text file.

    Output pressure is ordered top to bottom, matching py2sess layer helpers.
    GEOCAPE profile rows are surface to top, so they are reversed on read.
    """
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
) -> dict[str, np.ndarray]:
    """Build benchmark runtime inputs from profile text and scene YAML."""
    scene_file = Path(scene_path)
    scene = load_scene_yaml(scene_file)
    mode = _scene_mode(scene, kind)
    provider = _load_opacity_provider(scene, scene_file.parent, kind=kind)
    spectral = _spectral_arrays(scene, provider)
    gases = _scene_gases(scene)
    profile = load_profile_text(profile_path, gas_species=gases)
    if not gases:
        profile = replace(
            profile,
            gas_names=(),
            gas_vmr=np.zeros((profile.pressure_hpa.shape[0], 0), dtype=float),
        )
    provider_has_components = _has_layer_components(provider)
    gas_cross_sections = _gas_cross_sections(
        scene,
        scene_file.parent,
        spectral["wavelengths"],
        required=not provider_has_components,
    )

    bundle: dict[str, np.ndarray] = {
        "wavelengths": spectral["wavelengths"],
        "pressure_hpa": profile.pressure_hpa,
        "temperature_k": profile.temperature_k,
        "gas_vmr": profile.gas_vmr,
        "gas_cross_sections": gas_cross_sections,
        "surface_altitude_m": np.array(profile.surface_altitude_m, dtype=float),
    }
    if profile.heights_km is not None:
        bundle["heights"] = profile.heights_km
    for key in ("wavenumber_cm_inv", "wavenumber_band_cm_inv", "wavelength_microns"):
        if key in spectral:
            bundle[key] = spectral[key]

    _add_provider_arrays(bundle, provider)
    _add_surface_arrays(bundle, scene, mode=mode, nspec=spectral["wavelengths"].shape[0])
    _add_aerosol_arrays(bundle, scene, scene_file.parent)
    _add_reference_arrays(bundle, scene, scene_file.parent)

    rt_config = _section(scene, "rt")
    if "stream" in rt_config:
        bundle["stream_value"] = np.asarray(rt_config["stream"], dtype=float)

    if mode == "solar":
        if "user_obsgeom" not in bundle or "angles" in _section(scene, "geometry"):
            bundle["user_obsgeom"] = _solar_geometry(scene)
        if "flux_factor" not in bundle or "flux_factor" in _section(scene, "solar"):
            bundle["flux_factor"] = _broadcast_spectral(
                _section(scene, "solar").get("flux_factor", 1.0),
                spectral["wavelengths"].shape[0],
                "flux_factor",
            )
    else:
        if "user_angle" not in bundle or "view_angle" in _section(scene, "geometry"):
            bundle["user_angle"] = np.array([_thermal_view_angle(scene)], dtype=float)
        surface_temperature = _surface_temperature(scene, profile)
        if "level_temperature_k" not in bundle:
            bundle["level_temperature_k"] = profile.temperature_k
        if "surface_temperature_k" not in bundle or "temperature_k" in _section(scene, "surface"):
            bundle["surface_temperature_k"] = np.array([surface_temperature], dtype=float)

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
    pressure, temperature, gas, heights = _top_to_bottom(
        pressure=pressure,
        temperature=temperature,
        gas_vmr=gas,
        heights=None,
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
    gas = _select_gas_vmr(gas_columns, gas_names=gas_names, gas_species=gas_species)
    pressure, temperature, gas, heights = _top_to_bottom(
        pressure=pressure,
        temperature=temperature,
        gas_vmr=gas,
        heights=heights,
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
    )


def _scene_mode(scene: dict[str, Any], kind: str) -> str:
    raw = str(scene.get("mode", kind)).lower()
    if raw in {"uv", "solar"}:
        mode = "solar"
    elif raw in {"tir", "thermal"}:
        mode = "thermal"
    else:
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


def _spectral_arrays(
    scene: dict[str, Any],
    provider: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    spectral = _section(scene, "spectral")
    if "wavelengths_nm" in spectral:
        wavelengths = _as_1d_float(spectral["wavelengths_nm"], "spectral.wavelengths_nm")
        return {"wavelengths": wavelengths}
    if "wavelength_microns" in spectral:
        microns = _as_1d_float(spectral["wavelength_microns"], "spectral.wavelength_microns")
        return {"wavelengths": microns * 1000.0, "wavelength_microns": microns}
    if "wavenumber_cm_inv" in spectral:
        wavenumber = _as_1d_float(spectral["wavenumber_cm_inv"], "spectral.wavenumber_cm_inv")
        return {"wavelengths": 1.0e7 / wavenumber, "wavenumber_cm_inv": wavenumber}
    if "wavenumber_band_cm_inv" in spectral:
        bands = np.asarray(spectral["wavenumber_band_cm_inv"], dtype=float)
        if bands.ndim != 2 or bands.shape[1] != 2:
            raise ValueError("spectral.wavenumber_band_cm_inv must have shape (n, 2)")
        center = 0.5 * (bands[:, 0] + bands[:, 1])
        return {
            "wavelengths": 1.0e7 / center,
            "wavenumber_band_cm_inv": bands,
            "wavenumber_cm_inv": center,
        }
    if "wavelengths" in provider:
        arrays = {"wavelengths": np.asarray(provider["wavelengths"], dtype=float)}
        for key in ("wavenumber_cm_inv", "wavenumber_band_cm_inv", "wavelength_microns"):
            if key in provider:
                arrays[key] = np.asarray(provider[key], dtype=float)
        return arrays
    raise ValueError("scene spectral section must define wavelengths or wavenumbers")


def _gas_cross_sections(
    scene: dict[str, Any],
    base_dir: Path,
    wavelengths: np.ndarray,
    *,
    required: bool,
) -> np.ndarray:
    opacity = _section(scene, "opacity")
    gas_cfg = opacity.get("gas_cross_sections")
    gases = _scene_gases(scene)
    if gas_cfg is None:
        if required and gases:
            raise ValueError("scene gases require opacity.gas_cross_sections")
        return np.zeros((wavelengths.shape[0], 0), dtype=float)
    xsec = _load_array(gas_cfg, base_dir, "gas_cross_sections")
    if xsec.ndim == 1 and len(gases) == 1:
        xsec = xsec[:, np.newaxis]
    if xsec.shape[0] != wavelengths.shape[0]:
        raise ValueError("gas_cross_sections first axis must match spectral grid")
    if gases and xsec.shape[-1] != len(gases):
        raise ValueError("gas_cross_sections gas axis must match scene gases")
    return xsec


def _add_surface_arrays(
    bundle: dict[str, np.ndarray],
    scene: dict[str, Any],
    *,
    mode: str,
    nspec: int,
) -> None:
    surface = _section(scene, "surface")
    if "albedo" in surface or "albedo" not in bundle:
        bundle["albedo"] = _broadcast_spectral(surface.get("albedo", 0.0), nspec, "albedo")
    if mode == "thermal" and "emissivity" in surface:
        bundle["emissivity"] = _broadcast_spectral(surface["emissivity"], nspec, "emissivity")


def _load_opacity_provider(
    scene: dict[str, Any],
    base_dir: Path,
    *,
    kind: str,
) -> dict[str, np.ndarray]:
    provider = _section(scene, "opacity").get("provider")
    if provider is None:
        return {}
    if not isinstance(provider, dict):
        raise ValueError("opacity.provider must be a mapping")
    provider_kind = str(provider.get("kind", "fortran_createprops")).lower()
    if provider_kind not in {"fortran_createprops", "fortran_createprops_arrays"}:
        raise ValueError("opacity.provider.kind must be fortran_createprops")
    if "path" not in provider:
        raise ValueError("opacity.provider requires path")
    return load_createprops_provider(_resolve_path(provider["path"], base_dir), kind=kind)


def _has_layer_components(provider: dict[str, np.ndarray]) -> bool:
    return {
        "absorption_tau",
        "rayleigh_scattering_tau",
        "aerosol_scattering_tau",
    }.issubset(provider)


def _add_provider_arrays(
    bundle: dict[str, np.ndarray],
    provider: dict[str, np.ndarray],
) -> None:
    for key, value in provider.items():
        if key in {"wavelengths", "wavenumber_cm_inv", "wavenumber_band_cm_inv"} and key in bundle:
            _assert_matching_array(key, bundle[key], value)
            continue
        bundle[key] = value


def _add_aerosol_arrays(
    bundle: dict[str, np.ndarray], scene: dict[str, Any], base_dir: Path
) -> None:
    opacity = _section(scene, "opacity")
    aerosol = opacity.get("aerosol", {})
    if aerosol is None:
        aerosol = {}
    if not isinstance(aerosol, dict):
        raise ValueError("aerosol section must be a mapping")
    moments = aerosol.get("moments")
    loadings = aerosol.get("loadings")
    if loadings is not None and moments is None:
        raise ValueError("aerosol loadings require aerosol moments")
    if moments is None and "aerosol_moments" not in bundle:
        bundle["aerosol_moments"] = np.zeros((2, 3, 0), dtype=float)
    elif moments is not None:
        bundle["aerosol_moments"] = _load_array(moments, base_dir, "aerosol_moments")
    if loadings is None:
        return
    bundle["aerosol_loadings"] = _load_array(loadings, base_dir, "aerosol_loadings")
    bundle["aerosol_wavelengths_microns"] = _load_array(
        aerosol["wavelengths_microns"],
        base_dir,
        "aerosol_wavelengths_microns",
    )
    bundle["aerosol_bulk_iops"] = _load_array(aerosol["bulk_iops"], base_dir, "aerosol_bulk_iops")
    if "select_wavelength_microns" in aerosol:
        bundle["aerosol_select_wavelength_microns"] = np.asarray(
            aerosol["select_wavelength_microns"],
            dtype=float,
        )


def _add_reference_arrays(
    bundle: dict[str, np.ndarray], scene: dict[str, Any], base_dir: Path
) -> None:
    reference = scene.get("reference")
    if reference is None:
        return
    if not isinstance(reference, dict) or "path" not in reference:
        raise ValueError("reference must be a mapping with path")
    path = _resolve_path(reference["path"], base_dir)
    key = str(reference.get("total", "ref_total"))
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"reference file is missing {key}")
        bundle["ref_total"] = np.asarray(data[key], dtype=float)


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


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _assert_matching_array(name: str, left: np.ndarray, right: np.ndarray) -> None:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape or not np.allclose(left_arr, right_arr):
        raise ValueError(f"scene {name} does not match opacity.provider {name}")


def _broadcast_spectral(value: Any, nspec: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(nspec, float(arr), dtype=float)
    if arr.shape == (nspec,):
        return arr
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if np.all(np.diff(pressure) < 0.0):
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        gas_vmr = gas_vmr[::-1]
        heights = None if heights is None else heights[::-1]
    return pressure, temperature, gas_vmr, heights


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
