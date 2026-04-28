"""Adapters for Fortran CreateProps optical-provider outputs."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .phase import aerosol_interp_fraction

_COMMON_PROVIDER_KEYS = (
    "wavelengths",
    "heights",
    "albedo",
    "depol",
    "gas_absorption_tau",
    "absorption_tau",
    "rayleigh_scattering_tau",
    "aerosol_scattering_tau",
    "aerosol_moments",
    "aerosol_interp_fraction",
)

_UV_PROVIDER_KEYS = _COMMON_PROVIDER_KEYS + ("flux_factor", "user_obsgeom")
_TIR_PROVIDER_KEYS = _COMMON_PROVIDER_KEYS + (
    "wavenumber_cm_inv",
    "wavenumber_band_cm_inv",
    "level_temperature_k",
    "surface_temperature_k",
    "user_angle",
)


def load_createprops_provider(path: str | Path, *, kind: str) -> dict[str, np.ndarray]:
    """Load a local Fortran CreateProps provider directory."""
    provider_path = Path(path)
    if not provider_path.is_dir():
        raise ValueError("Fortran CreateProps provider path must be a directory")
    keys = _provider_keys(kind)
    arrays = {
        key: np.load(provider_path / f"{key}.npy", mmap_mode="r")
        for key in keys
        if (provider_path / f"{key}.npy").is_file()
    }
    _validate_provider_arrays(arrays, kind=kind)
    return arrays


def write_createprops_provider(
    arrays: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    kind: str,
) -> None:
    """Write CreateProps provider arrays as one ``.npy`` file per field."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    keys = _provider_keys(kind)
    for key in keys:
        if key in arrays:
            np.save(output_path / f"{key}.npy", np.asarray(arrays[key]))


def parse_createprops_dump(
    path: str | Path,
    *,
    kind: str,
    profile_file: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Parse a benchmark CreateProps text dump into provider arrays."""
    dump_path = Path(path)
    if kind == "uv":
        arrays = _parse_uv_dump(dump_path)
    elif kind == "tir":
        arrays = _parse_tir_dump(
            dump_path,
            profile_file=None if profile_file is None else Path(profile_file),
        )
    else:
        raise ValueError("kind must be 'uv' or 'tir'")
    _validate_provider_arrays(arrays, kind=kind)
    return arrays


def _provider_keys(kind: str) -> tuple[str, ...]:
    if kind == "uv":
        return _UV_PROVIDER_KEYS
    if kind == "tir":
        return _TIR_PROVIDER_KEYS
    raise ValueError("kind must be 'uv' or 'tir'")


def _validate_provider_arrays(arrays: dict[str, np.ndarray], *, kind: str) -> None:
    required = (
        "absorption_tau",
        "rayleigh_scattering_tau",
        "aerosol_scattering_tau",
        "depol",
        "aerosol_moments",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(
            "Fortran CreateProps provider is missing required arrays: " + ", ".join(missing)
        )

    absorption = np.asarray(arrays["absorption_tau"])
    rayleigh = np.asarray(arrays["rayleigh_scattering_tau"])
    aerosol = np.asarray(arrays["aerosol_scattering_tau"])
    if absorption.ndim != 2:
        raise ValueError("absorption_tau must have shape (nspec, nlayer)")
    if "gas_absorption_tau" in arrays:
        gas_absorption = np.asarray(arrays["gas_absorption_tau"])
        if gas_absorption.shape != absorption.shape:
            raise ValueError("gas_absorption_tau must match absorption_tau shape")
    if rayleigh.shape != absorption.shape:
        raise ValueError("rayleigh_scattering_tau must match absorption_tau shape")
    if aerosol.ndim != 3 or aerosol.shape[:2] != absorption.shape:
        raise ValueError("aerosol_scattering_tau must have shape (nspec, nlayer, naerosol)")

    nspec, nlayer = absorption.shape
    if "wavelengths" in arrays and np.asarray(arrays["wavelengths"]).shape != (nspec,):
        raise ValueError("wavelengths must have shape (nspec,)")
    if "heights" in arrays and np.asarray(arrays["heights"]).shape != (nlayer + 1,):
        raise ValueError("heights must have shape (nlayer + 1,)")
    if "albedo" in arrays and np.asarray(arrays["albedo"]).shape != (nspec,):
        raise ValueError("albedo must have shape (nspec,)")
    if "depol" in arrays and np.asarray(arrays["depol"]).shape != (nspec,):
        raise ValueError("depol must have shape (nspec,)")
    if kind == "uv" and "flux_factor" in arrays:
        if np.asarray(arrays["flux_factor"]).shape != (nspec,):
            raise ValueError("flux_factor must have shape (nspec,)")
    if kind == "tir" and "wavenumber_cm_inv" in arrays:
        if np.asarray(arrays["wavenumber_cm_inv"]).shape != (nspec,):
            raise ValueError("wavenumber_cm_inv must have shape (nspec,)")
    if kind == "tir" and not (
        "wavenumber_cm_inv" in arrays
        or "wavenumber_band_cm_inv" in arrays
        or "wavelength_microns" in arrays
    ):
        raise ValueError(
            "TIR CreateProps provider requires wavenumber_cm_inv, "
            "wavenumber_band_cm_inv, or wavelength_microns"
        )


def _read_aerosol_moments(handle, n_moments: int) -> np.ndarray:
    moments = np.empty((2, n_moments + 1, 5), dtype=np.float64)
    for _ in range(n_moments + 1):
        values = handle.readline().split()
        if len(values) != 11:
            raise ValueError("invalid aerosol-moment row in Fortran dump")
        index = int(values[0])
        moments[0, index] = [float(value) for value in values[1:6]]
        moments[1, index] = [float(value) for value in values[6:11]]
    return moments


def _component_optical_depths(
    *,
    gas_absorption_tau: np.ndarray,
    total_tau: np.ndarray,
    ssa: np.ndarray,
    rayleigh_fraction: np.ndarray,
    aerosol_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    gas_absorption_tau = np.asarray(gas_absorption_tau, dtype=np.float64)
    if gas_absorption_tau.shape != total_tau.shape:
        raise ValueError("gas_absorption_tau must match total_tau shape")
    if not np.all(np.isfinite(gas_absorption_tau)) or np.any(gas_absorption_tau < 0.0):
        raise ValueError("CreateProps dump has invalid gas_absorption_tau")
    scattering_tau = total_tau * ssa
    absorption_tau = total_tau - scattering_tau
    negative = absorption_tau < 0.0
    if np.any(negative):
        tolerance = 1.0e-12 * np.maximum(np.abs(total_tau), np.abs(scattering_tau))
        if np.any(negative & (np.abs(absorption_tau) > tolerance)):
            raise ValueError("CreateProps dump implies negative absorption_tau")
        absorption_tau = np.where(negative, 0.0, absorption_tau)
    return {
        "gas_absorption_tau": gas_absorption_tau,
        "absorption_tau": absorption_tau,
        "rayleigh_scattering_tau": scattering_tau * rayleigh_fraction,
        "aerosol_scattering_tau": scattering_tau[..., None] * aerosol_fraction,
    }


def _parse_uv_dump(dump_path: Path) -> dict[str, np.ndarray]:
    with dump_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) < 6:
            raise ValueError("invalid UV CreateProps dump header")
        n_layers = int(header[0])
        n_rows = int(header[1])
        n_moments = int(header[2])
        angles = np.array([float(header[3]), float(header[4]), float(header[5])], dtype=np.float64)
        heights = np.array([float(handle.readline().split()[0]) for _ in range(n_layers + 1)])
        aerosol_moments = _read_aerosol_moments(handle, n_moments)

        wavelengths = np.empty(n_rows, dtype=np.float64)
        albedo = np.empty(n_rows, dtype=np.float64)
        depol = np.empty(n_rows, dtype=np.float64)
        flux_factor = np.empty(n_rows, dtype=np.float64)
        gas_absorption_tau = np.empty((n_rows, n_layers), dtype=np.float64)
        total_tau = np.empty((n_rows, n_layers), dtype=np.float64)
        ssa = np.empty((n_rows, n_layers), dtype=np.float64)
        rayleigh_fraction = np.empty((n_rows, n_layers), dtype=np.float64)
        aerosol_fraction = np.empty((n_rows, n_layers, 5), dtype=np.float64)

        for row in range(n_rows):
            spec = handle.readline().split()
            if len(spec) < 6:
                raise ValueError(f"invalid UV spectral row {row + 1}")
            wavelengths[row] = float(spec[1])
            albedo[row] = float(spec[3])
            depol[row] = float(spec[4])
            flux_factor[row] = float(spec[5])
            for layer in range(n_layers):
                values = handle.readline().split()
                if len(values) < 10:
                    raise ValueError(f"invalid UV layer row {row + 1}, layer {layer + 1}")
                gas_absorption_tau[row, layer] = float(values[1])
                total_tau[row, layer] = float(values[2])
                ssa[row, layer] = float(values[3])
                rayleigh_fraction[row, layer] = float(values[4])
                aerosol_fraction[row, layer] = [float(value) for value in values[5:10]]

    arrays = {
        "wavelengths": wavelengths,
        "heights": heights,
        "albedo": albedo,
        "depol": depol,
        "flux_factor": flux_factor,
        "user_obsgeom": angles,
        "rayleigh_fraction": rayleigh_fraction,
        "aerosol_fraction": aerosol_fraction,
        "aerosol_moments": aerosol_moments,
        "aerosol_interp_fraction": aerosol_interp_fraction(wavelengths, reverse=True),
    }
    arrays.update(
        _component_optical_depths(
            gas_absorption_tau=gas_absorption_tau,
            total_tau=total_tau,
            ssa=ssa,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
        )
    )
    return arrays


def _parse_tir_dump(
    dump_path: Path,
    *,
    profile_file: Path | None,
) -> dict[str, np.ndarray]:
    with dump_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) < 5:
            raise ValueError("invalid TIR CreateProps dump header")
        n_layers = int(header[0])
        n_rows = int(header[1])
        n_moments = int(header[2])
        user_angle = np.array([float(header[3])], dtype=np.float64)
        heights = np.empty(n_layers + 1, dtype=np.float64)
        for level in range(n_layers + 1):
            heights[level] = float(handle.readline().split()[0])
        aerosol_moments = _read_aerosol_moments(handle, n_moments)

        wavelengths = np.empty(n_rows, dtype=np.float64)
        wavenumber = np.empty(n_rows, dtype=np.float64)
        albedo = np.empty(n_rows, dtype=np.float64)
        depol = np.empty(n_rows, dtype=np.float64)
        gas_absorption_tau = np.empty((n_rows, n_layers), dtype=np.float64)
        total_tau = np.empty((n_rows, n_layers), dtype=np.float64)
        ssa = np.empty((n_rows, n_layers), dtype=np.float64)
        rayleigh_fraction = np.empty((n_rows, n_layers), dtype=np.float64)
        aerosol_fraction = np.empty((n_rows, n_layers, 5), dtype=np.float64)

        for row in range(n_rows):
            spec = handle.readline().split()
            if len(spec) < 7:
                raise ValueError(f"invalid TIR spectral row {row + 1}")
            wavelengths[row] = float(spec[1])
            wavenumber[row] = float(spec[2])
            albedo[row] = float(spec[3])
            depol[row] = float(spec[4])
            for layer in range(n_layers):
                values = handle.readline().split()
                if len(values) < 11:
                    raise ValueError(f"invalid TIR layer row {row + 1}, layer {layer + 1}")
                gas_absorption_tau[row, layer] = float(values[1])
                total_tau[row, layer] = float(values[2])
                ssa[row, layer] = float(values[3])
                rayleigh_fraction[row, layer] = float(values[4])
                aerosol_fraction[row, layer] = [float(value) for value in values[5:10]]

    profile_path = _infer_tir_profile_path(dump_path) if profile_file is None else profile_file
    level_temperature, surface_temperature = _parse_tir_profile(
        profile_path,
        n_layers=n_layers,
    )
    arrays = {
        "wavelengths": wavelengths,
        "wavenumber_cm_inv": wavenumber,
        "wavenumber_band_cm_inv": np.column_stack((wavenumber - 0.5, wavenumber + 0.5)),
        "heights": heights,
        "albedo": albedo,
        "depol": depol,
        "user_angle": user_angle,
        "level_temperature_k": level_temperature,
        "surface_temperature_k": surface_temperature,
        "rayleigh_fraction": rayleigh_fraction,
        "aerosol_fraction": aerosol_fraction,
        "aerosol_moments": aerosol_moments,
        "aerosol_interp_fraction": aerosol_interp_fraction(wavelengths, reverse=True),
    }
    arrays.update(
        _component_optical_depths(
            gas_absorption_tau=gas_absorption_tau,
            total_tau=total_tau,
            ssa=ssa,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
        )
    )
    return arrays


def _infer_tir_profile_path(dump_path: Path) -> Path:
    match = re.match(
        r"Dump_(?P<location>[^_]+)_(?P<date>[^_]+)_(?P<time>[^.]+)\.dat(?:_.+)?$",
        dump_path.name,
    )
    if match is None:
        raise ValueError("could not infer TIR profile file from dump name; pass profile_file")
    profile_name = (
        f"Profiles_{match.group('location')}_20067{match.group('date')}_{match.group('time')}.dat"
    )
    return (dump_path.parent / "../../geocape_data/Profile_Data" / profile_name).resolve()


def _parse_tir_profile(profile_path: Path, *, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    surface_temperature = None
    rows: list[list[float]] = []
    with profile_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("surfaceTemperature"):
                surface_temperature = float(stripped.split("=", maxsplit=1)[1])
                continue
            values = stripped.split()
            if values and values[0].isdigit():
                rows.append([float(value) for value in values])

    if surface_temperature is None:
        raise ValueError(f"missing surface temperature in {profile_path}")
    if len(rows) < n_layers + 1:
        raise ValueError(
            f"profile {profile_path} has {len(rows)} levels; expected at least {n_layers + 1}"
        )

    profile_temperature = np.array([row[2] for row in rows], dtype=np.float64)
    level_temperature = profile_temperature[-(n_layers + 1) :][::-1]
    if not np.all(np.isfinite(level_temperature)) or np.any(level_temperature <= 0.0):
        raise ValueError(f"profile {profile_path} contains invalid level temperatures")
    return level_temperature, np.array([surface_temperature], dtype=np.float64)
