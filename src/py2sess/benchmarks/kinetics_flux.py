"""KINETICS/FLXOUT flux-table parsing for diagnostic comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

KINETICS_FLXOUT_COLUMNS = (
    "level",
    "altitude_km",
    "direct_flux",
    "diffuse_plus_flux",
    "diffuse_minus_flux",
    "net_flux",
    "diffuse_radiation_field",
    "total_radiation_field",
    "diffuse_factor",
)

_ALIASES = {
    "z": "altitude_km",
    "alt": "altitude_km",
    "altitude": "altitude_km",
    "altitudekm": "altitude_km",
    "ydf": "direct_flux",
    "radfd": "direct_flux",
    "direct": "direct_flux",
    "directdown": "direct_flux",
    "directflux": "direct_flux",
    "flp": "diffuse_plus_flux",
    "yflp": "diffuse_plus_flux",
    "diffuseplus": "diffuse_plus_flux",
    "diffuseplusflux": "diffuse_plus_flux",
    "up": "diffuse_plus_flux",
    "fluxup": "diffuse_plus_flux",
    "flm": "diffuse_minus_flux",
    "yflm": "diffuse_minus_flux",
    "diffuseminus": "diffuse_minus_flux",
    "diffuseminusflux": "diffuse_minus_flux",
    "down": "diffuse_minus_flux",
    "fluxdown": "diffuse_minus_flux",
    "ytot": "net_flux",
    "net": "net_flux",
    "netflux": "net_flux",
    "difflxr": "diffuse_radiation_field",
    "diffusefield": "diffuse_radiation_field",
    "diffuseradiationfield": "diffuse_radiation_field",
    "tradf": "total_radiation_field",
    "totalfield": "total_radiation_field",
    "totalradiationfield": "total_radiation_field",
    "diffac": "diffuse_factor",
    "diffusefactor": "diffuse_factor",
}


def _normalize_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.lower() if ch.isalnum())
    return _ALIASES.get(cleaned, cleaned)


def _header_from_text(text: str) -> list[str] | None:
    names = [_normalize_name(part) for part in text.replace(",", " ").split()]
    known = set(KINETICS_FLXOUT_COLUMNS)
    if len(names) > 1 and all(name in known for name in names):
        return names
    return None


def parse_kinetics_flux_table(text: str) -> dict[str, np.ndarray]:
    """Parses a captured KINETICS FLXOUT-style level table.

    The parser accepts whitespace- or comma-separated rows. A header is optional
    when the table columns follow ``KINETICS_FLXOUT_COLUMNS``.
    """
    header: list[str] | None = None
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[0] in "#!":
            candidate = line[1:].strip()
            parsed_header = _header_from_text(candidate)
            if parsed_header is not None:
                header = parsed_header
            continue
        parts = line.replace(",", " ").split()
        if any(any(ch.isalpha() for ch in part) for part in parts):
            parsed_header = _header_from_text(line)
            if parsed_header is None:
                raise ValueError(f"unrecognized KINETICS flux header: {line}")
            header = parsed_header
            continue
        rows.append([float(part) for part in parts])

    if not rows:
        raise ValueError("no numeric KINETICS flux rows found")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("KINETICS flux rows have inconsistent column counts")
    if header is None:
        if width > len(KINETICS_FLXOUT_COLUMNS):
            raise ValueError("KINETICS flux table has more columns than the default schema")
        names = list(KINETICS_FLXOUT_COLUMNS[:width])
    else:
        if len(header) != width:
            raise ValueError("KINETICS flux table header does not match row width")
        names = header

    data = np.asarray(rows, dtype=float)
    return {name: data[:, index] for index, name in enumerate(names)}


def read_kinetics_flux_table(path: str | Path) -> dict[str, np.ndarray]:
    """Reads and parses a captured KINETICS flux table."""
    return parse_kinetics_flux_table(Path(path).read_text())


def kinetics_flux_to_py2sess(table: dict[str, Any]) -> dict[str, Any]:
    """Maps diagnostic KINETICS flux columns to py2sess level-flux names."""
    direct = np.asarray(table["direct_flux"], dtype=float)
    diffuse_up = np.asarray(table["diffuse_plus_flux"], dtype=float)
    diffuse_down = np.asarray(table["diffuse_minus_flux"], dtype=float)
    flux_down = direct + diffuse_down
    flux_net = np.asarray(table.get("net_flux", diffuse_up - flux_down), dtype=float)
    flux_mean = np.asarray(table["total_radiation_field"], dtype=float)
    return {
        "flux_up": diffuse_up,
        "flux_down": flux_down,
        "flux_net": flux_net,
        "flux_mean": flux_mean,
        "direct_down": direct,
        "diffuse_down": diffuse_down,
        "diffuse_up": diffuse_up,
        "diffuse_radiation_field": np.asarray(
            table.get("diffuse_radiation_field", np.nan), dtype=float
        ),
        "total_radiation_field": flux_mean,
        "diffuse_factor": np.asarray(table.get("diffuse_factor", np.nan), dtype=float),
    }
