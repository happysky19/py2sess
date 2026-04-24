"""Build thermal source inputs from a temperature profile."""

from __future__ import annotations

import numpy as np

from py2sess import thermal_source_from_temperature_profile


def main() -> None:
    """Builds and prints one small thermal-source example."""
    level_temperature_k = np.array([220.0, 230.0, 245.0, 260.0, 275.0], dtype=float)
    surface_temperature_k = 288.0

    source = thermal_source_from_temperature_profile(
        level_temperature_k,
        surface_temperature_k,
        wavenumber_band_cm_inv=(900.0, 901.0),
    )

    print("planck:")
    print(source.planck)
    print()
    print(f"surface_planck: {source.surface_planck:.8e}")


if __name__ == "__main__":
    main()
