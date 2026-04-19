from __future__ import annotations

import unittest

import numpy as np

from py2sess import planck_radiance_wavenumber_band, thermal_source_from_temperature_profile


class ThermalSourceTests(unittest.TestCase):
    def test_band_planck_increases_with_temperature(self) -> None:
        values = planck_radiance_wavenumber_band(
            np.array([220.0, 240.0, 260.0], dtype=float),
            900.0,
            901.0,
        )
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_profile_helper_maps_level_and_surface_temperatures(self) -> None:
        source = thermal_source_from_temperature_profile(
            [220.0, 230.0, 240.0, 250.0],
            280.0,
            wavenumber_band_cm_inv=(900.0, 901.0),
        )
        self.assertEqual(source.thermal_bb_input.shape, (4,))
        self.assertGreater(source.surfbb, source.thermal_bb_input[-1])

    def test_profile_helper_returns_constant_planck_for_constant_temperature(self) -> None:
        temperature = 250.0
        source = thermal_source_from_temperature_profile(
            [temperature, temperature, temperature, temperature],
            temperature,
            wavenumber_band_cm_inv=(900.0, 901.0),
        )
        expected = planck_radiance_wavenumber_band(
            np.array([temperature], dtype=float), 900.0, 901.0
        )[0]
        np.testing.assert_allclose(
            source.thermal_bb_input, np.full(4, expected, dtype=float), rtol=0.0, atol=1.0e-12
        )
        self.assertAlmostEqual(source.surfbb, float(expected), places=12)


if __name__ == "__main__":
    unittest.main()
