from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    planck_radiance_wavelength,
    planck_radiance_wavenumber,
    planck_radiance_wavenumber_band,
    thermal_source_from_temperature_profile,
)
from py2sess.rtsolver.backend import has_torch


class ThermalSourceTests(unittest.TestCase):
    def test_band_planck_increases_with_temperature(self) -> None:
        values = planck_radiance_wavenumber_band(
            np.array([220.0, 240.0, 260.0]),
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
        self.assertEqual(source.planck.shape, (4,))
        self.assertGreater(source.surface_planck, source.planck[-1])

    def test_profile_helper_vectorizes_over_wavenumbers(self) -> None:
        level_temperature = np.array([220.0, 230.0, 240.0, 250.0])
        wavenumber = np.array([700.0, 800.0, 900.0])
        source = thermal_source_from_temperature_profile(
            level_temperature,
            280.0,
            wavenumber_cm_inv=wavenumber,
        )
        expected_planck = np.vstack(
            [planck_radiance_wavenumber(level_temperature, value) for value in wavenumber]
        )
        self.assertEqual(source.planck.shape, (3, 4))
        np.testing.assert_allclose(source.planck, expected_planck)

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_planck_matches_numpy_helpers(self) -> None:
        import torch

        from py2sess import planck_radiance_wavelength_torch, planck_radiance_wavenumber_torch

        temperature = torch.tensor([220.0, 250.0, 280.0], dtype=torch.float64)
        wavelength = torch.tensor([9.0, 10.0, 11.0], dtype=torch.float64)
        wavenumber = torch.tensor([700.0, 800.0, 900.0], dtype=torch.float64)
        wavelength_expected = np.array(
            [
                planck_radiance_wavelength(float(temp), float(wave))
                for temp, wave in zip(temperature.detach().numpy(), wavelength.detach().numpy())
            ]
        )
        wavenumber_expected = np.array(
            [
                planck_radiance_wavenumber(float(temp), float(wave))
                for temp, wave in zip(temperature.detach().numpy(), wavenumber.detach().numpy())
            ]
        )
        np.testing.assert_allclose(
            planck_radiance_wavelength_torch(temperature, wavelength).detach().numpy(),
            wavelength_expected,
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            planck_radiance_wavenumber_torch(temperature, wavenumber).detach().numpy(),
            wavenumber_expected,
            rtol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
