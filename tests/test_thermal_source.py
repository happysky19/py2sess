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
        self.assertEqual(source.planck.shape, (4,))
        self.assertGreater(source.surface_planck, source.planck[-1])

    def test_profile_helper_vectorizes_over_wavenumbers(self) -> None:
        level_temperature = np.array([220.0, 230.0, 240.0, 250.0], dtype=float)
        surface_temperature = 280.0
        wavenumber = np.array([700.0, 800.0, 900.0], dtype=float)
        source = thermal_source_from_temperature_profile(
            level_temperature,
            surface_temperature,
            wavenumber_cm_inv=wavenumber,
        )
        expected_planck = np.vstack(
            [planck_radiance_wavenumber(level_temperature, value) for value in wavenumber]
        )
        expected_surface = np.array(
            [planck_radiance_wavenumber(surface_temperature, value) for value in wavenumber]
        )
        self.assertEqual(source.planck.shape, (3, 4))
        self.assertEqual(source.surface_planck.shape, (3,))
        np.testing.assert_allclose(source.planck, expected_planck)
        np.testing.assert_allclose(source.surface_planck, expected_surface)

    def test_profile_helper_vectorizes_over_wavenumber_bands(self) -> None:
        level_temperature = np.array([220.0, 230.0, 240.0, 250.0], dtype=float)
        surface_temperature = 280.0
        bands = np.array([[899.5, 900.5], [900.5, 901.5], [901.5, 902.5]], dtype=float)
        source = thermal_source_from_temperature_profile(
            level_temperature,
            surface_temperature,
            wavenumber_band_cm_inv=bands,
        )
        expected_planck = np.vstack(
            [planck_radiance_wavenumber_band(level_temperature, low, high) for low, high in bands]
        )
        expected_surface = np.array(
            [planck_radiance_wavenumber_band(surface_temperature, low, high) for low, high in bands]
        )
        self.assertEqual(source.planck.shape, (3, 4))
        self.assertEqual(source.surface_planck.shape, (3,))
        np.testing.assert_allclose(source.planck, expected_planck, rtol=1.0e-14)
        np.testing.assert_allclose(source.surface_planck, expected_surface, rtol=1.0e-14)

    def test_profile_helper_reuses_uniform_sliding_band_grid(self) -> None:
        level_temperature = np.array([220.0, 230.0, 240.0, 250.0], dtype=float)
        surface_temperature = 280.0
        low = 899.5 + 0.001 * np.arange(8, dtype=float)
        bands = np.column_stack((low, low + 1.0))

        source = thermal_source_from_temperature_profile(
            level_temperature,
            surface_temperature,
            wavenumber_band_cm_inv=bands,
        )
        expected_planck = np.vstack(
            [planck_radiance_wavenumber_band(level_temperature, lo, hi) for lo, hi in bands]
        )
        expected_surface = np.array(
            [planck_radiance_wavenumber_band(surface_temperature, lo, hi) for lo, hi in bands]
        )

        np.testing.assert_allclose(source.planck, expected_planck, rtol=2.0e-13, atol=1.0e-14)
        np.testing.assert_allclose(
            source.surface_planck,
            expected_surface,
            rtol=2.0e-13,
            atol=1.0e-14,
        )

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
            source.planck, np.full(4, expected, dtype=float), rtol=0.0, atol=1.0e-12
        )
        self.assertAlmostEqual(source.surface_planck, float(expected), places=12)

    def test_torch_planck_matches_numpy_helpers(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        from py2sess import planck_radiance_wavelength_torch, planck_radiance_wavenumber_torch

        temperature = torch.tensor([220.0, 250.0, 280.0], dtype=torch.float64)
        wavelength = torch.tensor([9.0, 10.0, 11.0], dtype=torch.float64)
        wavenumber = torch.tensor([700.0, 800.0, 900.0], dtype=torch.float64)
        wavelength_expected = np.array(
            [
                planck_radiance_wavelength(float(temp), float(wave))
                for temp, wave in zip(temperature.detach().numpy(), wavelength.detach().numpy())
            ],
            dtype=float,
        )
        wavenumber_expected = np.array(
            [
                planck_radiance_wavenumber(float(temp), float(wave))
                for temp, wave in zip(temperature.detach().numpy(), wavenumber.detach().numpy())
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            planck_radiance_wavelength_torch(temperature, wavelength).detach().numpy(),
            wavelength_expected,
            rtol=1.0e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            planck_radiance_wavenumber_torch(temperature, wavenumber).detach().numpy(),
            wavenumber_expected,
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_torch_profile_builder_preserves_temperature_gradients(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        from py2sess import thermal_source_from_temperature_profile_torch

        level_temperature = torch.tensor(
            [220.0, 230.0, 240.0, 250.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        surface_temperature = torch.tensor(280.0, dtype=torch.float64, requires_grad=True)
        wavenumber = torch.tensor([700.0, 800.0, 900.0], dtype=torch.float64)
        source = thermal_source_from_temperature_profile_torch(
            level_temperature,
            surface_temperature,
            wavenumber_cm_inv=wavenumber,
        )
        self.assertEqual(tuple(source.planck.shape), (3, 4))
        self.assertEqual(tuple(source.surface_planck.shape), (3,))
        loss = source.planck.sum() + source.surface_planck.sum()
        loss.backward()
        self.assertIsNotNone(level_temperature.grad)
        self.assertIsNotNone(surface_temperature.grad)
        self.assertTrue(bool(torch.all(level_temperature.grad > 0.0)))
        self.assertGreater(float(surface_temperature.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
