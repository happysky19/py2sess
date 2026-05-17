from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from py2sess.optical.planck import planck_radiance_wavelength
from py2sess.optical.solar_reference import (
    ASTRONOMICAL_UNIT_M,
    IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
    IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2,
    OcoSolarModel,
    PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_S,
    ToonSolarReference,
    solar_planck_irradiance_w_m2_um,
    solar_planck_continuum_ratio,
)


class SolarReferenceTests(unittest.TestCase):
    def test_toon_reference_interpolates_wavenumber_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solar.out"
            path.write_text(
                "\n".join(
                    [
                        "           3           2",
                        " solar_test.out",
                        " Wavenumber  Transmittance",
                        " 1000.00 0.20",
                        " 2000.00 0.60",
                        " 3000.00 1.00",
                        "",
                    ]
                )
            )

            reference = ToonSolarReference.from_file(path)
            values = reference.at_wavelength_um(np.array([10.0, 5.0, 10.0 / 3.0]))

        np.testing.assert_allclose(values, [0.20, 0.60, 1.00])

    def test_toon_reference_accepts_scalar_wavelength(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solar.out"
            path.write_text(
                "\n".join(
                    [
                        "           2           2",
                        " solar_test.out",
                        " Wavenumber  Transmittance",
                        " 1000.00 0.20",
                        " 2000.00 0.60",
                        "",
                    ]
                )
            )

            values = ToonSolarReference.from_file(path).at_wavelength_um(5.0)

        self.assertEqual(values.shape, (1,))
        np.testing.assert_allclose(values, [0.60])

    def test_toon_reference_can_be_reused_without_reloading_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solar.out"
            path.write_text(
                "\n".join(
                    [
                        "           3           2",
                        " solar_test.out",
                        " Wavenumber  Transmittance",
                        " 1000.00 0.20",
                        " 2000.00 0.60",
                        "",
                    ]
                )
            )

            reference = ToonSolarReference.from_file(path)

        np.testing.assert_allclose(reference.at_wavelength_um(np.array([10.0, 5.0])), [0.20, 0.60])

    def test_toon_reference_rejects_uncovered_wavelengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solar.out"
            path.write_text(
                "\n".join(
                    [
                        "           3           2",
                        " solar_test.out",
                        " Wavenumber  Transmittance",
                        " 1000.00 0.20",
                        " 2000.00 0.60",
                        "",
                    ]
                )
            )

            with self.assertRaisesRegex(ValueError, "does not cover"):
                ToonSolarReference.from_file(path).at_wavelength_um(np.array([2.0]))

    def test_planck_continuum_ratio_is_reference_normalized(self) -> None:
        wavelengths = np.array([4.0, 5.0, 6.0])

        ratio = solar_planck_continuum_ratio(wavelengths, reference_wavelength_um=5.0)

        self.assertEqual(IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K, 5772.0)
        self.assertAlmostEqual(float(ratio[1]), 1.0)
        self.assertGreater(float(ratio[0]), 1.0)
        self.assertLess(float(ratio[2]), 1.0)

    def test_planck_helpers_accept_scalar_wavelength(self) -> None:
        ratio = solar_planck_continuum_ratio(1.615, reference_wavelength_um=1.615)
        irradiance = solar_planck_irradiance_w_m2_um(1.615)

        self.assertEqual(ratio.shape, (1,))
        self.assertEqual(irradiance.shape, (1,))
        self.assertAlmostEqual(float(ratio[0]), 1.0)
        self.assertGreater(float(irradiance[0]), 0.0)

    def test_planck_continuum_ratio_matches_wavelength_planck_function(self) -> None:
        wavelengths = np.array([0.755, 0.77, 1.60, 1.615, 2.05, 2.06, 2.08])
        reference = 1.615

        ratio = solar_planck_continuum_ratio(wavelengths, reference_wavelength_um=reference)
        expected = planck_radiance_wavelength(
            IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
            wavelengths,
        ) / planck_radiance_wavelength(
            IAU_NOMINAL_SOLAR_EFFECTIVE_TEMPERATURE_K,
            reference,
        )

        np.testing.assert_allclose(ratio, expected, rtol=1e-12, atol=0.0)

    def test_planck_irradiance_ratio_matches_continuum_ratio(self) -> None:
        wavelengths = np.array([1.60, 1.615, 1.63])

        irradiance = solar_planck_irradiance_w_m2_um(wavelengths)
        ratio = irradiance / solar_planck_irradiance_w_m2_um(np.array([1.615]))[0]

        np.testing.assert_allclose(
            ratio,
            solar_planck_continuum_ratio(wavelengths, reference_wavelength_um=1.615),
            rtol=1e-12,
            atol=0.0,
        )

    def test_planck_irradiance_scales_with_solar_distance(self) -> None:
        at_one_au = solar_planck_irradiance_w_m2_um(np.array([1.615]))[0]
        at_two_au = solar_planck_irradiance_w_m2_um(
            np.array([1.615]),
            observer_distance_m=2.0 * ASTRONOMICAL_UNIT_M,
        )[0]

        self.assertEqual(IAU_NOMINAL_TOTAL_SOLAR_IRRADIANCE_W_M2, 1361.0)
        self.assertAlmostEqual(float(at_one_au / at_two_au), 4.0)

    def test_toon_reference_can_include_relative_planck_continuum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solar.out"
            path.write_text(
                "\n".join(
                    [
                        "           3           2",
                        " solar_test.out",
                        " Wavenumber  Transmittance",
                        " 1000.00 1.00",
                        " 2000.00 1.00",
                        " 3000.00 1.00",
                        "",
                    ]
                )
            )
            wavelengths = np.array([10.0, 5.0, 10.0 / 3.0])

            values = ToonSolarReference.from_file(path).at_wavelength_um(
                wavelengths,
                planck_continuum_reference_um=5.0,
            )

        np.testing.assert_allclose(
            values,
            solar_planck_continuum_ratio(wavelengths, reference_wavelength_um=5.0),
        )

    def test_oco_solar_model_combines_continuum_absorption_and_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "l2_solar_model.h5"
            with h5py.File(path, "w") as handle:
                for band_index in range(1, 4):
                    continuum = handle.create_group(f"Solar/Continuum/Continuum_{band_index}")
                    continuum.create_dataset("wavenumber", data=np.array([1000.0, 2000.0]))
                    continuum.create_dataset("spectrum", data=np.array([2.0e21, 4.0e21]))
                    absorption = handle.create_group(f"Solar/Absorption/Absorption_{band_index}")
                    absorption.create_dataset("wavenumber", data=np.array([1000.0, 2000.0]))
                    absorption.create_dataset("spectrum", data=np.array([0.5, 1.0]))

            model = OcoSolarModel.from_hdf(path)
            wavelength = np.array([5.0])
            photon = model.photon_irradiance_m2_um(
                wavelength,
                band_index=1,
                observer_distance_m=2.0 * ASTRONOMICAL_UNIT_M,
            )
            energy = model.energy_irradiance_w_m2_um(
                wavelength,
                band_index=1,
                observer_distance_m=2.0 * ASTRONOMICAL_UNIT_M,
            )

        self.assertAlmostEqual(float(photon[0]), 1.0e21)
        expected_energy = photon[0] * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / 5.0e-6
        self.assertAlmostEqual(float(energy[0]), float(expected_energy))


if __name__ == "__main__":
    unittest.main()
