from __future__ import annotations

import unittest

import numpy as np

from py2sess.optical.scene import (
    aerosol_components_from_tables,
    atmospheric_profile_from_levels,
    build_scene_layer_optical_properties,
    build_scene_opacity_components,
    gas_absorption_tau_from_cross_sections,
    rayleigh_scattering_tau_from_air_columns,
)
from py2sess.optical.rayleigh import rayleigh_bodhaine


class OpticalSceneTests(unittest.TestCase):
    def test_rayleigh_bodhaine_matches_fortran_formula_values(self) -> None:
        rayleigh = rayleigh_bodhaine(np.array([300.0, 400.0, 500.0, 550.0, 1000.0]))

        np.testing.assert_allclose(
            rayleigh.cross_section,
            np.array(
                [
                    5.6524611694259e-26,
                    1.6738176053697816e-26,
                    6.661314194223813e-27,
                    4.510510396388468e-27,
                    4.0131596194371946e-28,
                ]
            ),
            rtol=0.0,
            atol=1.0e-39,
        )
        np.testing.assert_allclose(
            rayleigh.depolarization,
            np.array(
                [
                    0.03255221352837535,
                    0.02966853942251837,
                    0.02859921775110769,
                    0.028303524147044047,
                    0.02743789237994326,
                ]
            ),
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_atmospheric_profile_matches_geocape_profile_setter_formula(self) -> None:
        pressure = np.array([100.0, 300.0, 1000.0])
        temperature = np.array([220.0, 250.0, 290.0])
        gas_vmr = np.array([[1.0e-6, 2.0e-6], [3.0e-6, 4.0e-6], [5.0e-6, 6.0e-6]])

        profile = atmospheric_profile_from_levels(
            pressure_hpa=pressure,
            temperature_k=temperature,
            gas_vmr=gas_vmr,
        )

        ccon = -9.81 * 28.9 / 8314.0 * 500.0
        expected_heights = np.empty_like(pressure)
        expected_heights[-1] = 0.0
        for n in range(pressure.size - 1, 0, -1):
            avit = 1.0 / temperature[n - 1] + 1.0 / temperature[n]
            expected_heights[n - 1] = (
                expected_heights[n] - np.log(pressure[n] / pressure[n - 1]) / avit / ccon
            )

        rho_stand = 2.68675e19
        rho_zero = rho_stand * 273.15 / 1013.25
        expected_air_density = 1.0e5 * rho_zero * pressure / temperature
        expected_air_columns = (
            0.5
            * (expected_air_density[:-1] + expected_air_density[1:])
            * (expected_heights[:-1] - expected_heights[1:])
        )

        np.testing.assert_allclose(profile.heights_km, expected_heights)
        np.testing.assert_allclose(profile.air_density_per_km, expected_air_density)
        np.testing.assert_allclose(profile.air_columns, expected_air_columns)
        np.testing.assert_allclose(
            profile.gas_density_per_km, expected_air_density[:, None] * gas_vmr
        )

    def test_gas_absorption_tau_integrates_constant_and_level_cross_sections(self) -> None:
        heights = np.array([2.0, 1.0, 0.0])
        gas_density = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        constant = gas_absorption_tau_from_cross_sections(
            heights_km=heights,
            gas_density_per_km=gas_density,
            cross_sections=np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        np.testing.assert_allclose(constant, np.array([[0.8, 1.4], [1.8, 3.2]]))

        level = gas_absorption_tau_from_cross_sections(
            heights_km=heights,
            gas_density_per_km=gas_density,
            cross_sections=np.array([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]]),
        )
        expected = np.array(
            [
                [
                    0.5 * ((1.0 * 0.1 + 2.0 * 0.2) + (3.0 * 0.2 + 4.0 * 0.3)),
                    0.5 * ((3.0 * 0.2 + 4.0 * 0.3) + (5.0 * 0.3 + 6.0 * 0.4)),
                ]
            ]
        )
        np.testing.assert_allclose(level, expected)

    def test_rayleigh_scattering_tau_uses_air_columns(self) -> None:
        air_columns = np.array([1.0e24, 2.0e24])
        tau, depol = rayleigh_scattering_tau_from_air_columns(
            wavelengths_nm=np.array([500.0, 550.0]),
            air_columns=air_columns,
        )
        rayleigh = rayleigh_bodhaine(np.array([500.0, 550.0]))

        np.testing.assert_allclose(tau, rayleigh.cross_section[:, None] * air_columns[None, :])
        np.testing.assert_allclose(depol, rayleigh.depolarization)

    def test_aerosol_components_from_tables_matches_fortran_stage2_formula(self) -> None:
        components = aerosol_components_from_tables(
            wavelengths_microns=np.array([0.4, 0.6]),
            select_wavelength_microns=0.5,
            aerosol_loadings=np.array([[2.0, 4.0], [1.0, 8.0]]),
            aerosol_wavelengths_microns=np.array([0.3, 0.5, 0.7]),
            aerosol_bulk_iops=np.array(
                [
                    [[10.0, 20.0], [20.0, 40.0], [30.0, 80.0]],
                    [[5.0, 10.0], [10.0, 20.0], [15.0, 40.0]],
                ]
            ),
        )

        np.testing.assert_allclose(
            components.extinction_tau,
            np.array(
                [
                    [[1.5, 3.0], [0.75, 6.0]],
                    [[2.5, 6.0], [1.25, 12.0]],
                ]
            ),
        )
        np.testing.assert_allclose(
            components.scattering_tau,
            np.array(
                [
                    [[0.75, 1.5], [0.375, 3.0]],
                    [[1.25, 3.0], [0.625, 6.0]],
                ]
            ),
        )

    def test_scene_layer_builder_combines_profile_components(self) -> None:
        profile = atmospheric_profile_from_levels(
            pressure_hpa=np.array([100.0, 500.0, 1000.0]),
            temperature_k=np.array([220.0, 260.0, 290.0]),
            gas_vmr=np.array([[1.0e-8], [2.0e-8], [3.0e-8]]),
            heights_km=np.array([2.0, 1.0, 0.0]),
        )
        scene = build_scene_layer_optical_properties(
            wavelengths_nm=np.array([500.0, 600.0]),
            profile=profile,
            gas_cross_sections=np.array([[1.0e-22], [2.0e-22]]),
            aerosol_loadings=np.array([[2.0], [1.0]]),
            aerosol_wavelengths_microns=np.array([0.4, 0.5, 0.7]),
            aerosol_bulk_iops=np.array(
                [
                    [[10.0], [20.0], [40.0]],
                    [[5.0], [10.0], [20.0]],
                ]
            ),
            aerosol_select_wavelength_microns=0.5,
        )

        total_tau = (
            scene.gas_absorption_tau
            + scene.rayleigh_scattering_tau
            + scene.aerosol_extinction_tau.sum(axis=-1)
        )
        scattering_tau = scene.rayleigh_scattering_tau + scene.aerosol_scattering_tau.sum(axis=-1)
        np.testing.assert_allclose(scene.layer.tau, total_tau)
        np.testing.assert_allclose(
            scene.layer.ssa,
            np.divide(
                scattering_tau, total_tau, out=np.zeros_like(total_tau), where=total_tau > 0.0
            ),
        )
        self.assertEqual(scene.layer.tau.shape, (2, 2))
        self.assertEqual(scene.depol.shape, (2,))

    def test_scene_opacity_components_replace_createprops_layer_quantities(self) -> None:
        profile = atmospheric_profile_from_levels(
            pressure_hpa=np.array([100.0, 500.0, 1000.0]),
            temperature_k=np.array([220.0, 260.0, 290.0]),
            gas_vmr=np.array([[1.0e-8], [2.0e-8], [3.0e-8]]),
            heights_km=np.array([2.0, 1.0, 0.0]),
        )

        components = build_scene_opacity_components(
            wavelengths_nm=np.array([500.0, 600.0]),
            profile=profile,
            gas_cross_sections=np.array([[1.0e-22], [2.0e-22]]),
            aerosol_loadings=np.array([[2.0], [1.0]]),
            aerosol_wavelengths_microns=np.array([0.4, 0.5, 0.7]),
            aerosol_bulk_iops=np.array(
                [
                    [[10.0], [20.0], [40.0]],
                    [[5.0], [10.0], [20.0]],
                ]
            ),
            aerosol_select_wavelength_microns=0.5,
        )
        layer = components.layer_properties()

        scattering_tau = components.rayleigh_scattering_tau + components.aerosol_scattering_tau.sum(
            axis=-1
        )
        total_tau = (
            components.gas_absorption_tau
            + components.rayleigh_scattering_tau
            + components.aerosol_extinction_tau.sum(axis=-1)
        )
        np.testing.assert_allclose(layer.tau, total_tau)
        np.testing.assert_allclose(
            layer.ssa,
            np.divide(
                scattering_tau, total_tau, out=np.zeros_like(total_tau), where=total_tau > 0.0
            ),
        )
        np.testing.assert_allclose(
            layer.rayleigh_fraction,
            np.divide(
                components.rayleigh_scattering_tau,
                scattering_tau,
                out=np.zeros_like(scattering_tau),
                where=scattering_tau > 0.0,
            ),
        )
        np.testing.assert_allclose(
            layer.aerosol_fraction,
            np.divide(
                components.aerosol_scattering_tau,
                scattering_tau[..., None],
                out=np.zeros_like(components.aerosol_scattering_tau),
                where=scattering_tau[..., None] > 0.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
