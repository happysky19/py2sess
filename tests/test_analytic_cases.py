from __future__ import annotations

import math
import unittest

import numpy as np

from tests.analytic_oracles import (
    lambertian_surface_fo_radiance,
    solar_fo_single_scatter_isotropic_one_layer,
    thermal_fo_single_layer_uniform_source,
    thermal_surface_only_up_profile,
    twostream_upward_flux_pair_from_isotropic_intensity,
)
from py2sess import TwoStreamEss, TwoStreamEssOptions


STREAM_VALUE = 1.0 / math.sqrt(3.0)


class AnalyticCaseTests(unittest.TestCase):
    def test_solar_zero_flux_returns_zero_everywhere(self) -> None:
        result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", output_levels=True, downwelling=True)
        ).forward(
            tau=np.array([0.2, 0.3]),
            ssa=np.array([0.5, 0.4]),
            g=np.array([0.2, 0.1]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 40.0],
            stream=STREAM_VALUE,
            fbeam=0.0,
            albedo=0.3,
            include_fo=True,
        )
        np.testing.assert_allclose(result.radiance_total, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.radiance_profile_total, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.fluxes_toa, 0.0, atol=1.0e-14)

    def test_thermal_zero_sources_return_zero(self) -> None:
        result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="thermal", output_levels=True, downwelling=True)
        ).forward(
            tau=np.array([0.2, 0.3]),
            ssa=np.zeros(2),
            g=np.zeros(2),
            z=np.array([2.0, 1.0, 0.0]),
            angles=20.0,
            stream=STREAM_VALUE,
            planck=np.zeros(3),
            surface_planck=0.0,
            emissivity=0.5,
            albedo=0.5,
            include_fo=True,
        )
        np.testing.assert_allclose(result.radiance_total, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.radiance_profile_total, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.fluxes_toa, 0.0, atol=1.0e-14)

    def test_solar_surface_only_matches_lambertian_formula(self) -> None:
        sza = 30.0
        albedo = 0.3
        result = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar")).forward_fo(
            tau=np.zeros(2),
            ssa=np.zeros(2),
            g=np.zeros(2),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[sza, 0.0, 0.0],
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=albedo,
            delta_m_truncation_factor=np.zeros(2),
        )
        expected = lambertian_surface_fo_radiance(
            fbeam=1.0,
            albedo=albedo,
            solar_zenith_degrees=sza,
        )
        np.testing.assert_allclose(result.radiance, [expected], atol=1.0e-12)

    def test_thermal_surface_only_profile_matches_formula(self) -> None:
        tau = np.array([0.1, 0.2])
        surface = 1.4
        user_angle_degrees = 20.0
        result = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="thermal")).forward_fo(
            tau=tau,
            ssa=np.zeros(2),
            g=np.zeros(2),
            z=np.array([2.0, 1.0, 0.0]),
            angles=user_angle_degrees,
            stream=STREAM_VALUE,
            planck=np.zeros(3),
            surface_planck=surface,
            emissivity=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros(2),
        )
        expected = thermal_surface_only_up_profile(
            tau,
            user_angle_degrees=user_angle_degrees,
            surface_planck=surface,
            emissivity=1.0,
        )
        np.testing.assert_allclose(result.radiance_up_profile[0], expected, atol=8.0e-5)

    def test_single_layer_solar_scatter_matches_closed_form(self) -> None:
        tau = 0.05
        ssa = 0.2
        sza = 30.0
        vza = 20.0
        result = TwoStreamEss(TwoStreamEssOptions(nlyr=1, mode="solar")).forward_fo(
            tau=np.array([tau]),
            ssa=np.array([ssa]),
            g=np.zeros(1),
            z=np.array([1.0, 0.0]),
            angles=[sza, vza, 0.0],
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros(1),
            fo_scatter_term=np.array([ssa]),
        )
        expected = solar_fo_single_scatter_isotropic_one_layer(
            tau=tau,
            omega=ssa,
            solar_zenith_degrees=sza,
            view_zenith_degrees=vza,
            fbeam=1.0,
        )
        np.testing.assert_allclose(result.radiance, [expected], rtol=2.0e-5, atol=1.0e-14)

    def test_thermal_uniform_source_and_flux_pair_identities(self) -> None:
        tau = 0.2
        omega = 0.0
        source = 1.3
        user_angle_degrees = 20.0
        result = TwoStreamEss(TwoStreamEssOptions(nlyr=1, mode="thermal")).forward_fo(
            tau=np.array([tau]),
            ssa=np.array([omega]),
            g=np.zeros(1),
            z=np.array([1.0, 0.0]),
            angles=user_angle_degrees,
            stream=STREAM_VALUE,
            planck=np.full(2, source),
            surface_planck=0.0,
            emissivity=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros(1),
        )
        expected = thermal_fo_single_layer_uniform_source(
            tau=tau,
            omega=omega,
            user_angle_degrees=user_angle_degrees,
            blackbody_value=source,
        )
        np.testing.assert_allclose(result.radiance, [expected], rtol=2.0e-5, atol=1.0e-12)

        mean_intensity, flux_up = twostream_upward_flux_pair_from_isotropic_intensity(
            intensity=source,
            stream=STREAM_VALUE,
        )
        self.assertAlmostEqual(mean_intensity, 0.5 * source)
        self.assertAlmostEqual(flux_up, 2.0 * math.pi * STREAM_VALUE * source)


if __name__ == "__main__":
    unittest.main()
