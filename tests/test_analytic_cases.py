from __future__ import annotations

import math
import unittest

import numpy as np

from tests.analytic_oracles import (
    lambertian_surface_fo_radiance,
    solar_fo_single_scatter_isotropic_one_layer,
    thermal_atmosphere_only_down_profile,
    thermal_atmosphere_only_up_profile,
    thermal_fo_single_layer_uniform_source,
    thermal_surface_only_up_profile,
    twostream_upward_flux_pair_from_isotropic_intensity,
)
from py2sess import TwoStreamEss, TwoStreamEssOptions


STREAM_VALUE = 1.0 / math.sqrt(3.0)
THREE_LAYER_HEIGHT_GRID = np.array([3.0, 2.0, 1.0, 0.0], dtype=float)
THREE_LAYER_TAU = np.array([0.2, 0.3, 0.1], dtype=float)
THREE_LAYER_ZERO = np.zeros(3, dtype=float)
THREE_LAYER_ZERO_BB = np.zeros(4, dtype=float)
THREE_LAYER_ONE_BB = np.ones(4, dtype=float)


class BoundaryConditionAnalyticTests(unittest.TestCase):
    """Analytic checks for boundary-value identities and zero-source limits."""

    def test_solar_forward_zero_flux_returns_zero_everywhere(self) -> None:
        options = TwoStreamEssOptions(
            nlyr=3,
            mode="solar",
            output_levels=True,
            downwelling=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward(
            tau=THREE_LAYER_TAU,
            ssa=np.array([0.5, 0.4, 0.3], dtype=float),
            g=np.array([0.2, 0.1, 0.3], dtype=float),
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([[30.0, 20.0, 40.0]], dtype=float),
            stream=STREAM_VALUE,
            fbeam=0.0,
            albedo=0.3,
            delta_m_truncation_factor=np.array([0.01, 0.02, 0.03], dtype=float),
            include_fo=True,
        )
        np.testing.assert_allclose(
            result.intensity_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radiance_fo, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.combined_intensity_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_toa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_boa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_up, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_dn, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )

    def test_solar_fo_no_scattering_no_surface_returns_zero(self) -> None:
        options = TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_ZERO,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([[35.0, 50.0, 120.0]], dtype=float),
            stream=STREAM_VALUE,
            fbeam=2.5,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            geometry="pseudo_spherical",
        )
        np.testing.assert_allclose(
            result.radiance, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radiance_profile, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )

    def test_thermal_forward_zero_sources_returns_zero(self) -> None:
        options = TwoStreamEssOptions(
            nlyr=3,
            mode="thermal",
            output_levels=True,
            downwelling=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([20.0], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ZERO_BB,
            surface_planck=0.0,
            emissivity=1.0,
            include_fo=True,
        )
        np.testing.assert_allclose(
            result.intensity_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.intensity_boa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fo_thermal_total_up_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_toa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_boa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_up, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_dn, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )

    def test_thermal_forward_zero_sources_remain_zero_for_nonblack_surface(self) -> None:
        options = TwoStreamEssOptions(
            nlyr=3,
            mode="thermal",
            output_levels=True,
            downwelling=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([20.0], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.5,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ZERO_BB,
            surface_planck=0.0,
            emissivity=0.5,
            include_fo=True,
        )
        np.testing.assert_allclose(
            result.intensity_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.intensity_boa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fo_thermal_total_up_toa, np.zeros(1, dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_toa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.fluxes_boa, np.zeros((2, 1), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_up, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            result.radlevel_dn, np.zeros((1, 4), dtype=float), rtol=0.0, atol=1.0e-14
        )


class ClearSkyShortwaveAnalyticTests(unittest.TestCase):
    """Analytic checks for shortwave clear-sky surface and direct-source limits."""

    def test_solar_fo_surface_only_matches_lambertian_formula(self) -> None:
        options = TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True)
        solver = TwoStreamEss(options)
        albedo = 0.3
        sza = 30.0
        result = solver.forward_fo(
            tau=THREE_LAYER_ZERO,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([[sza, 0.0, 0.0]], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=albedo,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            geometry="pseudo_spherical",
        )
        expected = lambertian_surface_fo_radiance(
            fbeam=1.0, albedo=albedo, solar_zenith_degrees=sza
        )
        np.testing.assert_allclose(result.radiance, np.array([expected]), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            result.radiance_profile,
            np.full((1, 4), expected, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_solar_fo_oblique_surface_only_scales_with_flux_and_mu0(self) -> None:
        options = TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True)
        solver = TwoStreamEss(options)
        fbeam = 2.5
        albedo = 0.4
        sza = 35.0
        result = solver.forward_fo(
            tau=THREE_LAYER_ZERO,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([[sza, 50.0, 120.0]], dtype=float),
            stream=STREAM_VALUE,
            fbeam=fbeam,
            albedo=albedo,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            geometry="pseudo_spherical",
        )
        expected = lambertian_surface_fo_radiance(
            fbeam=fbeam,
            albedo=albedo,
            solar_zenith_degrees=sza,
        )
        np.testing.assert_allclose(result.radiance, np.array([expected]), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            result.radiance_profile,
            np.full((1, 4), expected, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_solar_fo_lambertian_surface_only_is_view_independent(self) -> None:
        options = TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_ZERO,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array(
                [
                    [35.0, 0.0, 0.0],
                    [35.0, 50.0, 120.0],
                ],
                dtype=float,
            ),
            stream=STREAM_VALUE,
            fbeam=2.5,
            albedo=0.4,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            geometry="pseudo_spherical",
        )
        expected = lambertian_surface_fo_radiance(
            fbeam=2.5,
            albedo=0.4,
            solar_zenith_degrees=35.0,
        )
        np.testing.assert_allclose(
            result.radiance, np.full(2, expected, dtype=float), rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result.radiance_profile,
            np.full((2, 4), expected, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )


class ClearSkyLongwaveAnalyticTests(unittest.TestCase):
    """Analytic checks for longwave pure-absorption and blackbody limits."""

    def test_thermal_fo_isothermal_absorbing_column_preserves_blackbody(self) -> None:
        options = TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([20.0], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ONE_BB,
            surface_planck=1.0,
            emissivity=1.0,
        )
        np.testing.assert_allclose(result.radiance_up_toa, np.array([1.0]), rtol=0.0, atol=5.0e-10)
        np.testing.assert_allclose(result.radiance_up_boa, np.array([1.0]), rtol=0.0, atol=5.0e-10)
        np.testing.assert_allclose(
            result.radiance_up_profile,
            np.ones((1, 4), dtype=float),
            rtol=0.0,
            atol=5.0e-10,
        )

    def test_thermal_full_isothermal_blackbody_has_exact_toa_flux_pair(self) -> None:
        options = TwoStreamEssOptions(
            nlyr=3,
            mode="thermal",
            output_levels=True,
            downwelling=True,
            additional_mvout=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([20.0], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ONE_BB,
            surface_planck=1.0,
            emissivity=1.0,
            include_fo=True,
        )
        expected_flux_toa = twostream_upward_flux_pair_from_isotropic_intensity(
            intensity=1.0,
            stream=STREAM_VALUE,
        )
        np.testing.assert_allclose(
            result.fluxes_toa[:, 0], expected_flux_toa, rtol=0.0, atol=2.0e-9
        )

    def test_thermal_fo_surface_only_matches_absorption_transmission_formula(self) -> None:
        user_angle_degrees = 20.0
        surface_planck = 5.0
        emissivity = 0.8
        options = TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=1.0 - emissivity,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ZERO_BB,
            surface_planck=surface_planck,
            emissivity=emissivity,
        )
        expected_profile = thermal_surface_only_up_profile(
            THREE_LAYER_TAU,
            user_angle_degrees=user_angle_degrees,
            surface_planck=surface_planck,
            emissivity=emissivity,
        )
        np.testing.assert_allclose(
            result.radiance_up_profile,
            expected_profile[np.newaxis, :],
            rtol=0.0,
            atol=8.0e-5,
        )
        np.testing.assert_allclose(
            result.radiance_up_toa,
            np.array([expected_profile[0]], dtype=float),
            rtol=0.0,
            atol=8.0e-5,
        )
        np.testing.assert_allclose(
            result.radiance_up_boa,
            np.array([expected_profile[-1]], dtype=float),
            rtol=0.0,
            atol=8.0e-5,
        )

    def test_thermal_fo_atmosphere_only_matches_absorption_emission_formula(self) -> None:
        user_angle_degrees = 20.0
        options = TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ONE_BB,
            surface_planck=0.0,
            emissivity=1.0,
        )
        expected_profile = thermal_atmosphere_only_up_profile(
            THREE_LAYER_TAU,
            user_angle_degrees=user_angle_degrees,
            blackbody_value=1.0,
        )
        np.testing.assert_allclose(
            result.radiance_up_profile,
            expected_profile[np.newaxis, :],
            rtol=0.0,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            result.radiance_up_toa,
            np.array([expected_profile[0]], dtype=float),
            rtol=0.0,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            result.radiance_up_boa,
            np.array([expected_profile[-1]], dtype=float),
            rtol=0.0,
            atol=2.0e-5,
        )

    def test_thermal_fo_atmosphere_only_downward_profile_matches_formula(self) -> None:
        user_angle_degrees = 20.0
        options = TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ONE_BB,
            surface_planck=0.0,
            emissivity=1.0,
        )
        expected_profile = thermal_atmosphere_only_down_profile(
            THREE_LAYER_TAU,
            user_angle_degrees=user_angle_degrees,
            blackbody_value=1.0,
        )
        np.testing.assert_allclose(
            result.intensity_atmos_dn_profile,
            expected_profile[np.newaxis, :],
            rtol=0.0,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            result.intensity_atmos_dn_toa,
            np.array([expected_profile[0]], dtype=float),
            rtol=0.0,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            result.intensity_atmos_dn_boa,
            np.array([expected_profile[-1]], dtype=float),
            rtol=0.0,
            atol=2.0e-5,
        )


class ScatteringAnalyticTests(unittest.TestCase):
    """Analytic checks for one-layer shortwave and longwave scattering cases."""

    def test_solar_fo_one_layer_isotropic_scattering_matches_closed_form(self) -> None:
        tau = 0.4
        omega = 0.6
        sza = 30.0
        vza = 20.0
        options = TwoStreamEssOptions(
            nlyr=1,
            mode="solar",
            output_levels=True,
            plane_parallel=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=np.array([tau], dtype=float),
            ssa=np.array([omega], dtype=float),
            g=np.array([0.0], dtype=float),
            z=np.array([1.0, 0.0], dtype=float),
            angles=np.array([[sza, vza, 0.0]], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros(1, dtype=float),
            geometry="pseudo_spherical",
        )
        expected = solar_fo_single_scatter_isotropic_one_layer(
            tau=tau,
            omega=omega,
            solar_zenith_degrees=sza,
            view_zenith_degrees=vza,
            fbeam=1.0,
        )
        np.testing.assert_allclose(result.radiance, np.array([expected]), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            result.radiance_profile,
            np.array([[expected, 0.0]], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_thermal_fo_one_layer_uniform_scattering_matches_closed_form(self) -> None:
        tau = 0.4
        omega = 0.3
        user_angle_degrees = 20.0
        blackbody_value = 2.0
        options = TwoStreamEssOptions(
            nlyr=1,
            mode="thermal",
            output_levels=True,
            downwelling=True,
            plane_parallel=True,
        )
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=np.array([tau], dtype=float),
            ssa=np.array([omega], dtype=float),
            g=np.array([0.0], dtype=float),
            z=np.array([1.0, 0.0], dtype=float),
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros(1, dtype=float),
            planck=np.array([blackbody_value, blackbody_value], dtype=float),
            surface_planck=0.0,
            emissivity=1.0,
        )
        expected = thermal_fo_single_layer_uniform_source(
            tau=tau,
            omega=omega,
            user_angle_degrees=user_angle_degrees,
            blackbody_value=blackbody_value,
        )
        np.testing.assert_allclose(
            result.radiance_up_toa, np.array([expected]), rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result.radiance_up_profile,
            np.array([[expected, 0.0]], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )


class ComplexAnalyticTests(unittest.TestCase):
    """Analytic checks that combine multiple exact properties in one case."""

    def test_thermal_fo_superposition_surface_plus_atmosphere_equals_total(self) -> None:
        user_angle_degrees = 20.0
        options = TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        surface_only = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.2,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=THREE_LAYER_ZERO_BB,
            surface_planck=5.0,
            emissivity=0.8,
        )
        atmosphere_only = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=np.full(4, 1.5, dtype=float),
            surface_planck=0.0,
            emissivity=1.0,
        )
        combined = solver.forward_fo(
            tau=THREE_LAYER_TAU,
            ssa=THREE_LAYER_ZERO,
            g=THREE_LAYER_ZERO,
            z=THREE_LAYER_HEIGHT_GRID,
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=0.2,
            delta_m_truncation_factor=THREE_LAYER_ZERO,
            planck=np.full(4, 1.5, dtype=float),
            surface_planck=5.0,
            emissivity=0.8,
        )
        np.testing.assert_allclose(
            combined.radiance_up_profile,
            surface_only.radiance_up_profile + atmosphere_only.radiance_up_profile,
            rtol=0.0,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            combined.intensity_atmos_dn_profile,
            atmosphere_only.intensity_atmos_dn_profile,
            rtol=0.0,
            atol=2.0e-10,
        )

    def test_thermal_fo_surface_only_irregular_multilayer_profile_matches_formula(self) -> None:
        tau_arr = np.array([0.05, 0.12, 0.27, 0.08, 0.19], dtype=float)
        user_angle_degrees = 37.0
        surface_planck = 2.7
        emissivity = 0.91
        options = TwoStreamEssOptions(nlyr=5, mode="thermal", output_levels=True, downwelling=True)
        solver = TwoStreamEss(options)
        result = solver.forward_fo(
            tau=tau_arr,
            ssa=np.zeros(5, dtype=float),
            g=np.zeros(5, dtype=float),
            z=np.arange(5, -1, -1, dtype=float),
            angles=np.array([user_angle_degrees], dtype=float),
            stream=STREAM_VALUE,
            fbeam=1.0,
            albedo=1.0 - emissivity,
            delta_m_truncation_factor=np.zeros(5, dtype=float),
            planck=np.zeros(6, dtype=float),
            surface_planck=surface_planck,
            emissivity=emissivity,
        )
        expected_profile = thermal_surface_only_up_profile(
            tau_arr,
            user_angle_degrees=user_angle_degrees,
            surface_planck=surface_planck,
            emissivity=emissivity,
        )
        np.testing.assert_allclose(
            result.radiance_up_profile,
            expected_profile[np.newaxis, :],
            rtol=0.0,
            atol=2.0e-4,
        )
