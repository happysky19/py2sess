from __future__ import annotations

import math
import unittest

import numpy as np

from py2sess.reference_cases import load_tir_benchmark_case, load_uv_benchmark_case
from py2sess.rtsolver.geometry import auxgeom_solar_obs, chapman_factors
from py2sess.optical.brdf_solar_obs import _ross_kernel as _solar_ross_kernel
from py2sess.optical.brdf_solar_obs import solar_obs_brdf_from_kernels
from py2sess.optical.brdf_thermal import _ross_kernel as _thermal_ross_kernel
from py2sess.optical.brdf_thermal import thermal_brdf_from_kernels
from py2sess.optical.delta_m import (
    default_delta_m_truncation_factor,
    delta_m_scale_optical_properties,
)
from py2sess.optical.planck import (
    _simpson_converged,
    planck_radiance_wavelength,
    planck_radiance_wavenumber,
    planck_radiance_wavenumber_band,
    thermal_source_from_temperature_profile,
)
from py2sess.optical.rayleigh import rayleigh_bodhaine
from py2sess.optical.surface_leaving import (
    morcasiwat_reflectance,
    seawater_refractive_index,
    surface_leaving_from_water,
)

GEOMETRY_RTOL = 1.0e-11
GEOMETRY_ATOL = 2.0e-11


class HelperParityTests(unittest.TestCase):
    def test_chapman_factors_regression(self) -> None:
        heights = np.array([3.0, 2.2, 1.4, 0.7, 0.0], dtype=float)
        expected = np.array(
            [
                [1.6614565464313844, 1.6610895422976520, 1.6607686508599673, 1.6604479805480710],
                [0.0, 1.6614565233907672, 1.6611353386694580, 1.6608143753981952],
                [0.0, 0.0, 1.6614794510031403, 1.6611582126771725],
                [0.0, 0.0, 0.0, 1.6614794333517637],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            chapman_factors(heights, 6371.0, 53.0),
            expected,
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        )

    def test_auxgeom_solar_obs_regression(self) -> None:
        x0 = np.cos(np.deg2rad(np.array([20.0, 60.0], dtype=float)))
        user_streams = np.cos(np.deg2rad(np.array([5.0, 45.0], dtype=float)))
        px11, pxsq, px0x, ulp = auxgeom_solar_obs(x0, user_streams, 1.0 / math.sqrt(3.0), True)
        self.assertAlmostEqual(px11, 0.5773502691896257)
        np.testing.assert_allclose(
            pxsq, np.array([0.3333333333333334, 0.3333333333333333], dtype=float)
        )
        np.testing.assert_allclose(
            px0x,
            np.array(
                [
                    [0.5425317875662492, 0.1396291388169096],
                    [0.2886751345948130, 0.3535533905932737],
                ],
                dtype=float,
            ),
        )
        np.testing.assert_allclose(ulp, np.array([-0.0616284167162192, -0.5], dtype=float))

    def test_uv_geometry_helpers_match_fixture(self) -> None:
        case = load_uv_benchmark_case()
        sza, vza, _raz = case.user_obsgeom
        x0 = np.cos(np.deg2rad(np.array([sza], dtype=float)))
        user_stream = np.cos(np.deg2rad(np.array([vza], dtype=float)))

        np.testing.assert_allclose(
            chapman_factors(case.heights, 6371.0, float(sza)),
            case.chapman,
            rtol=GEOMETRY_RTOL,
            atol=GEOMETRY_ATOL,
        )
        px11, pxsq, px0x, ulp = auxgeom_solar_obs(
            x0,
            user_stream,
            case.stream_value,
            True,
        )
        self.assertAlmostEqual(float(x0[0]), case.x0)
        self.assertAlmostEqual(float(user_stream[0]), case.user_stream)
        self.assertAlmostEqual(float(1.0 / user_stream[0]), case.user_secant)
        self.assertAlmostEqual(px11, case.px11)
        np.testing.assert_allclose(pxsq, case.pxsq, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        np.testing.assert_allclose(px0x[0], case.px0x, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        self.assertAlmostEqual(float(ulp[0]), case.ulp)

    def test_delta_m_scale_optical_properties_regression(self) -> None:
        tau = np.array([[0.2, 0.4, 0.7], [0.05, 0.1, 0.2]], dtype=float)
        omega = np.array([[0.9, 0.8, 0.7], [0.95, 0.6, 0.4]], dtype=float)
        asymm = np.array([[0.7, 0.6, 0.5], [0.2, 0.3, 0.4]], dtype=float)
        scaling = np.array([[0.2, 0.1, 0.05], [0.0, 0.1, 0.2]], dtype=float)
        delta_tau, omega_total, asymm_total = delta_m_scale_optical_properties(
            tau, omega, asymm, scaling
        )
        np.testing.assert_allclose(
            delta_tau,
            np.array([[0.1640, 0.3680, 0.6755], [0.05, 0.0940, 0.1840]], dtype=float),
        )
        np.testing.assert_allclose(
            omega_total,
            np.array(
                [
                    [0.8780487804878050, 0.7826086956521741, 0.6891191709844559],
                    [0.95, 0.5744680851063830, 0.3478260869565218],
                ],
                dtype=float,
            ),
        )
        np.testing.assert_allclose(
            asymm_total,
            np.array(
                [
                    [0.6250, 0.5555555555555556, 0.4736842105263158],
                    [0.2, 0.2222222222222222, 0.25],
                ],
                dtype=float,
            ),
        )

    def test_fixture_truncation_factors_are_explicit_optical_inputs(self) -> None:
        uv = load_uv_benchmark_case()
        tir = load_tir_benchmark_case()

        self.assertGreater(
            float(np.max(np.abs(uv.scaling - default_delta_m_truncation_factor(uv.asymm)))),
            1.0e-3,
        )
        self.assertGreater(
            float(
                np.max(np.abs(tir.d2s_scaling - default_delta_m_truncation_factor(tir.asymm_arr)))
            ),
            1.0e-3,
        )

    def test_solar_brdf_kernel_generation_regression(self) -> None:
        coeffs = solar_obs_brdf_from_kernels(
            kernel_specs=[
                {"which_brdf": 1, "factor": 0.2, "nstreams_brdf": 8},
                {"which_brdf": 2, "factor": 0.1, "nstreams_brdf": 8},
                {"which_brdf": 3, "factor": 0.05, "nstreams_brdf": 8},
            ],
            user_obsgeoms=np.array([[20.0, 10.0, 30.0], [60.0, 45.0, 170.0]], dtype=float),
            stream_value=1.0 / math.sqrt(3.0),
            n_geoms=2,
        )
        np.testing.assert_allclose(
            coeffs.brdf_f_0,
            np.array(
                [
                    [0.2589036739171707, -0.0353960164668122],
                    [0.4673643621922224, -0.0894308662748632],
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            coeffs.brdf_f,
            np.array([0.4091485401994730, -0.0848664477978274], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            coeffs.ubrdf_f,
            np.array(
                [
                    [0.2474226572159690, -0.0179186785097546],
                    [0.3388885476755556, -0.0737874968836241],
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_thermal_brdf_kernel_generation_regression(self) -> None:
        coeffs = thermal_brdf_from_kernels(
            kernel_specs=[
                {"which_brdf": 1, "factor": 0.3, "nstreams_brdf": 8, "stream_value": 0.5},
                {"which_brdf": 2, "factor": 0.1, "nstreams_brdf": 8, "stream_value": 0.5},
            ],
            user_angles=np.array([0.0, 55.0, 80.0], dtype=float),
            do_surface_emission=True,
        )
        self.assertAlmostEqual(coeffs.brdf_f, 0.6172584778749743)
        np.testing.assert_allclose(
            coeffs.ubrdf_f,
            np.array([0.3684853256372280, 0.5546405708618682, 1.5220074611559407], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(coeffs.emissivity, 0.3858475363873620)

    def test_ross_kernel_clamps_roundoff_both_sides(self) -> None:
        solar_value = _solar_ross_kernel(
            xi=1.0e-6,
            sxi=1.0,
            xj=1.0e-6,
            sxj=1.0,
            cphi=1.0 + 1.0e-15,
            thick=False,
        )
        thermal_value = _thermal_ross_kernel(
            xi=1.0e-6,
            sxi=1.0,
            xj=1.0e-6,
            sxj=1.0,
            cphi=1.0 + 1.0e-15,
            thick=True,
        )

        self.assertTrue(np.isfinite(solar_value))
        self.assertTrue(np.isfinite(thermal_value))

    def test_surface_leaving_regression(self) -> None:
        self.assertEqual(seawater_refractive_index(0.55, 34.3), (1.339, 1.96e-09))
        self.assertAlmostEqual(morcasiwat_reflectance(0.55, 0.2), 0.00875347764776946)
        coeffs = surface_leaving_from_water(
            n_beams=2,
            wavelength_microns=0.55,
            salinity_ppt=34.3,
            chlorophyll_mg_m3=0.2,
            do_isotropic=True,
        )
        np.testing.assert_allclose(
            coeffs.slterm_isotropic,
            np.array([0.0049030595467782, 0.0049030595467782], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(coeffs.slterm_f_0, np.zeros((2, 2), dtype=float))

    def test_planck_and_source_builders_regression(self) -> None:
        temperatures = np.array([210.0, 250.0, 290.0], dtype=float)
        np.testing.assert_allclose(
            planck_radiance_wavelength(temperatures, 11.0),
            np.array([1461607.4630400327, 3972817.0879451090, 8222035.1987223730], dtype=float),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            planck_radiance_wavenumber(temperatures, 950.0),
            np.array([1.52416035988e-04, 4.330071130164e-04, 9.248731341086e-04], dtype=float),
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            planck_radiance_wavenumber_band(temperatures, 800.0, 1200.0),
            np.array([5.439116824681846, 15.592570560191367, 33.825564060789596], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        source = thermal_source_from_temperature_profile(
            np.array([220.0, 240.0, 260.0, 280.0], dtype=float),
            292.0,
            wavenumber_band_cm_inv=(800.0, 1200.0),
        )
        np.testing.assert_allclose(
            source.planck,
            np.array(
                [
                    7.323563112133082,
                    12.365632666671635,
                    19.329655834530893,
                    28.427874121503780,
                ],
                dtype=float,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(source.surface_planck, 34.974800817351124)

    def test_planck_simpson_convergence_handles_zero_values(self) -> None:
        np.testing.assert_array_equal(
            _simpson_converged(np.array([0.0], dtype=float), np.array([0.0], dtype=float)),
            np.array([True]),
        )

    def test_rayleigh_rejects_peck_reeder_pole_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than 160 nm"):
            rayleigh_bodhaine(np.array([159.0], dtype=float))


if __name__ == "__main__":
    unittest.main()
