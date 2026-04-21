from __future__ import annotations

import unittest

import numpy as np

from py2sess import load_tir_benchmark_case, load_uv_benchmark_case
from py2sess.core.backend import has_torch, to_numpy
from py2sess.core.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.core.solar_obs_batch_numpy import solve_solar_obs_batch_numpy
from py2sess.core.thermal_batch_numpy import solve_thermal_batch_numpy


def _relative_diff(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Returns a stable elementwise relative difference."""
    scale = np.maximum(np.abs(reference), 1.0e-15)
    return np.abs(value - reference) / scale


def _assert_max_rel(
    testcase: unittest.TestCase, value: np.ndarray, reference: np.ndarray, limit: float
) -> None:
    """Checks the largest relative difference against a scalar limit."""
    testcase.assertLessEqual(float(np.max(_relative_diff(value, reference))), limit)


class ReferenceCaseTests(unittest.TestCase):
    def test_tir_fixture_matches_saved_components_and_total(self) -> None:
        case = load_tir_benchmark_case()
        result = solve_thermal_batch_numpy(
            tau_arr=case.tau_arr,
            omega_arr=case.omega_arr,
            asymm_arr=case.asymm_arr,
            d2s_scaling=case.d2s_scaling,
            thermal_bb_input=case.thermal_bb_input,
            surfbb=case.surfbb,
            albedo=case.albedo,
            emissivity=case.emissivity,
            heights=case.heights,
            user_angle_degrees=case.user_angle,
            stream_value=case.stream_value,
        )
        total = result.two_stream_toa + result.fo_total_up_toa
        _assert_max_rel(self, result.two_stream_toa, case.ref_2s, 1.0e-5)
        _assert_max_rel(self, result.fo_total_up_toa, case.ref_fo, 1.0e-5)
        _assert_max_rel(self, total, case.ref_total, 5.0e-4)

    def test_tir_torch_matches_numpy_component_split(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        from py2sess.core.thermal_batch_torch import solve_thermal_batch_torch

        case = load_tir_benchmark_case()
        numpy_result = solve_thermal_batch_numpy(
            tau_arr=case.tau_arr,
            omega_arr=case.omega_arr,
            asymm_arr=case.asymm_arr,
            d2s_scaling=case.d2s_scaling,
            thermal_bb_input=case.thermal_bb_input,
            surfbb=case.surfbb,
            albedo=case.albedo,
            emissivity=case.emissivity,
            heights=case.heights,
            user_angle_degrees=case.user_angle,
            stream_value=case.stream_value,
        )
        result = solve_thermal_batch_torch(
            tau_arr=case.tau_arr,
            omega_arr=case.omega_arr,
            asymm_arr=case.asymm_arr,
            d2s_scaling=case.d2s_scaling,
            thermal_bb_input=case.thermal_bb_input,
            surfbb=case.surfbb,
            albedo=case.albedo,
            emissivity=case.emissivity,
            heights=case.heights,
            user_angle_degrees=case.user_angle,
            stream_value=case.stream_value,
            device="cpu",
        )
        two_stream = to_numpy(result.two_stream_toa)
        fo = to_numpy(result.fo_total_up_toa)
        np.testing.assert_allclose(
            two_stream, numpy_result.two_stream_toa, rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(fo, numpy_result.fo_total_up_toa, rtol=1.0e-12, atol=1.0e-12)

    def test_uv_fixture_matches_saved_components_and_total(self) -> None:
        case = load_uv_benchmark_case()
        fo_precomputed = fo_solar_obs_batch_precompute(
            user_obsgeom=case.user_obsgeom,
            heights=case.heights,
            earth_radius=6371.0,
            nfine=3,
        )
        fo = solve_fo_solar_obs_eps_batch_numpy(
            tau=case.tau,
            omega=case.omega,
            scaling=case.scaling,
            albedo=case.albedo,
            flux_factor=case.flux_factor,
            exact_scatter=case.fo_exact_scatter,
            precomputed=fo_precomputed,
        )
        two_stream = solve_solar_obs_batch_numpy(
            tau=case.tau,
            omega=case.omega,
            asymm=case.asymm,
            scaling=case.scaling,
            albedo=case.albedo,
            flux_factor=case.flux_factor,
            stream_value=case.stream_value,
            chapman=case.chapman,
            x0=case.x0,
            user_stream=case.user_stream,
            user_secant=case.user_secant,
            azmfac=case.azmfac,
            px11=case.px11,
            pxsq=case.pxsq,
            px0x=case.px0x,
            ulp=case.ulp,
        )
        _assert_max_rel(self, two_stream, case.ref_2s, 2.0e-6)
        _assert_max_rel(self, fo, case.ref_fo, 3.0e-6)
        _assert_max_rel(self, two_stream + fo, case.ref_total, 3.0e-6)

    def test_uv_torch_matches_numpy_2s(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        from py2sess.core.solar_obs_batch_torch import solve_solar_obs_batch_torch

        case = load_uv_benchmark_case()
        numpy_two_stream = solve_solar_obs_batch_numpy(
            tau=case.tau,
            omega=case.omega,
            asymm=case.asymm,
            scaling=case.scaling,
            albedo=case.albedo,
            flux_factor=case.flux_factor,
            stream_value=case.stream_value,
            chapman=case.chapman,
            x0=case.x0,
            user_stream=case.user_stream,
            user_secant=case.user_secant,
            azmfac=case.azmfac,
            px11=case.px11,
            pxsq=case.pxsq,
            px0x=case.px0x,
            ulp=case.ulp,
        )
        two_stream = to_numpy(
            solve_solar_obs_batch_torch(
                tau=case.tau,
                omega=case.omega,
                asymm=case.asymm,
                scaling=case.scaling,
                albedo=case.albedo,
                flux_factor=case.flux_factor,
                stream_value=case.stream_value,
                chapman=case.chapman,
                x0=case.x0,
                user_stream=case.user_stream,
                user_secant=case.user_secant,
                azmfac=case.azmfac,
                px11=case.px11,
                pxsq=case.pxsq,
                px0x=case.px0x,
                ulp=case.ulp,
                device="cpu",
            )
        )
        np.testing.assert_allclose(two_stream, numpy_two_stream, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
