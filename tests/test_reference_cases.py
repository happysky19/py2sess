from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    load_tir_benchmark_case,
    load_uv_benchmark_case,
)
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

    def test_public_forward_tir_fixture_matches_batch_kernel(self) -> None:
        case = load_tir_benchmark_case()
        kernel = solve_thermal_batch_numpy(
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
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=case.n_layers, mode="thermal"))
        public = solver.forward(
            tau=case.tau_arr,
            ssa=case.omega_arr,
            g=case.asymm_arr,
            z=case.heights,
            angles=case.user_angle,
            stream=case.stream_value,
            albedo=case.albedo,
            delta_m_truncation_factor=case.d2s_scaling,
            planck=case.thermal_bb_input,
            surface_planck=case.surfbb,
            emissivity=case.emissivity,
            include_fo=True,
        )
        np.testing.assert_allclose(public.radiance_2s, kernel.two_stream_toa)
        np.testing.assert_allclose(public.radiance_fo, kernel.fo_total_up_toa)
        np.testing.assert_allclose(public.radiance_total, kernel.total_toa)

    def test_public_forward_tir_fixture_torch_matches_batch_kernel(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        from py2sess.core.thermal_batch_torch import solve_thermal_batch_torch

        case = load_tir_benchmark_case()
        kernel = solve_thermal_batch_torch(
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
        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=case.n_layers,
                mode="thermal",
                backend="torch",
                torch_enable_grad=False,
            )
        )
        public = solver.forward(
            tau=case.tau_arr,
            ssa=case.omega_arr,
            g=case.asymm_arr,
            z=case.heights,
            angles=case.user_angle,
            stream=case.stream_value,
            albedo=case.albedo,
            delta_m_truncation_factor=case.d2s_scaling,
            planck=case.thermal_bb_input,
            surface_planck=case.surfbb,
            emissivity=case.emissivity,
            include_fo=True,
        )
        np.testing.assert_allclose(
            to_numpy(public.radiance_2s), to_numpy(kernel.two_stream_toa), rtol=1.0e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            to_numpy(public.radiance_fo),
            to_numpy(kernel.fo_total_up_toa),
            rtol=1.0e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            to_numpy(public.radiance_total), to_numpy(kernel.total_toa), rtol=1.0e-12, atol=1e-12
        )

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

    def test_public_forward_uv_fixture_matches_batch_kernel(self) -> None:
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
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=case.n_layers, mode="solar"))
        public = solver.forward(
            tau=case.tau,
            ssa=case.omega,
            g=case.asymm,
            z=case.heights,
            angles=case.user_obsgeom,
            stream=case.stream_value,
            fbeam=case.flux_factor,
            albedo=case.albedo,
            delta_m_truncation_factor=case.scaling,
            include_fo=True,
            fo_scatter_term=case.fo_exact_scatter,
        )
        np.testing.assert_allclose(public.radiance_2s, two_stream)
        np.testing.assert_allclose(public.radiance_fo, fo)
        np.testing.assert_allclose(public.radiance_total, two_stream + fo)

    def test_public_forward_uv_fixture_torch_matches_batch_kernel(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
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
        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=case.n_layers,
                mode="solar",
                backend="torch",
                torch_enable_grad=False,
            )
        )
        public = solver.forward(
            tau=case.tau,
            ssa=case.omega,
            g=case.asymm,
            z=case.heights,
            angles=case.user_obsgeom,
            stream=case.stream_value,
            fbeam=case.flux_factor,
            albedo=case.albedo,
            delta_m_truncation_factor=case.scaling,
            include_fo=True,
            fo_scatter_term=case.fo_exact_scatter,
        )
        np.testing.assert_allclose(
            to_numpy(public.radiance_2s), two_stream, rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(to_numpy(public.radiance_fo), fo, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(
            to_numpy(public.radiance_total), two_stream + fo, rtol=1.0e-12, atol=1.0e-12
        )

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

    def test_uv_torch_matches_numpy_fo(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        from py2sess.core.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch

        case = load_uv_benchmark_case()
        fo_precomputed = fo_solar_obs_batch_precompute(
            user_obsgeom=case.user_obsgeom,
            heights=case.heights,
            earth_radius=6371.0,
            nfine=3,
        )
        numpy_fo = solve_fo_solar_obs_eps_batch_numpy(
            tau=case.tau,
            omega=case.omega,
            scaling=case.scaling,
            albedo=case.albedo,
            flux_factor=case.flux_factor,
            exact_scatter=case.fo_exact_scatter,
            precomputed=fo_precomputed,
        )
        torch_fo = to_numpy(
            solve_fo_solar_obs_eps_batch_torch(
                tau=case.tau,
                omega=case.omega,
                scaling=case.scaling,
                albedo=case.albedo,
                flux_factor=case.flux_factor,
                exact_scatter=case.fo_exact_scatter,
                precomputed=fo_precomputed,
                device="cpu",
            )
        )
        np.testing.assert_allclose(torch_fo, numpy_fo, rtol=1.0e-12, atol=1.0e-12)

    def test_uv_torch_fo_batch_supports_autograd(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        from py2sess.core.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch

        case = load_uv_benchmark_case()
        rows = 4
        fo_precomputed = fo_solar_obs_batch_precompute(
            user_obsgeom=case.user_obsgeom,
            heights=case.heights,
            earth_radius=6371.0,
            nfine=3,
        )
        tau = torch.tensor(case.tau[:rows], dtype=torch.float64, requires_grad=True)
        albedo = torch.tensor(case.albedo[:rows], dtype=torch.float64, requires_grad=True)
        fo = solve_fo_solar_obs_eps_batch_torch(
            tau=tau,
            omega=case.omega[:rows],
            scaling=case.scaling[:rows],
            albedo=albedo,
            flux_factor=case.flux_factor[:rows],
            exact_scatter=case.fo_exact_scatter[:rows],
            precomputed=fo_precomputed,
            device="cpu",
        )
        fo.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertIsNotNone(albedo.grad)
        self.assertTrue(torch.isfinite(tau.grad).all().item())
        self.assertTrue(torch.isfinite(albedo.grad).all().item())
        self.assertGreater(float(torch.abs(tau.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(albedo.grad).sum()), 0.0)

    def test_public_forward_uv_torch_batch_supports_autograd(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        case = load_uv_benchmark_case()
        rows = 4
        tau = torch.tensor(case.tau[:rows], dtype=torch.float64, requires_grad=True)
        albedo = torch.tensor(case.albedo[:rows], dtype=torch.float64, requires_grad=True)
        solver = TwoStreamEss(
            TwoStreamEssOptions(nlyr=case.n_layers, mode="solar", backend="torch")
        )
        result = solver.forward(
            tau=tau,
            ssa=case.omega[:rows],
            g=case.asymm[:rows],
            z=case.heights,
            angles=case.user_obsgeom,
            stream=case.stream_value,
            fbeam=case.flux_factor[:rows],
            albedo=albedo,
            delta_m_truncation_factor=case.scaling[:rows],
            include_fo=True,
            fo_scatter_term=case.fo_exact_scatter[:rows],
        )
        result.radiance_total.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertIsNotNone(albedo.grad)
        self.assertTrue(torch.isfinite(tau.grad).all().item())
        self.assertTrue(torch.isfinite(albedo.grad).all().item())
        self.assertGreater(float(torch.abs(tau.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(albedo.grad).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
