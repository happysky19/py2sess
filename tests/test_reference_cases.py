from __future__ import annotations

from importlib.resources import as_file, files
import unittest

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions, thermal_source_from_temperature_profile
from py2sess.optical.phase import build_solar_fo_scatter_term, build_two_stream_phase_inputs
from py2sess.reference_cases import load_tir_benchmark_case, load_uv_benchmark_case
from py2sess.rtsolver.backend import has_torch
from py2sess.rtsolver.fo_solar_obs_batch_numpy import (
    fo_solar_obs_batch_precompute,
    solve_fo_solar_obs_eps_batch_numpy,
)
from py2sess.rtsolver.solar_obs_batch_numpy import solve_solar_obs_batch_numpy
from py2sess.rtsolver.thermal_batch_numpy import solve_thermal_batch_numpy


def _relative_diff(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(reference), 1.0e-15)
    return np.abs(value - reference) / scale


def _assert_max_rel(
    testcase: unittest.TestCase,
    value: np.ndarray,
    reference: np.ndarray,
    limit: float,
) -> None:
    testcase.assertLessEqual(float(np.max(_relative_diff(value, reference))), limit)


def _generated_tir_phase(case):
    return build_two_stream_phase_inputs(
        ssa=case.omega_arr,
        depol=case.depol,
        rayleigh_fraction=case.rayleigh_fraction,
        aerosol_fraction=case.aerosol_fraction,
        aerosol_moments=case.aerosol_moments,
        aerosol_interp_fraction=case.aerosol_interp_fraction,
    )


def _generated_tir_source(case):
    source = thermal_source_from_temperature_profile(
        case.level_temperature_k,
        case.surface_temperature_k,
        wavenumber_band_cm_inv=case.wavenumber_band_cm_inv,
    )
    return np.asarray(source.planck, dtype=float), np.asarray(source.surface_planck, dtype=float)


def _generated_uv_phase(case):
    phase = build_two_stream_phase_inputs(
        ssa=case.omega,
        depol=case.depol,
        rayleigh_fraction=case.rayleigh_fraction,
        aerosol_fraction=case.aerosol_fraction,
        aerosol_moments=case.aerosol_moments,
        aerosol_interp_fraction=case.aerosol_interp_fraction,
    )
    scatter = build_solar_fo_scatter_term(
        ssa=case.omega,
        depol=case.depol,
        rayleigh_fraction=case.rayleigh_fraction,
        aerosol_fraction=case.aerosol_fraction,
        aerosol_moments=case.aerosol_moments,
        aerosol_interp_fraction=case.aerosol_interp_fraction,
        angles=case.user_obsgeom,
        delta_m_truncation_factor=phase.delta_m_truncation_factor,
    )
    return phase, scatter


def _has_cuda() -> bool:
    if not has_torch():
        return False
    import torch

    return bool(torch.cuda.is_available())


class ReferenceCaseTests(unittest.TestCase):
    def test_packaged_reference_outputs_are_separate_files(self) -> None:
        for input_name, reference_name in (
            ("uv_benchmark_fixture.npz", "uv_reference_outputs.npz"),
            ("tir_benchmark_fixture.npz", "tir_reference_outputs.npz"),
        ):
            with self.subTest(reference=reference_name):
                with (
                    as_file(files("py2sess.data.benchmark").joinpath(input_name)) as input_path,
                    as_file(
                        files("py2sess.data.benchmark").joinpath(reference_name)
                    ) as reference_path,
                    np.load(input_path) as input_data,
                    np.load(reference_path) as reference_data,
                ):
                    self.assertEqual(set(reference_data.files), {"ref_2s", "ref_fo", "ref_total"})
                    for key in reference_data.files:
                        np.testing.assert_array_equal(reference_data[key], input_data[key])

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
        _assert_max_rel(self, result.two_stream_toa, case.ref_2s, 1.0e-5)
        _assert_max_rel(self, result.fo_total_up_toa, case.ref_fo, 1.0e-5)
        _assert_max_rel(self, result.total_toa, case.ref_total, 5.0e-4)

    def test_public_forward_tir_fixture_matches_batch_kernel(self) -> None:
        case = load_tir_benchmark_case()
        phase = _generated_tir_phase(case)
        planck, surface_planck = _generated_tir_source(case)
        kernel = solve_thermal_batch_numpy(
            tau_arr=case.tau_arr,
            omega_arr=case.omega_arr,
            asymm_arr=case.asymm_arr,
            d2s_scaling=case.d2s_scaling,
            thermal_bb_input=planck,
            surfbb=surface_planck,
            albedo=case.albedo,
            emissivity=case.emissivity,
            heights=case.heights,
            user_angle_degrees=case.user_angle,
            stream_value=case.stream_value,
        )
        public = TwoStreamEss(TwoStreamEssOptions(nlyr=case.n_layers, mode="thermal")).forward(
            tau=case.tau_arr,
            ssa=case.omega_arr,
            g=phase.g,
            z=case.heights,
            angles=case.user_angle,
            stream=case.stream_value,
            albedo=case.albedo,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
            planck=planck,
            surface_planck=surface_planck,
            emissivity=case.emissivity,
            include_fo=True,
        )
        np.testing.assert_allclose(public.radiance_2s, kernel.two_stream_toa)
        np.testing.assert_allclose(public.radiance_fo, kernel.fo_total_up_toa)
        np.testing.assert_allclose(public.radiance_total, kernel.total_toa)

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
        phase, scatter = _generated_uv_phase(case)
        public = TwoStreamEss(TwoStreamEssOptions(nlyr=case.n_layers, mode="solar")).forward(
            tau=case.tau,
            ssa=case.omega,
            g=phase.g,
            z=case.heights,
            angles=case.user_obsgeom,
            stream=case.stream_value,
            fbeam=case.flux_factor,
            albedo=case.albedo,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
            include_fo=True,
            fo_scatter_term=scatter,
        )
        np.testing.assert_allclose(public.radiance_2s, case.ref_2s, rtol=2.0e-6, atol=1.0e-12)
        np.testing.assert_allclose(public.radiance_fo, case.ref_fo, rtol=3.0e-6, atol=1.0e-12)
        np.testing.assert_allclose(public.radiance_total, case.ref_total, rtol=3.0e-6, atol=1e-12)

    def test_torch_cuda_device_request_requires_available_cuda(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        if _has_cuda():
            self.skipTest("CUDA is available")
        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=1,
                mode="solar",
                backend="torch",
                torch_device="cuda",
                torch_dtype="float64",
                torch_enable_grad=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            solver.forward(
                tau=np.array([[0.01]]),
                ssa=np.array([[0.0]]),
                g=np.array([[0.0]]),
                z=np.array([1.0, 0.0]),
                angles=[30.0, 20.0, 0.0],
                albedo=np.array([0.1]),
            )


if __name__ == "__main__":
    unittest.main()
