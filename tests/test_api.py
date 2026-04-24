from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    TwoStreamEssBatchResult,
    load_tir_benchmark_case,
    load_uv_benchmark_case,
    thermal_source_from_temperature_profile,
    thermal_source_from_temperature_profile_torch,
)
from py2sess.core.backend import has_torch, to_numpy


class ApiTests(unittest.TestCase):
    def test_package_exports_are_available(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3))
        self.assertEqual(solver.options.nlyr, 3)
        self.assertEqual(solver.options.nlyr, 3)
        self.assertTrue(callable(thermal_source_from_temperature_profile_torch))

    def test_minimal_solar_fo_call_uses_public_names_and_defaults(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.01),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[[30.0, 0.0, 0.0]],
            albedo=0.3,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_single_solar_geometry_accepts_flat_angle_triplet(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.01),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 0.0, 0.0],
            albedo=0.3,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_minimal_thermal_fo_call_uses_public_names_and_defaults(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.1),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[0.0],
            planck=np.ones(4),
            surface_planck=1.0,
            emissivity=1.0,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_scalar_thermal_forward_total_is_two_stream_plus_fo(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
        result = solver.forward(
            tau=np.full(3, 0.1),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=0.0,
            planck=np.ones(4),
            surface_planck=1.0,
            emissivity=1.0,
            include_fo=True,
        )
        self.assertIsNotNone(result.radiance_fo)
        np.testing.assert_allclose(result.radiance_total, result.radiance_2s + result.radiance_fo)

    def test_single_thermal_geometry_accepts_scalar_view_angle(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.1),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=0.0,
            planck=np.ones(4),
            surface_planck=1.0,
            emissivity=1.0,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_advanced_fo_overrides_are_accepted(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.01),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[[30.0, 20.0, 0.0]],
            stream=0.5,
            delta_m_scaling=np.zeros(3),
            geometry="regular_pseudo_spherical",
            albedo=0.3,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_missing_solar_angles_error_is_public(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        with self.assertRaisesRegex(ValueError, "angles"):
            solver.forward_fo(
                tau=np.zeros(3),
                ssa=np.zeros(3),
                g=np.zeros(3),
                z=np.array([3.0, 2.0, 1.0, 0.0]),
            )

    def test_missing_thermal_planck_error_is_public(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
        with self.assertRaisesRegex(ValueError, "planck"):
            solver.forward_fo(
                tau=np.zeros(3),
                ssa=np.zeros(3),
                g=np.zeros(3),
                z=np.array([3.0, 2.0, 1.0, 0.0]),
                angles=[0.0],
            )

    def test_batched_solar_forward_matches_scalar_rows(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.zeros_like(tau)
        g = np.zeros_like(tau)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        albedo = np.array([0.1, 0.2])

        batch = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=[30.0, 0.0, 0.0],
            albedo=albedo,
        )

        self.assertIsInstance(batch, TwoStreamEssBatchResult)
        self.assertEqual(batch.radiance.shape, (2,))
        for i in range(tau.shape[0]):
            scalar = solver.forward(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=[30.0, 0.0, 0.0],
                albedo=float(albedo[i]),
            )
            np.testing.assert_allclose(batch.radiance_2s[i], scalar.radiance_2s[0])

    def test_batched_solar_forward_appends_geometry_axis(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.zeros_like(tau)
        g = np.zeros_like(tau)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        angles = np.array([[30.0, 0.0, 0.0], [45.0, 20.0, 90.0]])

        batch = solver.forward(tau=tau, ssa=ssa, g=g, z=z, angles=angles, albedo=0.2)

        self.assertEqual(batch.radiance.shape, (2, 2))
        self.assertEqual(batch.geometry_shape, (2,))
        for i in range(tau.shape[0]):
            for j in range(angles.shape[0]):
                scalar = solver.forward(
                    tau=tau[i],
                    ssa=ssa[i],
                    g=g[i],
                    z=z,
                    angles=angles[j],
                    albedo=0.2,
                )
                np.testing.assert_allclose(batch.radiance_2s[i, j], scalar.radiance_2s[0])

    def test_batched_thermal_forward_matches_scalar_rows(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.zeros_like(tau)
        g = np.zeros_like(tau)
        planck = np.array([[1.0, 1.1, 1.2, 1.3], [1.2, 1.1, 1.0, 0.9]])
        surface_planck = np.array([2.0, 2.5])
        angles = np.array([0.0, 30.0])

        batch = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            angles=angles,
            planck=planck,
            surface_planck=surface_planck,
            emissivity=1.0,
            albedo=0.0,
        )

        self.assertEqual(batch.radiance.shape, (2, 2))
        for i in range(tau.shape[0]):
            for j, angle in enumerate(angles):
                scalar = solver.forward(
                    tau=tau[i],
                    ssa=ssa[i],
                    g=g[i],
                    angles=float(angle),
                    planck=planck[i],
                    surface_planck=float(surface_planck[i]),
                    emissivity=1.0,
                    albedo=0.0,
                )
                np.testing.assert_allclose(
                    batch.radiance_2s[i, j], scalar.radiance_2s[0], rtol=1.0e-6, atol=1.0e-16
                )

    def test_batched_thermal_forward_can_attach_fo(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.zeros_like(tau)
        g = np.zeros_like(tau)
        planck = np.array([[1.0, 1.1, 1.2, 1.3], [1.2, 1.1, 1.0, 0.9]])
        result = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=0.0,
            planck=planck,
            surface_planck=np.array([2.0, 2.5]),
            emissivity=1.0,
            albedo=0.0,
            include_fo=True,
        )

        self.assertEqual(result.radiance.shape, (2,))
        self.assertIsNotNone(result.radiance_fo)
        np.testing.assert_allclose(result.radiance_total, result.radiance_2s + result.radiance_fo)

    def test_batched_solar_include_fo_requires_exact_scatter(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        with self.assertRaisesRegex(ValueError, "fo_exact_scatter"):
            solver.forward(
                tau=np.ones((2, 3)) * 0.01,
                ssa=np.zeros((2, 3)),
                g=np.zeros((2, 3)),
                z=np.array([3.0, 2.0, 1.0, 0.0]),
                angles=[30.0, 0.0, 0.0],
                include_fo=True,
            )

    def test_batched_solar_forward_fo_requires_and_uses_exact_scatter(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.2)
        g = np.full_like(tau, 0.1)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        scatter = np.full_like(tau, 0.5)
        with self.assertRaisesRegex(ValueError, "fo_exact_scatter"):
            solver.forward_fo(
                tau=tau,
                ssa=ssa,
                g=g,
                z=z,
                angles=[30.0, 20.0, 0.0],
            )

        batch = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            fo_exact_scatter=scatter,
        )

        self.assertEqual(batch.radiance.shape, (2,))
        self.assertEqual(batch.radiance_profile.shape, (2, 4))
        for i in range(tau.shape[0]):
            scalar = solver.forward_fo(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=[30.0, 20.0, 0.0],
                albedo=0.1,
                fo_exact_scatter=scatter[i],
            )
            np.testing.assert_allclose(batch.radiance[i], scalar.radiance[0])
            np.testing.assert_allclose(batch.radiance_profile[i], scalar.radiance_profile[0])

    def test_batched_forward_default_omits_level_profiles(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        result = solver.forward(
            tau=np.ones((2, 3)) * 0.01,
            ssa=np.zeros((2, 3)),
            g=np.zeros((2, 3)),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 0.0, 0.0],
        )
        self.assertIsNone(result.radiance_profile_2s)
        self.assertIsNone(result.radiance_profile_fo)
        self.assertIsNone(result.radiance_profile_total)

    def test_batched_solar_level_profiles_match_scalar_rows(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.2)
        g = np.full_like(tau, 0.1)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        scatter = np.full_like(tau, 0.5)

        batch = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            include_fo=True,
            fo_exact_scatter=scatter,
        )

        self.assertEqual(batch.radiance_profile_2s.shape, (2, 4))
        self.assertEqual(batch.radiance_profile_fo.shape, (2, 4))
        for i in range(tau.shape[0]):
            scalar_2s = solver.forward(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=[30.0, 20.0, 0.0],
                albedo=0.1,
            )
            scalar_fo = solver.forward_fo(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=[30.0, 20.0, 0.0],
                albedo=0.1,
                fo_exact_scatter=scatter[i],
            )
            np.testing.assert_allclose(batch.radiance_profile_2s[i], scalar_2s.radlevel_up[0])
            np.testing.assert_allclose(batch.radiance_profile_fo[i], scalar_fo.radiance_profile[0])
        np.testing.assert_allclose(
            batch.radiance_profile_total, batch.radiance_profile_2s + batch.radiance_profile_fo
        )

    def test_batched_thermal_level_profiles_match_scalar_rows(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.1)
        g = np.full_like(tau, 0.05)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        planck = np.array([[1.0, 1.1, 1.2, 1.3], [1.2, 1.1, 1.0, 0.9]])
        surface_planck = np.array([2.0, 2.5])

        batch = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=20.0,
            planck=planck,
            surface_planck=surface_planck,
            emissivity=0.9,
            albedo=0.1,
            include_fo=True,
        )

        self.assertEqual(batch.radiance_profile_2s.shape, (2, 4))
        self.assertEqual(batch.radiance_profile_fo.shape, (2, 4))
        for i in range(tau.shape[0]):
            scalar_2s = solver.forward(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=20.0,
                planck=planck[i],
                surface_planck=float(surface_planck[i]),
                emissivity=0.9,
                albedo=0.1,
            )
            scalar_fo = solver.forward_fo(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=20.0,
                planck=planck[i],
                surface_planck=float(surface_planck[i]),
                emissivity=0.9,
                albedo=0.1,
            )
            np.testing.assert_allclose(batch.radiance_profile_2s[i], scalar_2s.radlevel_up[0])
            np.testing.assert_allclose(
                batch.radiance_profile_fo[i], scalar_fo.radiance_up_profile[0]
            )
        np.testing.assert_allclose(
            batch.radiance_profile_total, batch.radiance_profile_2s + batch.radiance_profile_fo
        )

    def test_batched_torch_level_profiles_keep_gradients(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        solar_solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="torch",
                output_levels=True,
                torch_dtype="float64",
            )
        )
        solar_tau = torch.tensor(
            [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]],
            dtype=torch.float64,
            requires_grad=True,
        )
        solar_albedo = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        solar = solar_solver.forward(
            tau=solar_tau,
            ssa=torch.full_like(solar_tau, 0.2),
            g=torch.full_like(solar_tau, 0.1),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=solar_albedo,
            include_fo=True,
            fo_exact_scatter=torch.full_like(solar_tau, 0.5),
        )
        solar.radiance_profile_total.sum().backward()
        self.assertIsNotNone(solar_tau.grad)
        self.assertIsNotNone(solar_albedo.grad)
        self.assertTrue(torch.isfinite(solar_tau.grad).all().item())
        self.assertTrue(torch.isfinite(solar_albedo.grad).all().item())
        self.assertGreater(float(torch.abs(solar_tau.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(solar_albedo.grad).sum()), 0.0)

        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="torch",
                output_levels=True,
                torch_dtype="float64",
            )
        )
        tau = torch.tensor(
            [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]],
            dtype=torch.float64,
            requires_grad=True,
        )
        result = solver.forward(
            tau=tau,
            ssa=torch.zeros_like(tau),
            g=torch.zeros_like(tau),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=20.0,
            planck=torch.ones((2, 4), dtype=torch.float64),
            surface_planck=torch.ones(2, dtype=torch.float64),
            emissivity=1.0,
            albedo=0.0,
            include_fo=True,
        )
        result.radiance_profile_total.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertTrue(torch.isfinite(tau.grad).all().item())
        self.assertGreater(float(torch.abs(tau.grad).sum()), 0.0)

    def test_public_batched_thermal_torch_matches_numpy_endpoint(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")

        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.1)
        g = np.full_like(tau, 0.05)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        planck = np.array([[1.0, 1.1, 1.2, 1.3], [1.2, 1.1, 1.0, 0.9]])
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=20.0,
            planck=planck,
            surface_planck=np.array([2.0, 2.5]),
            emissivity=0.9,
            albedo=0.1,
            include_fo=True,
        )
        numpy_result = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal")).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="torch",
                torch_dtype="float64",
                torch_enable_grad=False,
            )
        ).forward(**kwargs)
        np.testing.assert_allclose(
            to_numpy(torch_result.radiance_2s), numpy_result.radiance_2s, rtol=1.0e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            to_numpy(torch_result.radiance_fo), numpy_result.radiance_fo, rtol=1.0e-12, atol=1e-12
        )

    def test_reference_case_loaders_return_expected_dimensions(self) -> None:
        tir = load_tir_benchmark_case()
        uv = load_uv_benchmark_case()
        self.assertEqual(tir.n_layers, 114)
        self.assertEqual(uv.n_layers, 114)
        self.assertGreater(tir.n_wavelengths, 0)
        self.assertGreater(uv.n_wavelengths, 0)

    def test_thermal_source_helper_returns_expected_sizes(self) -> None:
        source = thermal_source_from_temperature_profile(
            [220.0, 230.0, 240.0, 250.0],
            280.0,
            wavenumber_band_cm_inv=(900.0, 901.0),
        )
        self.assertEqual(source.thermal_bb_input.shape, (4,))
        self.assertGreater(source.surfbb, 0.0)


if __name__ == "__main__":
    unittest.main()
