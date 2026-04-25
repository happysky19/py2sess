from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    TwoStreamEssBatchResult,
    fo_scatter_term_henyey_greenstein,
    fo_scatter_term_henyey_greenstein_torch,
    thermal_source_from_temperature_profile_torch,
)
from py2sess.core.backend import has_torch, to_numpy


class ApiTests(unittest.TestCase):
    def test_package_exports_are_available(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3))
        self.assertEqual(solver.options.nlyr, 3)
        self.assertTrue(callable(fo_scatter_term_henyey_greenstein))
        self.assertTrue(callable(fo_scatter_term_henyey_greenstein_torch))
        self.assertTrue(callable(thermal_source_from_temperature_profile_torch))

    def test_fo_scatter_term_helper_matches_isotropic_formula(self) -> None:
        ssa = np.array([0.5, 0.25])
        scaling = np.array([0.0, 0.2])
        scatter = fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=np.zeros_like(ssa),
            delta_m_truncation_factor=scaling,
            angles=[30.0, 20.0, 0.0],
            n_moments=0,
        )
        np.testing.assert_allclose(scatter, ssa / (1.0 - scaling * ssa))

    def test_fo_scatter_term_helper_uses_closed_form_hg(self) -> None:
        ssa = np.array([0.4, 0.3])
        g = np.array([0.2, 0.6])
        scaling = g * g
        angles = [30.0, 20.0, 0.0]
        sza, vza, raz = np.deg2rad(angles)
        mu = -(np.cos(vza) * np.cos(sza)) + np.sin(vza) * np.sin(sza) * np.cos(raz)
        phase = (1.0 - g * g) / np.power(1.0 + g * g - 2.0 * g * mu, 1.5)
        expected = phase * ssa / (1.0 - scaling * ssa)

        low_order = fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=g,
            delta_m_truncation_factor=scaling,
            angles=angles,
            n_moments=1,
        )
        high_order = fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=g,
            delta_m_truncation_factor=scaling,
            angles=angles,
            n_moments=5000,
        )

        np.testing.assert_allclose(low_order, expected)
        np.testing.assert_allclose(high_order, expected)

    def test_fo_scatter_term_helper_supports_batches_and_geometries(self) -> None:
        scatter = fo_scatter_term_henyey_greenstein(
            ssa=np.full((2, 3), 0.4),
            g=np.full((2, 3), 0.1),
            angles=[[30.0, 20.0, 0.0], [40.0, 10.0, 90.0]],
            n_moments=3,
        )
        self.assertEqual(scatter.shape, (2, 3, 2))

    def test_fo_scatter_term_helper_matches_scalar_fo_phase_logic(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        tau = np.array([0.01, 0.02, 0.03])
        ssa = np.array([0.2, 0.25, 0.3])
        g = np.array([0.1, 0.2, 0.3])
        scaling = np.array([0.0, 0.05, 0.1])
        z = np.array([3.0, 2.0, 1.0, 0.0])
        angles = [30.0, 20.0, 0.0]
        scatter = fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=g,
            angles=angles,
            delta_m_truncation_factor=scaling,
            n_moments=5,
        )

        automatic = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            albedo=0.1,
            delta_m_truncation_factor=scaling,
            n_moments=5,
        )
        explicit = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            albedo=0.1,
            delta_m_truncation_factor=scaling,
            n_moments=5,
            fo_scatter_term=scatter,
        )
        np.testing.assert_allclose(automatic.radiance, explicit.radiance)
        np.testing.assert_allclose(automatic.radiance_profile, explicit.radiance_profile)

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_fo_scatter_term_torch_preserves_gradients(self) -> None:
        import torch

        ssa = torch.full((2, 3), 0.4, dtype=torch.float64, requires_grad=True)
        g = torch.full((2, 3), 0.1, dtype=torch.float64, requires_grad=True)
        scatter = fo_scatter_term_henyey_greenstein_torch(
            ssa=ssa,
            g=g,
            angles=[[30.0, 20.0, 0.0], [40.0, 10.0, 90.0]],
            n_moments=5,
        )
        self.assertEqual(tuple(scatter.shape), (2, 3, 2))
        scatter.sum().backward()
        self.assertTrue(torch.isfinite(ssa.grad).all())
        self.assertTrue(torch.isfinite(g.grad).all())

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

    def test_thermal_default_stream_is_gaussian_quadrature(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="thermal"))
        kwargs = dict(
            tau=np.full(3, 0.1),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=0.0,
            planck=np.ones(4),
            surface_planck=1.0,
            emissivity=1.0,
        )
        default = solver.forward(**kwargs)
        explicit = solver.forward(**kwargs, stream=1.0 / np.sqrt(3.0))
        np.testing.assert_allclose(default.radiance_2s, explicit.radiance_2s)

    def test_advanced_fo_overrides_are_accepted(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        result = solver.forward_fo(
            tau=np.full(3, 0.01),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[[30.0, 20.0, 0.0]],
            stream=0.5,
            delta_m_truncation_factor=np.zeros(3),
            geometry="regular_pseudo_spherical",
            albedo=0.3,
        )
        self.assertEqual(result.radiance.shape, (1,))

    def test_default_delta_m_truncation_factor_matches_g_squared_scalar(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        tau = np.array([0.05, 0.03, 0.02])
        ssa = np.array([0.4, 0.3, 0.2])
        g = np.array([0.6, 0.4, 0.2])
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            include_fo=True,
            fo_n_moments=8,
        )

        default = solver.forward(**kwargs)
        explicit = solver.forward(**kwargs, delta_m_truncation_factor=g * g)

        np.testing.assert_allclose(default.radiance_2s, explicit.radiance_2s)
        np.testing.assert_allclose(default.radiance_fo, explicit.radiance_fo)
        np.testing.assert_allclose(default.radiance_total, explicit.radiance_total)

    def test_delta_scaling_false_matches_zero_truncation_for_two_stream(self) -> None:
        tau = np.array([0.05, 0.03, 0.02])
        ssa = np.array([0.4, 0.3, 0.2])
        g = np.array([0.6, 0.4, 0.2])
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
        )

        off = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", delta_scaling=False))
        on = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", delta_scaling=True))

        r_off = off.forward(**kwargs)
        r_zero = on.forward(**kwargs, delta_m_truncation_factor=np.zeros(3))

        np.testing.assert_allclose(r_off.radiance_2s, r_zero.radiance_2s)

    def test_invalid_delta_m_truncation_factor_error_uses_public_name(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        with self.assertRaisesRegex(ValueError, "delta_m_truncation_factor"):
            solver.forward(
                tau=np.array([0.05, 0.03, 0.02]),
                ssa=np.array([0.4, 0.3, 0.2]),
                g=np.array([0.6, 0.4, 0.2]),
                z=np.array([3.0, 2.0, 1.0, 0.0]),
                angles=[30.0, 20.0, 0.0],
                delta_m_truncation_factor=np.array([0.1, 1.0, 0.2]),
            )

    def test_invalid_public_numeric_controls_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "thermal_tcutoff"):
            TwoStreamEssOptions(nlyr=3, mode="thermal", thermal_tcutoff=0.0)

        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        kwargs = dict(
            tau=np.full(3, 0.01),
            ssa=np.zeros(3),
            g=np.zeros(3),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "stream"):
            solver.forward(**kwargs, stream=0.0)
        with self.assertRaisesRegex(ValueError, "earth_radius"):
            solver.forward(**kwargs, earth_radius=-1.0)
        with self.assertRaisesRegex(ValueError, "fo_nfine"):
            solver.forward(**kwargs, include_fo=True, fo_nfine=0)
        with self.assertRaisesRegex(ValueError, "n_moments"):
            solver.forward_fo(**kwargs, n_moments=-1)

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

    def test_batched_default_delta_m_truncation_factor_matches_g_squared(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        tau = np.array([[0.05, 0.03, 0.02], [0.04, 0.02, 0.01]])
        ssa = np.array([[0.4, 0.3, 0.2], [0.3, 0.2, 0.1]])
        g = np.array([[0.6, 0.4, 0.2], [0.5, 0.3, 0.1]])
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
        )

        default = solver.forward(**kwargs)
        explicit = solver.forward(**kwargs, delta_m_truncation_factor=g * g)

        np.testing.assert_allclose(default.radiance_2s, explicit.radiance_2s)

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_default_delta_m_truncation_factor_matches_g_squared_and_gradients(
        self,
    ) -> None:
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="torch",
                torch_dtype="float64",
            )
        )
        tau = torch.tensor(
            [[0.05, 0.03, 0.02], [0.04, 0.02, 0.01]],
            dtype=torch.float64,
            requires_grad=True,
        )
        ssa = torch.tensor(
            [[0.4, 0.3, 0.2], [0.3, 0.2, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        g = torch.tensor(
            [[0.6, 0.4, 0.2], [0.5, 0.3, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
        )

        default = solver.forward(**kwargs)
        explicit = solver.forward(**kwargs, delta_m_truncation_factor=g * g)

        torch.testing.assert_close(default.radiance_2s, explicit.radiance_2s)
        default.radiance_2s.sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all().item())
        self.assertGreater(float(torch.abs(g.grad).sum()), 0.0)

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

    def test_batched_thermal_fo_delta_m_flags_match_scalar_rows(self) -> None:
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.2)
        g = np.full_like(tau, 0.05)
        scaling = np.array([[0.08, 0.12, 0.16], [0.10, 0.14, 0.18]])
        z = np.array([3.0, 2.0, 1.0, 0.0])
        planck = np.array([[1.0, 1.1, 1.2, 1.3], [1.2, 1.1, 1.0, 0.9]])
        surface_planck = np.array([2.0, 2.5])
        base_kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            delta_m_truncation_factor=scaling,
            z=z,
            angles=25.0,
            planck=planck,
            surface_planck=surface_planck,
            emissivity=0.9,
            albedo=0.1,
            include_fo=True,
        )
        option_cases = [
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                fo_optical_delta_m_scaling=False,
            ),
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                fo_thermal_source_delta_m_scaling=True,
            ),
        ]

        for options in option_cases:
            with self.subTest(options=options):
                batch = TwoStreamEss(options).forward(**base_kwargs)
                scalar_solver = TwoStreamEss(options)
                for row in range(tau.shape[0]):
                    scalar = scalar_solver.forward(
                        tau=tau[row],
                        ssa=ssa[row],
                        g=g[row],
                        delta_m_truncation_factor=scaling[row],
                        z=z,
                        angles=25.0,
                        planck=planck[row],
                        surface_planck=surface_planck[row],
                        emissivity=0.9,
                        albedo=0.1,
                        include_fo=True,
                    )
                    np.testing.assert_allclose(
                        batch.radiance_fo[row], scalar.radiance_fo[0], rtol=1.0e-12, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        batch.radiance_total[row],
                        scalar.radiance_total[0],
                        rtol=1.0e-12,
                        atol=1e-12,
                    )

    def test_batched_solar_include_fo_builds_scatter_term(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar"))
        tau = np.ones((2, 3)) * 0.01
        ssa = np.full_like(tau, 0.2)
        g = np.full_like(tau, 0.1)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        angles = [30.0, 0.0, 0.0]

        automatic = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            include_fo=True,
            fo_n_moments=5,
        )
        explicit = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            include_fo=True,
            fo_n_moments=5,
            fo_scatter_term=fo_scatter_term_henyey_greenstein(
                ssa=ssa,
                g=g,
                angles=angles,
                n_moments=5,
            ),
        )

        self.assertEqual(automatic.radiance_fo.shape, (2,))
        np.testing.assert_allclose(automatic.radiance_fo, explicit.radiance_fo)
        np.testing.assert_allclose(automatic.radiance_total, explicit.radiance_total)

    def test_batched_solar_forward_fo_builds_scatter_term(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
        tau = np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
        ssa = np.full_like(tau, 0.2)
        g = np.full_like(tau, 0.1)
        z = np.array([3.0, 2.0, 1.0, 0.0])
        angles = [30.0, 20.0, 0.0]
        scatter = fo_scatter_term_henyey_greenstein(
            ssa=ssa,
            g=g,
            angles=angles,
            n_moments=5,
        )

        batch = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            albedo=0.1,
            n_moments=5,
        )
        explicit = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            albedo=0.1,
            n_moments=5,
            fo_scatter_term=scatter,
        )

        self.assertEqual(batch.radiance.shape, (2,))
        self.assertEqual(batch.radiance_profile.shape, (2, 4))
        np.testing.assert_allclose(batch.radiance, explicit.radiance)
        np.testing.assert_allclose(batch.radiance_profile, explicit.radiance_profile)
        for i in range(tau.shape[0]):
            scalar = solver.forward_fo(
                tau=tau[i],
                ssa=ssa[i],
                g=g[i],
                z=z,
                angles=angles,
                albedo=0.1,
                n_moments=5,
                fo_scatter_term=scatter[i],
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
            fo_scatter_term=scatter,
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
                fo_scatter_term=scatter[i],
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

    def test_batched_torch_forward_fo_auto_scatter_term_keeps_gradients(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="torch",
                torch_dtype="float64",
            )
        )
        tau = torch.tensor(
            [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]],
            dtype=torch.float64,
            requires_grad=True,
        )
        ssa = torch.full(tau.shape, 0.2, dtype=torch.float64, requires_grad=True)
        g = torch.full(tau.shape, 0.1, dtype=torch.float64, requires_grad=True)
        result = solver.forward_fo(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            n_moments=5,
        )

        result.radiance.sum().backward()
        self.assertTrue(torch.isfinite(tau.grad).all().item())
        self.assertTrue(torch.isfinite(ssa.grad).all().item())
        self.assertTrue(torch.isfinite(g.grad).all().item())
        self.assertGreater(float(torch.abs(tau.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(ssa.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(g.grad).sum()), 0.0)

    def test_batched_torch_forward_include_fo_keeps_ssa_g_gradients(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="torch",
                torch_dtype="float64",
            )
        )
        tau = torch.tensor(
            [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]],
            dtype=torch.float64,
            requires_grad=True,
        )
        ssa = torch.full(tau.shape, 0.2, dtype=torch.float64, requires_grad=True)
        g = torch.full(tau.shape, 0.1, dtype=torch.float64, requires_grad=True)
        result = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            include_fo=True,
            fo_n_moments=5,
        )

        result.radiance_total.sum().backward()
        self.assertTrue(torch.isfinite(tau.grad).all().item())
        self.assertTrue(torch.isfinite(ssa.grad).all().item())
        self.assertTrue(torch.isfinite(g.grad).all().item())
        self.assertGreater(float(torch.abs(tau.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(ssa.grad).sum()), 0.0)
        self.assertGreater(float(torch.abs(g.grad).sum()), 0.0)

    def test_batched_thermal_torch_include_fo_keeps_scattering_gradients(self) -> None:
        if not has_torch():
            self.skipTest("torch not installed")
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="torch",
                torch_dtype="float64",
            )
        )
        tau = torch.tensor(
            [[0.2, 0.3, 0.4], [0.25, 0.35, 0.45]],
            dtype=torch.float64,
            requires_grad=True,
        )
        ssa = torch.tensor(
            [[0.15, 0.10, 0.05], [0.12, 0.08, 0.04]],
            dtype=torch.float64,
            requires_grad=True,
        )
        g = torch.tensor(
            [[0.1, 0.2, 0.3], [0.12, 0.22, 0.32]],
            dtype=torch.float64,
            requires_grad=True,
        )
        scaling = torch.zeros_like(tau, requires_grad=True)
        planck = torch.tensor(
            [[1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.1, 1.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        surface_planck = torch.tensor([1.4, 1.3], dtype=torch.float64, requires_grad=True)
        albedo = torch.tensor([0.05, 0.08], dtype=torch.float64, requires_grad=True)

        result = solver.forward(
            tau=tau,
            ssa=ssa,
            g=g,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=30.0,
            stream=0.5,
            albedo=albedo,
            delta_m_truncation_factor=scaling,
            planck=planck,
            surface_planck=surface_planck,
            emissivity=1.0 - albedo,
            include_fo=True,
        )

        result.radiance_total.sum().backward()
        for tensor in (tau, ssa, g, scaling, planck, surface_planck, albedo):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(float(torch.abs(tensor.grad).sum()), 0.0)

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
            fo_scatter_term=torch.full_like(solar_tau, 0.5),
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
        scaling = np.array([[0.08, 0.12, 0.16], [0.10, 0.14, 0.18]])
        kwargs = dict(
            tau=tau,
            ssa=ssa,
            g=g,
            delta_m_truncation_factor=scaling,
            z=z,
            angles=20.0,
            planck=planck,
            surface_planck=np.array([2.0, 2.5]),
            emissivity=0.9,
            albedo=0.1,
            include_fo=True,
        )
        numpy_result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                fo_thermal_source_delta_m_scaling=True,
            )
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="torch",
                torch_dtype="float64",
                torch_enable_grad=False,
                fo_thermal_source_delta_m_scaling=True,
            )
        ).forward(**kwargs)
        np.testing.assert_allclose(
            to_numpy(torch_result.radiance_2s), numpy_result.radiance_2s, rtol=1.0e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            to_numpy(torch_result.radiance_fo), numpy_result.radiance_fo, rtol=1.0e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
