from __future__ import annotations

import unittest

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.rtsolver.backend import has_torch, to_numpy


class ApiTests(unittest.TestCase):
    def test_scalar_solar_forward_returns_component_sum(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar", output_levels=True))
        result = solver.forward(
            tau=np.array([0.01, 0.02]),
            ssa=np.array([0.2, 0.1]),
            g=np.array([0.1, 0.2]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=0.1,
            include_fo=True,
        )

        self.assertEqual(result.radiance_total.shape, (1,))
        self.assertEqual(result.radiance_profile_total.shape, (1, 3))
        np.testing.assert_allclose(result.radiance_total, result.radiance_2s + result.radiance_fo)
        self.assertTrue(np.all(np.isfinite(result.radiance_profile_total)))

    def test_scalar_thermal_forward_returns_component_sum(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="thermal", output_levels=True))
        result = solver.forward(
            tau=np.array([0.1, 0.2]),
            ssa=np.array([0.05, 0.1]),
            g=np.array([0.1, 0.2]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=20.0,
            planck=np.array([1.0, 1.1, 1.2]),
            surface_planck=1.3,
            emissivity=0.9,
            albedo=0.05,
            include_fo=True,
        )

        self.assertEqual(result.radiance_total.shape, (1,))
        self.assertEqual(result.radiance_profile_total.shape, (1, 3))
        np.testing.assert_allclose(result.radiance_total, result.radiance_2s + result.radiance_fo)
        self.assertTrue(np.all(np.isfinite(result.radiance_profile_total)))

    def test_transparent_atmosphere_handles_solar_and_thermal_limits(self) -> None:
        z = np.array([2.0, 1.0, 0.0])
        zeros = np.zeros(2)
        solar = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar", output_levels=True))
        solar_result = solar.forward(
            tau=zeros,
            ssa=zeros,
            g=zeros,
            z=z,
            angles=[30.0, 20.0, 0.0],
            fbeam=0.0,
            albedo=0.0,
            delta_m_truncation_factor=zeros,
        )
        np.testing.assert_allclose(solar_result.radiance, np.zeros(1), atol=1.0e-12)

        thermal = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="thermal", output_levels=True))
        thermal_result = thermal.forward(
            tau=zeros,
            ssa=zeros,
            g=zeros,
            z=z,
            angles=20.0,
            planck=np.zeros(3),
            surface_planck=1.0,
            emissivity=0.9,
            albedo=0.1,
            delta_m_truncation_factor=zeros,
            include_fo=True,
        )
        np.testing.assert_allclose(thermal_result.radiance, np.array([0.9]), atol=1.0e-12)
        np.testing.assert_allclose(thermal_result.radiance_profile, np.full((1, 3), 0.9))

    def test_absorbing_solar_level_fluxes_match_beer_lambert(self) -> None:
        sza = 30.0
        mu0 = np.cos(np.deg2rad(sza))
        tau = np.array([0.1, 0.2])
        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=2,
                mode="solar",
                plane_parallel=True,
                delta_scaling=False,
                downwelling=True,
                output_levels=True,
                output_fluxes=True,
                fo_flux_n_mu=8,
            )
        )
        result = solver.forward(
            tau=tau,
            ssa=np.zeros_like(tau),
            g=np.zeros_like(tau),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[sza, 0.0, 0.0],
            fbeam=1.0,
            albedo=0.0,
            delta_m_truncation_factor=np.zeros_like(tau),
            include_fo=True,
        )
        level_tau = np.concatenate(([0.0], np.cumsum(tau)))
        np.testing.assert_allclose(result.flux_down[0], mu0 * np.exp(-level_tau / mu0), atol=1e-9)
        np.testing.assert_allclose(result.flux_up[0], 0.0, atol=1e-8)
        np.testing.assert_allclose(result.flux_net, result.flux_up - result.flux_down)

    def test_batched_solar_forward_matches_scalar_rows(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.03, 0.04]]),
            ssa=np.array([[0.2, 0.1], [0.1, 0.2]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
        )
        batch = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar")).forward(
            **kwargs, include_fo=True
        )
        for row in range(2):
            scalar = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar")).forward(
                tau=kwargs["tau"][row],
                ssa=kwargs["ssa"][row],
                g=kwargs["g"][row],
                z=kwargs["z"],
                angles=kwargs["angles"],
                albedo=kwargs["albedo"][row],
                include_fo=True,
            )
            np.testing.assert_allclose(batch.radiance_total[row], scalar.radiance_total[0])

    def test_batched_thermal_forward_matches_scalar_rows(self) -> None:
        kwargs = dict(
            tau=np.array([[0.1, 0.2], [0.2, 0.3]]),
            ssa=np.array([[0.05, 0.1], [0.1, 0.05]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=20.0,
            planck=np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
            surface_planck=np.array([1.3, 1.2]),
            emissivity=np.array([0.9, 0.85]),
            albedo=np.array([0.05, 0.08]),
        )
        batch = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="thermal")).forward(
            **kwargs, include_fo=True
        )
        for row in range(2):
            scalar = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="thermal")).forward(
                tau=kwargs["tau"][row],
                ssa=kwargs["ssa"][row],
                g=kwargs["g"][row],
                z=kwargs["z"],
                angles=kwargs["angles"],
                planck=kwargs["planck"][row],
                surface_planck=kwargs["surface_planck"][row],
                emissivity=kwargs["emissivity"][row],
                albedo=kwargs["albedo"][row],
                include_fo=True,
            )
            np.testing.assert_allclose(batch.radiance_total[row], scalar.radiance_total[0])

    def test_lambertian_brdf_matches_scalar_albedo(self) -> None:
        plain = TwoStreamEss(TwoStreamEssOptions(nlyr=1, mode="solar")).forward(
            tau=np.zeros(1),
            ssa=np.zeros(1),
            g=np.zeros(1),
            z=np.array([1.0, 0.0]),
            angles=[30.0, 20.0, 10.0],
            albedo=0.2,
            include_fo=True,
        )
        brdf = TwoStreamEss(TwoStreamEssOptions(nlyr=1, mode="solar", brdf_surface=True)).forward(
            tau=np.zeros(1),
            ssa=np.zeros(1),
            g=np.zeros(1),
            z=np.array([1.0, 0.0]),
            angles=[30.0, 20.0, 10.0],
            albedo=0.0,
            brdf={"kernel_specs": [{"which_brdf": 1, "factor": 0.2}]},
            include_fo=True,
        )
        np.testing.assert_allclose(brdf.radiance_total, plain.radiance_total)

    def test_invalid_public_inputs_fail_early(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(nlyr=1, mode="solar"))
        with self.assertRaisesRegex(ValueError, "angles"):
            solver.forward(
                tau=np.array([0.1]),
                ssa=np.array([0.0]),
                g=np.array([0.0]),
                z=np.array([1.0, 0.0]),
                albedo=0.0,
            )
        with self.assertRaisesRegex(ValueError, "delta_m_truncation_factor"):
            solver.forward(
                tau=np.array([0.1]),
                ssa=np.array([0.0]),
                g=np.array([0.0]),
                z=np.array([1.0, 0.0]),
                angles=[30.0, 20.0, 0.0],
                albedo=0.0,
                delta_m_truncation_factor=np.array([np.nan]),
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_solar_forward_matches_numpy(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.03, 0.04]]),
            ssa=np.array([[0.2, 0.1], [0.1, 0.2]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
        )
        numpy_result = TwoStreamEss(TwoStreamEssOptions(nlyr=2, mode="solar")).forward(
            **kwargs, include_fo=True
        )
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="torch", torch_dtype="float64")
        ).forward(**kwargs, include_fo=True)
        np.testing.assert_allclose(
            to_numpy(torch_result.radiance_total), numpy_result.radiance_total
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_forward_keeps_tau_gradients(self) -> None:
        import torch

        tau = torch.tensor([[0.01, 0.02]], dtype=torch.float64, requires_grad=True)
        solver = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="torch", torch_dtype="float64")
        )
        result = solver.forward(
            tau=tau,
            ssa=torch.tensor([[0.2, 0.1]], dtype=torch.float64),
            g=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=torch.tensor([0.1], dtype=torch.float64),
            include_fo=True,
        )
        result.radiance_total.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertTrue(torch.isfinite(tau.grad).all().item())

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_forward_flux_keeps_tau_gradients(self) -> None:
        import torch

        tau = torch.tensor([[0.01, 0.02]], dtype=torch.float64, requires_grad=True)
        solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=2,
                mode="solar",
                backend="torch",
                plane_parallel=True,
                torch_dtype="float64",
            )
        )
        result = solver.forward_flux(
            tau=tau,
            ssa=torch.tensor([[0.2, 0.1]], dtype=torch.float64),
            g=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=torch.tensor([0.1], dtype=torch.float64),
            include_fo=True,
            return_net=True,
        )
        result.flux_net.sum().backward()
        self.assertIsNotNone(tau.grad)
        self.assertTrue(torch.isfinite(tau.grad).all().item())


if __name__ == "__main__":
    unittest.main()
