from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    NoiseModel,
    OptimalEstimationProblem,
    TwoStreamEss,
    TwoStreamEssOptions,
    evaluate_jacobian,
    retrieval_diagnostics,
    solve_optimal_estimation,
    thermal_source_from_temperature_profile_torch,
)
from py2sess.rtsolver.backend import has_torch


def _finite_difference(forward_model, state, steps):
    import torch

    columns = []
    for index, step in enumerate(steps):
        offset = torch.zeros_like(state)
        offset[index] = step
        plus = forward_model(state + offset).detach()
        minus = forward_model(state - offset).detach()
        columns.append(((plus - minus) / (2.0 * step)).reshape(-1))
    return torch.stack(columns, dim=1)


@unittest.skipUnless(has_torch(), "torch not installed")
class RetrievalTests(unittest.TestCase):
    def test_noise_models_return_diagonal_covariance(self) -> None:
        import torch

        reference = torch.tensor([1.0, 2.0], dtype=torch.float64)
        absolute = NoiseModel(kind="absolute", level=0.2).covariance(reference)
        relative = NoiseModel(kind="relative", level=0.1).covariance(reference)
        hybrid = NoiseModel(kind="hybrid", level=0.1, absolute_floor=0.2).covariance(reference)

        np.testing.assert_allclose(absolute.numpy(), np.diag([0.04, 0.04]))
        np.testing.assert_allclose(relative.numpy(), np.diag([0.01, 0.04]))
        np.testing.assert_allclose(hybrid.numpy(), np.diag([0.05, 0.08]))

    def test_diagnostics_report_finite_oe_quantities(self) -> None:
        import torch

        jacobian = torch.tensor([[1.0, 0.2], [0.3, 0.7], [0.0, 1.0]], dtype=torch.float64)
        diagnostics = retrieval_diagnostics(
            jacobian,
            torch.eye(3, dtype=torch.float64) * 0.01,
            torch.eye(2, dtype=torch.float64),
        )

        self.assertEqual(tuple(diagnostics.posterior_covariance.shape), (2, 2))
        self.assertEqual(tuple(diagnostics.averaging_kernel.shape), (2, 2))
        self.assertGreater(float(diagnostics.dfs), 0.0)
        self.assertLessEqual(float(diagnostics.dfs), 2.0)
        self.assertTrue(bool(torch.isfinite(diagnostics.singular_values).all()))
        self.assertTrue(bool(torch.isfinite(diagnostics.hessian_condition)))

    def test_low_rank_full_state_reports_low_dfs(self) -> None:
        import torch

        jacobian = torch.zeros((3, 5), dtype=torch.float64)
        jacobian[:, 0] = torch.tensor([1.0, 0.5, 0.2], dtype=torch.float64)
        diagnostics = retrieval_diagnostics(
            jacobian,
            torch.eye(3, dtype=torch.float64),
            torch.eye(5, dtype=torch.float64),
        )

        self.assertLess(float(diagnostics.dfs), 1.0)
        self.assertEqual(tuple(diagnostics.averaging_kernel.shape), (5, 5))

    def test_solar_jacobian_matches_finite_difference(self) -> None:
        import torch

        forward_model, state, _truth = _solar_problem_parts()
        radiance, jacobian = evaluate_jacobian(forward_model, state)
        fd = _finite_difference(forward_model, state, torch.tensor([1.0e-5, 1.0e-5]))

        self.assertEqual(tuple(jacobian.shape), (radiance.numel(), state.numel()))
        np.testing.assert_allclose(jacobian.numpy(), fd.numpy(), rtol=2.0e-4, atol=1.0e-8)

    def test_thermal_jacobian_matches_finite_difference(self) -> None:
        import torch

        forward_model, state, _truth = _thermal_problem_parts()
        radiance, jacobian = evaluate_jacobian(forward_model, state)
        fd = _finite_difference(forward_model, state, torch.tensor([1.0e-5, 1.0e-3]))

        self.assertEqual(tuple(jacobian.shape), (radiance.numel(), state.numel()))
        np.testing.assert_allclose(jacobian.numpy(), fd.numpy(), rtol=3.0e-4, atol=1.0e-8)

    def test_zero_noise_solar_retrieval_recovers_truth(self) -> None:
        import torch

        forward_model, prior, truth = _solar_problem_parts()
        observation = forward_model(truth).detach()
        noise = NoiseModel(kind="absolute", level=0.0, absolute_floor=1.0e-10)
        problem = OptimalEstimationProblem(
            forward_model=forward_model,
            observation=observation,
            prior_state=prior,
            prior_covariance=torch.diag(torch.tensor([0.4**2, 0.25**2], dtype=torch.float64)),
            measurement_covariance=noise.covariance(observation),
            state_names=("tau_scale", "albedo"),
        )

        result = solve_optimal_estimation(problem, max_iter=10)
        np.testing.assert_allclose(result.state.numpy(), truth.numpy(), rtol=1.0e-7, atol=1.0e-7)
        self.assertLess(float(torch.linalg.norm(result.residual)), 1.0e-10)

    def test_zero_noise_thermal_retrieval_recovers_truth(self) -> None:
        import torch

        forward_model, prior, truth = _thermal_problem_parts()
        observation = forward_model(truth).detach()
        noise = NoiseModel(kind="absolute", level=0.0, absolute_floor=1.0e-10)
        problem = OptimalEstimationProblem(
            forward_model=forward_model,
            observation=observation,
            prior_state=prior,
            prior_covariance=torch.diag(torch.tensor([0.4**2, 15.0**2], dtype=torch.float64)),
            measurement_covariance=noise.covariance(observation),
            state_names=("tau_scale", "surface_temperature"),
        )

        result = solve_optimal_estimation(problem, max_iter=10)
        np.testing.assert_allclose(result.state.numpy(), truth.numpy(), rtol=1.0e-7, atol=1.0e-5)
        self.assertLess(float(torch.linalg.norm(result.residual)), 1.0e-10)

    def test_retrieval_forward_wrapper_matches_direct_forward(self) -> None:
        forward_model, _prior, truth = _solar_problem_parts()
        wrapped = forward_model(truth).detach().numpy()
        direct = _direct_solar_radiance(truth).detach().numpy()
        np.testing.assert_allclose(wrapped, direct, rtol=0.0, atol=0.0)


def _solar_problem_parts():
    import torch

    dtype = torch.float64
    base_tau = torch.tensor(
        [[0.020, 0.025, 0.030], [0.030, 0.020, 0.015], [0.015, 0.030, 0.020]],
        dtype=dtype,
    )
    ssa = torch.full_like(base_tau, 0.25)
    g = torch.full_like(base_tau, 0.15)
    z = np.array([3.0, 2.0, 1.0, 0.0])
    angles = [30.0, 20.0, 0.0]
    solver = TwoStreamEss(
        TwoStreamEssOptions(nlyr=3, mode="solar", backend="torch", torch_dtype="float64")
    )

    def forward_model(state):
        result = solver.forward(
            tau=state[0] * base_tau,
            ssa=ssa,
            g=g,
            z=z,
            angles=angles,
            albedo=state[1],
            include_fo=True,
            fo_n_moments=5,
        )
        return result.radiance_total.reshape(-1)

    return (
        forward_model,
        torch.tensor([0.82, 0.12], dtype=dtype),
        torch.tensor([1.0, 0.20], dtype=dtype),
    )


def _direct_solar_radiance(state):
    forward_model, _prior, _truth = _solar_problem_parts()
    return forward_model(state)


def _thermal_problem_parts():
    import torch

    dtype = torch.float64
    wavenumber = torch.linspace(700.0, 900.0, 5, dtype=dtype)
    layer_shape = torch.tensor([0.5, 0.3, 0.2], dtype=dtype)
    spectral_shape = torch.linspace(0.8, 1.2, wavenumber.numel(), dtype=dtype).reshape(-1, 1)
    tau = 0.06 * spectral_shape * layer_shape.reshape(1, -1)
    zeros = torch.zeros_like(tau)
    level_temperature = torch.tensor([225.0, 238.0, 252.0, 265.0], dtype=dtype)
    solver = TwoStreamEss(
        TwoStreamEssOptions(nlyr=3, mode="thermal", backend="torch", torch_dtype="float64")
    )

    def forward_model(state):
        source = thermal_source_from_temperature_profile_torch(
            level_temperature,
            state[1],
            wavenumber_cm_inv=wavenumber,
            dtype=dtype,
        )
        result = solver.forward(
            tau=state[0] * tau,
            ssa=zeros,
            g=zeros,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=25.0,
            albedo=0.04,
            emissivity=0.96,
            planck=source.planck,
            surface_planck=source.surface_planck,
            include_fo=True,
        )
        return result.radiance_total.reshape(-1)

    return (
        forward_model,
        torch.tensor([0.82, 280.0], dtype=dtype),
        torch.tensor([1.0, 288.0], dtype=dtype),
    )


if __name__ == "__main__":
    unittest.main()
