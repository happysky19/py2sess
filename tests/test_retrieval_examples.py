from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from py2sess.core.backend import has_torch


def _has_matplotlib() -> bool:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


if has_torch():
    import torch

    from examples.retrieve_synthetic_spectra import (
        retrieve_solar_synthetic,
        retrieve_thermal_synthetic,
        retrieve_uv_benchmark_synthetic,
    )
    from py2sess.retrieval import RodgersObjective, noise_std, save_retrieval_chart
else:  # pragma: no cover
    retrieve_solar_synthetic = None
    retrieve_thermal_synthetic = None
    retrieve_uv_benchmark_synthetic = None
    RodgersObjective = None
    noise_std = None
    save_retrieval_chart = None
    torch = None


@unittest.skipUnless(has_torch(), "torch not installed")
class RetrievalExampleTests(unittest.TestCase):
    def test_noise_models_are_positive_and_distinct(self) -> None:
        spectrum = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        absolute = noise_std(spectrum, 0.01, "absolute").numpy()
        relative = noise_std(spectrum, 0.01, "relative").numpy()
        hybrid = noise_std(spectrum, 0.01, "hybrid").numpy()
        self.assertTrue(np.all(absolute > 0.0))
        self.assertTrue(np.all(relative > 0.0))
        self.assertTrue(np.all(hybrid > 0.0))
        np.testing.assert_allclose(absolute, absolute[0])
        self.assertFalse(np.allclose(relative, relative[0]))
        self.assertTrue(np.all(hybrid > absolute))

    def test_noise_models_accept_numpy_arrays(self) -> None:
        spectrum = np.array([1.0, 2.0, 4.0], dtype=float)
        absolute = noise_std(spectrum, 0.01, "absolute")
        relative = noise_std(spectrum, 0.01, "relative")
        hybrid = noise_std(spectrum, 0.01, "hybrid")
        self.assertIsInstance(absolute, np.ndarray)
        self.assertTrue(np.all(absolute > 0.0))
        self.assertTrue(np.all(relative > 0.0))
        self.assertTrue(np.all(hybrid > absolute))

    def test_rodgers_objective_exposes_residual_jacobian_and_diagnostics(self) -> None:
        matrix = torch.tensor([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dtype=torch.float64)
        observed = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        objective = RodgersObjective.from_observation(
            observed=observed,
            measurement_error=np.ones(3),
            forward_model=lambda state: matrix @ state,
        )
        state = np.array([0.2, -0.1], dtype=float)
        self.assertEqual(objective.residual(state).shape, (3,))
        self.assertEqual(objective.jacobian(state).shape, (3, 2))
        diagnostics = objective.diagnostics(
            state,
            n_function_evaluations=1,
            n_jacobian_evaluations=1,
        )
        self.assertEqual(diagnostics.n_observations, 3)
        self.assertEqual(diagnostics.n_state, 2)
        self.assertGreater(diagnostics.degrees_of_freedom, 1.9)
        self.assertLessEqual(diagnostics.degrees_of_freedom, 2.0)

    def test_zero_noise_thermal_full_state_recovers_truth(self) -> None:
        result = retrieve_thermal_synthetic(noise_fraction=0.0, prior_mode="off")
        self.assertLess(result.final_data_loss, 1.0e-24)
        self.assertAlmostEqual(result.estimate["tau_scale"], result.truth["tau_scale"], places=10)
        self.assertAlmostEqual(
            result.estimate["surface_temperature"],
            result.truth["surface_temperature"],
            places=10,
        )
        np.testing.assert_allclose(
            result.estimate["temperature"],
            result.truth["temperature"],
            rtol=0.0,
            atol=1.0e-9,
        )
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.measurement_jacobian.shape, (192, 5))
        self.assertEqual(diagnostics.gauss_newton_hessian.shape, (5, 5))
        self.assertEqual(diagnostics.posterior_covariance.shape, (5, 5))
        self.assertEqual(diagnostics.averaging_kernel.shape, (5, 5))
        self.assertAlmostEqual(diagnostics.degrees_of_freedom, 5.0, places=6)
        self.assertEqual(result.spectrum.pre_noise.shape, (2, 96))
        self.assertEqual(result.spectrum.post_noise.shape, (2, 96))
        self.assertEqual(result.spectrum.fitted.shape, (2, 96))
        np.testing.assert_allclose(
            result.spectrum.pre_noise, result.spectrum.post_noise, atol=1.0e-12
        )
        np.testing.assert_allclose(result.spectrum.post_noise, result.spectrum.fitted, atol=1.0e-12)

    def test_zero_noise_solar_retrieval_recovers_truth(self) -> None:
        result = retrieve_solar_synthetic(noise_fraction=0.0, prior_mode="off")
        self.assertLess(result.final_data_loss, 1.0e-24)
        self.assertAlmostEqual(result.estimate["tau_scale"], result.truth["tau_scale"], places=10)
        self.assertAlmostEqual(result.estimate["albedo"], result.truth["albedo"], places=10)

    def test_zero_noise_uv_retrieval_recovers_truth(self) -> None:
        result = retrieve_uv_benchmark_synthetic(noise_fraction=0.0, prior_mode="off")
        self.assertLess(result.final_data_loss, 1.0e-24)
        self.assertAlmostEqual(result.estimate["tau_scale"], result.truth["tau_scale"], places=10)
        self.assertAlmostEqual(
            result.estimate["albedo_scale"],
            result.truth["albedo_scale"],
            places=10,
        )

    @unittest.skipUnless(_has_matplotlib(), "matplotlib not installed")
    def test_retrieval_chart_is_written(self) -> None:
        result = retrieve_solar_synthetic(noise_fraction=0.0, prior_mode="off")
        self.assertEqual(result.spectrum.pre_noise.shape, (1, 128))
        self.assertEqual(result.spectrum.post_noise.shape, (1, 128))
        with tempfile.TemporaryDirectory() as output_dir:
            path = save_retrieval_chart(result, output_dir)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
