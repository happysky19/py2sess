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
        _noise_std,
        retrieve_solar_synthetic,
        retrieve_thermal_synthetic,
        retrieve_uv_benchmark_synthetic,
        save_retrieval_chart,
    )
else:  # pragma: no cover
    retrieve_solar_synthetic = None
    retrieve_thermal_synthetic = None
    retrieve_uv_benchmark_synthetic = None
    save_retrieval_chart = None
    torch = None


@unittest.skipUnless(has_torch(), "torch not installed")
class RetrievalExampleTests(unittest.TestCase):
    def test_noise_models_are_positive_and_distinct(self) -> None:
        spectrum = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        absolute = _noise_std(spectrum, 0.01, "absolute").numpy()
        relative = _noise_std(spectrum, 0.01, "relative").numpy()
        hybrid = _noise_std(spectrum, 0.01, "hybrid").numpy()
        self.assertTrue(np.all(absolute > 0.0))
        self.assertTrue(np.all(relative > 0.0))
        self.assertTrue(np.all(hybrid > 0.0))
        np.testing.assert_allclose(absolute, absolute[0])
        self.assertFalse(np.allclose(relative, relative[0]))
        self.assertTrue(np.all(hybrid > absolute))

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
