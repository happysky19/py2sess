from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enrich_full_benchmark_optics.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("enrich_full_benchmark_optics", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load enrich_full_benchmark_optics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnrichFullBenchmarkOpticsTests(unittest.TestCase):
    def test_adds_rt_equivalent_component_optical_depths(self) -> None:
        module = _load_script_module()
        arrays = {
            "tau": np.array([[1.0, 2.0]]),
            "omega": np.array([[0.8, 0.25]]),
            "rayleigh_fraction": np.array([[0.25, 0.6]]),
            "aerosol_fraction": np.array([[[0.5, 0.25], [0.1, 0.3]]]),
        }

        module._add_rt_equivalent_components(arrays, tau_key="tau", ssa_key="omega")

        scattering_tau = arrays["tau"] * arrays["omega"]
        np.testing.assert_allclose(arrays["absorption_tau"], arrays["tau"] - scattering_tau)
        np.testing.assert_allclose(
            arrays["rayleigh_scattering_tau"],
            scattering_tau * arrays["rayleigh_fraction"],
        )
        np.testing.assert_allclose(
            arrays["aerosol_scattering_tau"],
            scattering_tau[..., None] * arrays["aerosol_fraction"],
        )
        self.assertNotIn("aerosol_extinction_tau", arrays)

    def test_keeps_tiny_positive_absorption(self) -> None:
        module = _load_script_module()
        arrays = {
            "tau": np.array([[1.0e-15]]),
            "omega": np.array([[1.0e-6]]),
            "rayleigh_fraction": np.array([[1.0]]),
            "aerosol_fraction": np.array([[[0.0]]]),
        }

        module._add_rt_equivalent_components(arrays, tau_key="tau", ssa_key="omega")

        self.assertGreater(float(arrays["absorption_tau"][0, 0]), 0.0)
        reconstructed = arrays["absorption_tau"] + arrays["rayleigh_scattering_tau"]
        reconstructed += arrays["aerosol_scattering_tau"].sum(axis=-1)
        np.testing.assert_allclose(reconstructed, arrays["tau"])


if __name__ == "__main__":
    unittest.main()
