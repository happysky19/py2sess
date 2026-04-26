from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
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

    def test_tir_dump_parser_adds_temperature_and_wavenumber_band_inputs(self) -> None:
        module = _load_script_module()
        dump_text = "\n".join(
            [
                "2 2 1 49.0 0.0 2",
                "2.0 0.0",
                "1.0 0.0",
                "0.0 0.0",
                "0 1 1 1 1 1 1 1 1 1 1",
                "1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
                "1 14285.0 500.0 0.1 0.03 0.2 0.01",
                "1 0.01 0.02 0.5 0.4 0.1 0.2 0.3 0.4 0.0 0.01",
                "2 0.02 0.03 0.6 0.5 0.2 0.3 0.4 0.5 0.0 0.02",
                "2 14280.0 501.0 0.1 0.03 0.2 0.01",
                "1 0.01 0.02 0.5 0.4 0.1 0.2 0.3 0.4 0.0 0.01",
                "2 0.02 0.03 0.6 0.5 0.2 0.3 0.4 0.5 0.0 0.02",
            ]
        )
        profile_text = "\n".join(
            [
                "surfaceTemperature(K) = 299.0",
                "1 1000.0 300.0",
                "2 500.0 250.0",
                "3 1.0 200.0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "Dump_1_26_0000.dat_25"
            profile_path = Path(tmpdir) / "profile.dat"
            dump_path.write_text(dump_text)
            profile_path.write_text(profile_text)

            parsed = module._parse_tir_dump(dump_path, profile_file=profile_path)

        np.testing.assert_allclose(parsed["wavenumber_cm_inv"], [500.0, 501.0])
        np.testing.assert_allclose(
            parsed["wavenumber_band_cm_inv"],
            [[499.5, 500.5], [500.5, 501.5]],
        )
        np.testing.assert_allclose(parsed["level_temperature_k"], [200.0, 250.0, 300.0])
        np.testing.assert_allclose(parsed["surface_temperature_k"], [299.0])


if __name__ == "__main__":
    unittest.main()
