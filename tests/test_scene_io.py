from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from py2sess.optical.scene_io import (
    build_benchmark_scene_inputs,
    load_profile_text,
    load_scene_yaml,
)
from py2sess.scene import SceneRun, load_scene


class SceneIoTests(unittest.TestCase):
    def _write_profile(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "Station 17 Day 2006726 Time 1200",
                    "surfaceTemperature(K) = 291.0",
                    "ZSUR(m) = 15.0",
                    "End_of_Header",
                    "Level Pressure TATM H2O O3 NO2",
                    "- hPa K VMR VMR VMR",
                    "1 1000.0 290.0 1.0e-2 3.0e-8 4.0e-8",
                    "2 500.0 260.0 2.0e-3 2.0e-8 3.0e-8",
                    "3 100.0 220.0 1.0e-4 1.0e-8 2.0e-8",
                    "",
                ]
            )
        )

    def _write_scene(self, path: Path, *, mode: str) -> None:
        if mode == "solar":
            spectral = "wavelengths_nm: [500.0, 600.0]"
            geometry = "angles: [30.0, 20.0, 0.0]"
            source = "solar:\n  flux_factor: [1.0, 1.0]"
        else:
            spectral = "wavenumber_band_cm_inv: [[899.5, 900.5], [900.5, 901.5]]"
            geometry = "view_angle: 20.0"
            source = ""
        path.write_text(
            f"""
mode: {mode}
gases: [O3]
spectral:
  {spectral}
geometry:
  {geometry}
surface:
  albedo: 0.1
opacity:
  gas_cross_sections:
    value:
      - [1.0e-22]
      - [2.0e-22]
  aerosol:
    moments:
      value:
        - [[], [], []]
        - [[], [], []]
{source}
"""
        )

    def test_profile_loader_reorders_top_to_bottom_and_selects_gases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            self._write_profile(profile_path)
            profile = load_profile_text(profile_path, gas_species=("O3",))

        np.testing.assert_allclose(profile.pressure_hpa, [100.0, 500.0, 1000.0])
        np.testing.assert_allclose(profile.temperature_k, [220.0, 260.0, 290.0])
        np.testing.assert_allclose(profile.gas_vmr[:, 0], [1.0e-8, 2.0e-8, 3.0e-8])
        self.assertEqual(profile.gas_names, ("O3",))

    def test_scene_yaml_builder_emits_runtime_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            scene_path = Path(tmpdir) / "scene.yaml"
            self._write_profile(profile_path)
            self._write_scene(scene_path, mode="solar")

            scene = load_scene_yaml(scene_path)
            bundle = build_benchmark_scene_inputs(
                profile_path=profile_path,
                scene_path=scene_path,
                kind="uv",
            )

        self.assertEqual(scene["mode"], "solar")
        self.assertNotIn("tau", bundle)
        self.assertEqual(bundle["gas_absorption_tau"].shape, (2, 2))
        np.testing.assert_allclose(bundle["user_obsgeom"], [30.0, 20.0, 0.0])

    def test_load_scene_generates_public_forward_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            scene_path = Path(tmpdir) / "scene.yaml"
            self._write_profile(profile_path)
            self._write_scene(scene_path, mode="solar")

            scene = load_scene(profile=profile_path, config=scene_path)
            inputs = scene.to_forward_inputs()

        self.assertEqual(inputs.mode, "solar")
        self.assertEqual(inputs.kwargs["tau"].shape, (2, 2))
        self.assertIn("fo_scatter_term", inputs.kwargs)

    def test_scene_api_rejects_mode_mismatch(self) -> None:
        scene = SceneRun.from_bundle(
            mode="solar",
            bundle={
                "wavelengths": np.array([500.0]),
                "tau": np.zeros((1, 1)),
                "omega": np.zeros((1, 1)),
                "asymm": np.zeros((1, 1)),
                "scaling": np.zeros((1, 1)),
                "heights": np.array([1.0, 0.0]),
                "user_obsgeom": np.array([30.0, 20.0, 0.0]),
                "albedo": np.zeros(1),
                "flux_factor": np.ones(1),
            },
        )
        from py2sess import TwoStreamEssOptions

        with self.assertRaisesRegex(ValueError, "scene mode"):
            scene.forward(options=TwoStreamEssOptions(nlyr=1, mode="thermal"))


if __name__ == "__main__":
    unittest.main()
