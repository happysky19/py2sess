from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from py2sess.optical.createprops import (
    load_createprops_provider,
    parse_createprops_dump,
    write_createprops_provider,
)
from py2sess.optical.scene_io import (
    build_benchmark_scene_inputs,
    load_profile_text,
    load_scene_yaml,
)


ROOT = Path(__file__).resolve().parents[1]


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

    def _write_provider(self, path: Path, *, mode: str) -> None:
        path.mkdir()
        arrays = {
            "wavelengths": np.array([500.0, 600.0]),
            "heights": np.array([2.0, 1.0, 0.0]),
            "albedo": np.array([0.1, 0.2]),
            "depol": np.array([0.03, 0.03]),
            "absorption_tau": np.array([[0.01, 0.02], [0.02, 0.01]]),
            "rayleigh_scattering_tau": np.array([[0.03, 0.04], [0.02, 0.03]]),
            "aerosol_scattering_tau": np.zeros((2, 2, 0)),
            "aerosol_moments": np.zeros((2, 3, 0)),
            "aerosol_interp_fraction": np.array([0.0, 1.0]),
        }
        if mode == "solar":
            arrays["user_obsgeom"] = np.array([30.0, 20.0, 0.0])
            arrays["flux_factor"] = np.array([1.0, 1.0])
        else:
            arrays["user_angle"] = np.array([20.0])
            arrays["wavenumber_band_cm_inv"] = np.array([[899.5, 900.5], [900.5, 901.5]])
            arrays["wavenumber_cm_inv"] = np.array([900.0, 901.0])
            arrays["level_temperature_k"] = np.array([220.0, 260.0, 290.0])
            arrays["surface_temperature_k"] = np.array([291.0])
        for key, value in arrays.items():
            np.save(path / f"{key}.npy", value)

    def _write_provider_scene(self, path: Path, provider: Path, *, mode: str) -> None:
        path.write_text(
            f"""
mode: {mode}
opacity:
  provider:
    kind: fortran_createprops
    path: {provider.name}
"""
        )

    def _run_benchmark_scene(self, script: str, profile: Path, scene: Path) -> str:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / script),
                "--profile",
                str(profile),
                "--scene",
                str(scene),
                "--backend",
                "numpy",
                "--require-python-generated-inputs",
                "--chunk-size",
                "1",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if result.returncode != 0:
            self.fail(f"{script} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result.stdout

    def test_geocape_profile_loader_reorders_top_to_bottom_and_selects_gases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            self._write_profile(profile_path)

            profile = load_profile_text(profile_path, gas_species=("O3",))

        np.testing.assert_allclose(profile.pressure_hpa, [100.0, 500.0, 1000.0])
        np.testing.assert_allclose(profile.temperature_k, [220.0, 260.0, 290.0])
        np.testing.assert_allclose(profile.gas_vmr[:, 0], [1.0e-8, 2.0e-8, 3.0e-8])
        self.assertEqual(profile.gas_names, ("O3",))
        self.assertEqual(profile.surface_temperature_k, 291.0)
        self.assertEqual(profile.surface_altitude_m, 15.0)

    def test_scene_yaml_loader_and_builder_emit_runtime_inputs_only(self) -> None:
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
        for forbidden in (
            "tau",
            "omega",
            "asymm",
            "scaling",
            "fo_exact_scatter",
            "gas_cross_sections",
        ):
            self.assertNotIn(forbidden, bundle)
        self.assertEqual(bundle["pressure_hpa"].shape, (3,))
        self.assertEqual(bundle["gas_absorption_tau"].shape, (2, 2))
        self.assertEqual(bundle["aerosol_moments"].shape, (2, 3, 0))
        np.testing.assert_allclose(bundle["user_obsgeom"], [30.0, 20.0, 0.0])

    def test_scene_builder_accepts_wavenumber_grid_variants(self) -> None:
        cases = (
            (
                "wavenumber_cm_inv:\n    start: 900.0\n    step: 2.0\n    count: 3\n"
                "  wavenumber_band_width_cm_inv: 1.0",
                1.0e7 / np.array([900.0, 902.0, 904.0]),
                None,
                np.array([[899.5, 900.5], [901.5, 902.5], [903.5, 904.5]]),
            ),
            (
                "wavenumber_cm_inv:\n    start: 1000.0\n    step: 100.0\n    count: 3\n"
                "  wavelength_order: reverse_from_wavenumber",
                np.array([1.0e7 / 1200.0, 1.0e7 / 1100.0, 1.0e7 / 1000.0]),
                np.array([1.0, 0.4545454545, 0.0]),
                None,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            self._write_profile(profile_path)
            for index, (spectral, expected_wavelengths, expected_fac, expected_bands) in enumerate(
                cases
            ):
                scene_path = Path(tmpdir) / f"scene_{index}.yaml"
                scene_path.write_text(
                    f"""
mode: thermal
spectral:
  {spectral}
geometry:
  view_angle: 20.0
surface:
  albedo: 0.1
"""
                )

                bundle = build_benchmark_scene_inputs(
                    profile_path=profile_path,
                    scene_path=scene_path,
                    kind="tir",
                )

                np.testing.assert_allclose(bundle["wavelengths"], expected_wavelengths)
                if expected_fac is not None:
                    np.testing.assert_allclose(bundle["aerosol_interp_fraction"], expected_fac)
                if expected_bands is not None:
                    np.testing.assert_allclose(bundle["wavenumber_band_cm_inv"], expected_bands)

    def test_scene_builder_applies_spectral_limit_before_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.dat"
            scene_path = Path(tmpdir) / "scene.yaml"
            self._write_profile(profile_path)
            scene_path.write_text(
                """
mode: solar
spectral:
  wavelengths_nm:
    start: 500.0
    step: 10.0
    count: 4
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo: 0.1
"""
            )

            bundle = build_benchmark_scene_inputs(
                profile_path=profile_path,
                scene_path=scene_path,
                kind="uv",
                spectral_limit=2,
            )

        np.testing.assert_allclose(bundle["wavelengths"], [500.0, 510.0])
        np.testing.assert_allclose(bundle["albedo"], [0.1, 0.1])

    def test_scene_builder_accepts_fortran_createprops_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            provider_path = root / "provider"
            scene_path = root / "scene.yaml"
            self._write_profile(profile_path)
            self._write_provider(provider_path, mode="solar")
            self._write_provider_scene(scene_path, provider_path, mode="solar")

            bundle = build_benchmark_scene_inputs(
                profile_path=profile_path,
                scene_path=scene_path,
                kind="uv",
            )

        for forbidden in ("tau", "omega", "asymm", "scaling", "fo_exact_scatter"):
            self.assertNotIn(forbidden, bundle)
        self.assertEqual(bundle["absorption_tau"].shape, (2, 2))
        self.assertEqual(bundle["rayleigh_scattering_tau"].shape, (2, 2))
        self.assertEqual(bundle["aerosol_scattering_tau"].shape, (2, 2, 0))
        np.testing.assert_allclose(bundle["user_obsgeom"], [30.0, 20.0, 0.0])

    def test_createprops_dump_parser_writes_provider_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dump_path = root / "uv_dump.dat"
            provider_path = root / "provider"
            dump_path.write_text(
                "\n".join(
                    [
                        "2 2 2 30.0 20.0 0.0",
                        "2.0",
                        "1.0",
                        "0.0",
                        "0 0 0 0 0 0 0 0 0 0 0",
                        "1 0 0 0 0 0 0 0 0 0 0",
                        "2 0 0 0 0 0 0 0 0 0 0",
                        "1 500.0 20000.0 0.1 0.03 1.0",
                        "1 0.0 0.4 0.75 1.0 0 0 0 0 0",
                        "2 0.0 0.2 0.50 1.0 0 0 0 0 0",
                        "2 600.0 16666.7 0.2 0.03 1.5",
                        "1 0.0 0.5 0.80 1.0 0 0 0 0 0",
                        "2 0.0 0.3 0.25 1.0 0 0 0 0 0",
                        "",
                    ]
                )
            )

            arrays = parse_createprops_dump(dump_path, kind="uv")
            write_createprops_provider(arrays, provider_path, kind="uv")
            loaded = load_createprops_provider(provider_path, kind="uv")

        np.testing.assert_allclose(loaded["wavelengths"], [500.0, 600.0])
        np.testing.assert_allclose(loaded["absorption_tau"], [[0.1, 0.1], [0.1, 0.225]])
        np.testing.assert_allclose(
            loaded["rayleigh_scattering_tau"],
            [[0.3, 0.1], [0.4, 0.075]],
        )
        self.assertEqual(loaded["aerosol_scattering_tau"].shape, (2, 2, 5))
        np.testing.assert_allclose(loaded["user_obsgeom"], [30.0, 20.0, 0.0])

    def test_tir_createprops_dump_parser_adds_temperature_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dump_path = root / "Dump_1_26_0000.dat_25"
            profile_path = root / "profile.dat"
            dump_path.write_text(
                "\n".join(
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
                ),
                encoding="utf-8",
            )
            profile_path.write_text(
                "\n".join(
                    [
                        "surfaceTemperature(K) = 299.0",
                        "1 1000.0 300.0",
                        "2 500.0 250.0",
                        "3 1.0 200.0",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_createprops_dump(dump_path, kind="tir", profile_file=profile_path)

        np.testing.assert_allclose(parsed["wavenumber_cm_inv"], [500.0, 501.0])
        np.testing.assert_allclose(
            parsed["wavenumber_band_cm_inv"],
            [[499.5, 500.5], [500.5, 501.5]],
        )
        np.testing.assert_allclose(parsed["level_temperature_k"], [200.0, 250.0, 300.0])
        np.testing.assert_allclose(parsed["surface_temperature_k"], [299.0])

    def test_benchmarks_accept_profile_scenes_strict_mode(self) -> None:
        cases = (
            (
                "solar",
                "benchmark_uv_full_spectrum.py",
                "layer optical properties: python-generated from scene/profile inputs",
                "optical preprocessing: python-generated",
            ),
            (
                "thermal",
                "benchmark_tir_full_spectrum.py",
                "layer optical properties: python-generated from scene/profile inputs",
                "thermal source: temperature (wavenumber_band_cm_inv)",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            self._write_profile(profile_path)
            for mode, script, layer_text, setup_text in cases:
                with self.subTest(mode=mode):
                    scene_path = root / f"{mode}_scene.yaml"
                    self._write_scene(scene_path, mode=mode)

                    output = self._run_benchmark_scene(script, profile_path, scene_path)

                    self.assertIn("input kind: profile+scene", output)
                    self.assertIn(layer_text, output)
                    self.assertIn(setup_text, output)

    def test_benchmarks_accept_createprops_provider_strict_mode(self) -> None:
        cases = (
            (
                "solar",
                "benchmark_uv_full_spectrum.py",
                "optical preprocessing: python-generated",
            ),
            (
                "thermal",
                "benchmark_tir_full_spectrum.py",
                "thermal source: temperature (wavenumber_band_cm_inv)",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            self._write_profile(profile_path)
            for mode, script, setup_text in cases:
                with self.subTest(mode=mode):
                    provider_path = root / f"{mode}_provider"
                    scene_path = root / f"{mode}_scene.yaml"
                    self._write_provider(provider_path, mode=mode)
                    self._write_provider_scene(scene_path, provider_path, mode=mode)

                    output = self._run_benchmark_scene(script, profile_path, scene_path)

                    self.assertIn("input kind: profile+scene", output)
                    self.assertIn(
                        "layer optical properties: python-generated from component optical depths",
                        output,
                    )
                    self.assertIn(setup_text, output)


if __name__ == "__main__":
    unittest.main()
