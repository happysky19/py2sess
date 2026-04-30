from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import netcdf_file

from py2sess.optical.geocape import (
    load_geocape_solar_flux,
    load_geocape_surface_albedo,
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

    def _write_scene(self, path: Path, *, mode: str, gas_table: str | None = None) -> None:
        if mode == "solar":
            spectral = "wavelengths_nm: [500.0, 600.0]"
            geometry = "angles: [30.0, 20.0, 0.0]"
            source = "solar:\n  flux_factor: [1.0, 1.0]"
        else:
            spectral = "wavenumber_band_cm_inv: [[899.5, 900.5], [900.5, 901.5]]"
            geometry = "view_angle: 20.0"
            source = ""
        gas_section = (
            f"    table3d: {{path: {gas_table}}}"
            if gas_table is not None
            else "    value:\n      - [1.0e-22]\n      - [2.0e-22]"
        )
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
{gas_section}
  aerosol:
    moments:
      value:
        - [[], [], []]
        - [[], [], []]
{source}
"""
        )

    def _write_provider_scene(self, path: Path, provider: Path, *, mode: str) -> None:
        path.write_text(
            f"""
mode: {mode}
opacity:
  provider:
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

    def test_geocape_surface_and_solar_tables_feed_scene_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            scene_path = root / "scene.yaml"
            solar_path = root / "solar.dat"
            emiss_path = root / "emiss.asc"
            self._write_profile(profile_path)
            solar_path.write_text(
                "header\nheader\n900 0.1\n1000 0.2\n1100 0.3\n1200 0.4\n",
                encoding="utf-8",
            )
            emiss_grid = np.column_stack(
                (np.linspace(900.0, 1200.0, 386), np.linspace(0.9, 0.6, 386))
            )
            with emiss_path.open("w", encoding="utf-8") as handle:
                handle.write("header\n" * 10)
                np.savetxt(handle, emiss_grid)
            scene_path.write_text(
                """
mode: solar
spectral:
  wavenumber_cm_inv: [1000.0, 1100.0]
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo:
    geocape_emissivity: {path: emiss.asc}
solar:
  flux_factor:
    geocape_solar_spectrum: {path: solar.dat, scale: 10.0}
"""
            )

            bundle = build_benchmark_scene_inputs(
                profile_path=profile_path,
                scene_path=scene_path,
                kind="uv",
                strict_runtime_inputs=True,
            )
            expected_albedo = load_geocape_surface_albedo(emiss_path, [1000.0, 1100.0])
            expected_flux = load_geocape_solar_flux(solar_path, [1000.0, 1100.0], scale=10.0)

        np.testing.assert_allclose(bundle["flux_factor"], [2.0, 3.0])
        np.testing.assert_allclose(bundle["albedo"], expected_albedo)
        np.testing.assert_allclose(bundle["flux_factor"], expected_flux)

    def test_strict_scene_rejects_opacity_provider_and_direct_hitran(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            provider_path = root / "provider"
            provider_scene = root / "provider.yaml"
            hitran_scene = root / "hitran.yaml"
            self._write_profile(profile_path)
            self._write_provider_scene(provider_scene, provider_path, mode="solar")
            hitran_scene.write_text(
                """
mode: solar
gases: [O3]
spectral:
  wavelengths_nm: [500.0, 600.0]
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo: 0.1
opacity:
  gas_cross_sections:
    hitran: {path: hitran}
"""
            )

            with self.assertRaisesRegex(ValueError, "opacity.provider"):
                build_benchmark_scene_inputs(
                    profile_path=profile_path,
                    scene_path=provider_scene,
                    kind="uv",
                    strict_runtime_inputs=True,
                )
            with self.assertRaisesRegex(ValueError, "direct HITRAN"):
                build_benchmark_scene_inputs(
                    profile_path=profile_path,
                    scene_path=hitran_scene,
                    kind="uv",
                    strict_runtime_inputs=True,
                )

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
            self._write_xsec_table(root / "solar_xsec.nc", mode="solar")
            self._write_xsec_table(root / "thermal_xsec.nc", mode="thermal")
            for mode, script, layer_text, setup_text in cases:
                with self.subTest(mode=mode):
                    scene_path = root / f"{mode}_scene.yaml"
                    self._write_scene(scene_path, mode=mode, gas_table=f"{mode}_xsec.nc")

                    output = self._run_benchmark_scene(script, profile_path, scene_path)

                    self.assertIn("input kind: profile+scene", output)
                    self.assertIn(layer_text, output)
                    self.assertIn(setup_text, output)

    def test_benchmarks_reject_opacity_provider_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "profile.dat"
            self._write_profile(profile_path)
            for mode in ("solar", "thermal"):
                with self.subTest(mode=mode):
                    provider_path = root / f"{mode}_provider"
                    scene_path = root / f"{mode}_scene.yaml"
                    self._write_provider_scene(scene_path, provider_path, mode=mode)

                    with self.assertRaisesRegex(ValueError, "opacity.provider"):
                        build_benchmark_scene_inputs(
                            profile_path=profile_path,
                            scene_path=scene_path,
                            kind="uv" if mode == "solar" else "tir",
                        )

    @staticmethod
    def _write_xsec_table(path: Path, *, mode: str) -> None:
        spectral_name = "wavelength_nm" if mode == "solar" else "wavenumber_cm_inv"
        spectral = np.array([500.0, 600.0] if mode == "solar" else [900.0, 901.0])
        pressure = np.array([100.0, 500.0, 1000.0])
        temperature = np.array([220.0, 260.0, 290.0])
        raw = np.full((1, 2, 3), 1.0e-22, dtype=float)
        with netcdf_file(path, "w") as data:
            data.createDimension("gas", 1)
            data.createDimension("wave", 2)
            data.createDimension("level", 3)
            data.createDimension("name_strlen", 2)
            data.createVariable(spectral_name, "d", ("wave",))[:] = spectral
            data.createVariable("pressure_hpa", "d", ("level",))[:] = pressure
            data.createVariable("temperature_k", "d", ("level",))[:] = temperature
            data.createVariable("gas_names", "c", ("gas", "name_strlen"))[:] = np.array(
                [[b"O", b"3"]], dtype="S1"
            )
            data.createVariable("cross_section", "d", ("gas", "wave", "level"))[:] = raw


if __name__ == "__main__":
    unittest.main()
