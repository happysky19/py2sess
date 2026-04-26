from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from py2sess.rtsolver.backend import has_torch


ROOT = Path(__file__).resolve().parents[1]


def _load_example_module(name: str):
    examples_dir = ROOT / "examples"
    examples_path = str(examples_dir)
    inserted_examples_path = False
    previous_module = sys.modules.get(name)
    if examples_path not in sys.path:
        sys.path.insert(0, examples_path)
        inserted_examples_path = True
    try:
        spec = importlib.util.spec_from_file_location(name, examples_dir / f"{name}.py")
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError(f"could not load example module: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted_examples_path:
            try:
                sys.path.remove(examples_path)
            except ValueError:  # pragma: no cover
                pass
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


class BenchmarkExampleTests(unittest.TestCase):
    def _component_optical_depths(
        self,
        *,
        tau: np.ndarray,
        ssa: np.ndarray,
        rayleigh_fraction: np.ndarray,
        aerosol_fraction: np.ndarray,
    ) -> dict[str, np.ndarray]:
        scattering_tau = tau * ssa
        return {
            "absorption_tau": np.maximum(tau - scattering_tau, 0.0),
            "rayleigh_scattering_tau": scattering_tau * rayleigh_fraction,
            "aerosol_scattering_tau": scattering_tau[..., None] * aerosol_fraction,
        }

    def _run_benchmark_process(
        self, script: str, fixture: str | Path
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        backend = "both" if has_torch() else "numpy"
        fixture_path = Path(fixture)
        if not fixture_path.is_absolute():
            fixture_path = ROOT / "src" / "py2sess" / "data" / "benchmark" / fixture_path
        command = [
            sys.executable,
            str(ROOT / "examples" / script),
            str(fixture_path),
            "--backend",
            backend,
            "--limit",
            "4",
            "--chunk-size",
            "2",
            "--torch-device",
            "cpu",
            "--torch-dtype",
            "float64",
            "--torch-threads",
            "1",
        ]
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )

    def _run_benchmark(self, script: str, fixture: str | Path) -> str:
        result = self._run_benchmark_process(script, fixture)
        if result.returncode != 0:
            self.fail(f"{script} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result.stdout

    def test_uv_full_spectrum_benchmark_smoke(self) -> None:
        output = self._run_benchmark(
            "benchmark_uv_full_spectrum.py",
            "uv_benchmark_fixture.npz",
        )
        self.assertIn("numpy", output)
        self.assertIn("numpy-forward", output)
        self.assertIn("layer optical properties: bundle", output)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("preprocessing total:", output)
        if has_torch():
            self.assertIn("torch-cpu-float64-forward", output)
        self.assertIn("max abs diff", output)

    def test_tir_full_spectrum_benchmark_smoke(self) -> None:
        output = self._run_benchmark(
            "benchmark_tir_full_spectrum.py",
            "tir_benchmark_fixture.npz",
        )
        self.assertIn("numpy", output)
        self.assertIn("numpy-forward", output)
        self.assertIn("layer optical properties: bundle", output)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("preprocessing total:", output)
        self.assertIn("emissivity: bundle", output)
        if has_torch():
            self.assertIn("torch-cpu-float64-forward", output)
        self.assertIn("max abs diff", output)

    def test_key_selection_prefers_python_generated_inputs(self) -> None:
        common = _load_example_module("_full_spectrum_benchmark_common")
        uv = _load_example_module("benchmark_uv_full_spectrum")
        tir = _load_example_module("benchmark_tir_full_spectrum")

        cases = [
            {
                "label": "uv",
                "module": uv,
                "available": {
                    "tau",
                    "omega",
                    "absorption_tau",
                    "rayleigh_scattering_tau",
                    "aerosol_scattering_tau",
                    "depol",
                    "rayleigh_fraction",
                    "aerosol_fraction",
                    "aerosol_moments",
                    "aerosol_interp_fraction",
                    "asymm",
                    "scaling",
                    "fo_exact_scatter",
                },
                "total_key": "tau",
                "ssa_key": "omega",
                "expected_layer": (
                    "absorption_tau",
                    "rayleigh_scattering_tau",
                    "aerosol_scattering_tau",
                ),
                "expected_optical": ("depol", "aerosol_moments", "aerosol_interp_fraction"),
                "forbidden": {"tau", "omega", "asymm", "fo_exact_scatter"},
            },
            {
                "label": "tir",
                "module": tir,
                "available": {
                    "tau_arr",
                    "omega_arr",
                    "absorption_tau",
                    "rayleigh_scattering_tau",
                    "aerosol_extinction_tau",
                    "aerosol_single_scattering_albedo",
                    "depol",
                    "rayleigh_fraction",
                    "aerosol_fraction",
                    "aerosol_moments",
                    "wavenumber_cm_inv",
                    "level_temperature_k",
                    "surface_temperature_k",
                    "thermal_bb_input",
                    "surfbb",
                    "asymm_arr",
                    "d2s_scaling",
                },
                "total_key": "tau_arr",
                "ssa_key": "omega_arr",
                "expected_layer": (
                    "absorption_tau",
                    "rayleigh_scattering_tau",
                    "aerosol_extinction_tau",
                    "aerosol_single_scattering_albedo",
                ),
                "expected_optical": ("depol", "aerosol_moments", "wavenumber_cm_inv"),
                "forbidden": {"tau_arr", "omega_arr", "asymm_arr", "d2s_scaling"},
            },
        ]

        for case in cases:
            with self.subTest(case=case["label"]):
                layer_keys = common.select_layer_optical_keys(
                    case["available"],
                    total_key=case["total_key"],
                    ssa_key=case["ssa_key"],
                )
                optical_keys = case["module"]._select_optical_keys(
                    case["available"],
                    use_dumped_derived_optics=False,
                    layer_optical_from_components=common.layer_optical_keys_are_components(
                        layer_keys
                    ),
                )

                self.assertEqual(layer_keys, case["expected_layer"])
                self.assertEqual(optical_keys, case["expected_optical"])
                self.assertTrue(case["forbidden"].isdisjoint(layer_keys + optical_keys))

        source_keys = tir._select_source_keys(
            cases[1]["available"], use_dumped_thermal_source=False
        )
        self.assertEqual(
            source_keys,
            ("level_temperature_k", "surface_temperature_k", "wavenumber_cm_inv"),
        )
        self.assertNotIn("thermal_bb_input", source_keys)

    def test_uv_benchmark_does_not_require_dumped_geometry_or_optics(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "uv_benchmark_fixture.npz"
        omitted = {
            "asymm",
            "scaling",
            "fo_exact_scatter",
            "chapman",
            "x0",
            "user_stream",
            "user_secant",
            "azmfac",
            "px11",
            "pxsq",
            "px0x",
            "ulp",
            "aerosol_interp_fraction",
        }
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "uv_minimal.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_uv_full_spectrum.py", trimmed)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)

    def test_uv_benchmark_can_generate_layer_optical_properties(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "uv_benchmark_fixture.npz"
        omitted = {
            "tau",
            "omega",
            "rayleigh_fraction",
            "aerosol_fraction",
            "asymm",
            "scaling",
            "fo_exact_scatter",
        }
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "uv_component_optics.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau"]),
                    ssa=np.array(data["omega"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_uv_full_spectrum.py", trimmed)
        self.assertIn(
            "layer optical properties: python-generated from component optical depths",
            output,
        )
        self.assertIn("optical preprocessing: python-generated", output)

    def test_uv_benchmark_rejects_row_index_wavelengths_for_generated_optics(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "uv_benchmark_fixture.npz"
        omitted = {"asymm", "scaling", "fo_exact_scatter", "aerosol_interp_fraction"}
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "uv_row_index_wavelengths.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays["wavelengths"] = np.arange(1, data["wavelengths"].size + 1, dtype=float)
            np.savez_compressed(trimmed, **arrays)
            result = self._run_benchmark_process("benchmark_uv_full_spectrum.py", trimmed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("physical wavelengths", result.stderr)

    def test_tir_benchmark_does_not_require_dumped_optics(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {"asymm_arr", "d2s_scaling", "emissivity", "aerosol_interp_fraction"}
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_minimal.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays["wavelength_microns"] = np.array(data["wavelengths"]) / 1000.0
            arrays["wavelengths"] = np.arange(1, data["wavelengths"].size + 1, dtype=float)
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("emissivity: 1 - albedo", output)

    def test_tir_benchmark_can_generate_layer_optical_properties(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {
            "tau_arr",
            "omega_arr",
            "rayleigh_fraction",
            "aerosol_fraction",
            "asymm_arr",
            "d2s_scaling",
        }
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_component_optics.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau_arr"]),
                    ssa=np.array(data["omega_arr"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn(
            "layer optical properties: python-generated from component optical depths",
            output,
        )
        self.assertIn("optical preprocessing: python-generated", output)

    def test_tir_benchmark_can_use_wavenumber_for_optical_interpolation(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {"asymm_arr", "d2s_scaling", "aerosol_interp_fraction"}
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_wavenumber_optics.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays["wavenumber_cm_inv"] = 1.0e7 / np.array(data["wavelengths"])
            arrays["wavelengths"] = np.arange(1, data["wavelengths"].size + 1, dtype=float)
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn(
            "optical preprocessing: python-generated (aerosol interpolation from wavenumber_cm_inv)",
            output,
        )

    def test_tir_benchmark_can_generate_thermal_source_from_temperature(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {"thermal_bb_input", "surfbb", "ref_2s", "ref_fo", "ref_total"}
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_temperature_source.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            n_rows = int(data["tau_arr"].shape[0])
            n_levels = int(data["tau_arr"].shape[1]) + 1
            arrays["level_temperature_k"] = np.linspace(220.0, 270.0, n_levels)
            arrays["surface_temperature_k"] = np.array([285.0], dtype=float)
            arrays["wavenumber_cm_inv"] = np.linspace(700.0, 900.0, n_rows)
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn("thermal source: temperature (wavenumber_cm_inv)", output)


if __name__ == "__main__":
    unittest.main()
