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


def _load_script_module(name: str):
    script = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load script module: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def _write_array_dir(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.mkdir()
        for key, value in arrays.items():
            np.save(path / f"{key}.npy", np.asarray(value))

    def _run_benchmark_process(
        self,
        script: str,
        fixture: str | Path,
        extra_args: tuple[str, ...] = (),
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
            *extra_args,
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

    def _run_benchmark(
        self,
        script: str,
        fixture: str | Path,
        extra_args: tuple[str, ...] = (),
    ) -> str:
        result = self._run_benchmark_process(script, fixture, extra_args)
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
        self.assertIn("input kind: npz", output)
        self.assertIn("layer optical properties: direct input", output)
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
        self.assertIn("input kind: npz", output)
        self.assertIn("layer optical properties: direct input", output)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("thermal source: temperature (wavenumber_band_cm_inv)", output)
        self.assertIn("preprocessing total:", output)
        self.assertIn("emissivity: direct input", output)
        if has_torch():
            self.assertIn("torch-cpu-float64-forward", output)
        self.assertIn("max abs diff", output)

    def test_scene_profile_inputs_generate_layer_optics(self) -> None:
        common = _load_example_module("_full_spectrum_benchmark_common")
        bundle = {
            "wavelengths": np.array([500.0, 600.0]),
            "heights": np.array([2.0, 1.0, 0.0]),
            "pressure_hpa": np.array([100.0, 500.0, 1000.0]),
            "temperature_k": np.array([220.0, 260.0, 290.0]),
            "gas_vmr": np.array([[1.0e-8], [2.0e-8], [3.0e-8]]),
            "gas_cross_sections": np.array([[1.0e-22], [2.0e-22]]),
            "aerosol_loadings": np.array([[2.0], [1.0]]),
            "aerosol_wavelengths_microns": np.array([0.4, 0.5, 0.7]),
            "aerosol_select_wavelength_microns": np.array(0.5),
            "aerosol_bulk_iops": np.array(
                [
                    [[10.0], [20.0], [40.0]],
                    [[5.0], [10.0], [20.0]],
                ]
            ),
        }

        layer_keys = common.select_layer_optical_keys(
            set(bundle),
            total_key="tau",
            ssa_key="omega",
        )
        self.assertTrue(common.layer_optical_keys_are_scene(layer_keys))

        prepared, _, mode = common.prepare_layer_optical_properties(
            bundle,
            total_key="tau",
            ssa_key="omega",
        )

        self.assertEqual(mode, "python-generated from scene/profile inputs")
        self.assertEqual(prepared["tau"].shape, (2, 2))
        self.assertEqual(prepared["omega"].shape, (2, 2))
        self.assertEqual(prepared["depol"].shape, (2,))
        self.assertEqual(prepared["rayleigh_fraction"].shape, (2, 2))
        self.assertEqual(prepared["aerosol_fraction"].shape, (2, 2, 1))

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
            trimmed = Path(tmpdir) / "uv_component_optics"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau"]),
                    ssa=np.array(data["omega"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
            self._write_array_dir(trimmed, arrays)
            output = self._run_benchmark(
                "benchmark_uv_full_spectrum.py",
                trimmed,
                extra_args=("--require-python-generated-inputs",),
            )
        self.assertIn("input kind: array-directory", output)
        self.assertIn(
            "layer optical properties: python-generated from component optical depths",
            output,
        )
        self.assertIn("optical preprocessing: python-generated", output)

    def test_strict_generated_input_mode_rejects_legacy_npz_input_store(self) -> None:
        for script, fixture in (
            ("benchmark_uv_full_spectrum.py", "uv_benchmark_fixture.npz"),
            ("benchmark_tir_full_spectrum.py", "tir_benchmark_fixture.npz"),
        ):
            with self.subTest(script=script):
                result = self._run_benchmark_process(
                    script,
                    fixture,
                    extra_args=("--require-python-generated-inputs",),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("array-directory input store", result.stderr)

    def test_strict_generated_input_mode_rejects_direct_layer_inputs(self) -> None:
        for script, fixture in (
            ("benchmark_uv_full_spectrum.py", "uv_benchmark_fixture.npz"),
            ("benchmark_tir_full_spectrum.py", "tir_benchmark_fixture.npz"),
        ):
            with self.subTest(script=script):
                fixture_path = ROOT / "src" / "py2sess" / "data" / "benchmark" / fixture
                with np.load(fixture_path) as data, tempfile.TemporaryDirectory() as tmpdir:
                    input_dir = Path(tmpdir) / "direct_inputs"
                    arrays = {key: np.array(data[key]) for key in data.files}
                    self._write_array_dir(input_dir, arrays)
                    result = self._run_benchmark_process(
                        script,
                        input_dir,
                        extra_args=("--require-python-generated-inputs",),
                    )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("strict generated-input mode requires component", result.stderr)

    def test_tir_strict_mode_uses_python_generated_runtime_inputs(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {
            "tau_arr",
            "omega_arr",
            "rayleigh_fraction",
            "aerosol_fraction",
            "asymm_arr",
            "d2s_scaling",
            "thermal_bb_input",
            "surfbb",
        }
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_strict_generated"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau_arr"]),
                    ssa=np.array(data["omega_arr"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
            self._write_array_dir(trimmed, arrays)
            output = self._run_benchmark(
                "benchmark_tir_full_spectrum.py",
                trimmed,
                extra_args=("--require-python-generated-inputs",),
            )
        self.assertIn("input kind: array-directory", output)
        self.assertIn(
            "layer optical properties: python-generated from component optical depths",
            output,
        )
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("thermal source: temperature (wavenumber_band_cm_inv)", output)

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
        self.assertIn("thermal source: temperature (wavenumber_band_cm_inv)", output)

    def test_runtime_minimal_uv_bundle_drops_regenerable_arrays(self) -> None:
        module = _load_script_module("create_runtime_minimal_benchmark_bundle")
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "uv_benchmark_fixture.npz"
        with np.load(fixture) as data:
            arrays = {key: np.array(data[key]) for key in data.files}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau"]),
                    ssa=np.array(data["omega"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
        minimal = module.minimal_input_arrays("uv", arrays)

        for key in ("tau", "omega", "asymm", "scaling", "fo_exact_scatter", "chapman"):
            self.assertNotIn(key, minimal)
        for key in (
            "wavelengths",
            "user_obsgeom",
            "heights",
            "albedo",
            "flux_factor",
            "absorption_tau",
            "rayleigh_scattering_tau",
            "aerosol_scattering_tau",
            "depol",
            "aerosol_moments",
            "ref_total",
        ):
            self.assertIn(key, minimal)

    def test_runtime_minimal_tir_bundle_drops_regenerable_arrays(self) -> None:
        module = _load_script_module("create_runtime_minimal_benchmark_bundle")
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        with np.load(fixture) as data:
            arrays = {key: np.array(data[key]) for key in data.files}
            arrays.update(
                self._component_optical_depths(
                    tau=np.array(data["tau_arr"]),
                    ssa=np.array(data["omega_arr"]),
                    rayleigh_fraction=np.array(data["rayleigh_fraction"]),
                    aerosol_fraction=np.array(data["aerosol_fraction"]),
                )
            )
            arrays["level_temperature_k"] = np.linspace(
                220.0,
                270.0,
                int(data["tau_arr"].shape[1]) + 1,
            )
            arrays["surface_temperature_k"] = np.array([285.0], dtype=float)
            arrays["wavenumber_cm_inv"] = np.linspace(
                700.0,
                900.0,
                int(data["tau_arr"].shape[0]),
            )
        minimal = module.minimal_input_arrays("tir", arrays)

        for key in (
            "tau_arr",
            "omega_arr",
            "asymm_arr",
            "d2s_scaling",
            "thermal_bb_input",
            "surfbb",
        ):
            self.assertNotIn(key, minimal)
        for key in (
            "wavelengths",
            "heights",
            "user_angle",
            "albedo",
            "absorption_tau",
            "rayleigh_scattering_tau",
            "aerosol_scattering_tau",
            "depol",
            "aerosol_moments",
            "level_temperature_k",
            "surface_temperature_k",
            "wavenumber_cm_inv",
            "ref_total",
        ):
            self.assertIn(key, minimal)


if __name__ == "__main__":
    unittest.main()
