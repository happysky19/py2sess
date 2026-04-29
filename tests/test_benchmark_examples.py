from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from py2sess.rtsolver.backend import has_torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DATA = ROOT / "src" / "py2sess" / "data" / "benchmark"


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

    def _fixture_arrays(self, fixture: str) -> dict[str, np.ndarray]:
        with np.load(BENCHMARK_DATA / fixture) as data:
            return {key: np.array(data[key]) for key in data.files}

    def _component_array_dir(
        self,
        tmpdir: str,
        *,
        fixture: str,
        omitted: set[str],
        tau_key: str,
        ssa_key: str,
    ) -> Path:
        data = self._fixture_arrays(fixture)
        arrays = {key: value for key, value in data.items() if key not in omitted}
        arrays.update(
            self._component_optical_depths(
                tau=data[tau_key],
                ssa=data[ssa_key],
                rayleigh_fraction=data["rayleigh_fraction"],
                aerosol_fraction=data["aerosol_fraction"],
            )
        )
        input_dir = Path(tmpdir) / "component_inputs"
        self._write_array_dir(input_dir, arrays)
        return input_dir

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
            fixture_path = BENCHMARK_DATA / fixture_path
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

    def test_uv_benchmark_does_not_require_dumped_geometry_or_optics(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "uv_minimal.npz"
            arrays = {
                key: value
                for key, value in self._fixture_arrays("uv_benchmark_fixture.npz").items()
                if key not in omitted
            }
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_uv_full_spectrum.py", trimmed)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)

    def test_uv_benchmark_can_generate_layer_optical_properties(self) -> None:
        omitted = {
            "tau",
            "omega",
            "rayleigh_fraction",
            "aerosol_fraction",
            "asymm",
            "scaling",
            "fo_exact_scatter",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = self._component_array_dir(
                tmpdir,
                fixture="uv_benchmark_fixture.npz",
                omitted=omitted,
                tau_key="tau",
                ssa_key="omega",
            )
            output = self._run_benchmark(
                "benchmark_uv_full_spectrum.py",
                input_dir,
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
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_dir = Path(tmpdir) / "direct_inputs"
                    self._write_array_dir(input_dir, self._fixture_arrays(fixture))
                    result = self._run_benchmark_process(
                        script,
                        input_dir,
                        extra_args=("--require-python-generated-inputs",),
                    )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("strict generated-input mode requires component", result.stderr)

    def test_tir_strict_mode_uses_python_generated_runtime_inputs(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = self._component_array_dir(
                tmpdir,
                fixture="tir_benchmark_fixture.npz",
                omitted=omitted,
                tau_key="tau_arr",
                ssa_key="omega_arr",
            )
            output = self._run_benchmark(
                "benchmark_tir_full_spectrum.py",
                input_dir,
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
        omitted = {"thermal_bb_input", "surfbb", "ref_2s", "ref_fo", "ref_total"}
        data = self._fixture_arrays("tir_benchmark_fixture.npz")
        with tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_temperature_source.npz"
            arrays = {key: value for key, value in data.items() if key not in omitted}
            n_rows = int(data["tau_arr"].shape[0])
            n_levels = int(data["tau_arr"].shape[1]) + 1
            arrays["level_temperature_k"] = np.linspace(220.0, 270.0, n_levels)
            arrays["surface_temperature_k"] = np.array([285.0], dtype=float)
            arrays["wavenumber_cm_inv"] = np.linspace(700.0, 900.0, n_rows)
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn("thermal source: temperature (wavenumber_band_cm_inv)", output)


if __name__ == "__main__":
    unittest.main()
