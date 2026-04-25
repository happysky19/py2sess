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


class BenchmarkExampleTests(unittest.TestCase):
    def _run_benchmark(self, script: str, fixture: str | Path) -> str:
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
        result = subprocess.run(
            command,
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

    def test_uv_full_spectrum_benchmark_smoke(self) -> None:
        output = self._run_benchmark(
            "benchmark_uv_full_spectrum.py",
            "uv_benchmark_fixture.npz",
        )
        self.assertIn("numpy", output)
        self.assertIn("numpy-forward", output)
        self.assertIn("geometry preprocessing: python-generated", output)
        self.assertIn("optical preprocessing: python-generated", output)
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
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("emissivity: bundle", output)
        if has_torch():
            self.assertIn("torch-cpu-float64-forward", output)
        self.assertIn("max abs diff", output)

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

    def test_tir_benchmark_does_not_require_dumped_optics(self) -> None:
        fixture = ROOT / "src" / "py2sess" / "data" / "benchmark" / "tir_benchmark_fixture.npz"
        omitted = {"asymm_arr", "d2s_scaling", "emissivity"}
        with np.load(fixture) as data, tempfile.TemporaryDirectory() as tmpdir:
            trimmed = Path(tmpdir) / "tir_minimal.npz"
            arrays = {key: np.array(data[key]) for key in data.files if key not in omitted}
            np.savez_compressed(trimmed, **arrays)
            output = self._run_benchmark("benchmark_tir_full_spectrum.py", trimmed)
        self.assertIn("optical preprocessing: python-generated", output)
        self.assertIn("emissivity: 1 - albedo", output)

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
