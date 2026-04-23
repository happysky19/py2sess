from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from py2sess.core.backend import has_torch


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkExampleTests(unittest.TestCase):
    def _run_benchmark(self, script: str, fixture: str) -> str:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        backend = "both" if has_torch() else "numpy"
        command = [
            sys.executable,
            str(ROOT / "examples" / script),
            str(ROOT / "src" / "py2sess" / "data" / "benchmark" / fixture),
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
        if has_torch():
            self.assertIn("torch-cpu-float64-forward", output)
        self.assertIn("max abs diff", output)


if __name__ == "__main__":
    unittest.main()
