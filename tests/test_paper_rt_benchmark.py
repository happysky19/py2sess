from __future__ import annotations

import csv
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


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaperRtBenchmarkTests(unittest.TestCase):
    def test_synthetic_builders_emit_valid_direct_rt_shapes(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        uv = bench.build_synthetic_uv_case(wavelengths=4, layers=3)
        tir = bench.build_synthetic_tir_case(wavelengths=4, layers=3)

        self.assertEqual(uv.case, "UV")
        self.assertEqual(uv.mode, "solar")
        self.assertEqual(uv.kwargs["tau"].shape, (4, 3))
        self.assertEqual(uv.kwargs["fo_scatter_term"].shape, (4, 3))
        self.assertEqual(uv.kwargs["z"].shape, (4,))
        self.assertTrue(np.isfinite(uv.kwargs["tau"]).all())
        self.assertTrue(np.all(uv.kwargs["tau"] > 0.0))
        self.assertTrue(np.all((uv.kwargs["ssa"] >= 0.0) & (uv.kwargs["ssa"] <= 1.0)))
        self.assertTrue(np.all((uv.kwargs["g"] >= 0.0) & (uv.kwargs["g"] <= 1.0)))

        self.assertEqual(tir.case, "TIR")
        self.assertEqual(tir.mode, "thermal")
        self.assertEqual(tir.kwargs["tau"].shape, (4, 3))
        self.assertEqual(tir.kwargs["planck"].shape, (4, 4))
        self.assertEqual(tir.kwargs["surface_planck"].shape, (4,))
        self.assertTrue(np.isfinite(tir.kwargs["planck"]).all())
        self.assertTrue(np.all(tir.kwargs["tau"] > 0.0))
        self.assertTrue(np.all((tir.kwargs["ssa"] >= 0.0) & (tir.kwargs["ssa"] <= 1.0)))
        self.assertTrue(np.all(tir.kwargs["g"] == 0.45))

    def test_torch_dtype_parser_is_float64_only(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        self.assertEqual(bench._parse_dtypes("float64"), ("float64",))
        with self.assertRaises(ValueError):
            bench._parse_dtypes("float32")

    def test_active_tau_layer_indices_are_deterministic(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        self.assertEqual(bench._active_layer_indices(6, 1).tolist(), [3])
        self.assertEqual(bench._active_layer_indices(6, 3).tolist(), [1, 3, 5])
        self.assertEqual(bench._active_layer_indices(4, 4).tolist(), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            bench._active_layer_indices(3, 4)

    def test_paper_preset_uses_50k_representative_wavelengths(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        self.assertEqual(bench.DEFAULT_BASE_WAVELENGTHS, 50000)
        forward_specs = bench._benchmark_specs(
            layer_counts=(5, bench.DEFAULT_BASE_LAYERS),
            wavelength_counts=(300, 1000),
            base_layers=bench.DEFAULT_BASE_LAYERS,
            base_wavelengths=bench.DEFAULT_BASE_WAVELENGTHS,
            full_grid=False,
        )
        self.assertIn((50000, 5, "layers"), forward_specs)
        self.assertIn((50000, bench.DEFAULT_BASE_LAYERS, "layers"), forward_specs)

        jacobian_specs = bench._jacobian_specs(
            layer_counts=(5, bench.DEFAULT_BASE_LAYERS),
            wavelength_counts=(300, 1000),
            grad_layer_counts=(1, bench.DEFAULT_BASE_LAYERS),
            base_layers=bench.DEFAULT_BASE_LAYERS,
            base_wavelengths=bench.DEFAULT_BASE_WAVELENGTHS,
            full_grid=False,
        )
        self.assertIn((50000, bench.DEFAULT_BASE_LAYERS, 1, "grad_vars", "tau"), jacobian_specs)
        self.assertIn(
            (
                50000,
                bench.DEFAULT_BASE_LAYERS,
                bench.DEFAULT_BASE_LAYERS,
                "omega_grad_vars",
                "omega",
            ),
            jacobian_specs,
        )
        self.assertIn(
            (50000, bench.DEFAULT_BASE_LAYERS, 0, "surface_albedo", "surface_albedo"),
            jacobian_specs,
        )

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_synthetic_jacobian_smoke_has_finite_gradients(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        config = bench.BackendConfig("torch", "Torch CPU", "cpu", "float64")
        for case in (
            bench.build_synthetic_uv_case(wavelengths=2, layers=2),
            bench.build_synthetic_tir_case(wavelengths=2, layers=2),
        ):
            with self.subTest(case=case.case):
                rows = bench._jacobian_runtime_rows(
                    case=case,
                    config=config,
                    sweep_axis="smoke",
                    active_tau_layers=1,
                    warmups=0,
                    repeats=1,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gradient_target"], "tau")
                self.assertEqual(rows[0]["active_tau_layers"], "1")
                self.assertEqual(rows[0]["n_grad_vars"], str(case.wavelengths))
                self.assertGreater(float(rows[0]["forward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["backward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["grad_l2"]), 0.0)
                self.assertTrue(np.isfinite(float(rows[0]["grad_checksum"])))

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_omega_jacobian_smoke_has_finite_gradients(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        config = bench.BackendConfig("torch", "Torch CPU", "cpu", "float64")
        for case in (
            bench.build_synthetic_uv_case(wavelengths=2, layers=2),
            bench.build_synthetic_tir_case(wavelengths=2, layers=2),
        ):
            with self.subTest(case=case.case):
                rows = bench._omega_jacobian_runtime_rows(
                    case=case,
                    config=config,
                    sweep_axis="omega_grad_vars",
                    active_tau_layers=1,
                    warmups=0,
                    repeats=1,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gradient_target"], "omega")
                self.assertEqual(rows[0]["active_tau_layers"], "1")
                self.assertEqual(rows[0]["n_grad_vars"], str(case.wavelengths))
                self.assertGreater(float(rows[0]["forward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["backward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["grad_l2"]), 0.0)
                self.assertTrue(np.isfinite(float(rows[0]["grad_checksum"])))

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_surface_albedo_jacobian_smoke_has_finite_gradients(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        config = bench.BackendConfig("torch", "Torch CPU", "cpu", "float64")
        for case in (
            bench.build_synthetic_uv_case(wavelengths=2, layers=2),
            bench.build_synthetic_tir_case(wavelengths=2, layers=2),
        ):
            with self.subTest(case=case.case):
                rows = bench._surface_albedo_jacobian_runtime_rows(
                    case=case,
                    config=config,
                    sweep_axis="surface_albedo",
                    warmups=0,
                    repeats=1,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gradient_target"], "surface_albedo")
                self.assertEqual(rows[0]["active_tau_layers"], "0")
                self.assertEqual(rows[0]["n_grad_vars"], str(case.wavelengths))
                self.assertGreater(float(rows[0]["forward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["backward_seconds"]), 0.0)
                self.assertGreater(float(rows[0]["grad_l2"]), 0.0)
                self.assertTrue(np.isfinite(float(rows[0]["grad_checksum"])))

    def test_benchmark_cli_smoke_writes_raw_summary_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "paper_rt"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "benchmark_paper_rt.py"),
                    "--preset",
                    "smoke",
                    "--groups",
                    "synthetic-forward",
                    "--backend-set",
                    "cpu",
                    "--torch-dtypes",
                    "float64",
                    "--wavelength-counts",
                    "2",
                    "--layer-counts",
                    "2",
                    "--base-wavelengths",
                    "2",
                    "--base-layers",
                    "2",
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                self.fail(
                    "benchmark_paper_rt.py failed\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )

            raw_path = output_dir / "raw_timings.csv"
            summary_path = output_dir / "summary.csv"
            manifest_path = output_dir / "manifest.csv"
            self.assertTrue(raw_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue({row["backend"] for row in rows}.issuperset({"NumPy", "Torch CPU"}))
            self.assertTrue(all(int(row["n_repeats"]) == 2 for row in rows))
            self.assertTrue(all(int(row["levels"]) == int(row["layers"]) + 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
