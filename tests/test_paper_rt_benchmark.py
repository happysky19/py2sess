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
        self.assertGreater(float(np.ptp(uv.kwargs["tau"])), 0.0)
        self.assertTrue(np.all((uv.kwargs["ssa"] > 0.0) & (uv.kwargs["ssa"] <= 1.0)))
        self.assertTrue(np.all((uv.kwargs["g"] >= 0.0) & (uv.kwargs["g"] < 1.0)))

        self.assertEqual(tir.case, "TIR")
        self.assertEqual(tir.mode, "thermal")
        self.assertEqual(tir.kwargs["tau"].shape, (4, 3))
        self.assertEqual(tir.kwargs["planck"].shape, (4, 4))
        self.assertEqual(tir.kwargs["surface_planck"].shape, (4,))
        self.assertTrue(np.isfinite(tir.kwargs["planck"]).all())
        self.assertTrue(np.all(tir.kwargs["tau"] > 0.0))
        self.assertGreater(float(np.ptp(tir.kwargs["tau"])), 0.0)
        self.assertTrue(np.all((tir.kwargs["ssa"] >= 0.0) & (tir.kwargs["ssa"] <= 1.0)))
        self.assertTrue(np.all((tir.kwargs["g"] >= 0.0) & (tir.kwargs["g"] < 1.0)))

    def test_torch_dtype_parser_is_float64_only(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        self.assertEqual(bench._parse_dtypes("float64"), ("float64",))
        with self.assertRaises(ValueError):
            bench._parse_dtypes("float32")

    def test_jacobian_target_parser_supports_retrieval_like_targets(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        self.assertEqual(
            bench._parse_jacobian_targets("tau,ssa,phase,emissivity,albedo"),
            ("tau", "omega", "g", "surface_emissivity", "surface_albedo"),
        )

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
            (
                50000,
                bench.DEFAULT_BASE_LAYERS,
                bench.DEFAULT_BASE_LAYERS,
                "g_grad_vars",
                "g",
            ),
            jacobian_specs,
        )
        self.assertIn(
            (50000, bench.DEFAULT_BASE_LAYERS, 0, "surface_albedo", "surface_albedo"),
            jacobian_specs,
        )
        self.assertIn(
            (50000, bench.DEFAULT_BASE_LAYERS, 0, "surface_emissivity", "surface_emissivity"),
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
    def test_g_jacobian_smoke_has_finite_gradients(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        config = bench.BackendConfig("torch", "Torch CPU", "cpu", "float64")
        for case in (
            bench.build_synthetic_uv_case(wavelengths=2, layers=2),
            bench.build_synthetic_tir_case(wavelengths=2, layers=2),
        ):
            with self.subTest(case=case.case):
                rows = bench._g_jacobian_runtime_rows(
                    case=case,
                    config=config,
                    sweep_axis="g_grad_vars",
                    active_tau_layers=1,
                    warmups=0,
                    repeats=1,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gradient_target"], "g")
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

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_surface_emissivity_jacobian_smoke_has_finite_gradients(self) -> None:
        bench = _load_script("benchmark_paper_rt.py")
        config = bench.BackendConfig("torch", "Torch CPU", "cpu", "float64")
        rows = bench._surface_emissivity_jacobian_runtime_rows(
            case=bench.build_synthetic_tir_case(wavelengths=2, layers=2),
            config=config,
            sweep_axis="surface_emissivity",
            warmups=0,
            repeats=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gradient_target"], "surface_emissivity")
        self.assertEqual(rows[0]["active_tau_layers"], "0")
        self.assertEqual(rows[0]["n_grad_vars"], "2")
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

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_jacobian_validation_summary_generator_is_finite(self) -> None:
        validation = _load_script("generate_jacobian_validation_summary.py")
        rows = validation.generate_rows()

        self.assertEqual(len(rows), 7)
        self.assertEqual(
            {row["regime"] for row in rows},
            {"UV solar", "TIR thermal", "Solar/VNIR", "Thermal"},
        )
        for row in rows:
            self.assertIn(
                row["case_size"],
                {"2 wavelengths x 3 layers", "1000 wavelengths x 50 layers"},
            )
            self.assertGreater(float(row["max_abs_fd"]), 0.0)
            self.assertLess(float(row["max_abs_error"]), 1.0e-8)
            self.assertLess(float(row["max_rel_error"]), 2.0e-3)
            self.assertLess(float(row["mean_rel_error"]), 3.0e-4)
            self.assertEqual(row["n_anomalous_columns"], "0")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "jacobian_validation.csv"
            validation.write_csv(output, rows)
            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written, rows)

    def test_jacobian_overhead_export_has_selected_figure_rows(self) -> None:
        exporter = _load_script("export_jacobian_overhead_summary.py")
        rows = exporter.build_rows(ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv")

        self.assertTrue(rows)
        selected = [row for row in rows if row["selected_for_figure"] == "true"]
        self.assertTrue(selected)
        selected_categories = {row["figure_category"] for row in selected}
        self.assertTrue(
            {
                "surface_albedo",
                "tau_1_layer",
                "tau_all_layers",
                "omega_1_layer",
                "omega_all_layers",
            }.issubset(selected_categories),
        )
        self.assertTrue(
            selected_categories.issubset(
                {
                    "surface_albedo",
                    "tau_1_layer",
                    "tau_all_layers",
                    "omega_1_layer",
                    "omega_all_layers",
                    "g_1_layer",
                    "g_all_layers",
                    "surface_emissivity",
                }
            )
        )
        for row in rows:
            self.assertGreater(float(row["forward_mean_s"]), 0.0)
            self.assertGreater(float(row["jacobian_mean_s"]), 0.0)
            self.assertGreater(float(row["jacobian_forward_ratio"]), 1.0)
            self.assertEqual(len(row["source_summary_sha256"]), 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "overhead.csv"
            exporter.write_csv(output, rows)
            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written, rows)

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_delta_m_chain_rule_diagnostic_is_finite(self) -> None:
        diagnostic = _load_script("analyze_delta_m_chain_rule.py")
        rows = diagnostic.generate_rows()

        self.assertTrue(rows)
        metrics = {(row["diagnostic"], row["regime"], row["metric"]) for row in rows}
        self.assertIn(
            (
                "delta_m_chain_rule",
                "Solar/VNIR",
                "omitted_chain_rule_relative_l2",
            ),
            metrics,
        )
        self.assertIn(
            (
                "thermal_fo_source_convention",
                "Thermal",
                "max_relative_radiance_difference_percent",
            ),
            metrics,
        )
        for row in rows:
            self.assertTrue(float(row["value"]) >= 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "delta_m.csv"
            diagnostic.write_csv(output, rows)
            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written, rows)

    def test_paper_rt_claim_audit_matches_assets(self) -> None:
        paper = ROOT / "docs" / "py2sess_rt_benchmark_paper.tex"
        if not paper.exists():
            self.skipTest("paper draft is not present")
        audit = _load_script("audit_paper_rt_claims.py")

        missing = audit.audit(
            paper_tex=paper,
            full_summary_csv=ROOT / "docs" / "assets" / "full_spectrum_paper_summary.csv",
            spectrum_csv=ROOT / "docs" / "assets" / "full_spectrum_spectrum_comparison.csv",
            validation_csv=ROOT / "docs" / "assets" / "jacobian_gradient_validation_summary.csv",
            overhead_csv=ROOT
            / "docs"
            / "assets"
            / "paper_rt_synthetic_jacobian_overhead_summary.csv",
            combined_summary_csv=ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv",
        )
        self.assertEqual(missing, [])

    def test_paper_artifact_manifest_has_hashes(self) -> None:
        manifest = _load_script("generate_paper_artifact_manifest.py")
        rows = manifest.build_rows()

        self.assertEqual(len(rows), len(manifest.DEFAULT_PATHS))
        paths = {row["path"] for row in rows}
        self.assertEqual(paths, set(manifest.DEFAULT_PATHS))
        self.assertIn("docs/paper_rt_release_checklist.md", paths)
        self.assertIn("docs/assets/paper_rt_all_timing_summary.csv", paths)
        self.assertIn("scripts/benchmark_paper_rt.py", paths)
        self.assertIn("scripts/export_full_spectrum_benchmark_bundle.py", paths)
        for row in rows:
            self.assertIn(row["kind"], {"document", "figure", "script", "table"})
            self.assertGreater(int(row["bytes"]), 0)
            self.assertEqual(len(row["sha256"]), 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.csv"
            manifest.write_csv(output, rows)
            with output.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written, rows)

    def test_paper_artifact_manifest_fails_for_missing_artifact(self) -> None:
        manifest = _load_script("generate_paper_artifact_manifest.py")
        with self.assertRaises(FileNotFoundError):
            manifest.build_rows(("docs/assets/not-a-paper-file.csv",))
        self.assertEqual(
            manifest.build_rows(("docs/assets/not-a-paper-file.csv",), allow_missing=True),
            [],
        )

    def test_paper_artifact_manifest_can_require_clean_git(self) -> None:
        manifest = _load_script("generate_paper_artifact_manifest.py")
        original_run_git = manifest._run_git
        try:
            manifest._run_git = lambda args: "abc123" if args[0] == "rev-parse" else " M file"
            with self.assertRaises(RuntimeError):
                manifest.build_rows(
                    ("docs/py2sess_rt_benchmark_paper.tex",),
                    require_clean_git=True,
                )
        finally:
            manifest._run_git = original_run_git

    def test_paper_archive_manifest_has_tracked_assets_by_default(self) -> None:
        archive = _load_script("prepare_paper_archive_manifest.py")
        rows = archive.build_rows()

        self.assertTrue(rows)
        paths = {row["path"] for row in rows}
        self.assertIn("docs/py2sess_rt_benchmark_paper.tex", paths)
        self.assertIn("docs/paper_rt_release_checklist.md", paths)
        self.assertIn("scripts/prepare_paper_archive_manifest.py", paths)
        self.assertFalse(any(path.startswith("outputs/") for path in paths))
        self.assertTrue({"paper-table", "script"} <= {row["kind"] for row in rows})
        for row in rows:
            self.assertEqual(row["required_for_submission"], "true")
            self.assertGreater(int(row["bytes"]), 0)
            self.assertEqual(len(row["sha256"]), 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "archive.csv"
            readme = Path(tmpdir) / "archive.md"
            archive.write_csv(output, rows)
            archive.write_readme(readme, rows)
            self.assertTrue(output.exists())
            self.assertTrue(readme.exists())

    def test_paper_archive_manifest_can_include_raw_outputs(self) -> None:
        archive = _load_script("prepare_paper_archive_manifest.py")
        raw = ROOT / "outputs" / "full_spectrum_m2pro_cpu" / "raw_full_spectrum_timings.csv"
        if not raw.exists():
            self.skipTest("local raw paper benchmark outputs are not present")

        rows = archive.build_rows(include_raw_outputs=True)
        paths = {row["path"] for row in rows}
        self.assertIn("outputs/full_spectrum_m2pro_cpu/raw_full_spectrum_timings.csv", paths)
        self.assertIn("outputs/synthetic_rt_m2pro_cpu_50k/raw_timings.csv", paths)
        self.assertIn("outputs/full_spectrum_benchmark/input_bundle/manifest.csv", paths)
        self.assertIn(
            "outputs/full_spectrum_m2pro_cpu/fortran_bundle/TIR/2S-ESS/Results_Exact_Opt/Exact_D01_Aer_ES_V49SI25_L1_D26_T0000.Tim",
            paths,
        )
        self.assertIn(
            "outputs/full_spectrum_m2pro_cpu/fortran_bundle/UVVSWIR/2S-ESS/Results_Exact_Opt/Exact_D01_Aer_ES_Obsg_S47V49A275SI11L1D26T1500.Tim",
            paths,
        )
        self.assertTrue(
            {"fortran-timing", "fortran-bundle", "input-bundle", "raw-benchmark-output"}
            <= {row["kind"] for row in rows}
        )

    def test_paper_archive_manifest_can_require_clean_git(self) -> None:
        archive = _load_script("prepare_paper_archive_manifest.py")
        original_git = archive._git
        try:
            archive._git = lambda args: "abc123" if args[0] == "rev-parse" else " M file"
            with self.assertRaises(RuntimeError):
                archive.build_rows(
                    ("docs/py2sess_rt_benchmark_paper.tex",),
                    require_clean_git=True,
                )
        finally:
            archive._git = original_git

    def test_publication_plot_scripts_create_png_outputs(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is required for plot smoke tests")

        paper_plot = _load_script("plot_paper_rt_benchmarks.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "synthetic"
            outputs = paper_plot.plot(
                ROOT / "docs" / "assets" / "paper_rt_all_timing_summary.csv",
                output_dir,
                ("png", "eps"),
            )
            names = {path.name for path in outputs}
            self.assertIn("paper_rt_synthetic_forward_publication.png", names)
            self.assertIn("paper_rt_synthetic_forward_publication.eps", names)
            self.assertIn("paper_rt_synthetic_jacobian_publication.png", names)
            self.assertIn("paper_rt_synthetic_jacobian_publication.eps", names)
            self.assertIn("paper_rt_synthetic_jacobian_overhead_publication.png", names)
            self.assertIn("paper_rt_synthetic_jacobian_overhead_publication.eps", names)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "full"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plot_full_spectrum_rt_benchmarks.py"),
                    "--summary",
                    str(ROOT / "docs" / "assets" / "full_spectrum_paper_summary.csv"),
                    "--spectrum-csv",
                    str(ROOT / "docs" / "assets" / "full_spectrum_spectrum_comparison.csv"),
                    "--output-dir",
                    str(output_dir),
                    "--formats",
                    "png,eps",
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
                    "plot_full_spectrum_rt_benchmarks.py failed\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            self.assertTrue((output_dir / "full_spectrum_rt_runtime.png").exists())
            self.assertTrue((output_dir / "full_spectrum_rt_runtime.eps").exists())
            self.assertTrue((output_dir / "full_spectrum_spectrum_comparison.png").exists())
            self.assertTrue((output_dir / "full_spectrum_spectrum_comparison.eps").exists())


if __name__ == "__main__":
    unittest.main()
