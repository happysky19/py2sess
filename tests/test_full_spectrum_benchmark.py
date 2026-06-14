from __future__ import annotations

import importlib.util
import argparse
import os
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FullSpectrumBenchmarkTests(unittest.TestCase):
    def test_parse_fortran_timing_text_handles_uv_and_tir_labels(self) -> None:
        bench = load_script("benchmark_full_spectrum_rt", "scripts/benchmark_full_spectrum_rt.py")
        text = """
 In Exact module:
   ExactRTMSetUps    =      0.54830933
   FOGeomTime        =      0.00062180
   FOSpherFnTime     =      0.00001526
   2SGeomTime        =      0.00021744
   ExactFO2SOpTime   =      3.99079895
   ExactFOCalcTime   =      7.59503174
   Exact2SCalcTime   =     14.49488068
 ExactModuleTime (2) =     26.99209976
 WriteTime (3)       =      0.39884186
 OverallRunTime      =     81.17389679
"""
        metrics = bench._parse_fortran_timing_text(text)
        self.assertAlmostEqual(metrics["module_seconds"], 26.99209976)
        self.assertAlmostEqual(metrics["fo_seconds"], 7.59503174)
        self.assertAlmostEqual(metrics["two_stream_seconds"], 14.49488068)
        self.assertGreater(metrics["setup_seconds"], 0.548)

    def test_summarize_rows_computes_best_and_component_means(self) -> None:
        bench = load_script(
            "benchmark_full_spectrum_rt_summary", "scripts/benchmark_full_spectrum_rt.py"
        )
        rows = [
            {
                "status": "ok",
                "system": "py2sess",
                "case": "TIR",
                "mode": "thermal",
                "backend": "NumPy",
                "device": "",
                "dtype": "float64",
                "timing_kind": "components",
                "seconds": 2.0,
                "fo_seconds": 0.5,
                "two_stream_seconds": 1.5,
                "wavelengths": 200000,
                "layers": 114,
                "chunk_size": 104000,
            },
            {
                "status": "ok",
                "system": "py2sess",
                "case": "TIR",
                "mode": "thermal",
                "backend": "NumPy",
                "device": "",
                "dtype": "float64",
                "timing_kind": "components",
                "seconds": 3.0,
                "fo_seconds": 0.7,
                "two_stream_seconds": 2.3,
                "wavelengths": 200000,
                "layers": 114,
                "chunk_size": 104000,
            },
        ]
        summary = bench._summarize(rows, "now")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["best_s"], 2.0)
        self.assertAlmostEqual(summary[0]["total_mean_s"], 2.5)
        self.assertAlmostEqual(summary[0]["fo_mean_s"], 0.6)
        self.assertAlmostEqual(summary[0]["two_stream_mean_s"], 1.9)

    def test_level_flux_timing_flags_select_scene_forward_fluxes(self) -> None:
        bench = load_script(
            "benchmark_full_spectrum_rt_flags", "scripts/benchmark_full_spectrum_rt.py"
        )

        args = argparse.Namespace(
            timing_kinds=None,
            components=False,
            output_levels=True,
            output_fluxes=True,
        )
        self.assertEqual(bench._normalize_timing_kinds(args), ("level-fluxes",))

        args.components = True
        self.assertEqual(bench._normalize_timing_kinds(args), ("level-fluxes", "components"))

    def test_level_flux_timing_uses_flux_only_scene_path_for_torch_runtime(self) -> None:
        bench = load_script(
            "benchmark_full_spectrum_rt_flux_only", "scripts/benchmark_full_spectrum_rt.py"
        )

        class DummyTorch:
            cuda = mock.Mock()

            def set_num_threads(self, threads: int) -> None:
                self.threads = threads

        class DummyScene:
            def __init__(self) -> None:
                self.forward_called = False
                self.forward_flux_options = None

            def forward(self, **kwargs):
                self.forward_called = True
                return mock.Mock(radiance_total=np.zeros(4))

            def forward_flux(self, **kwargs):
                self.forward_flux_options = kwargs
                return mock.Mock(flux_up=np.zeros((4, 3)), flux_down=np.zeros((4, 3)))

        inputs = mock.Mock(
            wavelengths=np.arange(4),
            kwargs={"tau": np.zeros((4, 2))},
            mode="solar",
            reference_total=np.zeros(4),
        )
        for backend in ("native", "torch"):
            scene = DummyScene()
            config = bench.BackendConfig(
                backend=backend,
                label=backend,
                device="cpu",
                dtype="float64",
            )
            with (
                mock.patch.object(bench, "_torch_module", return_value=DummyTorch()),
                mock.patch.object(bench, "native_backend_supports_device", return_value=True),
            ):
                row = bench._run_scene_forward_once(
                    scene,
                    inputs,
                    config,
                    torch_threads=4,
                    torch_bvp_engine="auto",
                    numpy_bvp_engine="auto",
                    output_levels=True,
                    output_fluxes=True,
                    fo_flux_n_mu=8,
                )
            self.assertFalse(scene.forward_called)
            self.assertIsNotNone(scene.forward_flux_options)
            self.assertNotIn("output_levels", scene.forward_flux_options)
            self.assertNotIn("output_fluxes", scene.forward_flux_options)
            self.assertTrue(scene.forward_flux_options["plane_parallel"])
            self.assertTrue(scene.forward_flux_options["include_fo"])
            self.assertTrue(scene.forward_flux_options["return_net"])
            self.assertEqual(row["max_abs_diff"], "")

    def test_pydisort_flux_export_requires_portable_input_root(self) -> None:
        exporter = load_script(
            "export_pydisort_full_spectrum_flux_nc",
            "scripts/export_pydisort_full_spectrum_flux_nc.py",
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--input-root"):
                exporter._case_specs(None)

        with mock.patch.dict(os.environ, {"PY2SESS_EXTERNAL_ROOT": "/tmp/2s-ess"}):
            specs = exporter._case_specs(None)
        self.assertEqual(
            specs["tir"].profile,
            Path("/tmp/2s-ess/geocape_data/Profile_Data/Profiles_1_2006726_0000.dat"),
        )
        self.assertEqual(specs["tir"].scene, ROOT / "benchmark_bundles" / "tir_scene_python.yaml")

        bundled = exporter._case_specs(Path("/tmp/input_bundle"))
        self.assertEqual(
            bundled["uv"].profile,
            Path("/tmp/input_bundle/profiles/Profiles_1_2006726_1500.dat"),
        )


if __name__ == "__main__":
    unittest.main()
