from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
