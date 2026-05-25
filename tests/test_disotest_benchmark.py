from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_disotest_flux.py"
    spec = importlib.util.spec_from_file_location("benchmark_disotest_flux", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DisotestBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_benchmark_module()

    def test_official_registry_covers_all_disotest_cases(self) -> None:
        expected = {
            *(f"DISOTEST 1{suffix} isotropic beam" for suffix in "abde"),
            "DISOTEST 1c isotropic top illumination",
            "DISOTEST 1f isotropic top illumination",
            *(f"DISOTEST 2{suffix} Rayleigh beam" for suffix in "abcd"),
            *(f"DISOTEST 3{suffix} HG beam" for suffix in "ab"),
            "DISOTEST 4a Haze-L beam",
            "DISOTEST 4b Haze-L absorbing beam",
            "DISOTEST 4c Haze-L oblique beam",
            "DISOTEST 5a Cloud C.1 conservative",
            "DISOTEST 5b Cloud C.1 absorbing",
            "DISOTEST 6a clear beam",
            "DISOTEST 6b absorbing beam",
            "DISOTEST 6c absorbing surface",
            "DISOTEST 6d absorbing non-Lambert surface",
            "DISOTEST 6e absorbing bottom emission",
            "DISOTEST 6f absorbing top/bottom emission",
            "DISOTEST 6g absorbing internal emission",
            "DISOTEST 6h absorbing thick emission",
            "DISOTEST 7a thermal internal",
            "DISOTEST 7b thermal thick",
            "DISOTEST 7c all sources Lambertian",
            "DISOTEST 7d all sources Hapke",
            "DISOTEST 7e all sources BDR",
            "DISOTEST 8a two-layer isotropic source",
            "DISOTEST 8b two-layer conservative source",
            "DISOTEST 8c two-layer thick source",
            "DISOTEST 9a multilayer isotropic source",
            "DISOTEST 9b multilayer anisotropic source",
            "DISOTEST 9c multilayer all sources",
            "DISOTEST 10a usrang true internal regression",
            "DISOTEST 10b usrang false internal regression",
            "DISOTEST 11a one-layer internal regression",
            "DISOTEST 11b multi-layer internal regression",
            "DISOTEST 12a absorption shortcut reference",
            "DISOTEST 12b absorption shortcut variant",
            "DISOTEST 13a albedo/transmissivity shortcut single",
            "DISOTEST 13b albedo/transmissivity regular single",
            "DISOTEST 13c albedo/transmissivity shortcut multi",
            "DISOTEST 13d albedo/transmissivity regular multi",
            "DISOTEST 14a disort/twostr reference",
            "DISOTEST 14b twostr comparison",
        }
        names = [case.name for case in self.module._selected_cases("disotest")]
        self.assertEqual(len(names), 48)
        self.assertEqual(len(set(names)), 48)
        self.assertEqual(set(names), expected)

    def test_official_registry_marks_publicly_runnable_subset(self) -> None:
        cases = self.module._selected_cases("disotest")
        runnable = [case for case in cases if case.unsupported_reason is None]
        unsupported = [case for case in cases if case.unsupported_reason is not None]
        self.assertEqual(len(runnable), 26)
        self.assertEqual(len(unsupported), 22)
        self.assertIn("DISOTEST 1c isotropic top illumination", {case.name for case in runnable})
        self.assertIn("DISOTEST 8c two-layer thick source", {case.name for case in runnable})
        self.assertIn("DISOTEST 4a Haze-L beam", {case.name for case in runnable})
        self.assertIn("DISOTEST 5b Cloud C.1 absorbing", {case.name for case in runnable})
        self.assertIn("DISOTEST 6d absorbing non-Lambert surface", {case.name for case in runnable})

    def test_disort_test_suite_aliases_runnable_disotest_subset(self) -> None:
        self.assertEqual(
            self.module._selected_cases("disort-test"),
            self.module._selected_cases("disotest-runnable"),
        )

    def test_top_isotropic_cases_use_benchmark_as_pydisort_reference(self) -> None:
        cases = {
            case.name: case
            for case in self.module._selected_cases("disotest")
            if case.name
            in {
                "DISOTEST 1c isotropic top illumination",
                "DISOTEST 1f isotropic top illumination",
                "DISOTEST 8a two-layer isotropic source",
                "DISOTEST 8b two-layer conservative source",
                "DISOTEST 8c two-layer thick source",
            }
        }
        self.assertEqual(len(cases), 5)
        for case in cases.values():
            benchmark = self.module._benchmark_flux(case)
            pydisort = self.module._run_pydisort(case)
            py2sess = self.module._run_py2sess(case)

            self.assertGreater(case.fisot, 0.0)
            np.testing.assert_allclose(pydisort["flux_down"], benchmark["flux_down"])
            self.assertFalse(np.isnan(py2sess["flux_down"]).any())

    def test_surface_first_cases_are_registered_as_diagnostic(self) -> None:
        cases = {case.name: case for case in self.module._selected_cases("disotest-surface")}
        self.assertEqual(
            set(cases),
            {
                "DISOTEST 6d absorbing non-Lambert surface",
                "DISOTEST 7d all sources Hapke",
                "DISOTEST 7e all sources BDR",
            },
        )

        test6d = cases["DISOTEST 6d absorbing non-Lambert surface"]
        self.assertEqual(test6d.surface_model, "hapke")
        self.assertIsNone(test6d.unsupported_category)
        self.assertIsNone(test6d.unsupported_reason)

        self.assertEqual(cases["DISOTEST 7d all sources Hapke"].surface_model, "hapke")
        self.assertEqual(
            cases["DISOTEST 7d all sources Hapke"].unsupported_category,
            "mixed_sources",
        )
        self.assertEqual(cases["DISOTEST 7e all sources BDR"].surface_model, "disort-bdr-function")

    def test_surface_diagnostic_uses_benchmark_as_pydisort_reference(self) -> None:
        case = self.module._selected_cases("disotest-surface")[0]
        benchmark = self.module._benchmark_flux(case)
        pydisort = self.module._run_pydisort(case)
        py2sess = self.module._run_py2sess(case)

        np.testing.assert_allclose(pydisort["flux_up"], benchmark["flux_up"])
        self.assertFalse(np.isnan(py2sess["flux_up"]).any())
        np.testing.assert_allclose(py2sess["flux_up"], benchmark["flux_up"], rtol=5.0e-4)

    def test_vijay_section6_fixture_has_expected_paper_rows(self) -> None:
        self.assertEqual(len(self.module.VIJAY_SECTION6_2SESS_FLUXES), 15)

    def test_vijay_section6_known_rows_match_zenodo_outputs(self) -> None:
        cases = {case.name: case for case in self.module._selected_cases("disort-test")}

        test1a = self.module._vijay_section6_flux(cases["DISOTEST 1a isotropic beam"])
        np.testing.assert_allclose(test1a["flux_up"][0], math.pi * 2.62195e-2)
        np.testing.assert_allclose(test1a["flux_down"][0], math.pi)

        test6c = self.module._vijay_section6_flux(cases["DISOTEST 6c absorbing surface"])
        np.testing.assert_allclose(test6c["flux_up"], [9.15782e-1, 2.48935, 6.76676])

    def test_tabulated_phase_cases_use_two_stream_moment_convention(self) -> None:
        cases = {case.name: case for case in self.module._selected_cases("disort-test")}
        haze = cases["DISOTEST 4a Haze-L beam"]
        cloud = cases["DISOTEST 5b Cloud C.1 absorbing"]

        np.testing.assert_allclose(haze.g, [2.41260 / 3.0])
        np.testing.assert_allclose(haze.delta_m_truncation_factor, [3.23047 / 5.0])
        np.testing.assert_allclose(cloud.g, [2.544 / 3.0])
        np.testing.assert_allclose(cloud.delta_m_truncation_factor, [3.883 / 5.0])
        self.assertEqual(cloud.compare_level_indices, (1, 2, 3))
        self.assertEqual(cloud.level_names, ("3.2", "12.8", "48"))

    def test_paper_table_rows_use_disort_py2sess_error_columns(self) -> None:
        rows = [
            {
                "case": "fixture",
                "field": "flux_up",
                "level": "TOA",
                "benchmark": 0.0,
                "pydisort": 0.0,
                "py2sess": 0.0,
                "py2sess_rel_percent": 0.0,
                "status": "run",
                "fo_flux_n_mu": 8,
                "py2sess_backend": "numpy",
            }
        ]
        paper_rows = self.module._paper_table_rows(rows)

        self.assertEqual(
            set(paper_rows[0]),
            {"case", "quantity", "level", "DISORT", "py2sess", "percent_error"},
        )
        self.assertEqual(rows[0]["fo_flux_n_mu"], 8)
        self.assertEqual(rows[0]["py2sess_backend"], "numpy")
        self.assertEqual(paper_rows[0]["DISORT"], rows[0]["benchmark"])
        self.assertEqual(paper_rows[0]["DISORT"], 0.0)
        self.assertEqual(paper_rows[0]["py2sess"], 0.0)
        self.assertEqual(paper_rows[0]["percent_error"], 0.0)
        self.assertEqual(self.module._format_percent(-0.004), "0")
        self.assertEqual(self.module._format_percent(0.004), "0")
        self.assertEqual(self.module._format_percent(0.005), "0.01")


if __name__ == "__main__":
    unittest.main()
