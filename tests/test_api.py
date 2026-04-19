from __future__ import annotations

import unittest

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    load_tir_benchmark_case,
    load_uv_benchmark_case,
    thermal_source_from_temperature_profile,
)


class ApiTests(unittest.TestCase):
    def test_package_exports_are_available(self) -> None:
        solver = TwoStreamEss(TwoStreamEssOptions(n_layers=3))
        self.assertEqual(solver.options.n_layers, 3)

    def test_reference_case_loaders_return_expected_dimensions(self) -> None:
        tir = load_tir_benchmark_case()
        uv = load_uv_benchmark_case()
        self.assertEqual(tir.n_layers, 114)
        self.assertEqual(uv.n_layers, 114)
        self.assertGreater(tir.n_wavelengths, 0)
        self.assertGreater(uv.n_wavelengths, 0)

    def test_thermal_source_helper_returns_expected_sizes(self) -> None:
        source = thermal_source_from_temperature_profile(
            [220.0, 230.0, 240.0, 250.0],
            280.0,
            wavenumber_band_cm_inv=(900.0, 901.0),
        )
        self.assertEqual(source.thermal_bb_input.shape, (4,))
        self.assertGreater(source.surfbb, 0.0)


if __name__ == "__main__":
    unittest.main()
