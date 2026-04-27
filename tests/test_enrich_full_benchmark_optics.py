from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from py2sess.optical.createprops import parse_createprops_dump


class EnrichFullBenchmarkOpticsTests(unittest.TestCase):
    def test_tir_dump_parser_adds_temperature_and_wavenumber_band_inputs(self) -> None:
        dump_text = "\n".join(
            [
                "2 2 1 49.0 0.0 2",
                "2.0 0.0",
                "1.0 0.0",
                "0.0 0.0",
                "0 1 1 1 1 1 1 1 1 1 1",
                "1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0",
                "1 14285.0 500.0 0.1 0.03 0.2 0.01",
                "1 0.01 0.02 0.5 0.4 0.1 0.2 0.3 0.4 0.0 0.01",
                "2 0.02 0.03 0.6 0.5 0.2 0.3 0.4 0.5 0.0 0.02",
                "2 14280.0 501.0 0.1 0.03 0.2 0.01",
                "1 0.01 0.02 0.5 0.4 0.1 0.2 0.3 0.4 0.0 0.01",
                "2 0.02 0.03 0.6 0.5 0.2 0.3 0.4 0.5 0.0 0.02",
            ]
        )
        profile_text = "\n".join(
            [
                "surfaceTemperature(K) = 299.0",
                "1 1000.0 300.0",
                "2 500.0 250.0",
                "3 1.0 200.0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "Dump_1_26_0000.dat_25"
            profile_path = Path(tmpdir) / "profile.dat"
            dump_path.write_text(dump_text, encoding="utf-8")
            profile_path.write_text(profile_text, encoding="utf-8")

            parsed = parse_createprops_dump(dump_path, kind="tir", profile_file=profile_path)

        np.testing.assert_allclose(parsed["wavenumber_cm_inv"], [500.0, 501.0])
        np.testing.assert_allclose(
            parsed["wavenumber_band_cm_inv"],
            [[499.5, 500.5], [500.5, 501.5]],
        )
        np.testing.assert_allclose(parsed["level_temperature_k"], [200.0, 250.0, 300.0])
        np.testing.assert_allclose(parsed["surface_temperature_k"], [299.0])


if __name__ == "__main__":
    unittest.main()
