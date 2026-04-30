from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from py2sess.optical.geocape import (
    _interp_indices,
    gas_cross_section_from_table,
    load_geocape_aerosol_loadings,
    load_geocape_aerosol_tables,
)
from py2sess.optical.scene import _interp_aerosol_table
from py2sess.optical.scene_io import _top_to_bottom, build_benchmark_scene_inputs


class GeocapeInputTests(unittest.TestCase):
    def test_gas_cross_section_table_interpolates_and_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            table = Path(tmpdir) / "xsec.txt"
            table.write_text("300 2\n400 4\n500 8\n")

            values = gas_cross_section_from_table(
                table,
                np.array([350.0, 450.0]),
                scale=1.0e-20,
            )

        np.testing.assert_allclose(values, [3.0e-20, 6.0e-20])

    def test_aerosol_loading_reader_matches_fortran_layer_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aer.dat"
            path.write_text("1 10 100 1000\n2 20 200 2000\n3 30 300 3000\n")

            loadings = load_geocape_aerosol_loadings(
                [path],
                n_layers=5,
                select_index=2,
                active_layers=3,
            )

        np.testing.assert_allclose(loadings[:, 0], [0.0, 0.0, 30.0, 20.0, 10.0])

    def test_aerosol_ssprops_reader_returns_bulk_iops_and_endpoint_moments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ssprops(root / "organic" / "70")

            tables = load_geocape_aerosol_tables(
                root,
                first_wavelength_microns=0.3,
                last_wavelength_microns=0.4,
                aggregates=("organic",),
                moment_cutoff=0.0,
                max_moments=2,
            )

        np.testing.assert_allclose(tables.wavelengths_microns, [0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(tables.bulk_iops[:, 1, 0], [0.30, 0.03])
        np.testing.assert_allclose(tables.moments[0, :, 0], [1.0, 0.3, 0.03])
        np.testing.assert_allclose(tables.moments[1, :, 0], [1.0, 0.4, 0.04])

    def test_aerosol_interpolation_accepts_table_endpoints(self) -> None:
        grid = np.array([0.3, 0.4, 0.5], dtype=float)
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)

        np.testing.assert_allclose(
            _interp_aerosol_table(np.array([0.3, 0.5]), grid, values),
            np.array([[1.0, 10.0], [3.0, 30.0]], dtype=float),
        )
        self.assertEqual(_interp_indices(grid, 0.3), (0, 0, 1.0, 0.0))
        self.assertEqual(_interp_indices(grid, 0.5), (2, 2, 1.0, 0.0))

    def test_profile_pressure_order_must_be_strictly_monotonic(self) -> None:
        pressure = np.array([1000.0, 900.0, 900.0], dtype=float)
        temperature = np.array([290.0, 280.0, 270.0], dtype=float)
        gas = np.ones((3, 1), dtype=float)

        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            _top_to_bottom(pressure=pressure, temperature=temperature, gas_vmr=gas, heights=None)

    def test_scene_builder_accepts_raw_geocape_table_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "profile.dat"
            scene = root / "scene.yaml"
            xsec = root / "o3.txt"
            loading = root / "organic_loading.dat"
            ssprops = root / "ssprops"
            self._write_profile(profile)
            xsec.write_text("300 1.0e-22\n400 2.0e-22\n450 2.5e-22\n")
            loading.write_text("1 0.2 0 0\n2 0.1 0 0\n")
            self._write_ssprops(ssprops / "organic" / "70")
            scene.write_text(
                f"""
mode: solar
gases: [O3]
spectral:
  wavelengths_nm: [300.0, 400.0, 450.0]
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo: 0.1
opacity:
  gas_cross_sections:
    tables:
      O3:
        path: {xsec.name}
  aerosol:
    loadings:
      kind: geocape_files
      select_index: 2
      active_layers: 2
      paths: [{loading.name}]
    ssprops:
      path: {ssprops.name}
      aggregates: [organic]
      moment_cutoff: 0.0
      max_moments: 2
"""
            )

            bundle = build_benchmark_scene_inputs(
                profile_path=profile,
                scene_path=scene,
                kind="uv",
                spectral_limit=2,
            )

        for forbidden in (
            "tau",
            "omega",
            "asymm",
            "scaling",
            "fo_exact_scatter",
            "gas_cross_sections",
        ):
            self.assertNotIn(forbidden, bundle)
        self.assertEqual(bundle["gas_absorption_tau"].shape, (2, 2))
        np.testing.assert_allclose(bundle["wavelengths"], [300.0, 400.0])
        self.assertGreater(float(bundle["gas_absorption_tau"].sum()), 0.0)
        np.testing.assert_allclose(bundle["aerosol_loadings"][:, 0], [0.1, 0.2])
        np.testing.assert_allclose(bundle["aerosol_select_wavelength_microns"], 0.4)
        self.assertEqual(bundle["aerosol_bulk_iops"].shape, (2, 4, 1))
        self.assertEqual(bundle["aerosol_moments"].shape, (2, 3, 1))
        np.testing.assert_allclose(bundle["aerosol_moments"][1, :, 0], [1.0, 0.45, 0.045])

    @staticmethod
    def _write_profile(path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "Level Pressure TATM O3",
                    "1 1000.0 290.0 3.0e-8",
                    "2 500.0 260.0 2.0e-8",
                    "3 100.0 220.0 1.0e-8",
                    "",
                ]
            )
        )

    @staticmethod
    def _write_ssprops(path: Path) -> None:
        path.mkdir(parents=True)
        path.joinpath("mie3_1.mie").write_text(
            "\n".join(f"{w:.1f} {w / 10:.2f} {w:.2f}" for w in (0.2, 0.3, 0.4, 0.5)) + "\n"
        )
        rows = []
        for wavelength in (0.2, 0.3, 0.4, 0.5):
            rows.extend(
                (
                    f" wavelength = {wavelength:.1f}",
                    "header",
                    "header",
                    " l=    0   1.0   0 0 0 0 0",
                    f" l=    1   {wavelength:.1f}   0 0 0 0 0",
                    f" l=    2   {wavelength / 10:.2f}  0 0 0 0 0",
                )
            )
        path.joinpath("mie3_1.mom").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    unittest.main()
