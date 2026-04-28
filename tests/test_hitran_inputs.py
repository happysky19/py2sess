from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import wofz

from py2sess.optical.hitran import (
    _lookup_partition_matrix,
    hitran_cross_sections,
    hitran_molecule_number,
    humlicek_voigt,
    load_hitran_partition_functions,
    read_hitran_lines,
)
from py2sess.optical.scene_io import build_benchmark_scene_inputs


class HitranInputTests(unittest.TestCase):
    def test_molecule_mapping_matches_fortran_numbers(self) -> None:
        self.assertEqual(hitran_molecule_number("H2O"), 1)
        self.assertEqual(hitran_molecule_number("SO2"), 9)

    def test_line_reader_filters_margin_and_skipped_isotopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_partition_file(root, {(1, 1): np.ones(195)})
            (root / "01_hit09.par").write_text(
                "".join(
                    [
                        _line(1, 1, 970.0),
                        _line(1, 1, 1000.0),
                        _line(2, 9, 1000.0),
                        _line(1, 1, 1030.1),
                    ]
                )
            )

            lines = read_hitran_lines(root, "H2O", np.array([1000.0, 1001.0]))

        self.assertEqual(lines.size, 1)
        np.testing.assert_allclose(lines.center_cm_inv, [1000.0])
        np.testing.assert_allclose(lines.molecular_mass_amu, [18.010565])

    def test_partition_lookup_uses_fortran_cubic_interpolant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table = np.arange(148.0, 343.0)
            self._write_partition_file(root, {(1, 1): table})

            partition = load_hitran_partition_functions(root)
            values = _lookup_partition_matrix(
                partition,
                molecule_number=1,
                isotopes=np.array([1]),
                temperature_k=np.array([200.0, 200.5]),
            )

        np.testing.assert_allclose(values[:, 0], [200.0, 200.5])

    def test_humlicek_voigt_matches_scipy_faddeeva_reference(self) -> None:
        x = np.array([-8.0, -7.4121099054, -5.0, -2.5, 0.0, 2.5, 5.0, 7.4121099054, 8.0])
        y = 0.7

        got = humlicek_voigt(x, y)
        expected = np.real(wofz(x + 1j * y))

        np.testing.assert_allclose(got, expected, rtol=2.0e-4, atol=2.0e-5)

    def test_humlicek_w4_region_2_matches_fortran_tail_regression(self) -> None:
        x = np.array([-7.4121099054, -2.1723622563, 4.3763589244, 7.4121099054])
        y = 1.19692721918

        got = humlicek_voigt(x, y)
        expected = np.array(
            [
                1.230067074865142e-02,
                1.292452925545625e-01,
                3.521612844247020e-02,
                1.230067074865142e-02,
            ]
        )

        np.testing.assert_allclose(got, expected, rtol=2.0e-8, atol=2.0e-10)

    def test_cross_section_tiny_line_case_is_finite_and_peaks_near_line_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_partition_file(root, {(1, 1): np.ones(195)})
            (root / "01_hit09.par").write_text(_line(1, 1, 1000.0, strength=1.0e-20))

            xsec = hitran_cross_sections(
                hitran_dir=root,
                molecule="H2O",
                spectral_grid=np.array([999.9, 1000.0, 1000.1]),
                pressure_atm=np.array([1.0]),
                temperature_k=np.array([296.0]),
            )

        self.assertEqual(xsec.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(xsec)))
        self.assertGreaterEqual(float(xsec.min()), 0.0)
        self.assertGreater(float(xsec[1, 0]), float(xsec[0, 0]))
        self.assertGreater(float(xsec[1, 0]), float(xsec[2, 0]))

    def test_cross_sections_batched_levels_match_level_by_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_partition_file(root, {(1, 1): np.ones(195)})
            (root / "01_hit09.par").write_text(_line(1, 1, 1000.0, strength=1.0e-20))
            grid = np.array([999.9, 1000.0, 1000.1])
            pressure = np.array([1.0, 0.5])
            temperature = np.array([296.0, 250.0])

            batched = hitran_cross_sections(
                hitran_dir=root,
                molecule="H2O",
                spectral_grid=grid,
                pressure_atm=pressure,
                temperature_k=temperature,
            )
            level_by_level = np.column_stack(
                [
                    hitran_cross_sections(
                        hitran_dir=root,
                        molecule="H2O",
                        spectral_grid=grid,
                        pressure_atm=np.array([press]),
                        temperature_k=np.array([temp]),
                    )[:, 0]
                    for press, temp in zip(pressure, temperature, strict=True)
                ]
            )

        np.testing.assert_allclose(batched, level_by_level, rtol=1.0e-14, atol=0.0)

    def test_scene_builder_accepts_hitran_gas_cross_section_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "profile.dat"
            scene = root / "scene.yaml"
            hitran = root / "hitran"
            hitran.mkdir()
            profile.write_text(
                "\n".join(
                    [
                        "Level Pressure TATM H2O",
                        "1 1000.0 296.0 1.0e-2",
                        "2 500.0 296.0 5.0e-3",
                        "3 100.0 296.0 1.0e-3",
                        "",
                    ]
                )
            )
            self._write_partition_file(hitran, {(1, 1): np.ones(195)})
            (hitran / "01_hit09.par").write_text(_line(1, 1, 1000.0, strength=1.0e-20))
            scene.write_text(
                f"""
mode: solar
gases: [H2O]
spectral:
  wavenumber_cm_inv: [999.9, 1000.0, 1000.1]
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo: 0.1
opacity:
  gas_cross_sections:
    hitran:
      path: {hitran.name}
"""
            )

            bundle = build_benchmark_scene_inputs(
                profile_path=profile,
                scene_path=scene,
                kind="uv",
            )

        self.assertEqual(bundle["gas_absorption_tau"].shape, (3, 2))
        self.assertGreater(float(bundle["gas_absorption_tau"][1, 0]), 0.0)

    @staticmethod
    def _write_partition_file(path: Path, tables: dict[tuple[int, int], np.ndarray]) -> None:
        rows = []
        for (mol, isotope), values in tables.items():
            rows.append(f"{mol} {isotope}")
            rows.extend(str(float(value)) for value in values)
        path.joinpath("hitran08-parsum.resorted").write_text("\n".join(rows) + "\n")


def _line(
    mol: int,
    isotope: int,
    center: float,
    *,
    strength: float = 1.0e-22,
    alpha: float = 0.1000,
    elow: float = 0.0,
    coeff: float = 0.50,
    pshift: float = 0.0,
) -> str:
    alpha_field = f"{alpha:.4f}"[-5:]
    self_field = f"{0.0:.4f}"[-5:]
    return (
        f"{mol:2d}{isotope:1d}{center:12.6f}{strength:10.3E}{0.0:10.3E}"
        f"{alpha_field}{self_field}{elow:10.4f}{coeff:4.2f}{pshift:8.6f}\n"
    )


if __name__ == "__main__":
    unittest.main()
