from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import netcdf_file

from py2sess.optical.opacity_table import gas_cross_sections_from_table3d
from py2sess.optical.scene import (
    atmospheric_profile_from_levels,
    gas_absorption_tau_from_cross_sections,
)
from py2sess.optical.scene_io import build_benchmark_scene_inputs


class OpacityTableTests(unittest.TestCase):
    def test_table3d_interpolates_profile_levels_and_reorders_gases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xsec.nc"
            raw = self._write_table(path)

            got = gas_cross_sections_from_table3d(
                path=path,
                gas_names=("H2O", "O3"),
                pressure_hpa=np.array([100.0, 500.0, 1000.0]),
                temperature_k=np.array([220.0, 260.0, 290.0]),
                spectral={"wavenumber_cm_inv": np.array([999.0, 1000.0])},
            )
            o3 = gas_cross_sections_from_table3d(
                path=path,
                gas_names=("O3",),
                pressure_hpa=np.array([100.0, 500.0, 1000.0]),
                temperature_k=np.array([220.0, 260.0, 290.0]),
                spectral={"wavenumber_cm_inv": np.array([999.0, 1000.0])},
            )

        expected = self._expected_profile_xsec(raw)
        np.testing.assert_allclose(got, expected)
        np.testing.assert_allclose(o3, expected[:, :, 1:2])

    def test_scene_builder_accepts_table3d_gas_cross_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            table = root / "xsec.nc"
            raw = self._write_table(table)
            profile = root / "profile.dat"
            scene = root / "scene.yaml"
            profile.write_text(
                "\n".join(
                    [
                        "Level Pressure TATM H2O O3",
                        "1 1000.0 290.0 1.0e-2 2.0e-8",
                        "2 500.0 260.0 5.0e-3 1.0e-8",
                        "3 100.0 220.0 1.0e-3 5.0e-9",
                        "",
                    ]
                )
            )
            scene.write_text(
                """
mode: solar
gases: [H2O, O3]
spectral:
  wavenumber_cm_inv: [999.0, 1000.0]
geometry:
  angles: [30.0, 20.0, 0.0]
surface:
  albedo: 0.1
opacity:
  gas_cross_sections:
    table3d:
      path: xsec.nc
"""
            )

            bundle = build_benchmark_scene_inputs(
                profile_path=profile,
                scene_path=scene,
                kind="uv",
            )

            atm = atmospheric_profile_from_levels(
                pressure_hpa=np.array([100.0, 500.0, 1000.0]),
                temperature_k=np.array([220.0, 260.0, 290.0]),
                gas_vmr=np.array([[1.0e-3, 5.0e-9], [5.0e-3, 1.0e-8], [1.0e-2, 2.0e-8]]),
            )
            expected_xsec = self._expected_profile_xsec(raw)
            expected_tau = gas_absorption_tau_from_cross_sections(
                heights_km=atm.heights_km,
                gas_density_per_km=atm.gas_density_per_km,
                cross_sections=expected_xsec,
            )

        np.testing.assert_allclose(bundle["gas_absorption_tau"], expected_tau)

    @staticmethod
    def _write_table(path: Path) -> np.ndarray:
        wavenumber = np.array([999.0, 1000.0])
        pressure = np.array([100.0, 500.0, 1000.0])
        temperature = np.array([220.0, 260.0, 290.0])
        raw = np.empty((2, 2, 3, 3), dtype=float)
        for gas in range(2):
            for wave in range(2):
                for press in range(3):
                    for temp in range(3):
                        raw[gas, wave, press, temp] = 1.0e-22 * (
                            1 + gas + 10 * wave + 100 * press + 1000 * temp
                        )

        with netcdf_file(path, "w") as data:
            data.createDimension("gas", 2)
            data.createDimension("wave", 2)
            data.createDimension("pressure", 3)
            data.createDimension("temperature", 3)
            data.createDimension("name_strlen", 3)
            data.createVariable("wavenumber_cm_inv", "d", ("wave",))[:] = wavenumber
            data.createVariable("pressure_hpa", "d", ("pressure",))[:] = pressure
            data.createVariable("temperature_k", "d", ("temperature",))[:] = temperature
            names = np.array([[b"O", b"3", b" "], [b"H", b"2", b"O"]], dtype="S1")
            data.createVariable("gas_names", "c", ("gas", "name_strlen"))[:] = names
            data.createVariable(
                "cross_section",
                "d",
                ("gas", "wave", "pressure", "temperature"),
            )[:] = raw
        return raw

    @staticmethod
    def _expected_profile_xsec(raw: np.ndarray) -> np.ndarray:
        reordered = raw[[1, 0]]
        out = np.empty((2, 3, 2), dtype=float)
        for level in range(3):
            out[:, level, :] = reordered[:, :, level, level].T
        return out


if __name__ == "__main__":
    unittest.main()
