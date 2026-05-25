from __future__ import annotations

import unittest

import numpy as np

from py2sess.benchmarks.flux_references import (
    pydisort_flux_to_py2sess,
    rayleigh_phase_moments,
    solar_isotropic_single_scatter_flux,
    solar_rayleigh_single_scatter_flux,
)
from py2sess.benchmarks.kinetics_flux import kinetics_flux_to_py2sess, parse_kinetics_flux_table


class FluxReferenceAdapterTests(unittest.TestCase):
    def test_pydisort_flux_channels_map_to_py2sess_conventions(self) -> None:
        raw = np.zeros((1, 1, 3, 8), dtype=float)
        raw[0, 0, :, 0] = [10.0, 8.0, 5.0]
        raw[0, 0, :, 1] = [1.0, 2.0, 3.0]
        raw[0, 0, :, 2] = [0.5, 0.4, 0.3]
        raw[0, 0, :, 4] = [2.5, 2.0, 1.5]

        mapped = pydisort_flux_to_py2sess(raw)
        np.testing.assert_allclose(mapped["flux_up"][0, 0], [0.5, 0.4, 0.3])
        np.testing.assert_allclose(mapped["flux_down"][0, 0], [11.0, 10.0, 8.0])
        np.testing.assert_allclose(mapped["flux_net"][0, 0], [-10.5, -9.6, -7.7])
        np.testing.assert_allclose(mapped["flux_mean"][0, 0], [2.5, 2.0, 1.5])

        flipped = pydisort_flux_to_py2sess(raw[..., ::-1, :], level_axis="boa_to_toa")
        np.testing.assert_allclose(flipped["flux_down"], mapped["flux_down"])

    def test_kinetics_flux_table_maps_to_py2sess_conventions(self) -> None:
        table = parse_kinetics_flux_table(
            """
            # level altitude_km direct_flux diffuse_plus_flux diffuse_minus_flux net_flux diffuse_radiation_field total_radiation_field diffuse_factor
            0 50.0 10.0 1.0 2.0 -11.0 3.0 13.0 0.30
            1 40.0  8.0 1.5 2.5  -9.0 4.0 12.0 0.50
            """
        )
        mapped = kinetics_flux_to_py2sess(table)
        np.testing.assert_allclose(mapped["flux_up"], [1.0, 1.5])
        np.testing.assert_allclose(mapped["flux_down"], [12.0, 10.5])
        np.testing.assert_allclose(mapped["flux_net"], [-11.0, -9.0])
        np.testing.assert_allclose(mapped["flux_mean"], [13.0, 12.0])

    def test_rayleigh_delta_zero_reduces_to_isotropic_reference(self) -> None:
        tau = np.array([0.1, 0.2], dtype=float)
        ssa = np.array([0.05, 0.08], dtype=float)
        isotropic = solar_isotropic_single_scatter_flux(
            tau,
            ssa=ssa,
            mu0=0.8,
            fbeam=1.7,
            include_direct=False,
        )
        rayleigh = solar_rayleigh_single_scatter_flux(
            tau,
            ssa=ssa,
            mu0=0.8,
            rayleigh_delta=0.0,
            fbeam=1.7,
            include_direct=False,
        )
        for field in ("flux_up", "flux_down", "flux_net", "flux_mean"):
            np.testing.assert_allclose(rayleigh[field], isotropic[field], atol=1.0e-12)

    def test_rayleigh_phase_moments_use_p2_coefficient(self) -> None:
        moments = rayleigh_phase_moments(5, rayleigh_delta=1.0)
        np.testing.assert_allclose(moments, [0.0, 0.1, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
