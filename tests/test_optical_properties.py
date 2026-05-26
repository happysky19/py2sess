from __future__ import annotations

import unittest

import numpy as np

from py2sess.optical.properties import build_layer_optical_properties


class OpticalPropertiesTests(unittest.TestCase):
    def test_builds_tau_ssa_and_scattering_fractions(self) -> None:
        absorption_tau = np.array([0.1, 0.2])
        rayleigh_scattering_tau = np.array([0.3, 0.1])
        aerosol_extinction_tau = np.array([[0.4, 0.2], [0.5, 0.1]])
        aerosol_single_scattering_albedo = np.array([0.75, 0.5])

        props = build_layer_optical_properties(
            absorption_tau=absorption_tau,
            rayleigh_scattering_tau=rayleigh_scattering_tau,
            aerosol_extinction_tau=aerosol_extinction_tau,
            aerosol_single_scattering_albedo=aerosol_single_scattering_albedo,
        )

        aerosol_scattering_tau = aerosol_extinction_tau * aerosol_single_scattering_albedo
        scattering_tau = rayleigh_scattering_tau + aerosol_scattering_tau.sum(axis=-1)
        total_tau = absorption_tau + rayleigh_scattering_tau + aerosol_extinction_tau.sum(axis=-1)
        np.testing.assert_allclose(props.tau, total_tau)
        np.testing.assert_allclose(props.ssa, scattering_tau / total_tau)
        np.testing.assert_allclose(
            props.rayleigh_fraction, rayleigh_scattering_tau / scattering_tau
        )
        np.testing.assert_allclose(
            props.aerosol_fraction, aerosol_scattering_tau / scattering_tau[:, None]
        )

    def test_handles_clear_absorbing_layers_safely(self) -> None:
        props = build_layer_optical_properties(absorption_tau=np.array([0.0, 0.2]))
        np.testing.assert_allclose(props.tau, [0.0, 0.2])
        np.testing.assert_allclose(props.ssa, [0.0, 0.0])
        np.testing.assert_allclose(props.rayleigh_fraction, [0.0, 0.0])
        self.assertEqual(props.aerosol_fraction.shape, (2, 0))

    def test_scattering_only_aerosol_does_not_require_extinction(self) -> None:
        props = build_layer_optical_properties(
            absorption_tau=np.array([0.2]),
            rayleigh_scattering_tau=np.array([0.3]),
            aerosol_scattering_tau=np.array([[0.4, 0.1]]),
        )
        np.testing.assert_allclose(props.tau, [0.2 + 0.3 + 0.5])
        np.testing.assert_allclose(props.ssa, [0.8])
        np.testing.assert_allclose(props.aerosol_fraction, [[0.5, 0.125]])

    def test_rejects_unphysical_or_ambiguous_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            build_layer_optical_properties(
                gas_absorption_tau=np.array([0.1]),
                aerosol_extinction_tau=np.array([[0.2]]),
                aerosol_scattering_tau=np.array([[0.3]]),
            )
        with self.assertRaisesRegex(ValueError, "aerosol axis"):
            build_layer_optical_properties(
                gas_absorption_tau=np.array([0.1, 0.2]),
                aerosol_extinction_tau=np.array([0.3, 0.4]),
                aerosol_single_scattering_albedo=np.array([0.9, 0.8]),
            )


if __name__ == "__main__":
    unittest.main()
