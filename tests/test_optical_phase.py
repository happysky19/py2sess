from __future__ import annotations

import unittest

import numpy as np

from py2sess.optical.phase import (
    aerosol_interp_fraction,
    build_solar_fo_scatter_term,
    build_two_stream_phase_inputs,
    ssa_from_optical_depth,
)


def _cos_scatter(angles: np.ndarray) -> float:
    sza, vza, raz = np.deg2rad(angles)
    return float(-(np.cos(vza) * np.cos(sza)) + np.sin(vza) * np.sin(sza) * np.cos(raz))


class OpticalPhaseFormulaTests(unittest.TestCase):
    def test_ssa_from_optical_depth_handles_zero_total(self) -> None:
        total = np.array([0.0, 2.0, 4.0])
        scattering = np.array([1.0, 0.5, 6.0])
        np.testing.assert_allclose(ssa_from_optical_depth(total, scattering), [0.0, 0.25, 1.5])

    def test_aerosol_interp_fraction_forward_and_reverse(self) -> None:
        wavelengths = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(aerosol_interp_fraction(wavelengths), [0.0, 1.0 / 3.0, 1.0])
        np.testing.assert_allclose(
            aerosol_interp_fraction(wavelengths, reverse=True),
            [1.0, 1.0 / 3.0, 0.0],
        )

    def test_pure_rayleigh_phase_inputs_and_fo_scatter(self) -> None:
        ssa = np.array([[0.4, 0.6]])
        depol = np.array([0.1])
        rayleigh_fraction = np.ones_like(ssa)
        aerosol_fraction = np.zeros((1, 2, 1))
        aerosol_moments = np.zeros((2, 3, 1))
        aerosol_moments[:, 0, :] = 1.0
        fac = np.array([0.0])

        phase = build_two_stream_phase_inputs(
            ssa=ssa,
            depol=depol,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=fac,
        )
        ray2mom = (1.0 - depol) / (2.0 + depol)
        expected_factor = np.broadcast_to(ray2mom[:, None] / 5.0, ssa.shape)
        np.testing.assert_allclose(phase.g, np.zeros_like(ssa))
        np.testing.assert_allclose(phase.delta_m_truncation_factor, expected_factor)

        angles = np.array([30.0, 20.0, 10.0])
        scatter = build_solar_fo_scatter_term(
            ssa=ssa,
            depol=depol,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=fac,
            angles=angles,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
        )
        delta = 2.0 * (1.0 - depol[0]) / (2.0 + depol[0])
        raypf = delta * 0.75 * (1.0 + _cos_scatter(angles) ** 2) + 1.0 - delta
        np.testing.assert_allclose(scatter, raypf * ssa / (1.0 - expected_factor * ssa))

    def test_phase_inputs_reject_nonphysical_fractions(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            build_two_stream_phase_inputs(
                ssa=np.array([[0.5]]),
                depol=np.array([0.0]),
                rayleigh_fraction=np.array([[0.2]]),
                aerosol_fraction=np.array([[[0.2]]]),
                aerosol_moments=np.ones((2, 3, 1)),
                aerosol_interp_fraction=np.array([0.0]),
            )


if __name__ == "__main__":
    unittest.main()
