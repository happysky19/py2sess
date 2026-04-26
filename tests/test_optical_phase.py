from __future__ import annotations

import unittest

import numpy as np

from py2sess import load_tir_benchmark_case, load_uv_benchmark_case
from py2sess.optical.phase import (
    aerosol_interp_fraction,
    build_solar_fo_scatter_term,
    build_two_stream_phase_inputs,
    ssa_from_optical_depth,
)
from py2sess.rtsolver.backend import has_torch


def _cos_scatter(angles: np.ndarray) -> float:
    sza, vza, raz = np.deg2rad(angles)
    return float(-(np.cos(vza) * np.cos(sza)) + np.sin(vza) * np.sin(sza) * np.cos(raz))


class OpticalPhaseFormulaTests(unittest.TestCase):
    def test_ssa_from_optical_depth_handles_zero_total(self) -> None:
        total = np.array([0.0, 2.0, 4.0])
        scattering = np.array([1.0, 0.5, 6.0])
        np.testing.assert_allclose(
            ssa_from_optical_depth(total, scattering),
            np.array([0.0, 0.25, 1.5]),
        )

    def test_aerosol_interp_fraction_forward_and_reverse(self) -> None:
        wavelengths = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(
            aerosol_interp_fraction(wavelengths),
            np.array([0.0, 1.0 / 3.0, 1.0]),
        )
        np.testing.assert_allclose(
            aerosol_interp_fraction(wavelengths, reverse=True),
            np.array([1.0, 1.0 / 3.0, 0.0]),
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
        expected = raypf * ssa / (1.0 - expected_factor * ssa)
        np.testing.assert_allclose(scatter, expected)

    def test_mixed_rayleigh_aerosol_formula(self) -> None:
        ssa = np.array([[0.5, 0.7]])
        depol = np.array([0.2])
        rayleigh_fraction = np.array([[0.25, 0.5]])
        aerosol_fraction = np.array([[[0.5, 0.25], [0.2, 0.3]]])
        aerosol_moments = np.zeros((2, 3, 2))
        aerosol_moments[:, 0, :] = 1.0
        aerosol_moments[0, 1, :] = [0.3, 0.6]
        aerosol_moments[1, 1, :] = [0.5, 0.8]
        aerosol_moments[0, 2, :] = [0.7, 0.2]
        aerosol_moments[1, 2, :] = [0.9, 0.4]
        fac = np.array([0.25])

        phase = build_two_stream_phase_inputs(
            ssa=ssa,
            depol=depol,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=fac,
        )

        m1 = np.array([0.35, 0.65])
        m2 = np.array([0.75, 0.25])
        ray2mom = (1.0 - depol[0]) / (2.0 + depol[0])
        expected_m1 = np.sum(aerosol_fraction[0] * m1, axis=-1)
        expected_m2 = rayleigh_fraction[0] * ray2mom + np.sum(aerosol_fraction[0] * m2, axis=-1)
        np.testing.assert_allclose(phase.g, expected_m1[None, :] / 3.0)
        np.testing.assert_allclose(phase.delta_m_truncation_factor, expected_m2[None, :] / 5.0)

        angles = np.array([40.0, 10.0, 30.0])
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
        mu = _cos_scatter(angles)
        p = np.array([1.0, mu, 0.5 * (3.0 * mu * mu - 1.0)])
        endpoint_phase = np.einsum("ema,m->ea", aerosol_moments, p)
        aerosol_phase = endpoint_phase[0] + fac[0] * (endpoint_phase[1] - endpoint_phase[0])
        delta = 2.0 * (1.0 - depol[0]) / (2.0 + depol[0])
        raypf = delta * 0.75 * (1.0 + mu * mu) + 1.0 - delta
        phase_total = rayleigh_fraction[0] * raypf + np.sum(
            aerosol_fraction[0] * aerosol_phase,
            axis=-1,
        )
        expected = phase_total * ssa[0] / (1.0 - phase.delta_m_truncation_factor[0] * ssa[0])
        np.testing.assert_allclose(scatter, expected[None, :])

    def test_phase_inputs_reject_nonphysical_fractions(self) -> None:
        ssa = np.array([[0.5, 0.7]])
        depol = np.array([0.2])
        aerosol_moments = np.zeros((2, 3, 1))
        aerosol_moments[:, 0, :] = 1.0
        fac = np.array([0.25])

        with self.assertRaisesRegex(ValueError, "not sum above 1"):
            build_two_stream_phase_inputs(
                ssa=ssa,
                depol=depol,
                rayleigh_fraction=np.array([[0.8, 0.8]]),
                aerosol_fraction=np.array([[[0.5], [0.5]]]),
                aerosol_moments=aerosol_moments,
                aerosol_interp_fraction=fac,
            )

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_solar_fo_scatter_term(
                ssa=ssa,
                depol=depol,
                rayleigh_fraction=np.array([[1.0, 1.0]]),
                aerosol_fraction=np.array([[[-0.1], [0.0]]]),
                aerosol_moments=aerosol_moments,
                aerosol_interp_fraction=fac,
                angles=np.array([40.0, 10.0, 30.0]),
                delta_m_truncation_factor=np.zeros_like(ssa),
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_phase_inputs_reject_nonphysical_fractions(self) -> None:
        import torch

        from py2sess.optical.phase_torch import build_two_stream_phase_inputs_torch

        ssa = torch.tensor([[0.5, 0.7]], dtype=torch.float64)
        depol = torch.tensor([0.2], dtype=torch.float64)
        aerosol_moments = torch.zeros((2, 3, 1), dtype=torch.float64)
        aerosol_moments[:, 0, :] = 1.0

        with self.assertRaisesRegex(ValueError, "not sum above 1"):
            build_two_stream_phase_inputs_torch(
                ssa=ssa,
                depol=depol,
                rayleigh_fraction=torch.tensor([[0.8, 0.8]], dtype=torch.float64),
                aerosol_fraction=torch.tensor([[[0.5], [0.5]]], dtype=torch.float64),
                aerosol_moments=aerosol_moments,
                aerosol_interp_fraction=torch.tensor([0.25], dtype=torch.float64),
            )


class OpticalPhaseFixtureTests(unittest.TestCase):
    def test_uv_fixture_optical_preprocessing_matches_fortran_dump(self) -> None:
        case = load_uv_benchmark_case()
        phase = build_two_stream_phase_inputs(
            ssa=case.omega,
            depol=case.depol,
            rayleigh_fraction=case.rayleigh_fraction,
            aerosol_fraction=case.aerosol_fraction,
            aerosol_moments=case.aerosol_moments,
            aerosol_interp_fraction=case.aerosol_interp_fraction,
        )
        scatter = build_solar_fo_scatter_term(
            ssa=case.omega,
            depol=case.depol,
            rayleigh_fraction=case.rayleigh_fraction,
            aerosol_fraction=case.aerosol_fraction,
            aerosol_moments=case.aerosol_moments,
            aerosol_interp_fraction=case.aerosol_interp_fraction,
            angles=case.user_obsgeom,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
        )
        np.testing.assert_allclose(phase.g, case.asymm, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            phase.delta_m_truncation_factor,
            case.scaling,
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(scatter, case.fo_exact_scatter, rtol=0.0, atol=1.0e-14)

    def test_tir_fixture_optical_preprocessing_matches_fortran_dump(self) -> None:
        case = load_tir_benchmark_case()
        phase = build_two_stream_phase_inputs(
            ssa=case.omega_arr,
            depol=case.depol,
            rayleigh_fraction=case.rayleigh_fraction,
            aerosol_fraction=case.aerosol_fraction,
            aerosol_moments=case.aerosol_moments,
            aerosol_interp_fraction=case.aerosol_interp_fraction,
        )
        np.testing.assert_allclose(phase.g, case.asymm_arr, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            phase.delta_m_truncation_factor,
            case.d2s_scaling,
            rtol=0.0,
            atol=1.0e-14,
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_phase_preprocessing_matches_numpy_and_keeps_gradients(self) -> None:
        import torch

        from py2sess.optical.phase_torch import (
            build_solar_fo_scatter_term_torch,
            build_two_stream_phase_inputs_torch,
        )

        ssa = torch.tensor([[0.5, 0.7]], dtype=torch.float64, requires_grad=True)
        depol = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        rayleigh_fraction = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
        aerosol_fraction = torch.tensor(
            [[[0.5, 0.25], [0.2, 0.3]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        aerosol_moments = torch.tensor(
            [
                [[1.0, 1.0], [0.3, 0.6], [0.7, 0.2]],
                [[1.0, 1.0], [0.5, 0.8], [0.9, 0.4]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        fac = torch.tensor([0.25], dtype=torch.float64)
        angles = torch.tensor([40.0, 10.0, 30.0], dtype=torch.float64)

        torch_phase = build_two_stream_phase_inputs_torch(
            ssa=ssa,
            depol=depol,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=fac,
        )
        torch_scatter = build_solar_fo_scatter_term_torch(
            ssa=ssa,
            depol=depol,
            rayleigh_fraction=rayleigh_fraction,
            aerosol_fraction=aerosol_fraction,
            aerosol_moments=aerosol_moments,
            aerosol_interp_fraction=fac,
            angles=angles,
            delta_m_truncation_factor=torch_phase.delta_m_truncation_factor,
        )
        numpy_phase = build_two_stream_phase_inputs(
            ssa=ssa.detach().numpy(),
            depol=depol.detach().numpy(),
            rayleigh_fraction=rayleigh_fraction.numpy(),
            aerosol_fraction=aerosol_fraction.detach().numpy(),
            aerosol_moments=aerosol_moments.detach().numpy(),
            aerosol_interp_fraction=fac.numpy(),
        )
        numpy_scatter = build_solar_fo_scatter_term(
            ssa=ssa.detach().numpy(),
            depol=depol.detach().numpy(),
            rayleigh_fraction=rayleigh_fraction.numpy(),
            aerosol_fraction=aerosol_fraction.detach().numpy(),
            aerosol_moments=aerosol_moments.detach().numpy(),
            aerosol_interp_fraction=fac.numpy(),
            angles=angles.numpy(),
            delta_m_truncation_factor=numpy_phase.delta_m_truncation_factor,
        )
        torch.testing.assert_close(torch_phase.g, torch.as_tensor(numpy_phase.g))
        torch.testing.assert_close(
            torch_phase.delta_m_truncation_factor,
            torch.as_tensor(numpy_phase.delta_m_truncation_factor),
        )
        torch.testing.assert_close(torch_scatter, torch.as_tensor(numpy_scatter))

        objective = (
            torch_phase.g.sum() + torch_phase.delta_m_truncation_factor.sum() + torch_scatter.sum()
        )
        objective.backward()
        for tensor in (ssa, depol, aerosol_fraction, aerosol_moments):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(float(torch.abs(tensor.grad).sum()), 0.0)

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_phase_gradients_match_finite_difference(self) -> None:
        import torch

        from py2sess.optical.phase_torch import (
            build_solar_fo_scatter_term_torch,
            build_two_stream_phase_inputs_torch,
        )

        rayleigh_fraction = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
        fac = torch.tensor([0.25], dtype=torch.float64)
        angles = torch.tensor([40.0, 10.0, 30.0], dtype=torch.float64)

        def objective(ssa, depol, aerosol_fraction, aerosol_moments):
            phase = build_two_stream_phase_inputs_torch(
                ssa=ssa,
                depol=depol,
                rayleigh_fraction=rayleigh_fraction,
                aerosol_fraction=aerosol_fraction,
                aerosol_moments=aerosol_moments,
                aerosol_interp_fraction=fac,
            )
            scatter = build_solar_fo_scatter_term_torch(
                ssa=ssa,
                depol=depol,
                rayleigh_fraction=rayleigh_fraction,
                aerosol_fraction=aerosol_fraction,
                aerosol_moments=aerosol_moments,
                aerosol_interp_fraction=fac,
                angles=angles,
                delta_m_truncation_factor=phase.delta_m_truncation_factor,
            )
            return phase.g.sum() + phase.delta_m_truncation_factor.sum() + scatter.sum()

        ssa = torch.tensor([[0.5, 0.7]], dtype=torch.float64, requires_grad=True)
        depol = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        aerosol_fraction = torch.tensor(
            [[[0.5, 0.25], [0.2, 0.3]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        aerosol_moments = torch.tensor(
            [
                [[1.0, 1.0], [0.3, 0.6], [0.7, 0.2]],
                [[1.0, 1.0], [0.5, 0.8], [0.9, 0.4]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        value = objective(ssa, depol, aerosol_fraction, aerosol_moments)
        value.backward()

        eps = 1.0e-6
        checks = (
            (ssa, (0, 0), 0),
            (depol, (0,), 1),
            (aerosol_fraction, (0, 0, 0), 2),
            (aerosol_moments, (0, 1, 0), 3),
        )
        for tensor, index, pos in checks:
            args = [
                ssa.detach().clone(),
                depol.detach().clone(),
                aerosol_fraction.detach().clone(),
                aerosol_moments.detach().clone(),
            ]
            args[pos][index] += eps
            plus = objective(*args)
            args[pos][index] -= 2.0 * eps
            minus = objective(*args)
            finite_difference = (plus - minus) / (2.0 * eps)
            self.assertAlmostEqual(
                float(tensor.grad[index]),
                float(finite_difference),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
