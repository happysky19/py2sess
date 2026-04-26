from __future__ import annotations

import unittest

import numpy as np

from py2sess.optical.properties import build_layer_optical_properties
from py2sess.rtsolver.backend import has_torch


class OpticalPropertiesTests(unittest.TestCase):
    def test_builds_tau_ssa_and_scattering_fractions(self) -> None:
        gas_absorption_tau = np.array([0.1, 0.2])
        rayleigh_scattering_tau = np.array([0.3, 0.1])
        aerosol_extinction_tau = np.array(
            [
                [0.4, 0.2],
                [0.5, 0.1],
            ]
        )
        aerosol_single_scattering_albedo = np.array([0.75, 0.5])

        props = build_layer_optical_properties(
            gas_absorption_tau=gas_absorption_tau,
            rayleigh_scattering_tau=rayleigh_scattering_tau,
            aerosol_extinction_tau=aerosol_extinction_tau,
            aerosol_single_scattering_albedo=aerosol_single_scattering_albedo,
        )

        aerosol_scattering_tau = aerosol_extinction_tau * aerosol_single_scattering_albedo
        scattering_tau = rayleigh_scattering_tau + aerosol_scattering_tau.sum(axis=-1)
        total_tau = (
            gas_absorption_tau + rayleigh_scattering_tau + aerosol_extinction_tau.sum(axis=-1)
        )
        np.testing.assert_allclose(props.tau, total_tau)
        np.testing.assert_allclose(props.ssa, scattering_tau / total_tau)
        np.testing.assert_allclose(
            props.rayleigh_fraction, rayleigh_scattering_tau / scattering_tau
        )
        np.testing.assert_allclose(
            props.aerosol_fraction,
            aerosol_scattering_tau / scattering_tau[:, None],
        )

    def test_handles_clear_absorbing_layers_safely(self) -> None:
        props = build_layer_optical_properties(gas_absorption_tau=np.array([0.0, 0.2]))
        np.testing.assert_allclose(props.tau, np.array([0.0, 0.2]))
        np.testing.assert_allclose(props.ssa, np.array([0.0, 0.0]))
        np.testing.assert_allclose(props.rayleigh_fraction, np.array([0.0, 0.0]))
        self.assertEqual(props.aerosol_fraction.shape, (2, 0))

    def test_rejects_unphysical_aerosol_scattering(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            build_layer_optical_properties(
                gas_absorption_tau=np.array([0.1]),
                aerosol_extinction_tau=np.array([[0.2]]),
                aerosol_scattering_tau=np.array([[0.3]]),
            )

    def test_requires_explicit_aerosol_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "aerosol axis"):
            build_layer_optical_properties(
                gas_absorption_tau=np.array([0.1, 0.2]),
                aerosol_extinction_tau=np.array([0.3, 0.4]),
                aerosol_single_scattering_albedo=np.array([0.9, 0.8]),
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    def test_torch_matches_numpy_and_preserves_gradients(self) -> None:
        import torch

        from py2sess.optical.properties_torch import build_layer_optical_properties_torch

        gas = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        rayleigh = torch.tensor([0.3, 0.1], dtype=torch.float64, requires_grad=True)
        aerosol_extinction = torch.tensor(
            [[0.4, 0.2], [0.5, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        aerosol_ssa = torch.tensor([0.75, 0.5], dtype=torch.float64, requires_grad=True)

        torch_props = build_layer_optical_properties_torch(
            gas_absorption_tau=gas,
            rayleigh_scattering_tau=rayleigh,
            aerosol_extinction_tau=aerosol_extinction,
            aerosol_single_scattering_albedo=aerosol_ssa,
        )
        numpy_props = build_layer_optical_properties(
            gas_absorption_tau=gas.detach().numpy(),
            rayleigh_scattering_tau=rayleigh.detach().numpy(),
            aerosol_extinction_tau=aerosol_extinction.detach().numpy(),
            aerosol_single_scattering_albedo=aerosol_ssa.detach().numpy(),
        )
        torch.testing.assert_close(torch_props.tau, torch.as_tensor(numpy_props.tau))
        torch.testing.assert_close(torch_props.ssa, torch.as_tensor(numpy_props.ssa))
        torch.testing.assert_close(
            torch_props.rayleigh_fraction,
            torch.as_tensor(numpy_props.rayleigh_fraction),
        )
        torch.testing.assert_close(
            torch_props.aerosol_fraction,
            torch.as_tensor(numpy_props.aerosol_fraction),
        )

        objective = (
            torch_props.tau.sum()
            + torch_props.ssa.sum()
            + torch_props.rayleigh_fraction.sum()
            + torch_props.aerosol_fraction.sum()
        )
        objective.backward()
        for tensor in (gas, rayleigh, aerosol_extinction, aerosol_ssa):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(float(torch.abs(tensor.grad).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
