from __future__ import annotations

import unittest

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions, thermal_source_from_temperature_profile_torch
from py2sess.core.backend import has_torch
from py2sess.retrieval import (
    finite_difference_jacobian,
    forward_value_and_jacobian,
    relative_jacobian_error,
)


@unittest.skipUnless(has_torch(), "torch not installed")
class GradientCheckTests(unittest.TestCase):
    def test_solar_public_forward_gradients_match_finite_difference(self) -> None:
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(nlyr=3, mode="solar", backend="torch", torch_dtype="float64")
        )
        base_tau = torch.tensor(
            [
                [0.03, 0.04, 0.05],
                [0.04, 0.05, 0.06],
                [0.05, 0.06, 0.07],
                [0.06, 0.07, 0.08],
            ],
            dtype=torch.float64,
        )
        z = np.array([3.0, 2.0, 1.0, 0.0])

        def forward_model(state: torch.Tensor) -> torch.Tensor:
            tau = base_tau * torch.exp(state[0])
            ssa = torch.sigmoid(state[1]).expand_as(base_tau)
            g = (0.5 * torch.tanh(state[2])).expand_as(base_tau)
            albedo = torch.sigmoid(state[3])
            return solver.forward(
                tau=tau,
                ssa=ssa,
                g=g,
                z=z,
                angles=[35.0, 20.0, 40.0],
                albedo=albedo,
            ).radiance

        state = np.array(
            [
                np.log(1.1),
                np.log(0.22 / 0.78),
                np.arctanh(0.30 / 0.5),
                np.log(0.27 / 0.73),
            ],
            dtype=float,
        )
        _value, ad_jacobian = forward_value_and_jacobian(forward_model, state)
        fd_jacobian = finite_difference_jacobian(forward_model, state, step=1.0e-5)

        self.assertLess(relative_jacobian_error(fd_jacobian, ad_jacobian), 2.0e-5)

    def test_thermal_source_and_surface_gradients_match_finite_difference(self) -> None:
        import torch

        solver = TwoStreamEss(
            TwoStreamEssOptions(nlyr=3, mode="thermal", backend="torch", torch_dtype="float64")
        )
        n_wavelengths = 5
        n_layers = 3
        z = np.array([3.0, 2.0, 1.0, 0.0])
        wavenumber = torch.linspace(720.0, 760.0, n_wavelengths, dtype=torch.float64)
        base_tau = torch.tensor(
            [
                [0.04, 0.05, 0.06],
                [0.05, 0.06, 0.07],
                [0.06, 0.07, 0.08],
                [0.07, 0.08, 0.09],
                [0.08, 0.09, 0.10],
            ],
            dtype=torch.float64,
        )
        omega = torch.full((n_wavelengths, n_layers), 0.06, dtype=torch.float64)
        asymm = torch.full((n_wavelengths, n_layers), 0.2, dtype=torch.float64)
        scaling = torch.full((n_wavelengths, n_layers), 0.01, dtype=torch.float64)
        temp_prior = torch.linspace(235.0, 285.0, n_layers + 1, dtype=torch.float64)
        temp_basis = torch.stack(
            (
                torch.ones(n_layers + 1, dtype=torch.float64),
                torch.linspace(-1.0, 1.0, n_layers + 1, dtype=torch.float64),
            ),
            dim=1,
        )

        def forward_model(state: torch.Tensor) -> torch.Tensor:
            tau = base_tau * torch.exp(state[0])
            temperature = temp_prior + temp_basis @ state[1:3]
            surface_temperature = 290.0 + 8.0 * torch.tanh(state[3])
            emissivity = torch.sigmoid(state[4])
            source = thermal_source_from_temperature_profile_torch(
                temperature,
                surface_temperature,
                wavenumber_cm_inv=wavenumber,
            )
            return solver.forward(
                tau=tau,
                ssa=omega,
                g=asymm,
                z=z,
                angles=25.0,
                delta_m_scaling=scaling,
                planck=source.thermal_bb_input,
                surface_planck=source.surfbb,
                emissivity=emissivity,
                albedo=1.0 - emissivity,
                include_fo=True,
            ).radiance

        state = np.array([np.log(1.05), 1.0, -1.5, np.arctanh(2.0 / 8.0), 2.0], dtype=float)
        _value, ad_jacobian = forward_value_and_jacobian(forward_model, state)
        fd_jacobian = finite_difference_jacobian(forward_model, state, step=1.0e-5)

        self.assertLess(relative_jacobian_error(fd_jacobian, ad_jacobian), 5.0e-5)


if __name__ == "__main__":
    unittest.main()
