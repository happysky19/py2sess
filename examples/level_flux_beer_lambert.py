"""DISORT-style level-flux output for a clear absorbing solar column."""

from __future__ import annotations

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions


def main() -> None:
    """Runs a two-layer Beer-Lambert case with an analytic flux solution."""
    sza = 30.0
    mu0 = np.cos(np.deg2rad(sza))
    fbeam = 1.0
    tau = np.array([0.1, 0.2], dtype=float)
    z = np.array([2.0, 1.0, 0.0], dtype=float)

    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tau.size,
            mode="solar",
            plane_parallel=True,
            delta_scaling=False,
            downwelling=True,
            output_levels=True,
            output_fluxes=True,
            fo_flux_n_mu=8,
        )
    )
    result = solver.forward(
        tau=tau,
        ssa=np.zeros_like(tau),
        g=np.zeros_like(tau),
        z=z,
        angles=[sza, 0.0, 0.0],
        fbeam=fbeam,
        albedo=0.0,
        delta_m_truncation_factor=np.zeros_like(tau),
        include_fo=True,
    )

    level_tau = np.concatenate(([0.0], np.cumsum(tau)))
    analytic_down = fbeam * mu0 * np.exp(-level_tau / mu0)

    flux_down = result.flux_down[0]
    flux_up = result.flux_up[0]
    flux_net = result.flux_net[0]

    np.testing.assert_allclose(flux_down, analytic_down, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(flux_up, 0.0, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(flux_net, flux_up - flux_down, rtol=0.0, atol=0.0)

    print("level  tau_cum  py2sess_down  analytic_down  flux_up      flux_net")
    for ilev, (tau_cum, down, expected, up, net) in enumerate(
        zip(level_tau, flux_down, analytic_down, flux_up, flux_net, strict=True)
    ):
        print(f"{ilev:5d}  {tau_cum:7.3f}  {down:12.8f}  {expected:13.8f}  {up:9.2e}  {net:11.8f}")


if __name__ == "__main__":
    main()
