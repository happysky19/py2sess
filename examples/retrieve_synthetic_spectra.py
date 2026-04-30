"""Rodgers-style optimal-estimation retrieval on small synthetic spectra."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from py2sess import (
    NoiseModel,
    OptimalEstimationProblem,
    TwoStreamEss,
    TwoStreamEssOptions,
    solve_optimal_estimation,
    thermal_source_from_temperature_profile_torch,
)
from py2sess.optical.phase_torch import (
    build_solar_fo_scatter_term_torch,
    build_two_stream_phase_inputs_torch,
)
from py2sess.reference_cases import load_uv_benchmark_case


@dataclass(frozen=True)
class SyntheticCase:
    """Container for a small retrieval problem."""

    name: str
    x_axis: np.ndarray
    x_label: str
    state_names: tuple[str, ...]
    truth_state: object
    prior_state: object
    prior_covariance: object
    forward_model: object


def _torch():
    import torch

    return torch


def _numpy(value) -> np.ndarray:
    torch = _torch()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _uv_case() -> SyntheticCase:
    torch = _torch()
    fixture = load_uv_benchmark_case()
    dtype = torch.float64
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=fixture.n_layers,
            mode="solar",
            backend="torch",
            torch_device="cpu",
            torch_dtype="float64",
        )
    )
    tau = torch.as_tensor(fixture.tau, dtype=dtype)
    ssa = torch.as_tensor(fixture.omega, dtype=dtype)
    depol = torch.as_tensor(fixture.depol, dtype=dtype)
    rayleigh_fraction = torch.as_tensor(fixture.rayleigh_fraction, dtype=dtype)
    aerosol_fraction = torch.as_tensor(fixture.aerosol_fraction, dtype=dtype)
    aerosol_moments = torch.as_tensor(fixture.aerosol_moments, dtype=dtype)
    aerosol_interp = torch.as_tensor(fixture.aerosol_interp_fraction, dtype=dtype)
    phase = build_two_stream_phase_inputs_torch(
        ssa=ssa,
        depol=depol,
        rayleigh_fraction=rayleigh_fraction,
        aerosol_fraction=aerosol_fraction,
        aerosol_moments=aerosol_moments,
        aerosol_interp_fraction=aerosol_interp,
    )
    fo_scatter = build_solar_fo_scatter_term_torch(
        ssa=ssa,
        depol=depol,
        rayleigh_fraction=rayleigh_fraction,
        aerosol_fraction=aerosol_fraction,
        aerosol_moments=aerosol_moments,
        aerosol_interp_fraction=aerosol_interp,
        angles=torch.as_tensor(fixture.user_obsgeom, dtype=dtype),
        delta_m_truncation_factor=phase.delta_m_truncation_factor,
    )
    albedo_truth = torch.as_tensor(0.20, dtype=dtype)

    def forward_model(state):
        tau_scale, albedo = state
        result = solver.forward(
            tau=tau_scale * tau,
            ssa=ssa,
            g=phase.g,
            z=fixture.heights,
            angles=fixture.user_obsgeom,
            stream=fixture.stream_value,
            fbeam=fixture.flux_factor,
            albedo=albedo,
            delta_m_truncation_factor=phase.delta_m_truncation_factor,
            include_fo=True,
            fo_scatter_term=fo_scatter,
        )
        return result.radiance_total.reshape(-1)

    return SyntheticCase(
        name="uv",
        x_axis=fixture.wavelengths,
        x_label="wavelength (nm)",
        state_names=("tau_scale", "albedo"),
        truth_state=torch.stack((torch.as_tensor(1.0, dtype=dtype), albedo_truth)),
        prior_state=torch.tensor([0.85, 0.12], dtype=dtype),
        prior_covariance=torch.diag(torch.tensor([0.35**2, 0.20**2], dtype=dtype)),
        forward_model=forward_model,
    )


def _thermal_sanity_case() -> SyntheticCase:
    torch = _torch()
    dtype = torch.float64
    nlyr = 3
    wavenumber = torch.linspace(700.0, 950.0, 12, dtype=dtype)
    layer_shape = torch.tensor([0.45, 0.35, 0.20], dtype=dtype)
    spectral_shape = torch.linspace(0.7, 1.3, wavenumber.numel(), dtype=dtype).reshape(-1, 1)
    tau = 0.08 * spectral_shape * layer_shape.reshape(1, -1)
    zeros = torch.zeros_like(tau)
    level_temperature = torch.tensor([225.0, 238.0, 252.0, 265.0], dtype=dtype)
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=nlyr,
            mode="thermal",
            backend="torch",
            torch_device="cpu",
            torch_dtype="float64",
        )
    )

    def forward_model(state):
        tau_scale, surface_temperature = state
        source = thermal_source_from_temperature_profile_torch(
            level_temperature,
            surface_temperature,
            wavenumber_cm_inv=wavenumber,
            dtype=dtype,
        )
        result = solver.forward(
            tau=tau_scale * tau,
            ssa=zeros,
            g=zeros,
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=25.0,
            albedo=0.04,
            emissivity=0.96,
            planck=source.planck,
            surface_planck=source.surface_planck,
            include_fo=True,
        )
        return result.radiance_total.reshape(-1)

    return SyntheticCase(
        name="thermal-sanity",
        x_axis=_numpy(wavenumber),
        x_label="wavenumber (cm^-1)",
        state_names=("tau_scale", "surface_temperature"),
        truth_state=torch.tensor([1.0, 288.0], dtype=dtype),
        prior_state=torch.tensor([0.80, 280.0], dtype=dtype),
        prior_covariance=torch.diag(torch.tensor([0.35**2, 12.0**2], dtype=dtype)),
        forward_model=forward_model,
    )


def _load_case(name: str) -> SyntheticCase:
    if name == "uv":
        return _uv_case()
    if name == "thermal-sanity":
        return _thermal_sanity_case()
    raise ValueError(f"unknown case {name!r}")


def _plot_spectrum(path: Path, case: SyntheticCase, truth, observed, fitted) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_axis = case.x_axis
    truth_np = _numpy(truth)
    observed_np = _numpy(observed)
    fitted_np = _numpy(fitted)
    residual = observed_np - fitted_np
    tiny = max(float(np.nanmin(truth_np[truth_np > 0.0])) * 0.1, 1.0e-30)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].semilogy(x_axis, np.maximum(truth_np, tiny), label="pre-noise truth")
    axes[0].semilogy(x_axis, np.maximum(observed_np, tiny), label="noisy observation")
    axes[0].semilogy(x_axis, np.maximum(fitted_np, tiny), label="fitted")
    axes[0].set_ylabel("radiance")
    axes[0].legend()
    axes[1].plot(x_axis, residual)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel(case.x_label)
    axes[1].set_ylabel("noisy - fitted")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["uv", "thermal-sanity"], default="uv")
    parser.add_argument(
        "--noise-kind", choices=["absolute", "relative", "hybrid"], default="hybrid"
    )
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    torch = _torch()
    case = _load_case(args.case)
    truth = case.forward_model(case.truth_state)
    noise = NoiseModel(kind=args.noise_kind, level=args.noise_level)
    generator = torch.Generator(device=truth.device).manual_seed(args.seed)
    observed = truth + noise.sample(truth, generator=generator)
    problem = OptimalEstimationProblem(
        forward_model=case.forward_model,
        observation=observed,
        prior_state=case.prior_state,
        prior_covariance=case.prior_covariance,
        measurement_covariance=noise.covariance(truth),
        state_names=case.state_names,
    )
    result = solve_optimal_estimation(problem, max_iter=10)

    print(f"case: {case.name}")
    print(
        f"status: {result.status.value} "
        f"converged: {result.converged} iterations: {result.n_iterations}"
    )
    print(f"cost: {result.cost_history[0]:.6e} -> {result.cost_history[-1]:.6e}")
    print(f"residual_norm: {float(torch.linalg.norm(result.residual)):.6e}")
    for name, truth_value, prior_value, fit_value in zip(
        case.state_names,
        _numpy(case.truth_state),
        _numpy(case.prior_state),
        _numpy(result.state),
    ):
        print(f"{name:20s} truth={truth_value:.8g} prior={prior_value:.8g} fit={fit_value:.8g}")
    print(f"dfs: {float(result.dfs):.6g}")
    print(f"hessian_condition: {float(result.hessian_condition):.6g}")
    print("singular_values:", " ".join(f"{value:.6g}" for value in _numpy(result.singular_values)))

    if args.plot is not None:
        _plot_spectrum(args.plot, case, truth, observed, result.radiance)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
