"""Synthetic retrieval workflows using py2sess and torch autograd.

This example keeps the py2sess core unchanged and demonstrates how to wrap the
existing torch forward kernels in small inverse problems.

The workflows are intentionally compact:

1. Thermal FO + 2S tau/surface-temperature sanity retrieval
   Retrieves optical-depth scale and surface temperature while holding the
   atmospheric temperature profile fixed. With zero noise and no priors, this
   should recover the synthetic truth to numerical precision.

2. Thermal FO + 2S full-state retrieval
   Retrieves a smooth temperature profile, surface temperature, and a scalar
   optical-depth scale. This is intentionally ill-conditioned because thermal
   emission, absorption, and surface temperature can compensate for one another.

3. Solar 2S retrieval
   Retrieves surface albedo and a scalar optical-depth scale from a synthetic
   solar-observation spectrum.

4. UV benchmark 2S retrieval
   Retrieves scalar optical-depth and albedo multipliers from the packaged UV
   benchmark geometry.

The solar and UV examples use the differentiable 2S torch kernel only. The
current standalone package has a batched NumPy FO solar helper for benchmark
parity, but not a torch-native FO solar helper, so those retrievals are limited
to the differentiable 2S path.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

import numpy as np
import torch
from scipy.optimize import least_squares

from py2sess import thermal_source_from_temperature_profile_torch
from py2sess.core.solar_obs_batch_torch import solve_solar_obs_batch_torch
from py2sess.core.thermal_batch_torch import solve_thermal_batch_torch
from py2sess.reference_cases import load_uv_benchmark_case


torch.set_default_dtype(torch.float64)


PriorMode = Literal["off", "weak"]
NoiseMode = Literal["absolute", "relative", "hybrid"]


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Optimal-estimation diagnostics at the retrieved state."""

    n_observations: int
    n_state: int
    n_function_evaluations: int
    n_jacobian_evaluations: int | None
    jacobian_condition: float
    hessian_condition: float
    jacobian_singular_values: np.ndarray
    measurement_jacobian: np.ndarray
    measurement_hessian: np.ndarray
    gauss_newton_hessian: np.ndarray
    prior_precision: np.ndarray
    posterior_covariance: np.ndarray
    averaging_kernel: np.ndarray
    degrees_of_freedom: float
    measurement_error: np.ndarray


@dataclass(frozen=True)
class SpectrumComparison:
    """Pre-noise, post-noise, and fitted spectra for charting retrieval results."""

    x: np.ndarray
    pre_noise: np.ndarray
    post_noise: np.ndarray
    fitted: np.ndarray
    x_label: str
    y_label: str
    group_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RodgersPrior:
    """Gaussian background prior in the retrieval state space."""

    background: np.ndarray
    inverse_sqrt_covariance: np.ndarray

    @classmethod
    def from_precision(cls, background: np.ndarray, precision: np.ndarray) -> "RodgersPrior":
        """Builds a prior from ``S_a^-1``."""
        background = np.asarray(background, dtype=float)
        precision = np.asarray(precision, dtype=float)
        cholesky = np.linalg.cholesky(precision)
        return cls(background=background, inverse_sqrt_covariance=cholesky.T)

    def residual_and_jacobian(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``S_a^-1/2 (x - x_a)`` and its Jacobian."""
        residual = self.inverse_sqrt_covariance @ (state - self.background)
        return residual, self.inverse_sqrt_covariance

    @property
    def precision(self) -> np.ndarray:
        """Returns ``S_a^-1``."""
        return self.inverse_sqrt_covariance.T @ self.inverse_sqrt_covariance


@dataclass(frozen=True)
class RetrievalResult:
    """Small container for retrieval summary values."""

    name: str
    initial_objective: float
    initial_data_loss: float
    final_objective: float
    final_data_loss: float
    prior_mode: PriorMode
    noise_model: NoiseMode
    noise_fraction: float
    truth: dict[str, float | np.ndarray]
    estimate: dict[str, float | np.ndarray]
    diagnostics: RetrievalDiagnostics
    spectrum: SpectrumComparison


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a NumPy array for printing."""
    return value.detach().cpu().numpy()


def _as_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts torch or NumPy values to a floating NumPy array."""
    if isinstance(value, torch.Tensor):
        return _as_numpy(value)
    return np.asarray(value, dtype=float)


def _noise_std(
    pre_noise: torch.Tensor,
    noise_fraction: float,
    noise_model: NoiseMode,
) -> torch.Tensor:
    """Returns per-channel radiance-noise standard deviations."""
    scale = torch.clamp(torch.mean(torch.abs(pre_noise)), min=1.0e-12)
    fraction = noise_fraction if noise_fraction > 0.0 else 1.0
    absolute = torch.full_like(pre_noise, fraction * scale)
    relative = fraction * torch.clamp(torch.abs(pre_noise), min=1.0e-6 * scale)
    if noise_model == "absolute":
        return absolute
    if noise_model == "relative":
        return relative
    if noise_model == "hybrid":
        return torch.sqrt(absolute**2 + relative**2)
    raise ValueError(f"unknown noise model: {noise_model}")


def _add_noise(
    pre_noise: torch.Tensor,
    noise_fraction: float,
    noise_model: NoiseMode,
) -> torch.Tensor:
    """Returns a noisy copy of a clean synthetic spectrum."""
    if noise_fraction <= 0.0:
        return pre_noise.clone()
    std = _noise_std(pre_noise, noise_fraction, noise_model)
    return pre_noise + std * torch.randn_like(pre_noise)


def _spectrum_comparison(
    *,
    x: torch.Tensor | np.ndarray,
    pre_noise: torch.Tensor,
    post_noise: torch.Tensor,
    fitted: torch.Tensor,
    n_spectra: int,
    n_wavelengths: int,
    x_label: str,
    y_label: str,
    group_labels: tuple[str, ...],
) -> SpectrumComparison:
    """Builds consistently shaped spectrum data for charting."""
    shape = (n_spectra, n_wavelengths)
    return SpectrumComparison(
        x=_as_array(x),
        pre_noise=_as_numpy(pre_noise).reshape(shape),
        post_noise=_as_numpy(post_noise).reshape(shape),
        fitted=_as_numpy(fitted).reshape(shape),
        x_label=x_label,
        y_label=y_label,
        group_labels=group_labels,
    )


def _condition_number(singular_values: np.ndarray) -> float:
    """Returns a stable condition number from descending singular values."""
    if singular_values.size == 0:
        return float("nan")
    smallest = float(singular_values[-1])
    if smallest <= np.finfo(float).eps:
        return float("inf")
    return float(singular_values[0] / smallest)


def _forward_value_and_jacobian(
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates a torch forward model and its dense state Jacobian."""
    state_tensor = torch.as_tensor(state, dtype=torch.float64)

    def flattened_forward(current_state: torch.Tensor) -> torch.Tensor:
        return forward_model(current_state).reshape(-1)

    value = flattened_forward(state_tensor)
    jacobian = torch.func.jacfwd(flattened_forward)(state_tensor)
    return _as_numpy(value), _as_numpy(jacobian)


def _forward_value(
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    state: np.ndarray,
) -> np.ndarray:
    """Evaluates a torch forward model as a NumPy vector."""
    state_tensor = torch.as_tensor(state, dtype=torch.float64)
    return _as_numpy(forward_model(state_tensor).reshape(-1))


def _empty_prior(_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns an empty residual/Jacobian block for no-prior retrievals."""
    return np.empty(0, dtype=float), np.empty((0, _state.size), dtype=float)


def _measurement_error(
    pre_noise: torch.Tensor,
    noise_fraction: float,
    noise_model: NoiseMode,
) -> np.ndarray:
    """Builds a diagonal measurement-error standard deviation vector."""
    return _as_numpy(_noise_std(pre_noise, noise_fraction, noise_model).reshape(-1))


def _optimal_estimation_least_squares(
    *,
    initial_state: np.ndarray,
    observed: torch.Tensor,
    measurement_error: np.ndarray,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    prior: RodgersPrior | None = None,
    max_nfev: int = 80,
) -> tuple[np.ndarray, float, float, float, float, RetrievalDiagnostics]:
    """Solves a small Rodgers-style retrieval with torch Jacobians.

    The minimized residual is

    ``[S_e^-1/2 (F(x) - y), S_a^-1/2 (x - x_a)]``.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    observed_np = _as_numpy(observed.reshape(-1))
    measurement_error = np.asarray(measurement_error, dtype=float).reshape(-1)
    if observed_np.shape != measurement_error.shape:
        raise ValueError("measurement_error must have the same flattened size as observed")
    if not np.all(np.isfinite(measurement_error)) or np.any(measurement_error <= 0.0):
        raise ValueError("measurement_error must contain finite positive values")

    def residual_parts(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = _forward_value(forward_model, state)
        data_residual = (model - observed_np) / measurement_error
        if prior is None:
            prior_residual, _prior_jacobian = _empty_prior(state)
        else:
            prior_residual, _prior_jacobian = prior.residual_and_jacobian(state)
        return data_residual, prior_residual

    def objective_values(state: np.ndarray) -> tuple[float, float]:
        data_residual, prior_residual = residual_parts(state)
        data_loss = float(np.mean(data_residual**2))
        objective = 0.5 * float(data_residual @ data_residual + prior_residual @ prior_residual)
        return objective, data_loss

    def residual(state: np.ndarray) -> np.ndarray:
        data_residual, prior_residual = residual_parts(state)
        return np.concatenate((data_residual, prior_residual))

    def jacobian(state: np.ndarray) -> np.ndarray:
        _model, model_jacobian = _forward_value_and_jacobian(forward_model, state)
        data_jacobian = model_jacobian / measurement_error[:, None]
        if prior is None:
            _prior, prior_jacobian = _empty_prior(state)
        else:
            _prior, prior_jacobian = prior.residual_and_jacobian(state)
        return np.vstack((data_jacobian, prior_jacobian))

    initial_objective, initial_data_loss = objective_values(initial_state)
    solution = least_squares(
        residual,
        initial_state,
        jac=jacobian,
        method="trf",
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=max_nfev,
    )
    if not solution.success:
        raise RuntimeError(f"least_squares did not converge: {solution.message}")
    final_objective, final_data_loss = objective_values(solution.x)

    _model, model_jacobian = _forward_value_and_jacobian(forward_model, solution.x)
    measurement_jacobian = model_jacobian / measurement_error[:, None]
    measurement_hessian = measurement_jacobian.T @ measurement_jacobian
    prior_precision = np.zeros((solution.x.size, solution.x.size), dtype=float)
    if prior is not None:
        prior_precision = prior.precision
    gauss_newton_hessian = measurement_hessian + prior_precision
    posterior_covariance = np.linalg.pinv(gauss_newton_hessian, rcond=1.0e-12)
    averaging_kernel = posterior_covariance @ measurement_hessian
    singular_values = np.linalg.svd(measurement_jacobian, compute_uv=False)
    hessian_singular_values = np.linalg.svd(gauss_newton_hessian, compute_uv=False)
    diagnostics = RetrievalDiagnostics(
        n_observations=int(observed_np.size),
        n_state=int(solution.x.size),
        n_function_evaluations=int(solution.nfev),
        n_jacobian_evaluations=None if solution.njev is None else int(solution.njev),
        jacobian_condition=_condition_number(singular_values),
        hessian_condition=_condition_number(hessian_singular_values),
        jacobian_singular_values=singular_values,
        measurement_jacobian=measurement_jacobian,
        measurement_hessian=measurement_hessian,
        gauss_newton_hessian=gauss_newton_hessian,
        prior_precision=prior_precision,
        posterior_covariance=posterior_covariance,
        averaging_kernel=averaging_kernel,
        degrees_of_freedom=float(np.trace(averaging_kernel)),
        measurement_error=measurement_error,
    )
    return (
        solution.x,
        initial_objective,
        initial_data_loss,
        final_objective,
        final_data_loss,
        diagnostics,
    )


def _thermal_forward(
    *,
    tau_scale_raw: torch.Tensor,
    temp_coeffs: torch.Tensor,
    surf_temp_raw: torch.Tensor,
    base_tau: torch.Tensor,
    omega: torch.Tensor,
    asymm: torch.Tensor,
    scaling: torch.Tensor,
    wavenumber: torch.Tensor,
    heights: np.ndarray,
    temp_basis: torch.Tensor,
    temp_prior: torch.Tensor,
    user_angle_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the differentiable thermal FO + 2S forward model."""
    tau_scale = torch.exp(tau_scale_raw)
    temperature = temp_prior + temp_basis @ temp_coeffs
    surface_temperature = 292.0 + 12.0 * torch.tanh(surf_temp_raw)
    source = thermal_source_from_temperature_profile_torch(
        temperature,
        surface_temperature,
        wavenumber_cm_inv=wavenumber,
    )

    result = solve_thermal_batch_torch(
        tau_arr=base_tau * tau_scale,
        omega_arr=omega,
        asymm_arr=asymm,
        d2s_scaling=scaling,
        thermal_bb_input=source.thermal_bb_input,
        surfbb=source.surfbb,
        albedo=torch.full((base_tau.shape[0],), 0.02, dtype=base_tau.dtype),
        emissivity=torch.full((base_tau.shape[0],), 0.98, dtype=base_tau.dtype),
        heights=heights,
        user_angle_degrees=user_angle_degrees,
        stream_value=0.5,
        # Dense BVP is slower than the production path, but avoids recurrence
        # updates that are awkward for reverse-mode autograd demos.
        bvp_engine="dense",
    )
    return result.total_toa, tau_scale, temperature, surface_temperature


def retrieve_thermal_synthetic(
    seed: int = 7,
    *,
    noise_fraction: float = 0.0,
    noise_model: NoiseMode = "absolute",
    prior_mode: PriorMode = "off",
    fit_temperature: bool = True,
) -> RetrievalResult:
    """Retrieves TIR temperature and optical-depth scale from synthetic radiances."""
    torch.manual_seed(seed)
    dtype = torch.float64
    n_wavelengths = 96
    n_layers = 6
    heights = np.linspace(18.0, 0.0, n_layers + 1)
    wavenumber = torch.linspace(680.0, 820.0, n_wavelengths, dtype=dtype)
    layer_shape = torch.linspace(0.7, 1.4, n_layers, dtype=dtype)
    phase = torch.linspace(0.0, 2.5 * np.pi, n_wavelengths)
    spectral_shape = (
        1.0 + 0.45 * torch.sin(phase) + 0.30 * torch.exp(-0.5 * ((wavenumber - 735.0) / 16.0) ** 2)
    )
    base_tau = 0.085 * spectral_shape[:, None] * layer_shape[None, :]
    omega = torch.full_like(base_tau, 0.08)
    asymm = torch.full_like(base_tau, 0.25)
    scaling = torch.full_like(base_tau, 0.02)

    temp_prior = torch.linspace(218.0, 286.0, n_layers + 1, dtype=dtype)
    temp_basis = torch.stack(
        (
            torch.ones(n_layers + 1, dtype=dtype),
            torch.linspace(-1.0, 1.0, n_layers + 1, dtype=dtype),
            torch.sin(torch.linspace(0.0, np.pi, n_layers + 1, dtype=dtype)),
        ),
        dim=1,
    )
    true_coeffs = torch.tensor([1.5, -2.0, 3.0], dtype=dtype)
    true_tau_raw = torch.tensor(np.log(1.18), dtype=dtype)
    true_surf_raw = torch.atanh(torch.tensor((296.0 - 292.0) / 12.0, dtype=dtype))

    def multi_angle_forward(
        tau_scale_raw: torch.Tensor,
        temp_coeffs: torch.Tensor,
        surf_temp_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenates two thermal viewing angles for a better constrained retrieval."""
        outputs = []
        tau_scale = temperature = surface_temperature = None
        for user_angle in (20.0, 55.0):
            model, tau_scale, temperature, surface_temperature = _thermal_forward(
                tau_scale_raw=tau_scale_raw,
                temp_coeffs=temp_coeffs,
                surf_temp_raw=surf_temp_raw,
                base_tau=base_tau,
                omega=omega,
                asymm=asymm,
                scaling=scaling,
                wavenumber=wavenumber,
                heights=heights,
                temp_basis=temp_basis,
                temp_prior=temp_prior,
                user_angle_degrees=user_angle,
            )
            outputs.append(model)
        if tau_scale is None or temperature is None or surface_temperature is None:
            raise RuntimeError("thermal multi-angle forward produced no outputs")
        return torch.cat(outputs), tau_scale, temperature, surface_temperature

    with torch.no_grad():
        pre_noise, true_tau_scale, true_temperature, true_surface_temperature = multi_angle_forward(
            true_tau_raw,
            true_coeffs,
            true_surf_raw,
        )
        post_noise = _add_noise(pre_noise, noise_fraction, noise_model)

    state_size = 5 if fit_temperature else 2
    initial_state = np.zeros(state_size, dtype=float)

    def unpack_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps the least-squares state vector to physical raw variables."""
        tau = state[0]
        if fit_temperature:
            temperature_coeffs = state[1:4]
            surface = state[4]
        else:
            temperature_coeffs = true_coeffs
            surface = state[1]
        return tau, temperature_coeffs, surface

    def forward_from_state(state: torch.Tensor) -> torch.Tensor:
        tau, temperature_coeffs, surface = unpack_state(state)
        model, _tau_scale, _temperature, _surface_temperature = multi_angle_forward(
            tau,
            temperature_coeffs,
            surface,
        )
        return model

    prior = None
    if prior_mode == "weak":
        prior_background = np.zeros(state_size, dtype=float)
        prior_background[0] = np.log(1.15)
        prior_stdev = np.array([0.18, 0.6], dtype=float)
        prior_precision = np.diag(1.0 / prior_stdev**2)
        if fit_temperature:
            prior_background = np.array([np.log(1.15), 0.0, 0.0, 0.0, 0.0], dtype=float)
            prior_stdev = np.array([0.18, 6.0, 6.0, 6.0, 0.6], dtype=float)
            prior_precision = np.diag(1.0 / prior_stdev**2)
            d2_matrix = np.diff(np.eye(n_layers + 1), n=2, axis=0)
            curvature_jacobian = d2_matrix @ _as_numpy(temp_basis)
            curvature_stdev = 10.0
            prior_precision[1:4, 1:4] += (
                curvature_jacobian.T @ curvature_jacobian
            ) / curvature_stdev**2
        prior = RodgersPrior.from_precision(prior_background, prior_precision)

    (
        retrieved_state,
        initial_objective,
        initial_data_loss,
        final_objective,
        final_data_loss,
        diagnostics,
    ) = _optimal_estimation_least_squares(
        initial_state=initial_state,
        observed=post_noise,
        measurement_error=_measurement_error(pre_noise, noise_fraction, noise_model),
        forward_model=forward_from_state,
        prior=prior,
        max_nfev=120,
    )

    retrieved_state_tensor = torch.as_tensor(retrieved_state, dtype=dtype)
    tau_raw, coeffs, surf_raw = unpack_state(retrieved_state_tensor)
    with torch.no_grad():
        fitted_spectrum, tau_scale, temperature, surface_temperature = multi_angle_forward(
            tau_raw,
            coeffs,
            surf_raw,
        )

    return RetrievalResult(
        name=(
            "thermal FO+2S full-state retrieval"
            if fit_temperature
            else "thermal FO+2S tau/surface-temperature sanity retrieval"
        ),
        initial_objective=initial_objective,
        initial_data_loss=initial_data_loss,
        final_objective=final_objective,
        final_data_loss=final_data_loss,
        prior_mode=prior_mode,
        noise_model=noise_model,
        noise_fraction=noise_fraction,
        truth={
            "tau_scale": float(true_tau_scale),
            "surface_temperature": float(true_surface_temperature),
            "temperature": _as_numpy(true_temperature),
        },
        estimate={
            "tau_scale": float(tau_scale),
            "surface_temperature": float(surface_temperature),
            "temperature": _as_numpy(temperature),
        },
        diagnostics=diagnostics,
        spectrum=_spectrum_comparison(
            x=wavenumber,
            pre_noise=pre_noise,
            post_noise=post_noise,
            fitted=fitted_spectrum,
            n_spectra=2,
            n_wavelengths=n_wavelengths,
            x_label="wavenumber (cm^-1)",
            y_label="TOA radiance",
            group_labels=("20 deg", "55 deg"),
        ),
    )


def _solar_geometry() -> dict[str, float | np.ndarray]:
    """Builds a small fixed solar-observation geometry dictionary."""
    n_layers = 6
    solar_zenith_degrees = 35.0
    view_zenith_degrees = 20.0
    azimuth_degrees = 40.0
    solar_zenith = np.deg2rad(solar_zenith_degrees)
    view_zenith = np.deg2rad(view_zenith_degrees)
    azimuth = np.deg2rad(azimuth_degrees)
    stream_value = 1.0 / np.sqrt(3.0)
    px11 = np.sqrt(0.5 * (1.0 - stream_value**2))
    x0 = float(np.cos(solar_zenith))
    user_stream = float(np.cos(view_zenith))
    chapman = np.triu(np.ones((n_layers, n_layers), dtype=float)) / x0
    pxsq = np.array([stream_value**2, px11**2], dtype=float)
    px0x = np.array([x0 * stream_value, np.sqrt(0.5 * (1.0 - x0**2)) * px11], dtype=float)
    return {
        "stream_value": stream_value,
        "chapman": chapman,
        "x0": x0,
        "user_stream": user_stream,
        "user_secant": 1.0 / user_stream,
        "azmfac": float(np.cos(azimuth)),
        "px11": px11,
        "pxsq": pxsq,
        "px0x": px0x,
        "ulp": -np.sqrt(0.5 * (1.0 - user_stream**2)),
    }


def _solar_forward(
    *,
    tau_scale_raw: torch.Tensor,
    albedo_raw: torch.Tensor,
    base_tau: torch.Tensor,
    omega: torch.Tensor,
    asymm: torch.Tensor,
    scaling: torch.Tensor,
    flux_factor: torch.Tensor,
    geometry: dict[str, float | np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the differentiable solar-observation 2S forward model."""
    tau_scale = torch.exp(tau_scale_raw)
    albedo = torch.sigmoid(albedo_raw)
    radiance = solve_solar_obs_batch_torch(
        tau=base_tau * tau_scale,
        omega=omega,
        asymm=asymm,
        scaling=scaling,
        albedo=albedo.expand(base_tau.shape[0]),
        flux_factor=flux_factor,
        stream_value=float(geometry["stream_value"]),
        chapman=geometry["chapman"],
        x0=float(geometry["x0"]),
        user_stream=float(geometry["user_stream"]),
        user_secant=float(geometry["user_secant"]),
        azmfac=float(geometry["azmfac"]),
        px11=float(geometry["px11"]),
        pxsq=geometry["pxsq"],
        px0x=geometry["px0x"],
        ulp=float(geometry["ulp"]),
        dtype=base_tau.dtype,
        device="cpu",
        # See the thermal example above: dense is the conservative autograd path.
        bvp_engine="dense",
    )
    return radiance, tau_scale, albedo


def retrieve_solar_synthetic(
    seed: int = 11,
    *,
    noise_fraction: float = 0.0,
    noise_model: NoiseMode = "absolute",
    prior_mode: PriorMode = "off",
) -> RetrievalResult:
    """Retrieves surface albedo and optical-depth scale from a synthetic solar spectrum."""
    torch.manual_seed(seed)
    dtype = torch.float64
    n_wavelengths = 128
    n_layers = 6
    phase = torch.linspace(0.0, 2.0 * np.pi, n_wavelengths)
    spectral_shape = (
        1.0
        + 0.50 * torch.cos(phase)
        + 0.35 * torch.exp(-0.5 * ((torch.linspace(-1.0, 1.0, n_wavelengths)) / 0.28) ** 2)
    )
    layer_shape = torch.linspace(1.2, 0.7, n_layers, dtype=dtype)
    base_tau = 0.055 * spectral_shape[:, None] * layer_shape[None, :]
    omega = torch.full_like(base_tau, 0.22)
    asymm = torch.full_like(base_tau, 0.35)
    scaling = torch.full_like(base_tau, 0.03)
    flux_factor = torch.ones(n_wavelengths, dtype=dtype)
    geometry = _solar_geometry()

    true_tau_raw = torch.tensor(np.log(0.82), dtype=dtype)
    true_albedo = torch.tensor(0.27, dtype=dtype)
    true_albedo_raw = torch.logit(true_albedo)

    with torch.no_grad():
        pre_noise, true_tau_scale, true_albedo_est = _solar_forward(
            tau_scale_raw=true_tau_raw,
            albedo_raw=true_albedo_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )
        post_noise = _add_noise(pre_noise, noise_fraction, noise_model)

    albedo_background = torch.tensor(0.25, dtype=dtype)
    initial_state = np.array([0.0, float(torch.logit(albedo_background))], dtype=float)

    def forward_from_state(state: torch.Tensor) -> torch.Tensor:
        model, _tau_scale, albedo = _solar_forward(
            tau_scale_raw=state[0],
            albedo_raw=state[1],
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )
        return model

    prior = None
    if prior_mode == "weak":
        albedo_raw_background = float(torch.logit(albedo_background))
        prior_background = np.array([0.0, albedo_raw_background], dtype=float)
        prior_stdev = np.array([0.35, 0.45], dtype=float)
        prior = RodgersPrior.from_precision(prior_background, np.diag(1.0 / prior_stdev**2))

    (
        retrieved_state,
        initial_objective,
        initial_data_loss,
        final_objective,
        final_data_loss,
        diagnostics,
    ) = _optimal_estimation_least_squares(
        initial_state=initial_state,
        observed=post_noise,
        measurement_error=_measurement_error(pre_noise, noise_fraction, noise_model),
        forward_model=forward_from_state,
        prior=prior,
        max_nfev=80,
    )

    tau_raw = torch.tensor(retrieved_state[0], dtype=dtype)
    albedo_raw = torch.tensor(retrieved_state[1], dtype=dtype)
    with torch.no_grad():
        fitted_spectrum, tau_scale, albedo = _solar_forward(
            tau_scale_raw=tau_raw,
            albedo_raw=albedo_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )

    return RetrievalResult(
        name="solar 2S albedo/tau retrieval",
        initial_objective=initial_objective,
        initial_data_loss=initial_data_loss,
        final_objective=final_objective,
        final_data_loss=final_data_loss,
        prior_mode=prior_mode,
        noise_model=noise_model,
        noise_fraction=noise_fraction,
        truth={
            "tau_scale": float(true_tau_scale),
            "albedo": float(true_albedo_est),
        },
        estimate={
            "tau_scale": float(tau_scale),
            "albedo": float(albedo),
        },
        diagnostics=diagnostics,
        spectrum=_spectrum_comparison(
            x=np.arange(n_wavelengths, dtype=float),
            pre_noise=pre_noise,
            post_noise=post_noise,
            fitted=fitted_spectrum,
            n_spectra=1,
            n_wavelengths=n_wavelengths,
            x_label="spectral index",
            y_label="2S TOA radiance",
            group_labels=("solar 2S",),
        ),
    )


def _uv_benchmark_forward(
    *,
    tau_scale_raw: torch.Tensor,
    albedo_scale_raw: torch.Tensor,
    base_tau: torch.Tensor,
    omega: torch.Tensor,
    asymm: torch.Tensor,
    scaling: torch.Tensor,
    albedo: torch.Tensor,
    flux_factor: torch.Tensor,
    case,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the differentiable 2S path for the packaged UV benchmark geometry."""
    tau_scale = torch.exp(tau_scale_raw)
    albedo_scale = torch.exp(albedo_scale_raw)
    radiance = solve_solar_obs_batch_torch(
        tau=base_tau * tau_scale,
        omega=omega,
        asymm=asymm,
        scaling=scaling,
        albedo=albedo * albedo_scale,
        flux_factor=flux_factor,
        stream_value=case.stream_value,
        chapman=case.chapman,
        x0=case.x0,
        user_stream=case.user_stream,
        user_secant=case.user_secant,
        azmfac=case.azmfac,
        px11=case.px11,
        pxsq=case.pxsq,
        px0x=case.px0x,
        ulp=case.ulp,
        dtype=base_tau.dtype,
        device="cpu",
        bvp_engine="dense",
    )
    return radiance, tau_scale, albedo_scale


def retrieve_uv_benchmark_synthetic(
    seed: int = 13,
    *,
    noise_fraction: float = 0.0,
    noise_model: NoiseMode = "absolute",
    prior_mode: PriorMode = "off",
) -> RetrievalResult:
    """Retrieves scalar tau/albedo multipliers from the packaged UV 2S case."""
    torch.manual_seed(seed)
    dtype = torch.float64
    case = load_uv_benchmark_case()
    base_tau = torch.as_tensor(case.tau, dtype=dtype)
    omega = torch.as_tensor(case.omega, dtype=dtype)
    asymm = torch.as_tensor(case.asymm, dtype=dtype)
    scaling = torch.as_tensor(case.scaling, dtype=dtype)
    albedo = torch.as_tensor(case.albedo, dtype=dtype)
    flux_factor = torch.as_tensor(case.flux_factor, dtype=dtype)

    true_tau_raw = torch.tensor(0.0, dtype=dtype)
    true_albedo_scale_raw = torch.tensor(0.0, dtype=dtype)
    with torch.no_grad():
        pre_noise, true_tau_scale, true_albedo_scale = _uv_benchmark_forward(
            tau_scale_raw=true_tau_raw,
            albedo_scale_raw=true_albedo_scale_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            albedo=albedo,
            flux_factor=flux_factor,
            case=case,
        )
        post_noise = _add_noise(pre_noise, noise_fraction, noise_model)

    initial_state = np.array([np.log(0.9), np.log(1.1)], dtype=float)

    def forward_from_state(state: torch.Tensor) -> torch.Tensor:
        model, _tau_scale, _albedo_scale = _uv_benchmark_forward(
            tau_scale_raw=state[0],
            albedo_scale_raw=state[1],
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            albedo=albedo,
            flux_factor=flux_factor,
            case=case,
        )
        return model

    prior = None
    if prior_mode == "weak":
        prior_background = np.zeros(2, dtype=float)
        prior_stdev = np.array([0.35, 0.25], dtype=float)
        prior = RodgersPrior.from_precision(prior_background, np.diag(1.0 / prior_stdev**2))

    (
        retrieved_state,
        initial_objective,
        initial_data_loss,
        final_objective,
        final_data_loss,
        diagnostics,
    ) = _optimal_estimation_least_squares(
        initial_state=initial_state,
        observed=post_noise,
        measurement_error=_measurement_error(pre_noise, noise_fraction, noise_model),
        forward_model=forward_from_state,
        prior=prior,
        max_nfev=80,
    )

    with torch.no_grad():
        fitted_spectrum, tau_scale, albedo_scale = _uv_benchmark_forward(
            tau_scale_raw=torch.tensor(retrieved_state[0], dtype=dtype),
            albedo_scale_raw=torch.tensor(retrieved_state[1], dtype=dtype),
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            albedo=albedo,
            flux_factor=flux_factor,
            case=case,
        )

    return RetrievalResult(
        name="UV benchmark 2S albedo-scale/tau-scale retrieval",
        initial_objective=initial_objective,
        initial_data_loss=initial_data_loss,
        final_objective=final_objective,
        final_data_loss=final_data_loss,
        prior_mode=prior_mode,
        noise_model=noise_model,
        noise_fraction=noise_fraction,
        truth={
            "tau_scale": float(true_tau_scale),
            "albedo_scale": float(true_albedo_scale),
        },
        estimate={
            "tau_scale": float(tau_scale),
            "albedo_scale": float(albedo_scale),
        },
        diagnostics=diagnostics,
        spectrum=_spectrum_comparison(
            x=case.wavelengths,
            pre_noise=pre_noise,
            post_noise=post_noise,
            fitted=fitted_spectrum,
            n_spectra=1,
            n_wavelengths=case.n_wavelengths,
            x_label="wavelength (nm)",
            y_label="2S TOA radiance",
            group_labels=("UV benchmark 2S",),
        ),
    )


def _format_array(values: np.ndarray) -> str:
    """Formats a compact numeric profile."""
    return np.array2string(values, precision=2, floatmode="fixed", max_line_width=120)


def _format_singular_values(values: np.ndarray) -> str:
    """Formats singular values for compact console output."""
    return np.array2string(values, precision=3, suppress_small=False, max_line_width=120)


def _slugify(value: str) -> str:
    """Builds a stable filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "retrieval"


def _import_pyplot():
    """Imports matplotlib lazily for optional chart generation."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Plotting requires matplotlib. Install py2sess with the plot extra."
        ) from exc
    return plt


def _plot_spectrum_comparison(
    *,
    ax_fit,
    ax_diff,
    spectrum: SpectrumComparison,
) -> None:
    """Plots pre-noise, post-noise, fitted spectra, and signed fit residuals."""
    colors = ("#0f766e", "#7c3aed", "#0369a1", "#be123c")
    labels = spectrum.group_labels or tuple(
        f"spectrum {index + 1}" for index in range(spectrum.pre_noise.shape[0])
    )
    for index, (pre_noise, post_noise, fitted) in enumerate(
        zip(spectrum.pre_noise, spectrum.post_noise, spectrum.fitted)
    ):
        color = colors[index % len(colors)]
        label = labels[index] if index < len(labels) else f"spectrum {index + 1}"
        ax_fit.plot(
            spectrum.x,
            pre_noise,
            color=color,
            linewidth=2.2,
            label=f"{label} pre-noise",
        )
        ax_fit.plot(
            spectrum.x,
            post_noise,
            color=color,
            linewidth=1.7,
            linestyle=":",
            label=f"{label} post-noise",
        )
        ax_fit.plot(
            spectrum.x,
            fitted,
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label} fitted",
        )
        ax_diff.plot(
            spectrum.x,
            post_noise - fitted,
            color=color,
            linewidth=2.0,
            label=f"{label} post-noise - fitted",
        )

    ax_fit.set_xlabel(spectrum.x_label)
    ax_fit.set_ylabel(spectrum.y_label)
    ax_fit.set_yscale("log")
    ax_fit.legend(frameon=False, fontsize=9)
    ax_fit.grid(True, color="#d6d3d1", linewidth=0.8, alpha=0.7)

    ax_diff.set_xlabel(spectrum.x_label)
    ax_diff.set_ylabel(f"post-noise - fitted {spectrum.y_label}")
    ax_diff.axhline(0.0, color="#57534e", linewidth=0.9, alpha=0.7)
    ax_diff.legend(frameon=False, fontsize=9)
    ax_diff.grid(True, color="#d6d3d1", linewidth=0.8, alpha=0.7)


def save_retrieval_chart(result: RetrievalResult, output_dir: Path | str) -> Path:
    """Saves a two-panel pre-noise/post-noise/fitted spectrum chart."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _import_pyplot()

    fig, (ax_fit, ax_diff) = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(result.name, fontsize=14, fontweight="bold")
    ax_fit.set_title("Pre-noise, post-noise, and fitted radiance")
    ax_diff.set_title("Post-noise - fitted")
    _plot_spectrum_comparison(ax_fit=ax_fit, ax_diff=ax_diff, spectrum=result.spectrum)

    for axis in (ax_fit, ax_diff):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    path = output_path / f"{_slugify(result.name)}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _print_result(result: RetrievalResult) -> None:
    """Prints one retrieval result in a readable block."""
    print(f"\n{result.name}")
    print("-" * len(result.name))
    print(f"prior mode: {result.prior_mode}")
    print(f"noise model: {result.noise_model}")
    print(f"noise fraction: {result.noise_fraction:.3e}")
    print(f"initial objective: {result.initial_objective:.6e}")
    print(f"initial data loss: {result.initial_data_loss:.6e}")
    print(f"final objective:   {result.final_objective:.6e}")
    print(f"final data loss:   {result.final_data_loss:.6e}")
    for key, truth_value in result.truth.items():
        estimate_value = result.estimate[key]
        if isinstance(truth_value, np.ndarray):
            print(f"{key} truth:    {_format_array(truth_value)}")
            print(f"{key} estimate: {_format_array(np.asarray(estimate_value))}")
        else:
            print(f"{key}: truth={truth_value:.6f}, estimate={float(estimate_value):.6f}")
    diagnostics = result.diagnostics
    posterior_stdev = np.sqrt(np.maximum(np.diag(diagnostics.posterior_covariance), 0.0))
    print(
        "Jacobian:"
        f" observations={diagnostics.n_observations}"
        f" state={diagnostics.n_state}"
        f" condition={diagnostics.jacobian_condition:.3e}"
    )
    print(f"Gauss-Newton Hessian condition: {diagnostics.hessian_condition:.3e}")
    print(f"degrees of freedom for signal: {diagnostics.degrees_of_freedom:.3f}")
    print(
        f"Jacobian singular values: {_format_singular_values(diagnostics.jacobian_singular_values)}"
    )
    print(f"posterior state stdev: {_format_singular_values(posterior_stdev)}")
    print(
        "least_squares:"
        f" nfev={diagnostics.n_function_evaluations}"
        f" njev={diagnostics.n_jacobian_evaluations}"
    )


def _parse_args() -> argparse.Namespace:
    """Parses command-line options for the synthetic retrieval demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Run small differentiable synthetic retrieval examples. The default "
            "is a zero-noise/no-prior sanity test; use --prior-mode weak and "
            "nonzero noise fractions for a stabilized noisy demonstration."
        )
    )
    parser.add_argument(
        "--prior-mode",
        choices=("off", "weak"),
        default="off",
        help="Use no priors or weak background priors in the objective.",
    )
    parser.add_argument(
        "--noise-model",
        choices=("absolute", "relative", "hybrid"),
        default="absolute",
        help="absolute=mean radiance, relative=per-channel radiance, hybrid=quadrature sum.",
    )
    parser.add_argument(
        "--thermal-noise",
        type=float,
        default=0.0,
        help="Thermal white-noise fraction relative to mean absolute radiance.",
    )
    parser.add_argument(
        "--solar-noise",
        type=float,
        default=0.0,
        help="Solar white-noise fraction relative to mean absolute radiance.",
    )
    parser.add_argument(
        "--uv-noise",
        type=float,
        default=0.0,
        help="UV benchmark white-noise fraction relative to mean absolute 2S radiance.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory where retrieval spectrum-comparison charts are saved as PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the synthetic retrieval examples."""
    args = _parse_args()
    print("Synthetic py2sess retrieval examples")
    print("Default mode is a zero-noise/no-prior synthetic sanity check.")
    print("Use --prior-mode weak with nonzero noise for a noisy stabilized example.")
    results = (
        retrieve_thermal_synthetic(
            noise_fraction=args.thermal_noise,
            noise_model=args.noise_model,
            prior_mode=args.prior_mode,
            fit_temperature=False,
        ),
        retrieve_thermal_synthetic(
            noise_fraction=args.thermal_noise,
            noise_model=args.noise_model,
            prior_mode=args.prior_mode,
            fit_temperature=True,
        ),
        retrieve_solar_synthetic(
            noise_fraction=args.solar_noise,
            noise_model=args.noise_model,
            prior_mode=args.prior_mode,
        ),
        retrieve_uv_benchmark_synthetic(
            noise_fraction=args.uv_noise,
            noise_model=args.noise_model,
            prior_mode=args.prior_mode,
        ),
    )
    for result in results:
        _print_result(result)
    if args.plot_dir is not None:
        print("\nSaved retrieval charts:")
        for result in results:
            print(f"  {save_retrieval_chart(result, args.plot_dir)}")


if __name__ == "__main__":
    main()
