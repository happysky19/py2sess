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

The solar example uses the differentiable 2S torch kernel only. The current
standalone package has a batched NumPy FO solar helper for benchmark parity, but
not a torch-native FO solar helper, so the solar retrieval is limited to the
differentiable 2S path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from py2sess import thermal_source_from_temperature_profile_torch
from py2sess.core.solar_obs_batch_torch import solve_solar_obs_batch_torch
from py2sess.core.thermal_batch_torch import solve_thermal_batch_torch


torch.set_default_dtype(torch.float64)


PriorMode = Literal["off", "weak"]


@dataclass(frozen=True)
class RetrievalResult:
    """Small container for retrieval summary values."""

    name: str
    initial_objective: float
    initial_data_loss: float
    final_objective: float
    final_data_loss: float
    prior_mode: PriorMode
    noise_fraction: float
    truth: dict[str, float | np.ndarray]
    estimate: dict[str, float | np.ndarray]


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a NumPy array for printing."""
    return value.detach().cpu().numpy()


def _relative_rmse(model: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Returns a scale-normalized mean-squared residual."""
    scale = torch.clamp(torch.mean(torch.abs(observed)), min=1.0e-12)
    return torch.mean(((model - observed) / scale) ** 2)


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
        clean, true_tau_scale, true_temperature, true_surface_temperature = multi_angle_forward(
            true_tau_raw,
            true_coeffs,
            true_surf_raw,
        )
        noise_sigma = noise_fraction * torch.mean(torch.abs(clean))
        observed = clean + noise_sigma * torch.randn_like(clean)

    tau_raw = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    if fit_temperature:
        coeffs = torch.zeros(3, dtype=dtype, requires_grad=True)
    else:
        coeffs = true_coeffs.clone()
    surf_raw = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    optimizer_parameters = [tau_raw, surf_raw]
    if fit_temperature:
        optimizer_parameters.insert(1, coeffs)
    optimizer = torch.optim.Adam(optimizer_parameters, lr=0.06)
    initial_loss: tuple[float, float] | None = None

    def objective() -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the thermal retrieval objective."""
        model, _tau_scale, temperature, _surface_temperature = multi_angle_forward(
            tau_raw, coeffs, surf_raw
        )
        data_loss = _relative_rmse(model, observed)
        if prior_mode == "weak":
            smoothness = torch.mean(torch.diff(temperature, n=2) ** 2)
            # Weak background priors stabilize the tau/temperature tradeoff in
            # noisy examples. Keep them opt-in: in a zero-noise sanity test,
            # priors can deliberately move the solution away from truth.
            tau_background_raw = torch.log(torch.tensor(1.15, dtype=dtype))
            tau_prior = ((tau_raw - tau_background_raw) / 0.18) ** 2
            coeff_prior = torch.mean((coeffs / torch.tensor([6.0, 6.0, 6.0], dtype=dtype)) ** 2)
            surface_prior = (surf_raw / 0.6) ** 2
            prior_loss = 2.0e-5 * smoothness + 4.0e-4 * tau_prior + 1.0e-4 * coeff_prior
            prior_loss = prior_loss + 5.0e-5 * surface_prior
        else:
            prior_loss = torch.zeros((), dtype=dtype)
        return data_loss + prior_loss, data_loss

    for _step in range(520):
        optimizer.zero_grad()
        loss, data_loss = objective()
        if initial_loss is None:
            initial_loss = (float(loss.detach()), float(data_loss.detach()))
        loss.backward()
        optimizer.step()

    lbfgs = torch.optim.LBFGS(optimizer_parameters, lr=0.7, max_iter=40)

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, _data_loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        _model, tau_scale, temperature, surface_temperature = multi_angle_forward(
            tau_raw, coeffs, surf_raw
        )
        final_objective, final_data_loss = objective()

    if initial_loss is None:
        raise RuntimeError("thermal optimizer did not run")

    return RetrievalResult(
        name=(
            "thermal FO+2S full-state retrieval"
            if fit_temperature
            else "thermal FO+2S tau/surface-temperature sanity retrieval"
        ),
        initial_objective=float(initial_loss[0]),
        initial_data_loss=float(initial_loss[1]),
        final_objective=float(final_objective),
        final_data_loss=float(final_data_loss),
        prior_mode=prior_mode,
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
        clean, true_tau_scale, true_albedo_est = _solar_forward(
            tau_scale_raw=true_tau_raw,
            albedo_raw=true_albedo_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )
        noise_sigma = noise_fraction * torch.mean(torch.abs(clean))
        observed = clean + noise_sigma * torch.randn_like(clean)

    tau_raw = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    albedo_background = torch.tensor(0.25, dtype=dtype)
    albedo_raw = torch.logit(albedo_background).requires_grad_()
    optimizer = torch.optim.Adam([tau_raw, albedo_raw], lr=0.05)
    initial_loss: tuple[float, float] | None = None

    def objective() -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the solar retrieval objective."""
        model, _tau_scale, albedo = _solar_forward(
            tau_scale_raw=tau_raw,
            albedo_raw=albedo_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )
        data_loss = _relative_rmse(model, observed)
        if prior_mode == "weak":
            tau_prior = (tau_raw / 0.35) ** 2
            albedo_prior = ((albedo - albedo_background) / 0.08) ** 2
            prior_loss = 2.0e-4 * tau_prior + 1.0e-4 * albedo_prior
        else:
            prior_loss = torch.zeros((), dtype=dtype)
        return data_loss + prior_loss, data_loss

    for _step in range(420):
        optimizer.zero_grad()
        loss, data_loss = objective()
        if initial_loss is None:
            initial_loss = (float(loss.detach()), float(data_loss.detach()))
        loss.backward()
        optimizer.step()

    lbfgs = torch.optim.LBFGS([tau_raw, albedo_raw], lr=0.8, max_iter=40)

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, _data_loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        _model, tau_scale, albedo = _solar_forward(
            tau_scale_raw=tau_raw,
            albedo_raw=albedo_raw,
            base_tau=base_tau,
            omega=omega,
            asymm=asymm,
            scaling=scaling,
            flux_factor=flux_factor,
            geometry=geometry,
        )
        final_objective, final_data_loss = objective()

    if initial_loss is None:
        raise RuntimeError("solar optimizer did not run")

    return RetrievalResult(
        name="solar 2S albedo/tau retrieval",
        initial_objective=float(initial_loss[0]),
        initial_data_loss=float(initial_loss[1]),
        final_objective=float(final_objective),
        final_data_loss=float(final_data_loss),
        prior_mode=prior_mode,
        noise_fraction=noise_fraction,
        truth={
            "tau_scale": float(true_tau_scale),
            "albedo": float(true_albedo_est),
        },
        estimate={
            "tau_scale": float(tau_scale),
            "albedo": float(albedo),
        },
    )


def _format_array(values: np.ndarray) -> str:
    """Formats a compact numeric profile."""
    return np.array2string(values, precision=2, floatmode="fixed", max_line_width=120)


def _print_result(result: RetrievalResult) -> None:
    """Prints one retrieval result in a readable block."""
    print(f"\n{result.name}")
    print("-" * len(result.name))
    print(f"prior mode: {result.prior_mode}")
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
    return parser.parse_args()


def main() -> None:
    """Runs the synthetic retrieval examples."""
    args = _parse_args()
    print("Synthetic py2sess retrieval examples")
    print("Default mode is a zero-noise/no-prior synthetic sanity check.")
    print("Use --prior-mode weak with nonzero noise for a noisy stabilized example.")
    for result in (
        retrieve_thermal_synthetic(
            noise_fraction=args.thermal_noise,
            prior_mode=args.prior_mode,
            fit_temperature=False,
        ),
        retrieve_thermal_synthetic(
            noise_fraction=args.thermal_noise,
            prior_mode=args.prior_mode,
            fit_temperature=True,
        ),
        retrieve_solar_synthetic(
            noise_fraction=args.solar_noise,
            prior_mode=args.prior_mode,
        ),
    ):
        _print_result(result)


if __name__ == "__main__":
    main()
