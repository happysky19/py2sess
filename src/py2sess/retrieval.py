"""Small retrieval utilities built around differentiable py2sess forward models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from .core.backend import _load_torch


PriorMode = Literal["off", "weak"]
NoiseMode = Literal["absolute", "relative", "hybrid"]


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Optimal-estimation diagnostics at one retrieved state."""

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
    """Container for one compact retrieval result."""

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


def _require_torch():
    """Returns torch or raises a clear optional-dependency error."""
    torch = _load_torch()
    if torch is None:  # pragma: no cover
        raise RuntimeError("retrieval helpers require PyTorch")
    return torch


def as_numpy(value: Any) -> np.ndarray:
    """Converts torch or NumPy-like values to a floating NumPy array."""
    torch = _load_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _is_torch_tensor(value: Any) -> bool:
    """Returns whether ``value`` is a torch tensor without importing torch eagerly."""
    torch = _load_torch()
    return torch is not None and isinstance(value, torch.Tensor)


def noise_std(pre_noise: Any, noise_fraction: float, noise_model: NoiseMode) -> Any:
    """Returns per-channel radiance-noise standard deviations."""
    fraction = noise_fraction if noise_fraction > 0.0 else 1.0
    if _is_torch_tensor(pre_noise):
        torch = _require_torch()
        scale = torch.clamp(torch.mean(torch.abs(pre_noise)), min=1.0e-12)
        absolute = torch.full_like(pre_noise, fraction * scale)
        relative = fraction * torch.clamp(torch.abs(pre_noise), min=1.0e-6 * scale)
    else:
        array = np.asarray(pre_noise, dtype=float)
        scale = max(float(np.mean(np.abs(array))), 1.0e-12)
        absolute = np.full_like(array, fraction * scale, dtype=float)
        relative = fraction * np.maximum(np.abs(array), 1.0e-6 * scale)
    if noise_model == "absolute":
        return absolute
    if noise_model == "relative":
        return relative
    if noise_model == "hybrid":
        if _is_torch_tensor(pre_noise):
            torch = _require_torch()
            return torch.sqrt(absolute**2 + relative**2)
        return np.sqrt(absolute**2 + relative**2)
    raise ValueError(f"unknown noise model: {noise_model}")


def add_noise(pre_noise: Any, noise_fraction: float, noise_model: NoiseMode) -> Any:
    """Returns a noisy copy of a clean synthetic spectrum."""
    if noise_fraction <= 0.0:
        return pre_noise.clone() if _is_torch_tensor(pre_noise) else np.array(pre_noise, copy=True)
    if _is_torch_tensor(pre_noise):
        torch = _require_torch()
        return pre_noise + noise_std(pre_noise, noise_fraction, noise_model) * torch.randn_like(
            pre_noise
        )
    array = np.asarray(pre_noise, dtype=float)
    return array + noise_std(array, noise_fraction, noise_model) * np.random.standard_normal(
        array.shape
    )


def measurement_error(
    pre_noise: Any,
    noise_fraction: float,
    noise_model: NoiseMode,
) -> np.ndarray:
    """Builds a diagonal measurement-error standard deviation vector."""
    return as_numpy(noise_std(pre_noise, noise_fraction, noise_model).reshape(-1))


def spectrum_comparison(
    *,
    x: Any,
    pre_noise: Any,
    post_noise: Any,
    fitted: Any,
    n_spectra: int,
    n_wavelengths: int,
    x_label: str,
    y_label: str,
    group_labels: tuple[str, ...],
) -> SpectrumComparison:
    """Builds consistently shaped spectrum data for charting."""
    shape = (n_spectra, n_wavelengths)
    return SpectrumComparison(
        x=as_numpy(x),
        pre_noise=as_numpy(pre_noise).reshape(shape),
        post_noise=as_numpy(post_noise).reshape(shape),
        fitted=as_numpy(fitted).reshape(shape),
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


def forward_value_and_jacobian(
    forward_model: Callable[[Any], Any],
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates a torch forward model and its dense state Jacobian."""
    torch = _require_torch()
    state_tensor = torch.as_tensor(state, dtype=torch.float64)

    def flattened_forward(current_state):
        return forward_model(current_state).reshape(-1)

    value = flattened_forward(state_tensor)
    jacobian = torch.func.jacfwd(flattened_forward)(state_tensor)
    return as_numpy(value), as_numpy(jacobian)


def forward_value(forward_model: Callable[[Any], Any], state: np.ndarray) -> np.ndarray:
    """Evaluates a torch forward model as a NumPy vector."""
    torch = _require_torch()
    state_tensor = torch.as_tensor(state, dtype=torch.float64)
    return as_numpy(forward_model(state_tensor).reshape(-1))


def finite_difference_jacobian(
    forward_model: Callable[[Any], Any],
    state: np.ndarray,
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    """Returns a central finite-difference Jacobian for a small state vector."""
    state = np.asarray(state, dtype=float)
    base_value = forward_value(forward_model, state)
    jacobian = np.empty((base_value.size, state.size), dtype=float)
    for index in range(state.size):
        delta = np.zeros_like(state)
        delta[index] = step
        plus = forward_value(forward_model, state + delta)
        minus = forward_value(forward_model, state - delta)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def relative_jacobian_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    floor: float = 1.0e-12,
) -> float:
    """Returns the largest relative Jacobian mismatch with a stable floor."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    scale = np.maximum(np.abs(reference), floor)
    return float(np.max(np.abs(candidate - reference) / scale))


def _empty_prior(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns an empty residual/Jacobian block for no-prior retrievals."""
    return np.empty(0, dtype=float), np.empty((0, state.size), dtype=float)


@dataclass(frozen=True)
class RodgersObjective:
    """Rodgers-style residual and Jacobian around a differentiable forward model."""

    observed: np.ndarray
    measurement_error: np.ndarray
    forward_model: Callable[[Any], Any]
    prior: RodgersPrior | None = None

    @classmethod
    def from_observation(
        cls,
        *,
        observed: Any,
        measurement_error: np.ndarray,
        forward_model: Callable[[Any], Any],
        prior: RodgersPrior | None = None,
    ) -> "RodgersObjective":
        """Builds and validates an objective from observation-space inputs."""
        observed_np = as_numpy(observed).reshape(-1)
        measurement_error = np.asarray(measurement_error, dtype=float).reshape(-1)
        if observed_np.shape != measurement_error.shape:
            raise ValueError("measurement_error must have the same flattened size as observed")
        if not np.all(np.isfinite(measurement_error)) or np.any(measurement_error <= 0.0):
            raise ValueError("measurement_error must contain finite positive values")
        return cls(
            observed=observed_np,
            measurement_error=measurement_error,
            forward_model=forward_model,
            prior=prior,
        )

    def residual_parts(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns data and prior residual blocks."""
        model = forward_value(self.forward_model, state)
        data_residual = (model - self.observed) / self.measurement_error
        if self.prior is None:
            prior_residual, _prior_jacobian = _empty_prior(np.asarray(state, dtype=float))
        else:
            prior_residual, _prior_jacobian = self.prior.residual_and_jacobian(state)
        return data_residual, prior_residual

    def residual(self, state: np.ndarray) -> np.ndarray:
        """Returns the stacked Rodgers residual vector."""
        data_residual, prior_residual = self.residual_parts(state)
        return np.concatenate((data_residual, prior_residual))

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Returns the stacked residual Jacobian."""
        _model, model_jacobian = forward_value_and_jacobian(self.forward_model, state)
        data_jacobian = model_jacobian / self.measurement_error[:, None]
        if self.prior is None:
            _prior, prior_jacobian = _empty_prior(np.asarray(state, dtype=float))
        else:
            _prior, prior_jacobian = self.prior.residual_and_jacobian(state)
        return np.vstack((data_jacobian, prior_jacobian))

    def objective_values(self, state: np.ndarray) -> tuple[float, float]:
        """Returns total objective and mean squared data residual."""
        data_residual, prior_residual = self.residual_parts(state)
        data_loss = float(np.mean(data_residual**2))
        objective = 0.5 * float(data_residual @ data_residual + prior_residual @ prior_residual)
        return objective, data_loss

    def diagnostics(
        self,
        state: np.ndarray,
        *,
        n_function_evaluations: int,
        n_jacobian_evaluations: int | None,
    ) -> RetrievalDiagnostics:
        """Builds optimal-estimation diagnostics at ``state``."""
        _model, model_jacobian = forward_value_and_jacobian(self.forward_model, state)
        measurement_jacobian = model_jacobian / self.measurement_error[:, None]
        measurement_hessian = measurement_jacobian.T @ measurement_jacobian
        prior_precision = np.zeros((state.size, state.size), dtype=float)
        if self.prior is not None:
            prior_precision = self.prior.precision
        gauss_newton_hessian = measurement_hessian + prior_precision
        posterior_covariance = np.linalg.pinv(gauss_newton_hessian, rcond=1.0e-12)
        averaging_kernel = posterior_covariance @ measurement_hessian
        singular_values = np.linalg.svd(measurement_jacobian, compute_uv=False)
        hessian_singular_values = np.linalg.svd(gauss_newton_hessian, compute_uv=False)
        return RetrievalDiagnostics(
            n_observations=int(self.observed.size),
            n_state=int(state.size),
            n_function_evaluations=int(n_function_evaluations),
            n_jacobian_evaluations=n_jacobian_evaluations,
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
            measurement_error=self.measurement_error,
        )


def optimal_estimation_least_squares(
    *,
    initial_state: np.ndarray,
    observed: Any,
    measurement_error: np.ndarray,
    forward_model: Callable[[Any], Any],
    prior: RodgersPrior | None = None,
    max_nfev: int = 80,
) -> tuple[np.ndarray, float, float, float, float, RetrievalDiagnostics]:
    """Solves a small Rodgers-style least-squares retrieval."""
    initial_state = np.asarray(initial_state, dtype=float)
    objective = RodgersObjective.from_observation(
        observed=observed,
        measurement_error=measurement_error,
        forward_model=forward_model,
        prior=prior,
    )
    initial_objective, initial_data_loss = objective.objective_values(initial_state)
    solution = least_squares(
        objective.residual,
        initial_state,
        jac=objective.jacobian,
        method="trf",
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=max_nfev,
    )
    if not solution.success:
        raise RuntimeError(f"least_squares did not converge: {solution.message}")
    final_objective, final_data_loss = objective.objective_values(solution.x)
    diagnostics = objective.diagnostics(
        solution.x,
        n_function_evaluations=int(solution.nfev),
        n_jacobian_evaluations=None if solution.njev is None else int(solution.njev),
    )
    return (
        solution.x,
        initial_objective,
        initial_data_loss,
        final_objective,
        final_data_loss,
        diagnostics,
    )


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


def _plot_spectrum_comparison(*, ax_fit, ax_diff, spectrum: SpectrumComparison) -> None:
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
