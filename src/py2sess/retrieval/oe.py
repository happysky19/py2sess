"""Rodgers-style optimal-estimation retrieval utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.backend import _load_torch


def _require_torch():
    torch = _load_torch()
    if torch is None:  # pragma: no cover
        raise RuntimeError("py2sess retrieval requires PyTorch")
    return torch


def _context_from_values(*values):
    torch = _require_torch()
    for value in values:
        if torch.is_tensor(value):
            return value.dtype, value.device
    return torch.float64, torch.device("cpu")


def _as_tensor(value, *, dtype, device):
    torch = _require_torch()
    if torch.is_tensor(value):
        if value.dtype != dtype or value.device != device:
            return value.to(dtype=dtype, device=device)
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_flat_vector(value, *, dtype, device):
    return _as_tensor(value, dtype=dtype, device=device).reshape(-1)


def _as_covariance(name: str, value, *, size: int, dtype, device):
    torch = _require_torch()
    tensor = _as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        if float(tensor.detach().cpu()) <= 0.0:
            raise ValueError(f"{name} scalar covariance must be positive")
        return tensor * torch.eye(size, dtype=dtype, device=device)
    if tensor.ndim == 1:
        if tensor.numel() != size:
            raise ValueError(f"{name} diagonal must have length {size}")
        if not bool((tensor > 0.0).all()):
            raise ValueError(f"{name} diagonal entries must be positive")
        return torch.diag(tensor)
    if tensor.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    return tensor


def _solve_matrix(system, rhs):
    return _require_torch().linalg.solve(system, rhs)


def _inverse_matrix(matrix):
    torch = _require_torch()
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return torch.linalg.solve(matrix, eye)


@dataclass(frozen=True)
class NoiseModel:
    """Diagonal measurement-noise model for synthetic retrieval tests."""

    kind: str = "absolute"
    level: float = 0.0
    absolute_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.kind not in {"absolute", "relative", "hybrid"}:
            raise ValueError("kind must be 'absolute', 'relative', or 'hybrid'")
        if self.level < 0.0:
            raise ValueError("level must be non-negative")
        if self.absolute_floor <= 0.0:
            raise ValueError("absolute_floor must be positive")

    def stddev(self, reference):
        """Returns per-channel standard deviations."""
        torch = _require_torch()
        dtype, device = _context_from_values(reference)
        ref = _as_tensor(reference, dtype=dtype, device=device)
        level = torch.as_tensor(self.level, dtype=dtype, device=device)
        floor = torch.as_tensor(self.absolute_floor, dtype=dtype, device=device)
        if self.kind == "absolute":
            sigma = level * torch.ones_like(ref)
        elif self.kind == "relative":
            sigma = level * torch.abs(ref)
        else:
            sigma = torch.sqrt((level * torch.abs(ref)) ** 2 + floor**2)
        return torch.clamp(sigma, min=floor)

    def covariance(self, reference):
        """Returns the diagonal measurement covariance matrix."""
        torch = _require_torch()
        sigma = self.stddev(reference).reshape(-1)
        return torch.diag(sigma * sigma)

    def sample(self, reference, *, generator=None):
        """Draws synthetic noise with zero exactly when ``level`` is zero."""
        torch = _require_torch()
        sigma = self.stddev(reference)
        if self.level == 0.0:
            return torch.zeros_like(sigma)
        return sigma * torch.randn(
            sigma.shape, dtype=sigma.dtype, device=sigma.device, generator=generator
        )


@dataclass(frozen=True)
class OptimalEstimationProblem:
    """Inputs for a Rodgers-style optimal-estimation retrieval."""

    forward_model: Callable[[Any], Any]
    observation: Any
    prior_state: Any
    prior_covariance: Any
    measurement_covariance: Any
    state_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Standard optimal-estimation diagnostics at one linearization point."""

    posterior_covariance: Any
    averaging_kernel: Any
    dfs: Any
    singular_values: Any
    hessian_condition: Any


@dataclass(frozen=True)
class OptimalEstimationResult:
    """Result from ``solve_optimal_estimation``."""

    state: Any
    radiance: Any
    residual: Any
    cost_history: tuple[float, ...]
    jacobian: Any
    posterior_covariance: Any
    averaging_kernel: Any
    dfs: Any
    singular_values: Any
    hessian_condition: Any
    converged: bool
    n_iterations: int
    state_names: tuple[str, ...] = ()


def evaluate_jacobian(forward_model: Callable[[Any], Any], state):
    """Evaluates ``forward_model`` and its dense Jacobian with torch autograd."""
    torch = _require_torch()
    dtype, device = _context_from_values(state)
    x = _as_flat_vector(state, dtype=dtype, device=device).detach().clone().requires_grad_(True)

    def wrapped(current):
        return forward_model(current).reshape(-1)

    y = wrapped(x)
    jacobian = torch.autograd.functional.jacobian(wrapped, x, create_graph=False)
    return y.detach(), jacobian.reshape(y.numel(), x.numel()).detach()


def retrieval_diagnostics(
    jacobian, measurement_covariance, prior_covariance
) -> RetrievalDiagnostics:
    """Computes posterior covariance, averaging kernel, DFS, and conditioning."""
    torch = _require_torch()
    dtype, device = _context_from_values(jacobian, measurement_covariance, prior_covariance)
    k = _as_tensor(jacobian, dtype=dtype, device=device)
    se = _as_covariance(
        "measurement_covariance",
        measurement_covariance,
        size=k.shape[0],
        dtype=dtype,
        device=device,
    )
    sa = _as_covariance(
        "prior_covariance",
        prior_covariance,
        size=k.shape[1],
        dtype=dtype,
        device=device,
    )
    se_inv_k = _solve_matrix(se, k)
    sa_inv = _inverse_matrix(sa)
    gain_hessian = k.T @ se_inv_k
    hessian = gain_hessian + sa_inv
    posterior = _inverse_matrix(hessian)
    averaging_kernel = posterior @ gain_hessian
    se_chol = torch.linalg.cholesky(se)
    sa_chol = torch.linalg.cholesky(sa)
    whitened = torch.linalg.solve(se_chol, k @ sa_chol)
    singular_values = torch.linalg.svdvals(whitened)
    hessian_singular_values = torch.linalg.svdvals(hessian)
    condition = hessian_singular_values[0] / hessian_singular_values[-1]
    return RetrievalDiagnostics(
        posterior_covariance=posterior.detach(),
        averaging_kernel=averaging_kernel.detach(),
        dfs=torch.trace(averaging_kernel).detach(),
        singular_values=singular_values.detach(),
        hessian_condition=condition.detach(),
    )


def _prepare_problem(problem: OptimalEstimationProblem):
    dtype, device = _context_from_values(
        problem.observation,
        problem.prior_state,
        problem.prior_covariance,
        problem.measurement_covariance,
    )
    observation = _as_flat_vector(problem.observation, dtype=dtype, device=device)
    prior = _as_flat_vector(problem.prior_state, dtype=dtype, device=device)
    prior_covariance = _as_covariance(
        "prior_covariance",
        problem.prior_covariance,
        size=prior.numel(),
        dtype=dtype,
        device=device,
    )
    measurement_covariance = _as_covariance(
        "measurement_covariance",
        problem.measurement_covariance,
        size=observation.numel(),
        dtype=dtype,
        device=device,
    )
    if problem.state_names and len(problem.state_names) != prior.numel():
        raise ValueError("state_names length must match prior_state")
    return observation, prior, prior_covariance, measurement_covariance


def _oe_cost(forward_model, state, observation, prior, se, sa):
    y = forward_model(state).reshape(-1)
    residual = observation - y
    departure = state - prior
    return residual @ _solve_matrix(se, residual) + departure @ _solve_matrix(sa, departure)


def solve_optimal_estimation(
    problem: OptimalEstimationProblem,
    *,
    initial_state: Any | None = None,
    max_iter: int = 8,
    step_tolerance: float = 1.0e-10,
    cost_tolerance: float = 1.0e-12,
) -> OptimalEstimationResult:
    """Solves a small Rodgers-style optimal-estimation inverse problem."""
    torch = _require_torch()
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    observation, prior, sa, se = _prepare_problem(problem)
    state = (
        prior.clone()
        if initial_state is None
        else _as_flat_vector(initial_state, dtype=prior.dtype, device=prior.device)
    )
    cost = float(_oe_cost(problem.forward_model, state, observation, prior, se, sa).detach().cpu())
    cost_history = [cost]
    converged = False
    accepted_iterations = 0
    eye_state = torch.eye(prior.numel(), dtype=prior.dtype, device=prior.device)

    for _iteration in range(max_iter):
        radiance, jacobian = evaluate_jacobian(problem.forward_model, state)
        se_inv_k = _solve_matrix(se, jacobian)
        sa_inv = _solve_matrix(sa, eye_state)
        hessian = jacobian.T @ se_inv_k + sa_inv
        rhs = jacobian.T @ _solve_matrix(
            se,
            observation - radiance + jacobian @ (state - prior),
        )
        target = prior + _solve_matrix(hessian, rhs)
        step = target - state
        if float(torch.linalg.norm(step).detach().cpu()) <= step_tolerance * (
            1.0 + float(torch.linalg.norm(state).detach().cpu())
        ):
            converged = True
            break

        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = state + damping * step
            candidate_cost = float(
                _oe_cost(problem.forward_model, candidate, observation, prior, se, sa)
                .detach()
                .cpu()
            )
            if candidate_cost <= cost:
                state = candidate.detach()
                accepted_iterations += 1
                accepted = True
                cost_drop = cost - candidate_cost
                cost = candidate_cost
                cost_history.append(cost)
                if cost_drop <= cost_tolerance * (1.0 + abs(cost_history[-2])):
                    converged = True
                break
        if not accepted or converged:
            break

    final_radiance, final_jacobian = evaluate_jacobian(problem.forward_model, state)
    diagnostics = retrieval_diagnostics(final_jacobian, se, sa)
    residual = observation - final_radiance
    return OptimalEstimationResult(
        state=state.detach(),
        radiance=final_radiance.detach(),
        residual=residual.detach(),
        cost_history=tuple(cost_history),
        jacobian=final_jacobian.detach(),
        posterior_covariance=diagnostics.posterior_covariance,
        averaging_kernel=diagnostics.averaging_kernel,
        dfs=diagnostics.dfs,
        singular_values=diagnostics.singular_values,
        hessian_condition=diagnostics.hessian_condition,
        converged=converged,
        n_iterations=accepted_iterations,
        state_names=problem.state_names,
    )
