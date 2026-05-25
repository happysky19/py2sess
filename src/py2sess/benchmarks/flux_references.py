"""Flux-reference adapters used by benchmark and parity tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import expn

PYDISORT_FLUX_CHANNELS = (
    "direct_down",
    "diffuse_down",
    "diffuse_up",
    "flux_divergence",
    "mean_intensity",
    "mean_diffuse_down",
    "mean_diffuse_up",
    "mean_direct",
)


def _flip_level_axis(values: Any) -> Any:
    if type(values).__module__.startswith("torch") and hasattr(values, "flip"):
        return values.flip(dims=(-2,))
    return values[..., ::-1, :]


def pydisort_flux_to_py2sess(flux: Any, *, level_axis: str = "toa_to_boa") -> dict[str, Any]:
    """Maps pydisort ``gather_flx()`` output to py2sess level-flux names.

    pydisort returns ``(..., nlvl, 8)`` with direct downward, diffuse
    downward, diffuse upward, flux divergence, and mean-intensity channels.
    py2sess stores total downward flux, so the direct and diffuse downward
    channels are summed here.
    """
    if not hasattr(flux, "shape"):
        flux = np.asarray(flux)
    if len(flux.shape) < 2 or int(flux.shape[-1]) < len(PYDISORT_FLUX_CHANNELS):
        raise ValueError("pydisort flux output must have shape (..., nlvl, 8)")
    if level_axis == "boa_to_toa":
        flux = _flip_level_axis(flux)
    elif level_axis != "toa_to_boa":
        raise ValueError("level_axis must be 'toa_to_boa' or 'boa_to_toa'")

    direct_down = flux[..., 0]
    diffuse_down = flux[..., 1]
    diffuse_up = flux[..., 2]
    flux_down = direct_down + diffuse_down
    flux_up = diffuse_up
    flux_mean = flux[..., 4]
    flux_net = flux_up - flux_down
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_net,
        "flux_mean": flux_mean,
        "direct_down": direct_down,
        "diffuse_down": diffuse_down,
        "diffuse_up": diffuse_up,
        "flux_divergence": flux[..., 3],
        "mean_intensity": flux_mean,
        "mean_diffuse_down": flux[..., 5],
        "mean_diffuse_up": flux[..., 6],
        "mean_direct": flux[..., 7],
    }


def run_pydisort_absorbing_solar_flux(
    tau: Any,
    *,
    mu0: float,
    fbeam: float = math.pi,
    albedo: float = 0.0,
    nstr: int = 4,
    nmom: int = 4,
    dtype: str = "float64",
) -> dict[str, Any]:
    """Runs a small pure-absorbing pydisort solar-flux reference case."""
    import pydisort
    import torch

    tau_np = np.asarray(tau, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")
    torch_dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
    nlyr = int(tau_np.shape[0])
    nprop = 2 + int(nmom)

    options = pydisort.DisortOptions().flags("onlyfl,lamber").nwave(1).ncol(1)
    options.ds().nlyr = nlyr
    options.ds().nstr = int(nstr)
    options.ds().nmom = int(nmom)
    options.ds().nphase = int(nmom)

    prop = torch.zeros((1, 1, nlyr, nprop), dtype=torch_dtype)
    prop[0, 0, :, 0] = torch.as_tensor(tau_np, dtype=torch_dtype)
    prop[0, 0, :, 1] = 0.0
    prop[0, 0, :, 2] = 1.0

    solver = pydisort.Disort(options)
    solver.forward(
        prop,
        umu0=torch.tensor([mu0], dtype=torch_dtype),
        fbeam=torch.tensor([[fbeam]], dtype=torch_dtype),
        albedo=torch.tensor([[albedo]], dtype=torch_dtype),
    )
    return pydisort_flux_to_py2sess(solver.gather_flx())


def solar_isotropic_single_scatter_flux(
    tau: Any,
    *,
    ssa: Any,
    mu0: float,
    fbeam: float = math.pi,
    include_direct: bool = True,
) -> dict[str, np.ndarray]:
    """Analytic-kernel single-scatter solar flux for isotropic scattering.

    The case assumes plane-parallel geometry, no surface reflection, no
    delta-M scaling, and a direct solar beam incident at cosine ``mu0``.
    """
    tau_np = np.asarray(tau, dtype=float)
    omega = np.asarray(ssa, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if omega.ndim == 0:
        omega = np.full_like(tau_np, float(omega))
    if omega.shape != tau_np.shape:
        raise ValueError("ssa must be scalar or match tau")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")

    levels = np.concatenate(([0.0], np.cumsum(tau_np)))
    nlevels = levels.size
    flux_up = np.zeros(nlevels, dtype=float)
    flux_down = np.zeros(nlevels, dtype=float)
    flux_mean = np.zeros(nlevels, dtype=float)

    def integrate_level(level: float, *, upward: bool, order: int) -> float:
        total = 0.0
        for layer, layer_omega in enumerate(omega):
            top = levels[layer]
            bottom = levels[layer + 1]
            if upward:
                lower = max(level, top)
                upper = bottom
            else:
                lower = top
                upper = min(level, bottom)
            if upper <= lower:
                continue

            def integrand(t: float) -> float:
                distance = (t - level) if upward else (level - t)
                return float(layer_omega) * math.exp(-t / mu0) * float(expn(order, distance))

            total += quad(
                integrand,
                lower,
                upper,
                points=[level] if lower <= level <= upper else None,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=100,
            )[0]
        return total

    for idx, level in enumerate(levels):
        flux_up[idx] = 0.5 * fbeam * integrate_level(float(level), upward=True, order=2)
        flux_down[idx] = 0.5 * fbeam * integrate_level(float(level), upward=False, order=2)
        flux_mean[idx] = (
            fbeam
            / (8.0 * math.pi)
            * (
                integrate_level(float(level), upward=True, order=1)
                + integrate_level(float(level), upward=False, order=1)
            )
        )

    if include_direct:
        direct_trans = np.exp(-levels / mu0)
        flux_down = flux_down + fbeam * mu0 * direct_trans
        flux_mean = flux_mean + fbeam * direct_trans / (4.0 * math.pi)

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def solar_isotropic_twostream_single_scatter_flux(
    tau: Any,
    *,
    ssa: Any,
    mu0: float,
    stream: float,
    fbeam: float = math.pi,
) -> dict[str, np.ndarray]:
    """Two-stream quadrature counterpart for isotropic solar single scatter."""
    tau_np = np.asarray(tau, dtype=float)
    omega = np.asarray(ssa, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if omega.ndim == 0:
        omega = np.full_like(tau_np, float(omega))
    if omega.shape != tau_np.shape:
        raise ValueError("ssa must be scalar or match tau")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")
    if not 0.0 < stream <= 1.0:
        raise ValueError("stream must satisfy 0 < stream <= 1")

    levels = np.concatenate(([0.0], np.cumsum(tau_np)))
    nlevels = levels.size
    intensity_up = np.zeros(nlevels, dtype=float)
    intensity_down = np.zeros(nlevels, dtype=float)

    def integrate_level(level: float, *, upward: bool) -> float:
        total = 0.0
        for layer, layer_omega in enumerate(omega):
            top = levels[layer]
            bottom = levels[layer + 1]
            if upward:
                lower = max(level, top)
                upper = bottom
            else:
                lower = top
                upper = min(level, bottom)
            if upper <= lower:
                continue

            def integrand(t: float) -> float:
                distance = (t - level) if upward else (level - t)
                return (
                    float(layer_omega) * math.exp(-t / mu0) * math.exp(-distance / stream) / stream
                )

            total += quad(
                integrand,
                lower,
                upper,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=100,
            )[0]
        return fbeam * total / (4.0 * math.pi)

    for idx, level in enumerate(levels):
        intensity_up[idx] = integrate_level(float(level), upward=True)
        intensity_down[idx] = integrate_level(float(level), upward=False)

    flux_up = 2.0 * math.pi * stream * intensity_up
    flux_down = 2.0 * math.pi * stream * intensity_down
    flux_mean = 0.5 * (intensity_up + intensity_down)
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def rayleigh_phase_moments(nmom: int, *, rayleigh_delta: float = 1.0) -> np.ndarray:
    """Returns pydisort phase moments for the unpolarized Rayleigh phase family.

    ``pydisort`` stores ``P0 = 1`` internally, so the input vector starts at
    ``P1``. Rayleigh has only ``P2 = 0.1`` in DISORT's convention.
    """
    if nmom <= 0:
        raise ValueError("nmom must be positive")
    if not 0.0 <= rayleigh_delta <= 1.0:
        raise ValueError("rayleigh_delta must satisfy 0 <= rayleigh_delta <= 1")
    moments = np.zeros(int(nmom), dtype=float)
    if moments.size > 1:
        moments[1] = 0.1 * float(rayleigh_delta)
    return moments


def _layer_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"{name} must be scalar or match tau")
    return arr


def _rayleigh_phase_coefficients(
    mu0: float, rayleigh_delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a0 = 1.0 - rayleigh_delta
    a0 += rayleigh_delta * 0.75 * (1.5 - 0.5 * mu0 * mu0)
    a2 = rayleigh_delta * 0.75 * (1.5 * mu0 * mu0 - 0.5)
    return a0, a2


def solar_rayleigh_single_scatter_flux(
    tau: Any,
    *,
    ssa: Any,
    mu0: float,
    rayleigh_delta: Any = 1.0,
    delta_m_truncation_factor: Any = 0.0,
    fbeam: float = math.pi,
    include_direct: bool = True,
) -> dict[str, np.ndarray]:
    """Analytic-kernel single-scatter solar flux for Rayleigh scattering."""
    tau_np = np.asarray(tau, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")
    omega = _layer_array("ssa", ssa, tau_np.shape)
    delta = _layer_array("rayleigh_delta", rayleigh_delta, tau_np.shape)
    trunc = _layer_array("delta_m_truncation_factor", delta_m_truncation_factor, tau_np.shape)
    if np.any((delta < 0.0) | (delta > 1.0)):
        raise ValueError("rayleigh_delta must satisfy 0 <= rayleigh_delta <= 1")

    source = omega / (1.0 - trunc * omega)
    phase0, phase2 = _rayleigh_phase_coefficients(mu0, delta)
    levels = np.concatenate(([0.0], np.cumsum(tau_np)))
    nlevels = levels.size
    flux_up = np.zeros(nlevels, dtype=float)
    flux_down = np.zeros(nlevels, dtype=float)
    flux_mean = np.zeros(nlevels, dtype=float)

    def integrate_level(level: float, *, upward: bool, orders: tuple[int, int]) -> float:
        total = 0.0
        for layer, layer_source in enumerate(source):
            top = levels[layer]
            bottom = levels[layer + 1]
            if upward:
                lower = max(level, top)
                upper = bottom
            else:
                lower = top
                upper = min(level, bottom)
            if upper <= lower:
                continue

            def integrand(t: float) -> float:
                distance = (t - level) if upward else (level - t)
                kernel = phase0[layer] * float(expn(orders[0], distance))
                kernel += phase2[layer] * float(expn(orders[1], distance))
                return float(layer_source) * math.exp(-t / mu0) * kernel

            total += quad(
                integrand,
                lower,
                upper,
                points=[level] if lower <= level <= upper else None,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=100,
            )[0]
        return total

    for idx, level in enumerate(levels):
        flux_up[idx] = 0.5 * fbeam * integrate_level(float(level), upward=True, orders=(2, 4))
        flux_down[idx] = 0.5 * fbeam * integrate_level(float(level), upward=False, orders=(2, 4))
        flux_mean[idx] = (
            fbeam
            / (8.0 * math.pi)
            * (
                integrate_level(float(level), upward=True, orders=(1, 3))
                + integrate_level(float(level), upward=False, orders=(1, 3))
            )
        )

    if include_direct:
        direct_trans = np.exp(-levels / mu0)
        flux_down = flux_down + fbeam * mu0 * direct_trans
        flux_mean = flux_mean + fbeam * direct_trans / (4.0 * math.pi)

    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def solar_rayleigh_twostream_single_scatter_flux(
    tau: Any,
    *,
    ssa: Any,
    mu0: float,
    stream: float,
    rayleigh_delta: Any = 1.0,
    delta_m_truncation_factor: Any = 0.0,
    fbeam: float = math.pi,
) -> dict[str, np.ndarray]:
    """Two-stream quadrature counterpart for Rayleigh solar single scatter."""
    tau_np = np.asarray(tau, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")
    if not 0.0 < stream <= 1.0:
        raise ValueError("stream must satisfy 0 < stream <= 1")
    omega = _layer_array("ssa", ssa, tau_np.shape)
    delta = _layer_array("rayleigh_delta", rayleigh_delta, tau_np.shape)
    trunc = _layer_array("delta_m_truncation_factor", delta_m_truncation_factor, tau_np.shape)
    if np.any((delta < 0.0) | (delta > 1.0)):
        raise ValueError("rayleigh_delta must satisfy 0 <= rayleigh_delta <= 1")

    source = omega / (1.0 - trunc * omega)
    phase0, phase2 = _rayleigh_phase_coefficients(mu0, delta)
    phase_stream = phase0 + phase2 * stream * stream
    levels = np.concatenate(([0.0], np.cumsum(tau_np)))
    nlevels = levels.size
    intensity_up = np.zeros(nlevels, dtype=float)
    intensity_down = np.zeros(nlevels, dtype=float)

    def integrate_level(level: float, *, upward: bool) -> float:
        total = 0.0
        for layer, layer_source in enumerate(source):
            top = levels[layer]
            bottom = levels[layer + 1]
            if upward:
                lower = max(level, top)
                upper = bottom
            else:
                lower = top
                upper = min(level, bottom)
            if upper <= lower:
                continue

            def integrand(t: float) -> float:
                distance = (t - level) if upward else (level - t)
                return (
                    float(layer_source)
                    * phase_stream[layer]
                    * math.exp(-t / mu0)
                    * math.exp(-distance / stream)
                    / stream
                )

            total += quad(
                integrand,
                lower,
                upper,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=100,
            )[0]
        return fbeam * total / (4.0 * math.pi)

    for idx, level in enumerate(levels):
        intensity_up[idx] = integrate_level(float(level), upward=True)
        intensity_down[idx] = integrate_level(float(level), upward=False)

    flux_up = 2.0 * math.pi * stream * intensity_up
    flux_down = 2.0 * math.pi * stream * intensity_down
    flux_mean = 0.5 * (intensity_up + intensity_down)
    return {
        "flux_up": flux_up,
        "flux_down": flux_down,
        "flux_net": flux_up - flux_down,
        "flux_mean": flux_mean,
    }


def run_pydisort_solar_flux(
    tau: Any,
    *,
    ssa: Any,
    phase_moments: Any,
    mu0: float,
    fbeam: float = math.pi,
    albedo: float = 0.0,
    nstr: int = 16,
    nmom: int = 16,
    dtype: str = "float64",
) -> dict[str, Any]:
    """Runs a small pydisort solar-flux reference case."""
    import pydisort
    import torch

    tau_np = np.asarray(tau, dtype=float)
    omega = np.asarray(ssa, dtype=float)
    moments = np.asarray(phase_moments, dtype=float)
    if tau_np.ndim != 1:
        raise ValueError("tau must be one-dimensional")
    if omega.ndim == 0:
        omega = np.full_like(tau_np, float(omega))
    if omega.shape != tau_np.shape:
        raise ValueError("ssa must be scalar or match tau")
    if moments.ndim == 1:
        moments = np.broadcast_to(moments, (tau_np.size, moments.size)).copy()
    if moments.shape[0] != tau_np.size:
        raise ValueError("phase_moments must have one row per layer")
    if not 0.0 < mu0 <= 1.0:
        raise ValueError("mu0 must satisfy 0 < mu0 <= 1")

    torch_dtype = {"float64": torch.float64, "float32": torch.float32}[dtype]
    nlyr = int(tau_np.shape[0])
    nprop = 2 + int(nmom)
    moments = moments[:, :nmom]
    if moments.shape[1] < nmom:
        moments = np.pad(moments, ((0, 0), (0, nmom - moments.shape[1])))
    moments = np.ascontiguousarray(moments)

    options = pydisort.DisortOptions().flags("onlyfl,lamber").nwave(1).ncol(1)
    options.ds().nlyr = nlyr
    options.ds().nstr = int(nstr)
    options.ds().nmom = int(nmom)
    options.ds().nphase = int(nmom)

    prop = torch.zeros((1, 1, nlyr, nprop), dtype=torch_dtype)
    prop[0, 0, :, 0] = torch.as_tensor(tau_np, dtype=torch_dtype)
    prop[0, 0, :, 1] = torch.as_tensor(omega, dtype=torch_dtype)
    prop[0, 0, :, 2:] = torch.as_tensor(moments, dtype=torch_dtype)

    solver = pydisort.Disort(options)
    solver.forward(
        prop,
        umu0=torch.tensor([mu0], dtype=torch_dtype),
        fbeam=torch.tensor([[fbeam]], dtype=torch_dtype),
        albedo=torch.tensor([[albedo]], dtype=torch_dtype),
    )
    return pydisort_flux_to_py2sess(solver.gather_flx())
