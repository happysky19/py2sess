"""Batched torch helpers for solar observation-geometry FO calculations."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import _load_torch
from .fo_solar_obs_batch_numpy import FoSolarObsBatchPrecompute

torch = _load_torch()

MAX_TAU_PATH = 88.0


@dataclass(frozen=True)
class FoSolarObsBatchTorchResult:
    """Batched torch solar FO endpoint and optional profile components."""

    total: Any
    single_scatter: Any
    direct_beam: Any
    total_profile: Any | None = None
    single_scatter_profile: Any | None = None
    direct_beam_profile: Any | None = None


def _infer_context(values: tuple[Any, ...], *, dtype, device):
    """Returns a torch dtype/device from explicit options or tensor inputs."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    detected = None
    for value in values:
        if isinstance(value, torch.Tensor):
            detected = value
            break
    if device is None:
        device = detected.device if detected is not None else torch.device("cpu")
    else:
        device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "torch device 'mps' is not available in this process. On macOS this "
            "usually means the current sandbox cannot access Metal/MPS; rerun outside "
            "the sandbox/escalated environment, or use device='cpu'."
        )
    if dtype is None:
        if detected is not None:
            dtype = detected.dtype
        elif device.type == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float64
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("torch device 'mps' requires float32 or lower precision")
    return dtype, device


def _as_tensor(value, *, dtype, device):
    """Converts ``value`` to a torch tensor on the requested context."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            return torch.as_tensor(value, dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _broadcast_rows(name: str, value, *, batch_size: int, dtype, device):
    """Returns a 1D tensor broadcast to the spectral batch size."""
    tensor = _as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    try:
        tensor = torch.broadcast_to(tensor, (batch_size,))
    except RuntimeError as exc:
        raise ValueError(f"{name} must be scalar or have shape ({batch_size},)") from exc
    return tensor


def _exp_cutoff_torch(values):
    """Applies the Fortran 88-optical-depth cutoff."""
    result = torch.exp(-values)
    return torch.where(values >= MAX_TAU_PATH, torch.zeros_like(result), result)


def solve_fo_solar_obs_eps_batch_torch(
    *,
    tau,
    omega,
    scaling,
    albedo,
    flux_factor,
    exact_scatter,
    precomputed: FoSolarObsBatchPrecompute,
    dtype=None,
    device=None,
    return_profile: bool = False,
    return_components: bool = False,
):
    """Evaluates FO EPS TOA radiance for a spectral batch with torch.

    Parameters
    ----------
    tau, omega, scaling, exact_scatter
        Layer arrays with shape ``(n_spectral, n_layers)``. ``exact_scatter``
        is required, matching the optimized NumPy UV full-spectrum path.
    albedo, flux_factor
        Scalar or ``(n_spectral,)`` surface/source values.
    precomputed
        Geometry terms from :func:`fo_solar_obs_batch_precompute`.

    Returns
    -------
    torch.Tensor
        Upwelling TOA FO radiance with shape ``(n_spectral,)``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed")
    dtype, device = _infer_context(
        (tau, omega, scaling, albedo, flux_factor, exact_scatter), dtype=dtype, device=device
    )
    tau_t = _as_tensor(tau, dtype=dtype, device=device)
    omega_t = _as_tensor(omega, dtype=dtype, device=device)
    scaling_t = _as_tensor(scaling, dtype=dtype, device=device)
    phase_terms = _as_tensor(exact_scatter, dtype=dtype, device=device)
    if tau_t.ndim != 2:
        raise ValueError("tau must have shape (n_spectral, n_layers)")
    if omega_t.shape != tau_t.shape:
        raise ValueError("omega must have the same shape as tau")
    if scaling_t.shape != tau_t.shape:
        raise ValueError("scaling must have the same shape as tau")
    if phase_terms.shape != tau_t.shape:
        raise ValueError("exact_scatter must have the same shape as tau")

    batch_size, nlayers = tau_t.shape
    if precomputed.inv_layer_thickness.shape[0] != nlayers:
        raise ValueError("precomputed geometry does not match tau layer count")
    albedo_t = _broadcast_rows("albedo", albedo, batch_size=batch_size, dtype=dtype, device=device)
    flux_t = _broadcast_rows(
        "flux_factor", flux_factor, batch_size=batch_size, dtype=dtype, device=device
    )

    inv_layer_thickness = _as_tensor(precomputed.inv_layer_thickness, dtype=dtype, device=device)
    tau_scaled = tau_t * (1.0 - omega_t * scaling_t)
    extinction = tau_scaled * inv_layer_thickness.unsqueeze(0)
    sunpathsnl = _as_tensor(precomputed.sunpathsnl, dtype=dtype, device=device)
    total_tau = extinction[:, : precomputed.ntrav_nl] @ sunpathsnl
    attenuation_nl = _exp_cutoff_torch(total_tau)
    cumsource_up = torch.zeros(batch_size, dtype=dtype, device=device)
    cumsource_db = 4.0 * precomputed.mu0 * albedo_t * attenuation_nl
    profile_up = None
    profile_db = None
    if return_profile:
        profile_up = torch.zeros((batch_size, nlayers + 1), dtype=dtype, device=device)
        profile_db = torch.empty((batch_size, nlayers + 1), dtype=dtype, device=device)
        profile_db[:, nlayers] = cumsource_db

    if precomputed.do_nadir:
        v = 0
        xfine = _as_tensor(precomputed.xfine[:, :, v], dtype=dtype, device=device)
        wfine = _as_tensor(precomputed.wfine, dtype=dtype, device=device)
        sunpathsfine = _as_tensor(precomputed.sunpathsfine[:, :, :, v], dtype=dtype, device=device)
        for n in range(nlayers, 0, -1):
            layer = n - 1
            kn = extinction[:, layer]
            layer_sum = torch.zeros(batch_size, dtype=dtype, device=device)
            nfine_layer = int(precomputed.nfinedivs[layer])
            for j in range(nfine_layer):
                ntrav = int(precomputed.ntraversefine[j, layer, v])
                paths = sunpathsfine[:ntrav, j, layer]
                fine_tau = extinction[:, :ntrav] @ paths
                attenuation = _exp_cutoff_torch(fine_tau)
                layer_sum = layer_sum + (
                    phase_terms[:, layer]
                    * attenuation
                    * torch.exp(-xfine[j, layer] * kn)
                    * wfine[j, layer]
                )
            lostrans = _exp_cutoff_torch(tau_scaled[:, layer])
            source = layer_sum * kn
            cumsource_db = lostrans * cumsource_db
            cumsource_up = lostrans * cumsource_up + source
            if return_profile:
                profile_up[:, layer] = cumsource_up
                profile_db[:, layer] = cumsource_db
    else:
        if precomputed.fine_path_matrix is None or precomputed.fine_column_index is None:
            raise ValueError("missing non-nadir FO batch geometry terms")
        cota = _as_tensor(precomputed.cota, dtype=dtype, device=device)
        cotfine = _as_tensor(precomputed.cotfine, dtype=dtype, device=device)
        csqfine = _as_tensor(precomputed.csqfine, dtype=dtype, device=device)
        wfine = _as_tensor(precomputed.wfine, dtype=dtype, device=device)
        fine_path_matrix = _as_tensor(precomputed.fine_path_matrix, dtype=dtype, device=device)
        fine_attenuation = _exp_cutoff_torch(extinction @ fine_path_matrix)
        cot_1 = cota[nlayers]
        for n in range(nlayers, 0, -1):
            layer = n - 1
            cot_2 = cota[layer]
            ke = precomputed.rayconv * extinction[:, layer]
            lostrans = torch.exp(-ke * (cot_2 - cot_1))
            layer_sum = torch.zeros(batch_size, dtype=dtype, device=device)
            nfine_layer = int(precomputed.nfinedivs[layer])
            for j in range(nfine_layer):
                column = int(precomputed.fine_column_index[j, layer])
                tran = torch.exp(-ke * (cot_2 - cotfine[j, layer]))
                layer_sum = layer_sum + (
                    phase_terms[:, layer]
                    * fine_attenuation[:, column]
                    * csqfine[j, layer]
                    * tran
                    * wfine[j, layer]
                )
            source = layer_sum * ke
            cumsource_db = lostrans * cumsource_db
            cumsource_up = lostrans * cumsource_up + source
            if return_profile:
                profile_up[:, layer] = cumsource_up
                profile_db[:, layer] = cumsource_db
            cot_1 = cot_2

    scale = 0.25 * flux_t / math.pi
    single_scatter = scale * cumsource_up
    direct_beam = scale * cumsource_db
    if return_profile:
        single_scatter_profile = scale.unsqueeze(1) * profile_up
        direct_beam_profile = scale.unsqueeze(1) * profile_db
        total_profile = single_scatter_profile + direct_beam_profile
        if return_components:
            return FoSolarObsBatchTorchResult(
                total=single_scatter + direct_beam,
                single_scatter=single_scatter,
                direct_beam=direct_beam,
                total_profile=total_profile,
                single_scatter_profile=single_scatter_profile,
                direct_beam_profile=direct_beam_profile,
            )
        return total_profile
    if return_components:
        return FoSolarObsBatchTorchResult(
            total=single_scatter + direct_beam,
            single_scatter=single_scatter,
            direct_beam=direct_beam,
        )
    return single_scatter + direct_beam
