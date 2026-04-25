"""Backend conversion helpers for optional torch support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

torch = None
_TORCH_IMPORT_ATTEMPTED = False


def _load_torch():
    """Imports torch on demand and returns the module when available."""
    global torch, _TORCH_IMPORT_ATTEMPTED
    if not _TORCH_IMPORT_ATTEMPTED:
        try:
            import torch as torch_module
        except ImportError:  # pragma: no cover
            torch_module = None
        torch = torch_module
        _TORCH_IMPORT_ATTEMPTED = True
    return torch


def _looks_like_torch_tensor(value: Any) -> bool:
    """Returns true when a value appears to be a torch tensor without importing torch."""
    value_type = type(value)
    return value_type.__module__.startswith("torch") and value_type.__name__ == "Tensor"


@dataclass(frozen=True)
class TorchContext:
    """Tensor dtype/device context used when converting NumPy inputs to torch."""

    dtype: Any
    device: Any


def has_torch() -> bool:
    """Returns whether PyTorch is importable in the active environment."""
    return _load_torch() is not None


def detect_torch_context(*values: Any) -> TorchContext | None:
    """Detects dtype/device from the first torch tensor in ``values``.

    Parameters
    ----------
    values
        Candidate inputs that may include torch tensors.

    Returns
    -------
    TorchContext or None
        The detected torch context, or ``None`` if no tensor input is present
        or PyTorch is unavailable.
    """
    torch_module = (
        _load_torch() if any(_looks_like_torch_tensor(value) for value in values) else torch
    )
    if torch_module is None:
        return None
    for value in values:
        if isinstance(value, torch_module.Tensor):
            return TorchContext(dtype=value.dtype, device=value.device)
    return None


def to_numpy(value: Any) -> Any:
    """Converts a torch tensor to a detached CPU NumPy array."""
    torch_module = _load_torch() if _looks_like_torch_tensor(value) else torch
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return value.detach().cpu().numpy()
    return value


def _copy_if_readonly_numpy(value: Any) -> Any:
    """Copies read-only NumPy arrays before exposing them to torch."""
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        return np.array(value, copy=True)
    return value


def arrays_to_torch(solved: dict[str, np.ndarray], context: TorchContext | None) -> dict[str, Any]:
    """Converts a dictionary of NumPy arrays to torch tensors."""
    torch_module = _load_torch()
    if torch_module is None:
        raise RuntimeError("PyTorch backend requested but torch is not installed")
    if context is None:
        context = TorchContext(dtype=torch_module.float64, device=torch_module.device("cpu"))
    return {
        key: torch_module.as_tensor(
            _copy_if_readonly_numpy(value), dtype=context.dtype, device=context.device
        )
        for key, value in solved.items()
    }


def value_to_torch(value: Any, context: TorchContext | None) -> Any:
    """Converts a single value to a torch tensor using ``context``."""
    torch_module = _load_torch()
    if torch_module is None:
        raise RuntimeError("PyTorch backend requested but torch is not installed")
    if value is None:
        return None
    if isinstance(value, torch_module.Tensor):
        if context is not None and (value.dtype != context.dtype or value.device != context.device):
            return value.to(dtype=context.dtype, device=context.device)
        return value
    if context is None:
        context = TorchContext(dtype=torch_module.float64, device=torch_module.device("cpu"))
    return torch_module.as_tensor(
        _copy_if_readonly_numpy(value), dtype=context.dtype, device=context.device
    )
