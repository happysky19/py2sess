"""Shared batched recurrence helpers for NumPy CPU solvers."""

from __future__ import annotations

import numpy as np


_NUMBA_MIN_BATCH = 256
_UPWELLING_ACCUM_KERNEL = None
_UPWELLING_ACCUM_IMPORT_FAILED = False


def _get_upwelling_accumulation_kernel():
    """Lazily builds the optional Numba upwelling recurrence kernel."""
    global _UPWELLING_ACCUM_KERNEL, _UPWELLING_ACCUM_IMPORT_FAILED
    if _UPWELLING_ACCUM_KERNEL is not None:
        return _UPWELLING_ACCUM_KERNEL
    if _UPWELLING_ACCUM_IMPORT_FAILED:
        return None
    try:  # pragma: no cover - optional acceleration dependency
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _kernel(layer_source, layer_trans, surface_source):
            batch, nlay = layer_source.shape
            out = np.empty(batch, np.float64)
            for b in prange(batch):
                accum = surface_source[b]
                for n in range(nlay - 1, -1, -1):
                    accum = layer_source[b, n] + layer_trans[b, n] * accum
                out[b] = accum
            return out

        _UPWELLING_ACCUM_KERNEL = _kernel
        return _UPWELLING_ACCUM_KERNEL
    except Exception:  # pragma: no cover - optional acceleration dependency
        _UPWELLING_ACCUM_IMPORT_FAILED = True
        return None


def accumulate_upwelling_sources_numpy(
    *,
    layer_source: np.ndarray,
    layer_trans: np.ndarray,
    surface_source: np.ndarray,
) -> np.ndarray:
    """Evaluates the backward layer recurrence for batched NumPy solves."""
    kernel = _get_upwelling_accumulation_kernel()
    if kernel is not None and layer_source.shape[0] >= _NUMBA_MIN_BATCH:
        return kernel(layer_source, layer_trans, surface_source)
    prefix = np.cumprod(layer_trans, axis=1)
    prefix_exclusive = np.empty_like(prefix)
    prefix_exclusive[:, 0] = 1.0
    prefix_exclusive[:, 1:] = prefix[:, :-1]
    return np.sum(prefix_exclusive * layer_source, axis=1) + prefix[:, -1] * surface_source
