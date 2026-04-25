"""Taylor-series multiplier helpers matching the Fortran 2S implementation."""

from __future__ import annotations

import numpy as np


def taylor_series_1(order: int, eps: float, delta: float, udel: float, sm: float) -> float:
    """Evaluates the first Fortran 2S Taylor-series multiplier."""
    mterms = order + 1
    dm1 = delta
    mult = dm1
    power = 1.0
    for m in range(2, mterms + 1):
        dm1 = delta * dm1 / float(m)
        power = power * eps
        mult = mult + power * dm1
    return mult * udel * sm


def vectorized_taylor_series_1(
    order: int,
    eps: np.ndarray,
    delta: np.ndarray,
    udel: np.ndarray,
    sm: float,
) -> np.ndarray:
    """Evaluates the first Fortran 2S Taylor multiplier for arrays.

    Parameters
    ----------
    order
        Taylor expansion order.
    eps, delta, udel
        Array inputs matching the scalar Fortran multiplier convention.
    sm
        Scalar multiplier applied to the final result.

    Returns
    -------
    ndarray
        Multiplier values with the broadcasted input shape.
    """
    mterms = order + 1
    dm1 = delta
    mult = dm1.copy()
    power = np.ones_like(delta)
    for m in range(2, mterms + 1):
        dm1 = delta * dm1 / float(m)
        power = power * eps
        mult = mult + power * dm1
    return mult * udel * sm


def taylor_series_2(
    order: int,
    small: float,
    eps: float,
    y: float,
    delta: float,
    fac1: float,
    fac2: float,
    sm: float,
) -> float:
    """Evaluates the second Fortran 2S Taylor-series multiplier."""
    max_terms = 10
    mterms = order + 1
    if abs(y) < small:
        mterms += 1

    d = np.zeros(max_terms + 1, dtype=float)
    dm1 = 1.0
    d[0] = dm1
    for m in range(1, mterms + 1):
        dm1 = delta * dm1 / float(m)
        d[m] = dm1

    if abs(y) < small:
        power = 1.0
        power2 = 1.0
        mult = d[2]
        for m in range(3, mterms + 1):
            power = power * (eps - y)
            power2 = power - y * power2
            mult = mult + d[m] * power2
        return mult * fac1 * sm

    y1 = 1.0 / y
    ac = np.zeros(max_terms + 1, dtype=float)
    acm1 = 1.0
    ac[0] = acm1
    for m in range(1, mterms + 1):
        acm1 = y1 * acm1
        ac[m] = acm1

    cc = np.zeros(max_terms + 1, dtype=float)
    cc[0] = 1.0
    for m in range(1, mterms + 1):
        total = 0.0
        for j in range(m + 1):
            total += ac[j] * d[m - j]
        cc[m] = total

    term_1 = np.zeros(max_terms + 1, dtype=float)
    for m in range(mterms + 1):
        term_1[m] = fac1 * ac[m] - fac2 * cc[m]

    power = 1.0
    mult = term_1[1]
    for m in range(2, mterms + 1):
        power = eps * power
        mult = mult + power * term_1[m]
    return mult * sm * y1


def vectorized_taylor_series_2(
    order: int,
    small: float,
    eps: np.ndarray,
    y: np.ndarray,
    delta: np.ndarray,
    fac1: np.ndarray,
    fac2: np.ndarray,
    sm: float,
) -> np.ndarray:
    """Evaluates the second Fortran 2S Taylor multiplier for arrays.

    The implementation mirrors :func:`taylor_series_2` but evaluates near and
    regular branches over array masks.
    """
    result = np.empty_like(eps)
    near = np.abs(y) < small
    far = ~near

    if np.any(near):
        mterms = order + 2
        d = [np.ones_like(eps[near])]
        for m in range(1, mterms + 1):
            d.append(delta[near] * d[-1] / float(m))
        power = np.ones_like(eps[near])
        power2 = np.ones_like(eps[near])
        mult = d[2].copy()
        for m in range(3, mterms + 1):
            power = power * (eps[near] - y[near])
            power2 = power - y[near] * power2
            mult = mult + d[m] * power2
        result[near] = mult * fac1[near] * sm

    if np.any(far):
        mterms = order + 1
        y1 = 1.0 / y[far]
        d = [np.ones_like(eps[far])]
        for m in range(1, mterms + 1):
            d.append(delta[far] * d[-1] / float(m))
        ac = [np.ones_like(eps[far])]
        for _ in range(1, mterms + 1):
            ac.append(y1 * ac[-1])
        cc = [np.ones_like(eps[far])]
        for m in range(1, mterms + 1):
            total = np.zeros_like(eps[far])
            for j in range(m + 1):
                total = total + ac[j] * d[m - j]
            cc.append(total)
        term_1 = [fac1[far] * ac[m] - fac2[far] * cc[m] for m in range(mterms + 1)]
        power = np.ones_like(eps[far])
        mult = term_1[1].copy()
        for m in range(2, mterms + 1):
            power = eps[far] * power
            mult = mult + power * term_1[m]
        result[far] = mult * sm * y1

    return result
