"""Torch Taylor-series stabilization helpers."""

from __future__ import annotations

from .backend import _load_torch

torch = _load_torch()


def taylor_series_1_torch(order: int, eps, delta, udel, sm):
    """Evaluates the first Fortran 2S Taylor multiplier on torch tensors."""
    mterms = order + 1
    dm1 = delta
    mult = dm1
    power = torch.ones_like(delta)
    for m in range(2, mterms + 1):
        dm1 = delta * dm1 / float(m)
        power = power * eps
        mult = mult + power * dm1
    return mult * udel * sm


def taylor_series_2_torch(order: int, small: float, eps, y, delta, fac1, fac2, sm):
    """Evaluates the second Fortran 2S Taylor multiplier on torch tensors."""
    mterms = order + 1
    if bool(torch.abs(y) < small):
        mterms += 1

    d = [torch.ones_like(delta)]
    dm1 = d[0]
    for m in range(1, mterms + 1):
        dm1 = delta * dm1 / float(m)
        d.append(dm1)

    if bool(torch.abs(y) < small):
        power = torch.ones_like(delta)
        power2 = torch.ones_like(delta)
        mult = d[2]
        for m in range(3, mterms + 1):
            power = power * (eps - y)
            power2 = power - y * power2
            mult = mult + d[m] * power2
        return mult * fac1 * sm

    y1 = 1.0 / y
    ac = [torch.ones_like(delta)]
    acm1 = ac[0]
    for _m in range(1, mterms + 1):
        acm1 = y1 * acm1
        ac.append(acm1)

    cc = [torch.ones_like(delta)]
    for m in range(1, mterms + 1):
        total = torch.zeros_like(delta)
        for j in range(m + 1):
            total = total + ac[j] * d[m - j]
        cc.append(total)

    term_1 = []
    for m in range(mterms + 1):
        term_1.append(fac1 * ac[m] - fac2 * cc[m])

    power = torch.ones_like(delta)
    mult = term_1[1]
    for m in range(2, mterms + 1):
        power = eps * power
        mult = mult + power * term_1[m]
    return mult * sm * y1
