"""Torch-native batched boundary-value problem solvers."""

from __future__ import annotations

from .backend import _load_torch

torch = _load_torch()


def _canonical_torch_bvp_engine(engine: str) -> str:
    """Normalizes the public torch BVP engine names."""
    normalized = engine.lower()
    if normalized not in {"auto", "dense", "pentadiagonal", "block"}:
        raise ValueError("Torch BVP engine must be 'auto', 'dense', 'pentadiagonal', or 'block'")
    return normalized


def default_auto_bvp_context_torch(*, device, bvp_device=None, bvp_dtype=None):
    """Returns the default torch BVP solve context for ``bvp_engine='auto'``.

    The supported parity-oriented path stays on CPU float64. On Apple MPS, the
    current development path keeps the outer tensors on MPS but runs the BVP on
    CPU float64. Callers can override this by passing ``bvp_device`` and/or
    ``bvp_dtype`` explicitly.
    """
    resolved_bvp_device = None if bvp_device is None else torch.device(bvp_device)
    resolved_bvp_dtype = bvp_dtype
    if device.type == "mps":
        if resolved_bvp_device is None:
            resolved_bvp_device = torch.device("cpu")
        if resolved_bvp_device.type == "cpu" and resolved_bvp_dtype is None:
            resolved_bvp_dtype = torch.float64
    return resolved_bvp_device, resolved_bvp_dtype


def _move_tensor_for_solve(value, *, solve_device, solve_dtype):
    """Moves a tensor to an optional BVP solve context."""
    if not torch.is_tensor(value):
        return value
    target_device = value.device if solve_device is None else torch.device(solve_device)
    target_dtype = value.dtype if solve_dtype is None else solve_dtype
    if value.device == target_device and value.dtype == target_dtype:
        return value
    if value.device.type == "mps" and target_dtype == torch.float64:
        return value.detach().cpu().to(dtype=target_dtype)
    return value.to(device=target_device, dtype=target_dtype)


def _move_value_for_solve(value, *, solve_device, solve_dtype):
    """Moves tensor or tuple-of-tensor BVP inputs to the solve context."""
    if isinstance(value, tuple):
        return tuple(
            _move_tensor_for_solve(item, solve_device=solve_device, solve_dtype=solve_dtype)
            for item in value
        )
    return _move_tensor_for_solve(value, solve_device=solve_device, solve_dtype=solve_dtype)


def _slice_bvp_batch_value(value, bad):
    """Returns the subset of batch-shaped BVP inputs selected by ``bad``."""
    if isinstance(value, tuple):
        return tuple(_slice_bvp_batch_value(item, bad) for item in value)
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == bad.shape[0]:
        return value[bad]
    return value


def repair_nonfinite_bvp_rows_torch(
    *,
    lcon,
    mcon,
    bad_row_solver,
    solver_kwargs,
    solve_device=None,
    solve_dtype=None,
):
    """Repairs non-finite BVP rows with the dense fallback solver."""
    if bool(torch.isfinite(lcon).all()) and bool(torch.isfinite(mcon).all()):
        return lcon, mcon

    finite = torch.isfinite(lcon).all(dim=1) & torch.isfinite(mcon).all(dim=1)
    bad = ~finite
    if not bool(bad.any()):
        return lcon, mcon

    dense_kwargs = {key: _slice_bvp_batch_value(value, bad) for key, value in solver_kwargs.items()}
    lcon_bad, mcon_bad = bad_row_solver(
        **dense_kwargs,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
    )
    lcon = lcon.clone()
    mcon = mcon.clone()
    lcon[bad] = lcon_bad
    mcon[bad] = mcon_bad
    return lcon, mcon


def _matmul_2x2_batch_torch(lhs, rhs):
    """Returns the batched product of two ``(..., 2, 2)`` torch tensors."""
    out = torch.empty_like(lhs)
    out[..., 0, 0] = lhs[..., 0, 0] * rhs[..., 0, 0] + lhs[..., 0, 1] * rhs[..., 1, 0]
    out[..., 0, 1] = lhs[..., 0, 0] * rhs[..., 0, 1] + lhs[..., 0, 1] * rhs[..., 1, 1]
    out[..., 1, 0] = lhs[..., 1, 0] * rhs[..., 0, 0] + lhs[..., 1, 1] * rhs[..., 1, 0]
    out[..., 1, 1] = lhs[..., 1, 0] * rhs[..., 0, 1] + lhs[..., 1, 1] * rhs[..., 1, 1]
    return out


def _matvec_2x2_batch_torch(mat, vec):
    """Returns the batched product of ``(..., 2, 2)`` and ``(..., 2)`` tensors."""
    out = torch.empty_like(vec)
    out[..., 0] = mat[..., 0, 0] * vec[..., 0] + mat[..., 0, 1] * vec[..., 1]
    out[..., 1] = mat[..., 1, 0] * vec[..., 0] + mat[..., 1, 1] * vec[..., 1]
    return out


def _invert_2x2_batch_torch(mat):
    """Returns the batched inverse of ``(..., 2, 2)`` torch tensors."""
    det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]
    inv = torch.empty_like(mat)
    scale = torch.clamp(torch.amax(torch.abs(mat), dim=(-2, -1)), min=1.0)
    threshold = torch.finfo(mat.dtype).eps * 64.0 * scale * scale
    safe_det = torch.where(torch.abs(det) <= threshold, torch.full_like(det, torch.nan), det)
    inv[..., 0, 0] = mat[..., 1, 1] / safe_det
    inv[..., 0, 1] = -mat[..., 0, 1] / safe_det
    inv[..., 1, 0] = -mat[..., 1, 0] / safe_det
    inv[..., 1, 1] = mat[..., 0, 0] / safe_det
    return inv


def _build_block_tridiagonal_system_torch(
    *,
    albedo,
    bottom_source,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
):
    """Builds the batched 2x2 block-tridiagonal torch BVP coefficients."""
    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    batch, nlay = xpos1.shape
    dtype = xpos1.dtype
    device = xpos1.device
    lower = torch.zeros((batch, nlay, 2, 2), dtype=dtype, device=device)
    diag = torch.zeros((batch, nlay, 2, 2), dtype=dtype, device=device)
    upper = torch.zeros((batch, nlay, 2, 2), dtype=dtype, device=device)
    rhs = torch.zeros((batch, nlay, 2), dtype=dtype, device=device)

    factor = surface_factor * albedo
    xpnet = xpos2[:, -1] - factor * xpos1[:, -1] * stream_value
    xmnet = xpos1[:, -1] - factor * xpos2[:, -1] * stream_value
    bottom_rhs = -wlower1[:, -1] + wlower0[:, -1] * stream_value * factor + bottom_source

    diag[:, 0, 0, 0] = xpos1[:, 0]
    diag[:, 0, 0, 1] = xpos2[:, 0] * eigentrans[:, 0]
    rhs[:, 0, 0] = -wupper0[:, 0]

    if nlay == 1:
        diag[:, 0, 1, 0] = xpnet * eigentrans[:, 0]
        diag[:, 0, 1, 1] = xmnet
        rhs[:, 0, 1] = bottom_rhs
        return lower, diag, upper, rhs

    diag[:, 0, 1, 0] = xpos1[:, 0] * eigentrans[:, 0]
    diag[:, 0, 1, 1] = xpos2[:, 0]
    upper[:, 0, 1, 0] = -xpos1[:, 1]
    upper[:, 0, 1, 1] = -xpos2[:, 1] * eigentrans[:, 1]
    rhs[:, 0, 1] = wupper0[:, 1] - wlower0[:, 0]

    for n in range(1, nlay - 1):
        prev = n - 1
        lower[:, n, 0, 0] = xpos2[:, prev] * eigentrans[:, prev]
        lower[:, n, 0, 1] = xpos1[:, prev]
        diag[:, n, 0, 0] = -xpos2[:, n]
        diag[:, n, 0, 1] = -xpos1[:, n] * eigentrans[:, n]
        rhs[:, n, 0] = wupper1[:, n] - wlower1[:, prev]

        diag[:, n, 1, 0] = xpos1[:, n] * eigentrans[:, n]
        diag[:, n, 1, 1] = xpos2[:, n]
        upper[:, n, 1, 0] = -xpos1[:, n + 1]
        upper[:, n, 1, 1] = -xpos2[:, n + 1] * eigentrans[:, n + 1]
        rhs[:, n, 1] = wupper0[:, n + 1] - wlower0[:, n]

    lower[:, -1, 0, 0] = xpos2[:, -2] * eigentrans[:, -2]
    lower[:, -1, 0, 1] = xpos1[:, -2]
    diag[:, -1, 0, 0] = -xpos2[:, -1]
    diag[:, -1, 0, 1] = -xpos1[:, -1] * eigentrans[:, -1]
    rhs[:, -1, 0] = wupper1[:, -1] - wlower1[:, -2]
    diag[:, -1, 1, 0] = xpnet * eigentrans[:, -1]
    diag[:, -1, 1, 1] = xmnet
    rhs[:, -1, 1] = bottom_rhs
    return lower, diag, upper, rhs


def _solve_block_bvp_batch_torch(
    *,
    albedo,
    bottom_source,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
):
    """Solves regular two-stream BVP systems with a 2x2 block Thomas method."""
    lower, diag, upper, rhs = _build_block_tridiagonal_system_torch(
        albedo=albedo,
        bottom_source=bottom_source,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )
    batch, nlay = xpos1.shape
    cprime = torch.zeros_like(upper)
    dprime = torch.zeros_like(rhs)

    inv = _invert_2x2_batch_torch(diag[:, 0])
    if nlay > 1:
        cprime[:, 0] = _matmul_2x2_batch_torch(inv, upper[:, 0])
    dprime[:, 0] = _matvec_2x2_batch_torch(inv, rhs[:, 0])

    for n in range(1, nlay):
        schur = diag[:, n] - _matmul_2x2_batch_torch(lower[:, n], cprime[:, n - 1])
        inv = _invert_2x2_batch_torch(schur)
        if n < nlay - 1:
            cprime[:, n] = _matmul_2x2_batch_torch(inv, upper[:, n])
        dprime[:, n] = _matvec_2x2_batch_torch(
            inv,
            rhs[:, n] - _matvec_2x2_batch_torch(lower[:, n], dprime[:, n - 1]),
        )

    solution = torch.empty((batch, nlay, 2), dtype=dprime.dtype, device=dprime.device)
    solution[:, -1] = dprime[:, -1]
    for n in range(nlay - 2, -1, -1):
        solution[:, n] = dprime[:, n] - _matvec_2x2_batch_torch(cprime[:, n], solution[:, n + 1])
    return solution[:, :, 0], solution[:, :, 1]


def _solve_pentadiagonal_bvp_batch_torch(
    *,
    albedo,
    bottom_source,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
):
    """Solves regular two-stream BVP systems with pentadiagonal elimination."""
    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    x1 = xpos1.mT.contiguous()
    x2 = xpos2.mT.contiguous()
    et = eigentrans.mT.contiguous()
    wupper0 = wupper0.mT.contiguous()
    wupper1 = wupper1.mT.contiguous()
    wlower0 = wlower0.mT.contiguous()
    wlower1 = wlower1.mT.contiguous()
    nlay, batch = x1.shape
    ntotal = 2 * nlay
    dtype = x1.dtype
    device = x1.device
    col = torch.zeros((ntotal, batch), dtype=dtype, device=device)

    factor = surface_factor * albedo
    xpnet = x2[-1] - factor * x1[-1] * stream_value
    xmnet = x1[-1] - factor * x2[-1] * stream_value

    col[0] = -wupper0[0]
    for n in range(1, nlay):
        prev = n - 1
        row_m = 2 * n - 1
        row_p = row_m + 1
        col[row_m] = wupper0[n] - wlower0[prev]
        col[row_p] = wupper1[n] - wlower1[prev]
    col[-1] = -wlower1[-1] + wlower0[-1] * stream_value * factor + bottom_source

    elm1 = torch.zeros((ntotal - 1, batch), dtype=dtype, device=device)
    elm2 = torch.zeros((ntotal - 2, batch), dtype=dtype, device=device)

    elm31 = 1.0 / x1[0]
    elm1_i2 = -(x2[0] * et[0]) * elm31
    elm1[0] = elm1_i2
    elm2_i2 = torch.zeros(batch, dtype=dtype, device=device)
    elm2[0] = elm2_i2

    col_i2 = col[0] * elm31
    col[0] = col_i2

    mat22 = x1[0] * et[0]
    bet = x2[0] + mat22 * elm1_i2
    bet = -1.0 / bet
    elm1_i1 = -x1[1] * bet
    elm1[1] = elm1_i1
    elm2_i1 = (-x2[1] * et[1]) * bet
    elm2[1] = elm2_i1
    col_i1 = (mat22 * col_i2 - col[1]) * bet
    col[1] = col_i1

    for i in range(2, ntotal - 2):
        if i % 2 == 0:
            n = i // 2
            prev = n - 1
            mat1_i = x2[prev] * et[prev]
            mat2_i = x1[prev]
            mat3_i = -x2[n]
            mat4_i = -x1[n] * et[n]
            mat5_i = 0.0
        else:
            n = (i + 1) // 2
            prev = n - 1
            mat1_i = 0.0
            mat2_i = x1[prev] * et[prev]
            mat3_i = x2[prev]
            mat4_i = -x1[n]
            mat5_i = -x2[n] * et[n]
        bet = mat2_i + mat1_i * elm1_i2
        den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
        den = -1.0 / den
        elm1_i2 = elm1_i1
        elm1_i1 = (mat4_i + bet * elm2_i1) * den
        elm1[i] = elm1_i1
        elm2_i2 = elm2_i1
        elm2_i1 = mat5_i * den
        elm2[i] = elm2_i1

        col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
        col_i2 = col_i1
        col_i1 = col_i
        col[i] = col_i

    i = ntotal - 2
    n = i // 2
    prev = n - 1
    mat1_i = x2[prev] * et[prev]
    mat2_i = x1[prev]
    mat3_i = -x2[n]
    mat4_i = -x1[n] * et[n]
    bet = mat2_i + mat1_i * elm1_i2
    den = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
    den = -1.0 / den
    elm1_i2 = elm1_i1
    elm1_i1 = (mat4_i + bet * elm2_i1) * den
    elm1[i] = elm1_i1
    elm2_i2 = elm2_i1

    col_i = (mat1_i * col_i2 + bet * col_i1 - col[i]) * den
    col_i2 = col_i1
    col_i1 = col_i
    col[i] = col_i

    i = ntotal - 1
    bet = xpnet * et[-1]
    den = xmnet + bet * elm1_i1
    den = -1.0 / den
    col_i = (bet * col_i1 - col[i]) * den
    col_i2 = col_i1
    col_i1 = col_i
    col[i] = col_i

    i = ntotal - 2
    col_i = col_i2 + elm1[i] * col_i1
    col[i] = col_i
    col_i2 = col_i1
    col_i1 = col_i
    for i in range(ntotal - 3, -1, -1):
        col_i = col[i] + elm1[i] * col_i1 + elm2[i] * col_i2
        col[i] = col_i
        col_i2 = col_i1
        col_i1 = col_i
    return col[0::2].mT, col[1::2].mT


def solve_solar_observation_bvp_batch_torch(
    *,
    albedo,
    direct_beam,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves batched solar-observation BVP systems."""
    output_device = xpos1.device
    output_dtype = xpos1.dtype
    if solve_device is not None or solve_dtype is not None:
        albedo_s = _move_value_for_solve(albedo, solve_device=solve_device, solve_dtype=solve_dtype)
        direct_beam_s = _move_value_for_solve(
            direct_beam, solve_device=solve_device, solve_dtype=solve_dtype
        )
        xpos1_s = _move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype)
        xpos2_s = _move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype)
        eigentrans_s = _move_value_for_solve(
            eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
        )
        wupper_s = _move_value_for_solve(wupper, solve_device=solve_device, solve_dtype=solve_dtype)
        wlower_s = _move_value_for_solve(wlower, solve_device=solve_device, solve_dtype=solve_dtype)
        lcon, mcon = _solve_pentadiagonal_bvp_batch_torch(
            albedo=albedo_s,
            bottom_source=direct_beam_s,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1_s,
            xpos2=xpos2_s,
            eigentrans=eigentrans_s,
            wupper=wupper_s,
            wlower=wlower_s,
        )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )
    return _solve_pentadiagonal_bvp_batch_torch(
        albedo=albedo,
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def solve_solar_observation_block_bvp_batch_torch(
    *,
    albedo,
    direct_beam,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves batched solar-observation BVP systems with 2x2 block Thomas."""
    output_device = xpos1.device
    output_dtype = xpos1.dtype
    if solve_device is not None or solve_dtype is not None:
        lcon, mcon = solve_solar_observation_block_bvp_batch_torch(
            albedo=_move_value_for_solve(
                albedo, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            direct_beam=_move_value_for_solve(
                direct_beam, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=_move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype),
            xpos2=_move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype),
            eigentrans=_move_value_for_solve(
                eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wupper=_move_value_for_solve(
                wupper, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wlower=_move_value_for_solve(
                wlower, solve_device=solve_device, solve_dtype=solve_dtype
            ),
        )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )
    return _solve_block_bvp_batch_torch(
        albedo=albedo,
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def solve_thermal_bvp_batch_torch(
    *,
    albedo,
    emissivity,
    surfbb,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves regular thermal two-stream BVP systems."""
    output_device = xpos1.device
    output_dtype = xpos1.dtype
    if solve_device is not None or solve_dtype is not None:
        albedo_s = _move_value_for_solve(albedo, solve_device=solve_device, solve_dtype=solve_dtype)
        bottom_source_s = _move_value_for_solve(
            surfbb * emissivity,
            solve_device=solve_device,
            solve_dtype=solve_dtype,
        )
        xpos1_s = _move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype)
        xpos2_s = _move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype)
        eigentrans_s = _move_value_for_solve(
            eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
        )
        wupper_s = _move_value_for_solve(wupper, solve_device=solve_device, solve_dtype=solve_dtype)
        wlower_s = _move_value_for_solve(wlower, solve_device=solve_device, solve_dtype=solve_dtype)
        lcon, mcon = _solve_pentadiagonal_bvp_batch_torch(
            albedo=albedo_s,
            bottom_source=bottom_source_s,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1_s,
            xpos2=xpos2_s,
            eigentrans=eigentrans_s,
            wupper=wupper_s,
            wlower=wlower_s,
        )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )
    return _solve_pentadiagonal_bvp_batch_torch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def solve_thermal_block_bvp_batch_torch(
    *,
    albedo,
    emissivity,
    surfbb,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves regular thermal two-stream BVP systems with 2x2 block Thomas."""
    output_device = xpos1.device
    output_dtype = xpos1.dtype
    if solve_device is not None or solve_dtype is not None:
        lcon, mcon = solve_thermal_block_bvp_batch_torch(
            albedo=_move_value_for_solve(
                albedo, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            emissivity=_move_value_for_solve(
                emissivity, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            surfbb=_move_value_for_solve(
                surfbb, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=_move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype),
            xpos2=_move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype),
            eigentrans=_move_value_for_solve(
                eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wupper=_move_value_for_solve(
                wupper, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wlower=_move_value_for_solve(
                wlower, solve_device=solve_device, solve_dtype=solve_dtype
            ),
        )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )
    return _solve_block_bvp_batch_torch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )


def _solve_dense_bvp_batch_torch(
    *,
    albedo,
    bottom_source,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves two-stream BVP systems with dense matrices for float32 stability."""
    output_device = xpos1.device
    output_dtype = xpos1.dtype
    if solve_device is not None or solve_dtype is not None:
        lcon, mcon = _solve_dense_bvp_batch_torch(
            albedo=_move_value_for_solve(
                albedo, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            bottom_source=_move_value_for_solve(
                bottom_source, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=_move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype),
            xpos2=_move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype),
            eigentrans=_move_value_for_solve(
                eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wupper=_move_value_for_solve(
                wupper, solve_device=solve_device, solve_dtype=solve_dtype
            ),
            wlower=_move_value_for_solve(
                wlower, solve_device=solve_device, solve_dtype=solve_dtype
            ),
        )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )

    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    batch, nlay = xpos1.shape
    ntotal = 2 * nlay
    dtype = xpos1.dtype
    device = xpos1.device
    mat = torch.zeros((batch, ntotal, ntotal), dtype=dtype, device=device)
    rhs = torch.zeros((batch, ntotal), dtype=dtype, device=device)

    factor = surface_factor * albedo
    r2_homp = factor * xpos1[:, -1] * stream_value
    r2_homm = factor * xpos2[:, -1] * stream_value
    xpnet = xpos2[:, -1] - r2_homp
    xmnet = xpos1[:, -1] - r2_homm

    mat[:, 0, 0] = xpos1[:, 0]
    mat[:, 0, 1] = xpos2[:, 0] * eigentrans[:, 0]
    rhs[:, 0] = -wupper0[:, 0]
    row = 1
    for n in range(1, nlay):
        prev = n - 1
        mat[:, row, 2 * prev] = xpos1[:, prev] * eigentrans[:, prev]
        mat[:, row, 2 * prev + 1] = xpos2[:, prev]
        mat[:, row, 2 * n] = -xpos1[:, n]
        mat[:, row, 2 * n + 1] = -xpos2[:, n] * eigentrans[:, n]
        rhs[:, row] = wupper0[:, n] - wlower0[:, prev]
        row += 1

        mat[:, row, 2 * prev] = xpos2[:, prev] * eigentrans[:, prev]
        mat[:, row, 2 * prev + 1] = xpos1[:, prev]
        mat[:, row, 2 * n] = -xpos2[:, n]
        mat[:, row, 2 * n + 1] = -xpos1[:, n] * eigentrans[:, n]
        rhs[:, row] = wupper1[:, n] - wlower1[:, prev]
        row += 1

    mat[:, -1, -2] = xpnet * eigentrans[:, -1]
    mat[:, -1, -1] = xmnet
    h_partic = wlower0[:, -1] * stream_value
    rhs[:, -1] = -wlower1[:, -1] + h_partic * factor + bottom_source

    sol = torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)
    return sol[:, 0::2], sol[:, 1::2]


def solve_solar_observation_dense_bvp_batch_torch(
    *,
    albedo,
    direct_beam,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves solar-observation BVP systems with dense matrices."""
    return _solve_dense_bvp_batch_torch(
        albedo=albedo,
        bottom_source=direct_beam,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
    )


def solve_thermal_dense_bvp_batch_torch(
    *,
    albedo,
    emissivity,
    surfbb,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
    wupper,
    wlower,
    solve_device=None,
    solve_dtype=None,
):
    """Solves thermal BVP systems with dense matrices for float32 stability."""
    return _solve_dense_bvp_batch_torch(
        albedo=albedo,
        bottom_source=surfbb * emissivity,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
        solve_device=solve_device,
        solve_dtype=solve_dtype,
    )
