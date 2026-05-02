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


def _broadcast_bvp_batch_vector(value, *, batch, dtype, device):
    """Broadcasts a scalar-or-vector BVP input to a row vector."""
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    return torch.broadcast_to(tensor, (batch,))


def _detect_singular_pentadiagonal_rows_torch(
    *,
    albedo,
    surface_factor,
    stream_value,
    xpos1,
    xpos2,
    eigentrans,
):
    """Flags rows whose pentadiagonal elimination pivots collapse numerically."""
    with torch.no_grad():
        x1 = xpos1.transpose(0, 1).contiguous()
        x2 = xpos2.transpose(0, 1).contiguous()
        et = eigentrans.transpose(0, 1).contiguous()
        nlay = x1.shape[0]
        batch = x1.shape[1]
        albedo_vec = _broadcast_bvp_batch_vector(
            albedo,
            batch=batch,
            dtype=x1.dtype,
            device=x1.device,
        )
        bad = torch.zeros((batch,), dtype=torch.bool, device=x1.device)

        elm31 = 1.0 / x1[0]
        elm1_i2 = -(x2[0] * et[0]) * elm31
        bet = x2[0] + x1[0] * et[0] * elm1_i2
        bad |= ~torch.isfinite(bet) | (bet == 0.0)

        inv_bet = -1.0 / bet
        elm1_i1 = -x1[1] * inv_bet
        elm2_i2 = torch.zeros((batch,), dtype=x1.dtype, device=x1.device)
        elm2_i1 = (-x2[1] * et[1]) * inv_bet

        for i in range(2, 2 * nlay - 2):
            if i % 2 == 0:
                n = i // 2
                prev = n - 1
                mat1_i = x2[prev] * et[prev]
                mat2_i = x1[prev]
                mat3_i = -x2[n]
                mat4_i = -x1[n] * et[n]
                mat5_i = torch.zeros_like(mat1_i)
            else:
                n = (i + 1) // 2
                prev = n - 1
                mat1_i = torch.zeros_like(x1[prev])
                mat2_i = x1[prev] * et[prev]
                mat3_i = x2[prev]
                mat4_i = -x1[n]
                mat5_i = -x2[n] * et[n]
            bet = mat2_i + mat1_i * elm1_i2
            den0 = mat3_i + mat1_i * elm2_i2 + bet * elm1_i1
            bad |= ~torch.isfinite(den0) | (den0 == 0.0)
            inv_den = -1.0 / den0
            elm1_i2 = elm1_i1
            elm1_i1 = (mat4_i + bet * elm2_i1) * inv_den
            elm2_i2 = elm2_i1
            elm2_i1 = mat5_i * inv_den

        factor = surface_factor * albedo_vec
        xmnet = x1[-1] - factor * x2[-1] * stream_value
        final_bet = (x2[-1] - factor * x1[-1] * stream_value) * et[-1]
        final_den = xmnet + final_bet * elm1_i1
        bad |= ~torch.isfinite(final_den) | (final_den == 0.0)
        return bad


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


def _build_pentadiagonal_system_torch(
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
    """Builds batched pentadiagonal coefficients for the thermal/solar BVP."""
    wupper0, wupper1 = wupper
    wlower0, wlower1 = wlower
    batch, nlay = xpos1.shape
    ntotal = 2 * nlay
    dtype = xpos1.dtype
    device = xpos1.device
    mat1 = torch.zeros((batch, ntotal), dtype=dtype, device=device)
    mat2 = torch.zeros((batch, ntotal), dtype=dtype, device=device)
    mat3 = torch.zeros((batch, ntotal), dtype=dtype, device=device)
    mat4 = torch.zeros((batch, ntotal), dtype=dtype, device=device)
    mat5 = torch.zeros((batch, ntotal), dtype=dtype, device=device)
    rhs = torch.zeros((batch, ntotal), dtype=dtype, device=device)

    factor = surface_factor * albedo
    xpnet = xpos2[:, -1] - factor * xpos1[:, -1] * stream_value
    xmnet = xpos1[:, -1] - factor * xpos2[:, -1] * stream_value

    mat3[:, 0] = xpos1[:, 0]
    mat4[:, 0] = xpos2[:, 0] * eigentrans[:, 0]
    rhs[:, 0] = -wupper0[:, 0]

    for n in range(1, nlay):
        prev = n - 1
        row_m = 2 * n - 1
        row_p = row_m + 1
        mat2[:, row_m] = xpos1[:, prev] * eigentrans[:, prev]
        mat3[:, row_m] = xpos2[:, prev]
        mat4[:, row_m] = -xpos1[:, n]
        mat5[:, row_m] = -xpos2[:, n] * eigentrans[:, n]
        rhs[:, row_m] = wupper0[:, n] - wlower0[:, prev]

        mat1[:, row_p] = xpos2[:, prev] * eigentrans[:, prev]
        mat2[:, row_p] = xpos1[:, prev]
        mat3[:, row_p] = -xpos2[:, n]
        mat4[:, row_p] = -xpos1[:, n] * eigentrans[:, n]
        rhs[:, row_p] = wupper1[:, n] - wlower1[:, prev]

    mat2[:, -1] = xpnet * eigentrans[:, -1]
    mat3[:, -1] = xmnet
    rhs[:, -1] = -wlower1[:, -1] + wlower0[:, -1] * stream_value * factor + bottom_source
    return mat1, mat2, mat3, mat4, mat5, rhs


def _solve_block_tridiagonal_coefficients_torch(lower, diag, upper, rhs):
    """Solves a batched 2x2 block-tridiagonal linear system."""
    batch, nlay = rhs.shape[:2]
    cprime = torch.zeros_like(upper)
    dprime = torch.zeros_like(rhs)

    if nlay > 1:
        cprime[:, 0] = torch.linalg.solve(diag[:, 0], upper[:, 0])
    dprime[:, 0] = torch.linalg.solve(diag[:, 0], rhs[:, 0].unsqueeze(-1)).squeeze(-1)

    for n in range(1, nlay):
        schur = diag[:, n] - _matmul_2x2_batch_torch(lower[:, n], cprime[:, n - 1])
        if n < nlay - 1:
            cprime[:, n] = torch.linalg.solve(schur, upper[:, n])
        rhs_eff = rhs[:, n] - _matvec_2x2_batch_torch(lower[:, n], dprime[:, n - 1])
        dprime[:, n] = torch.linalg.solve(schur, rhs_eff.unsqueeze(-1)).squeeze(-1)

    solution = torch.empty((batch, nlay, 2), dtype=dprime.dtype, device=dprime.device)
    solution[:, -1] = dprime[:, -1]
    for n in range(nlay - 2, -1, -1):
        solution[:, n] = dprime[:, n] - _matvec_2x2_batch_torch(cprime[:, n], solution[:, n + 1])
    return solution


def _transpose_pentadiagonal_system_torch(mat1, mat2, mat3, mat4, mat5):
    """Builds pentadiagonal coefficients for the transpose system."""
    mat1_t = torch.zeros_like(mat1)
    mat2_t = torch.zeros_like(mat2)
    mat3_t = mat3.clone()
    mat4_t = torch.zeros_like(mat4)
    mat5_t = torch.zeros_like(mat5)
    mat1_t[:, 2:] = mat5[:, :-2]
    mat2_t[:, 1:] = mat4[:, :-1]
    mat4_t[:, :-1] = mat2[:, 1:]
    mat5_t[:, :-2] = mat1[:, 2:]
    return mat1_t, mat2_t, mat3_t, mat4_t, mat5_t


def _solve_pentadiagonal_coefficients_torch(mat1, mat2, mat3, mat4, mat5, rhs):
    """Solves a batched pentadiagonal linear system."""
    batch, ntotal = rhs.shape
    if ntotal == 2:
        det = mat3[:, 0] * mat3[:, 1] - mat4[:, 0] * mat2[:, 1]
        x0 = (rhs[:, 0] * mat3[:, 1] - mat4[:, 0] * rhs[:, 1]) / det
        x1 = (mat3[:, 0] * rhs[:, 1] - rhs[:, 0] * mat2[:, 1]) / det
        return torch.stack((x0, x1), dim=1)

    elm1 = torch.zeros((batch, ntotal - 1), dtype=rhs.dtype, device=rhs.device)
    elm2 = torch.zeros((batch, ntotal - 2), dtype=rhs.dtype, device=rhs.device)
    elm3 = torch.zeros((batch, ntotal), dtype=rhs.dtype, device=rhs.device)
    elm4 = torch.zeros((batch, ntotal), dtype=rhs.dtype, device=rhs.device)
    col = rhs.clone()

    elm31 = 1.0 / mat3[:, 0]
    elm3[:, 0] = elm31
    elm1_i2 = -mat4[:, 0] * elm31
    elm1[:, 0] = elm1_i2
    elm2_i2 = -mat5[:, 0] * elm31
    elm2[:, 0] = elm2_i2

    mat22 = mat2[:, 1]
    bet = mat3[:, 1] + mat22 * elm1_i2
    bet = -1.0 / bet
    elm1_i1 = (mat4[:, 1] + mat22 * elm2_i2) * bet
    elm1[:, 1] = elm1_i1
    elm2_i1 = mat5[:, 1] * bet
    elm2[:, 1] = elm2_i1
    elm3[:, 1] = bet

    for i in range(2, ntotal - 2):
        mat1_i = mat1[:, i]
        bet = mat2[:, i] + mat1_i * elm1_i2
        den = mat3[:, i] + mat1_i * elm2_i2 + bet * elm1_i1
        den = -1.0 / den
        elm1_i2 = elm1_i1
        elm1_i1 = (mat4[:, i] + bet * elm2_i1) * den
        elm1[:, i] = elm1_i1
        elm2_i2 = elm2_i1
        elm2_i1 = mat5[:, i] * den
        elm2[:, i] = elm2_i1
        elm3[:, i] = bet
        elm4[:, i] = den

    i = ntotal - 2
    mat1_i = mat1[:, i]
    bet = mat2[:, i] + mat1_i * elm1_i2
    den = mat3[:, i] + mat1_i * elm2_i2 + bet * elm1_i1
    den = -1.0 / den
    elm1_i2 = elm1_i1
    elm1_i1 = (mat4[:, i] + bet * elm2_i1) * den
    elm1[:, i] = elm1_i1
    elm2_i2 = elm2_i1
    elm3[:, i] = bet
    elm4[:, i] = den

    i = ntotal - 1
    mat1_i = mat1[:, i]
    bet = mat2[:, i] + mat1_i * elm1_i2
    den = mat3[:, i] + mat1_i * elm2_i2 + bet * elm1_i1
    den = -1.0 / den
    elm3[:, i] = bet
    elm4[:, i] = den

    col_i2 = col[:, 0] * elm3[:, 0]
    col[:, 0] = col_i2
    col_i1 = (mat22 * col_i2 - col[:, 1]) * elm3[:, 1]
    col[:, 1] = col_i1
    for i in range(2, ntotal):
        col_i = (mat1[:, i] * col_i2 + elm3[:, i] * col_i1 - col[:, i]) * elm4[:, i]
        col_i2 = col_i1
        col_i1 = col_i
        col[:, i] = col_i

    i = ntotal - 2
    col_i = col_i2 + elm1[:, i] * col_i1
    col[:, i] = col_i
    col_i2 = col_i1
    col_i1 = col_i
    for i in range(ntotal - 3, -1, -1):
        col_i = col[:, i] + elm1[:, i] * col_i1 + elm2[:, i] * col_i2
        col[:, i] = col_i
        col_i2 = col_i1
        col_i1 = col_i
    return col


def _solve_dense_block_tridiagonal_coefficients_torch(lower, diag, upper, rhs):
    """Solves a batched block-tridiagonal system through a dense matrix build."""
    batch, nlay = rhs.shape[:2]
    ntotal = 2 * nlay
    mat = torch.zeros((batch, ntotal, ntotal), dtype=rhs.dtype, device=rhs.device)
    vec = torch.zeros((batch, ntotal), dtype=rhs.dtype, device=rhs.device)
    for n in range(nlay):
        row = 2 * n
        mat[:, row : row + 2, row : row + 2] = diag[:, n]
        vec[:, row : row + 2] = rhs[:, n]
        if n > 0:
            mat[:, row : row + 2, row - 2 : row] = lower[:, n]
        if n < nlay - 1:
            mat[:, row : row + 2, row + 2 : row + 4] = upper[:, n]
    return torch.linalg.solve(mat, vec.unsqueeze(-1)).squeeze(-1).reshape(batch, nlay, 2)


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
    solution = _solve_block_tridiagonal_coefficients_torch(lower, diag, upper, rhs)
    return solution[:, :, 0], solution[:, :, 1]


def _transpose_block_tridiagonal_coefficients_torch(lower, diag, upper):
    """Builds block coefficients for the transpose system ``A^T``."""
    lower_t = torch.zeros_like(lower)
    diag_t = diag.transpose(-1, -2).contiguous()
    upper_t = torch.zeros_like(upper)
    lower_t[:, 1:] = upper[:, :-1].transpose(-1, -2)
    upper_t[:, :-1] = lower[:, 1:].transpose(-1, -2)
    return lower_t, diag_t, upper_t


class _ThermalBvpAutogradFn(torch.autograd.Function):
    """Custom VJP for the fast thermal pentadiagonal BVP solve."""

    @staticmethod
    def forward(
        ctx,
        albedo,
        emissivity,
        surfbb,
        xpos1,
        xpos2,
        eigentrans,
        wupper0,
        wupper1,
        wlower0,
        wlower1,
        surface_factor,
        stream_value,
    ):
        bottom_source = surfbb * emissivity
        lcon, mcon = _solve_pentadiagonal_bvp_batch_torch(
            albedo=albedo,
            bottom_source=bottom_source,
            surface_factor=surface_factor,
            stream_value=stream_value,
            xpos1=xpos1,
            xpos2=xpos2,
            eigentrans=eigentrans,
            wupper=(wupper0, wupper1),
            wlower=(wlower0, wlower1),
        )
        bad_rows = ~(torch.isfinite(lcon).all(dim=1) & torch.isfinite(mcon).all(dim=1))
        if bool(bad_rows.any()):
            lcon_bad, mcon_bad = _solve_dense_bvp_batch_torch(
                albedo=albedo[bad_rows],
                bottom_source=bottom_source[bad_rows],
                surface_factor=surface_factor,
                stream_value=stream_value,
                xpos1=xpos1[bad_rows],
                xpos2=xpos2[bad_rows],
                eigentrans=eigentrans[bad_rows],
                wupper=(wupper0[bad_rows], wupper1[bad_rows]),
                wlower=(wlower0[bad_rows], wlower1[bad_rows]),
            )
            lcon = lcon.clone()
            mcon = mcon.clone()
            lcon[bad_rows] = lcon_bad
            mcon[bad_rows] = mcon_bad
        ctx.surface_factor = float(surface_factor)
        ctx.stream_value = float(stream_value)
        ctx.bad_rows = bad_rows
        ctx.save_for_backward(
            albedo,
            emissivity,
            surfbb,
            xpos1,
            xpos2,
            eigentrans,
            wupper0,
            wupper1,
            wlower0,
            wlower1,
            lcon,
            mcon,
        )
        return lcon, mcon

    @staticmethod
    def backward(ctx, grad_lcon, grad_mcon):
        (
            albedo,
            emissivity,
            surfbb,
            xpos1,
            xpos2,
            eigentrans,
            wupper0,
            wupper1,
            wlower0,
            wlower1,
            lcon,
            mcon,
        ) = ctx.saved_tensors
        prepared_values = []

        def _prep(value, slot):
            if not ctx.needs_input_grad[slot]:
                prepared_values.append((value.detach(), False))
                return prepared_values[-1][0]
            prepared = value.detach().requires_grad_(True)
            prepared_values.append((prepared, True))
            return prepared

        with torch.enable_grad():
            albedo = _prep(albedo, 0)
            emissivity = _prep(emissivity, 1)
            surfbb = _prep(surfbb, 2)
            xpos1 = _prep(xpos1, 3)
            xpos2 = _prep(xpos2, 4)
            eigentrans = _prep(eigentrans, 5)
            wupper0 = _prep(wupper0, 6)
            wupper1 = _prep(wupper1, 7)
            wlower0 = _prep(wlower0, 8)
            wlower1 = _prep(wlower1, 9)

            mat1, mat2, mat3, mat4, mat5, rhs = _build_pentadiagonal_system_torch(
                albedo=albedo,
                bottom_source=surfbb * emissivity,
                surface_factor=ctx.surface_factor,
                stream_value=ctx.stream_value,
                xpos1=xpos1,
                xpos2=xpos2,
                eigentrans=eigentrans,
                wupper=(wupper0, wupper1),
                wlower=(wlower0, wlower1),
            )

        solution = torch.empty((mat1.shape[0], mat1.shape[1]), dtype=lcon.dtype, device=lcon.device)
        solution[:, 0::2] = lcon
        solution[:, 1::2] = mcon
        grad_solution = torch.empty_like(solution)
        grad_solution[:, 0::2] = grad_lcon
        grad_solution[:, 1::2] = grad_mcon
        mat1_t, mat2_t, mat3_t, mat4_t, mat5_t = _transpose_pentadiagonal_system_torch(
            mat1.detach(),
            mat2.detach(),
            mat3.detach(),
            mat4.detach(),
            mat5.detach(),
        )
        bad = (
            _detect_singular_pentadiagonal_rows_torch(
                albedo=albedo.detach(),
                surface_factor=ctx.surface_factor,
                stream_value=ctx.stream_value,
                xpos1=xpos1.detach(),
                xpos2=xpos2.detach(),
                eigentrans=eigentrans.detach(),
            )
            | ctx.bad_rows
        )
        if bool(bad.any()):
            adjoint = torch.empty_like(grad_solution)
            good = ~bad
            if bool(good.any()):
                adjoint[good] = _solve_pentadiagonal_coefficients_torch(
                    mat1_t[good],
                    mat2_t[good],
                    mat3_t[good],
                    mat4_t[good],
                    mat5_t[good],
                    grad_solution[good],
                )
            if bool(bad.any()):
                lower, diag, upper, _ = _build_block_tridiagonal_system_torch(
                    albedo=albedo.detach()[bad],
                    bottom_source=(surfbb.detach() * emissivity.detach())[bad],
                    surface_factor=ctx.surface_factor,
                    stream_value=ctx.stream_value,
                    xpos1=xpos1.detach()[bad],
                    xpos2=xpos2.detach()[bad],
                    eigentrans=eigentrans.detach()[bad],
                    wupper=(wupper0.detach()[bad], wupper1.detach()[bad]),
                    wlower=(wlower0.detach()[bad], wlower1.detach()[bad]),
                )
                lower_t, diag_t, upper_t = _transpose_block_tridiagonal_coefficients_torch(
                    lower, diag, upper
                )
                adjoint[bad] = _solve_dense_block_tridiagonal_coefficients_torch(
                    lower_t,
                    diag_t,
                    upper_t,
                    grad_solution[bad].reshape(-1, grad_solution.shape[1] // 2, 2),
                ).reshape(-1, grad_solution.shape[1])
        else:
            adjoint = _solve_pentadiagonal_coefficients_torch(
                mat1_t,
                mat2_t,
                mat3_t,
                mat4_t,
                mat5_t,
                grad_solution,
            )
        adjoint_bad = ~torch.isfinite(adjoint).all(dim=1)
        if bool(adjoint_bad.any()):
            lower, diag, upper, _ = _build_block_tridiagonal_system_torch(
                albedo=albedo.detach()[adjoint_bad],
                bottom_source=(surfbb.detach() * emissivity.detach())[adjoint_bad],
                surface_factor=ctx.surface_factor,
                stream_value=ctx.stream_value,
                xpos1=xpos1.detach()[adjoint_bad],
                xpos2=xpos2.detach()[adjoint_bad],
                eigentrans=eigentrans.detach()[adjoint_bad],
                wupper=(wupper0.detach()[adjoint_bad], wupper1.detach()[adjoint_bad]),
                wlower=(wlower0.detach()[adjoint_bad], wlower1.detach()[adjoint_bad]),
            )
            lower_t, diag_t, upper_t = _transpose_block_tridiagonal_coefficients_torch(
                lower, diag, upper
            )
            adjoint[adjoint_bad] = _solve_dense_block_tridiagonal_coefficients_torch(
                lower_t,
                diag_t,
                upper_t,
                grad_solution[adjoint_bad].reshape(-1, grad_solution.shape[1] // 2, 2),
            ).reshape(-1, grad_solution.shape[1])

        grad_rhs = adjoint
        grad_mat1 = torch.zeros_like(mat1)
        grad_mat2 = torch.zeros_like(mat2)
        grad_mat3 = -adjoint * solution
        grad_mat4 = torch.zeros_like(mat4)
        grad_mat5 = torch.zeros_like(mat5)
        grad_mat2[:, 1:] = -adjoint[:, 1:] * solution[:, :-1]
        grad_mat4[:, :-1] = -adjoint[:, :-1] * solution[:, 1:]
        grad_mat1[:, 2:] = -adjoint[:, 2:] * solution[:, :-2]
        grad_mat5[:, :-2] = -adjoint[:, :-2] * solution[:, 2:]

        grad_inputs = tuple(value for value, needed in prepared_values if needed)
        outputs = []
        grad_outputs = []
        for output, grad_output in (
            (mat1, grad_mat1),
            (mat2, grad_mat2),
            (mat3, grad_mat3),
            (mat4, grad_mat4),
            (mat5, grad_mat5),
            (rhs, grad_rhs),
        ):
            if output.requires_grad:
                outputs.append(output)
                grad_outputs.append(grad_output)
        raw_grads = torch.autograd.grad(
            outputs=tuple(outputs),
            inputs=grad_inputs,
            grad_outputs=tuple(grad_outputs),
            allow_unused=True,
        )
        grads = []
        index = 0
        for _value, needed in prepared_values:
            if needed:
                grads.append(raw_grads[index])
                index += 1
            else:
                grads.append(None)
        return (*grads, None, None)


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
    return _solve_pentadiagonal_bvp_batch_torch_impl(
        albedo,
        bottom_source,
        surface_factor,
        stream_value,
        xpos1,
        xpos2,
        eigentrans,
        wupper[0],
        wupper[1],
        wlower[0],
        wlower[1],
    )


def _solve_pentadiagonal_bvp_batch_torch_eager(
    albedo: torch.Tensor,
    bottom_source: torch.Tensor,
    surface_factor: float,
    stream_value: float,
    xpos1: torch.Tensor,
    xpos2: torch.Tensor,
    eigentrans: torch.Tensor,
    wupper0_in: torch.Tensor,
    wupper1_in: torch.Tensor,
    wlower0_in: torch.Tensor,
    wlower1_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = xpos1.transpose(0, 1).contiguous()
    x2 = xpos2.transpose(0, 1).contiguous()
    et = eigentrans.transpose(0, 1).contiguous()
    wupper0 = wupper0_in.transpose(0, 1).contiguous()
    wupper1 = wupper1_in.transpose(0, 1).contiguous()
    wlower0 = wlower0_in.transpose(0, 1).contiguous()
    wlower1 = wlower1_in.transpose(0, 1).contiguous()
    nlay = x1.shape[0]
    batch = x1.shape[1]
    ntotal = 2 * nlay
    col = torch.zeros((ntotal, batch), dtype=x1.dtype, device=x1.device)

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

    elm1 = torch.zeros((ntotal - 1, batch), dtype=x1.dtype, device=x1.device)
    elm2 = torch.zeros((ntotal - 2, batch), dtype=x1.dtype, device=x1.device)

    elm31 = 1.0 / x1[0]
    elm1_i2 = -(x2[0] * et[0]) * elm31
    elm1[0] = elm1_i2
    elm2_i2 = torch.zeros(batch, dtype=x1.dtype, device=x1.device)
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
            mat5_i = torch.zeros_like(mat1_i)
        else:
            n = (i + 1) // 2
            prev = n - 1
            mat1_i = torch.zeros_like(x1[prev])
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
    return col[0::2].transpose(0, 1), col[1::2].transpose(0, 1)


_solve_pentadiagonal_bvp_batch_torch_impl = (
    torch.jit.script(_solve_pentadiagonal_bvp_batch_torch_eager)
    if torch is not None
    else _solve_pentadiagonal_bvp_batch_torch_eager
)


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
    if xpos1.shape[1] == 1:
        return solve_thermal_dense_bvp_batch_torch(
            albedo=albedo,
            emissivity=emissivity,
            surfbb=surfbb,
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
    needs_grad = any(
        torch.is_tensor(value) and bool(value.requires_grad)
        for value in (
            albedo,
            emissivity,
            surfbb,
            xpos1,
            xpos2,
            eigentrans,
            *wupper,
            *wlower,
        )
    )

    if solve_device is not None or solve_dtype is not None:
        albedo_s = _move_value_for_solve(albedo, solve_device=solve_device, solve_dtype=solve_dtype)
        emissivity_s = _move_value_for_solve(
            emissivity, solve_device=solve_device, solve_dtype=solve_dtype
        )
        surfbb_s = _move_value_for_solve(surfbb, solve_device=solve_device, solve_dtype=solve_dtype)
        xpos1_s = _move_value_for_solve(xpos1, solve_device=solve_device, solve_dtype=solve_dtype)
        xpos2_s = _move_value_for_solve(xpos2, solve_device=solve_device, solve_dtype=solve_dtype)
        eigentrans_s = _move_value_for_solve(
            eigentrans, solve_device=solve_device, solve_dtype=solve_dtype
        )
        wupper_s = _move_value_for_solve(wupper, solve_device=solve_device, solve_dtype=solve_dtype)
        wlower_s = _move_value_for_solve(wlower, solve_device=solve_device, solve_dtype=solve_dtype)
        batch = xpos1_s.shape[0]
        albedo_s = _broadcast_bvp_batch_vector(
            albedo_s, batch=batch, dtype=xpos1_s.dtype, device=xpos1_s.device
        )
        emissivity_s = _broadcast_bvp_batch_vector(
            emissivity_s, batch=batch, dtype=xpos1_s.dtype, device=xpos1_s.device
        )
        surfbb_s = _broadcast_bvp_batch_vector(
            surfbb_s, batch=batch, dtype=xpos1_s.dtype, device=xpos1_s.device
        )
        if needs_grad:
            lcon, mcon = _ThermalBvpAutogradFn.apply(
                albedo_s,
                emissivity_s,
                surfbb_s,
                xpos1_s,
                xpos2_s,
                eigentrans_s,
                wupper_s[0],
                wupper_s[1],
                wlower_s[0],
                wlower_s[1],
                surface_factor,
                stream_value,
            )
        else:
            lcon, mcon = _solve_pentadiagonal_bvp_batch_torch(
                albedo=albedo_s,
                bottom_source=surfbb_s * emissivity_s,
                surface_factor=surface_factor,
                stream_value=stream_value,
                xpos1=xpos1_s,
                xpos2=xpos2_s,
                eigentrans=eigentrans_s,
                wupper=wupper_s,
                wlower=wlower_s,
            )
            lcon, mcon = repair_nonfinite_bvp_rows_torch(
                lcon=lcon,
                mcon=mcon,
                bad_row_solver=solve_thermal_dense_bvp_batch_torch,
                solver_kwargs={
                    "albedo": albedo_s,
                    "emissivity": emissivity_s,
                    "surfbb": surfbb_s,
                    "surface_factor": surface_factor,
                    "stream_value": stream_value,
                    "xpos1": xpos1_s,
                    "xpos2": xpos2_s,
                    "eigentrans": eigentrans_s,
                    "wupper": wupper_s,
                    "wlower": wlower_s,
                },
            )
        return lcon.to(device=output_device, dtype=output_dtype), mcon.to(
            device=output_device, dtype=output_dtype
        )
    batch = xpos1.shape[0]
    albedo_s = _broadcast_bvp_batch_vector(
        albedo, batch=batch, dtype=xpos1.dtype, device=xpos1.device
    )
    emissivity_s = _broadcast_bvp_batch_vector(
        emissivity, batch=batch, dtype=xpos1.dtype, device=xpos1.device
    )
    surfbb_s = _broadcast_bvp_batch_vector(
        surfbb, batch=batch, dtype=xpos1.dtype, device=xpos1.device
    )
    if needs_grad:
        return _ThermalBvpAutogradFn.apply(
            albedo_s,
            emissivity_s,
            surfbb_s,
            xpos1,
            xpos2,
            eigentrans,
            wupper[0],
            wupper[1],
            wlower[0],
            wlower[1],
            surface_factor,
            stream_value,
        )
    lcon, mcon = _solve_pentadiagonal_bvp_batch_torch(
        albedo=albedo_s,
        bottom_source=surfbb_s * emissivity_s,
        surface_factor=surface_factor,
        stream_value=stream_value,
        xpos1=xpos1,
        xpos2=xpos2,
        eigentrans=eigentrans,
        wupper=wupper,
        wlower=wlower,
    )
    return repair_nonfinite_bvp_rows_torch(
        lcon=lcon,
        mcon=mcon,
        bad_row_solver=solve_thermal_dense_bvp_batch_torch,
        solver_kwargs={
            "albedo": albedo_s,
            "emissivity": emissivity_s,
            "surfbb": surfbb_s,
            "surface_factor": surface_factor,
            "stream_value": stream_value,
            "xpos1": xpos1,
            "xpos2": xpos2,
            "eigentrans": eigentrans,
            "wupper": wupper,
            "wlower": wlower,
        },
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
