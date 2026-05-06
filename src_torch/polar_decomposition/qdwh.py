from typing import Optional, Sequence
import torch
import numpy as np


def _as_tensor_like(x, like: torch.Tensor) -> torch.Tensor:
    """Convert scalars/arrays to a tensor on the same device/dtype as `like`."""
    if torch.is_tensor(x):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


# Helpers for working with padded shapes
def _mask(
    x: torch.Tensor,
    dims: Sequence[Optional[int]],
    alternative = 0,
) -> torch.Tensor:
    """Masks `x` up to the dynamic shape `dims`.

    Replaces values outside those dimensions with `alternative`. `alternative` is
    broadcast with `x`.
    """
    assert x.ndim == len(dims), (x.ndim, len(dims))
    alt = _as_tensor_like(alternative, x)

    mask = None
    for i, d in enumerate(dims):
        if d is None:
            continue
        # broadcasted_iota equivalent: compare index along dim i to d
        idx = torch.arange(x.size(i), device=x.device)
        view = [1] * x.ndim
        view[i] = -1
        mask_i = idx.view(*view) < int(d)
        mask = mask_i if mask is None else (mask & mask_i)

    return x if mask is None else torch.where(mask, x, alt)

def _pad_in_dim(
    x: torch.Tensor,
    low: int = 0,
    high: int = 0,
    interior: int = 0,
    fill_value = 0,
    axis: int = 0,
) -> torch.Tensor:
    """Pads tensor `x` along `axis` with `low/high` and optional `interior` gaps."""
    axis = axis % x.ndim
    old = x.size(axis)
    if old == 0:
        new_len = low + high
    else:
        new_len = low + old + high + (old - 1) * interior

    out_shape = list(x.shape)
    out_shape[axis] = new_len
    out = torch.full(out_shape, fill_value, dtype=x.dtype, device=x.device)

    if old == 0:
        return out

    step = interior + 1
    positions = low + torch.arange(old, device=x.device) * step
    out.index_copy_(axis, positions.to(dtype=torch.long), x)
    return out

def _dynamic_concat(a, b, m, axis=0):
    "Concatenates padded arrays `a` and `b` where the true size of `a` is `m`."
    if m is None:
        return torch.cat([a, b], dim=axis)
    return torch.narrow(
        _pad_in_dim(a, high=b.shape[axis], axis=axis), axis, 0, m + b.shape[axis])


def _use_qr(u, m, n, params):
    """QDWH iteration using QR decomposition.

    Args:
    u: a matrix, with static (padded) shape M x N.
    m, n: the dynamic shape of the matrix, where m <= M and n <= N.
    params: the QDWH parameters.
    """
    a_minus_e_by_sqrt_c, sqrt_c, e = params
    M, N = u.shape

    y = _dynamic_concat(sqrt_c * u, torch.eye(N, dtype=u.dtype), m)
    q, _ = torch.linalg.qr(y, mode='reduced')
    # q1 = q[:m, :]
    q1 = _mask(q[:M, :], (m, n))
    # q2 = (q[m:, :]).T.conj()
    q2 = q[m:, :]
    q2 = _mask(q2, (n, n)).T.conj()
    return e * u + a_minus_e_by_sqrt_c * (q1 @ q2)


def _use_cholesky(u, m, n, params):
    """QDWH iteration using Cholesky decomposition.

    Args:
    u: a matrix, with static (padded) shape M x N
    m, n: the dynamic shape of the matrix, where m <= M and n <= N.
    params: the QDWH parameters.
    """
    a_minus_e, c, e = params
    _, N = u.shape
    x = c * (u.T.conj() @ u) + torch.eye(N, dtype=u.dtype)
    # Pads the lower-right corner with the identity matrix to prevent the Cholesky
    # decomposition from failing due to the matrix not being PSD if padded with
    # zeros.
    x = _mask(x, (n, n), torch.eye(N, dtype=x.dtype))
    # `y` is lower triangular.
    y = torch.linalg.cholesky(x, upper=False)

    z = torch.linalg.solve_triangular(y.conj(), u.transpose(-2, -1), upper=False, left=True).conj()

    z = torch.linalg.solve_triangular(y.conj().transpose(-2, -1), z, upper=True, left=True)
    z = z.transpose(-2, -1).conj()

    return e * u + a_minus_e * z


def polar_decomposition_qdwh(A, num_iters=10, tol_norm=1e-6, **kwargs):
    """QR-based dynamically weighted Halley iteration for polar decomposition."""

    # Estimates `alpha` and `beta = alpha * l`, where `alpha` is an estimate of
    # norm(x, 2) such that `alpha >= norm(x, 2)` and `beta` is a lower bound for
    # the smallest singular value of x.
    assert A.dtype is torch.float32
    one_norm = torch.linalg.norm(A, ord=1)
    inf_norm = torch.linalg.norm(A, ord=torch.inf)
    alpha_inverse = torch.rsqrt(one_norm) * torch.rsqrt(inf_norm)
    u = A * alpha_inverse.to(A.dtype)
    m, n = A.shape

    def get_qr_params(a, b, c):
        e = b / c
        a_minus_e = a - e
        sqrt_c = c ** (1 / 2)
        return (a_minus_e / sqrt_c, sqrt_c, e)

    def get_chol_params(a, b, c):
        e = b / c
        a_minus_e = a - e
        return (a_minus_e, c, e)

    CHOLESKY_CUTOFF = 100

    qr_coefs = []
    chol_coefs = []
    eps = torch.finfo(A.dtype).eps
    l = eps
    tol_l = 10.0 * eps / 2.0
    tol_norm = np.cbrt(tol_l)
    k = 0
    while l + tol_l < 1 and k < num_iters:
        k += 1
        l2 = l * l
        dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
        sqd = (1.0 + dd) ** (1 / 2)
        a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** (1 / 2)
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)
        if c > CHOLESKY_CUTOFF:
            qr_coefs.append(get_qr_params(a, b, c))
        else:
            chol_coefs.append(get_chol_params(a, b, c))

    def iteration(k, state, update_fn, coefs, test_convergence):
        u, _ = state

        if coefs is None:
        # As l → 1, the coefficients a, b, c → 3, 1, 3, which is Halley's method.
            params = get_chol_params(3, 1, 3)
        else:
            params = coefs[k]

        u_prev = u
        u = update_fn(u, m, n, params)

        is_not_converged = True
        if test_convergence:
            is_not_converged = torch.linalg.norm(u - u_prev) > tol_norm
        return u, is_not_converged

    def iterate(u, coefs, **kwargs):
        if not coefs:
            return u, True
        coefs = torch.tensor(coefs).to(u.dtype)
        for i in range(len(coefs)):
            u, is_not_converged = iteration(i, (u, True), coefs=coefs, **kwargs)
        return u, is_not_converged

    u, _ = iterate(
        u, coefs=qr_coefs, update_fn=_use_qr, test_convergence=False
    )
    u, is_not_converged = iterate(
        u, coefs=chol_coefs, update_fn=_use_cholesky, test_convergence=True
    )

    # If l has converged but u still has not, continue with Halley's method
    # (coef = None) until convergence.
    k = len(qr_coefs) + len(chol_coefs)
    while is_not_converged and k < num_iters:
        u, is_not_converged = iteration(
            k,
            (u, is_not_converged),
            coefs=None,
            update_fn=_use_cholesky,
            test_convergence=True,
        )
        k += 1

    # Applies Newton-Schulz refinement for better accuracy.
    u = 1.5 * u - 0.5 * u @ (u.T.conj() @ u)

    h = u.T.conj() @ A
    h = (h + h.T.conj()) / 2

    # Converged within the maximum number of iterations.
    is_converged = not(is_not_converged)

    return u, h
