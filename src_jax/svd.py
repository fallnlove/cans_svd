from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from src_jax.eigenvalue_decomposition import _make_K


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "polar_method",
        "cans_tol",
        "threshold",
        "eps",
        "eps_qr",
    ),
)
@partial(
    jax.jit,
    static_argnames=[
        "polar_method",
        "cans_tol",
        "threshold",
        "eps",
        "eps_qr",
    ],
)
def svd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    threshold: float=1e-8,
    eps: float=1e-8,
    eps_qr: float=1e-5,
) -> List[jnp.array]:
    """
    Function to compute the Singular Value Decomposition (SVD).

    Args:
        A: Input matrix of shape (m, n).
        polar_method: Method to compute the polar decomposition. Can be "cans" or "qdwh".
        cans_tol: Tolerance for the CANS iteration (if polar_method is "cans").
        threshold: Threshold for small singular values to be treated as zero.
        eps: Small constant to avoid division by zero in gradient computation.
        eps_qr: Tolerance for performing QR decomposition to ensure orthogonality.
    Returns:
        U: Left singular vectors.
        S: Singular values.
        Vt: Right singular vectors.
    """
    n_start = A.shape[0]
    if A.shape[0] < A.shape[1]:
        A = A.T
    if polar_method == "cans":
        from src_jax.cans_iteration import cans_iteration
        W = cans_iteration(A, tol=cans_tol)
        H = W.T @ A
        H = (H + H.T.conj()) / 2
    elif polar_method == "qdwh":
        from jax.lax.linalg import qdwh
        W, H, _, _ = qdwh(A)
    else:
        raise ValueError(f"Unknown polar method: {polar_method}")

    V, s = jax.lax.linalg.eigh(H)
    U = W @ V

    def perform_qr(U, s):
        U, R = jax.lax.linalg.qr(U, full_matrices=False)
        s = jnp.diag(R) * s
        return U, s    
    U = U[:, jnp.arange(U.shape[-1] - 1, -1, -1)]
    V = V[:, jnp.arange(V.shape[-1] - 1, -1, -1)]
    s = s[jnp.arange(s.shape[-1] - 1, -1, -1)]

    U, s = jax.lax.cond(
        jnp.any(jax.lax.abs(jnp.linalg.norm(U, axis=0) - 1) > eps_qr),
        perform_qr,
        lambda U, s: (U, s),
        U,
        s,
    )
    V = jnp.where(s < 0, -V, V)
    s = jnp.abs(s)

    idx = jnp.argsort(s, descending=True)

    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    if n_start == A.shape[0]:
        return U, s, V.T 
    else:
        return V, s, U.T


def _svd_fwd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    threshold: float=1e-8,
    eps: float=1e-8,
    eps_qr: float=1e-5,
):
    U, s, Vt = svd(A, polar_method, cans_tol, threshold, eps, eps_qr)
    return (U, s, Vt), (U, s, Vt)


def _svd_bwd(polar_method, cans_tol, threshold, eps, eps_qr, res, g):
    U, s, Vt = res
    grad_U, grad_s, grad_Vt = g

    n_start = U.shape[0]
    if U.shape[0] < Vt.shape[1]:
        U, Vt = Vt.T, U.T
        grad_U, grad_Vt = grad_Vt.T, grad_U.T


    r = jnp.sum(s > threshold)
    mask = (jnp.arange(U.shape[1]) < r).astype(U.dtype)

    U_r = U * mask[None, :]
    Vt_r = Vt * mask[:, None]
    s_r = s * mask

    gU = grad_U * mask[None, :]
    gVt = grad_Vt * mask[:, None]
    gS = grad_s * mask

    grad_A = jnp.zeros((U.shape[0], Vt.shape[1]), dtype=U.dtype)

    inv_Sr = jnp.where(s_r > 0, 1.0 / s_r, 0.0)

    grad_A += (gU * inv_Sr[None, :]) @ Vt_r

    tmp = jnp.diag(gS)

    K = _make_K(s_r, eps).mT

    tmp += K * (Vt_r @ gVt.mT + U_r.mT @ gU)
    tmp -= (U_r.mT @ gU) * inv_Sr[None, :]
    tmp = 0.5 * (tmp + tmp.mT)

    denom = s_r[:, None] + s_r[None, :] + eps
    tmp = tmp / denom

    grad_A += 2.0 * (U_r * s_r[None, :]) @ tmp @ Vt_r

    if n_start == grad_A.shape[0]:
        return (grad_A,)
    else:
        return (grad_A.T,)


svd.defvjp(_svd_fwd, _svd_bwd)
