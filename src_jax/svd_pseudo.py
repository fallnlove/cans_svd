from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from src_jax.eigenvalue_decomposition_pseudo import symmetric_eigh_pseudo
from src_jax.polar_decomposition_pseudo import polar_decomposition_pseudo

from src_jax.eigenvalue_decomposition_pseudo import _eigh_pseudo_bwd
from src_jax.polar_decomposition_pseudo import _polar_pseudo_bwd


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
def svd_pseudo(
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


def _svd_pseudo_fwd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    threshold: float=1e-8,
    eps: float=1e-8,
    eps_qr: float=1e-5,
):
    U, s, Vt = svd_pseudo(A, polar_method, cans_tol, threshold, eps, eps_qr)
    return (U, s, Vt), (U, s, Vt, A)


@partial(
    jax.jit,
    static_argnames=[
        "polar_method",
        "threshold",
        "eps",
    ],
)
def svd_pseudo_grad(
    A: jnp.array,
    U: jnp.array,
    s: jnp.array,
    Vt: jnp.array,
    grad_U: jnp.array,
    grad_s: jnp.array,
    grad_Vt: jnp.array,
    polar_method: str="cans",
    threshold: float=1e-8,
    eps: float=1e-8,
) -> jnp.array:
    """
    Compute dL/dA for svd_pseudo from output cotangents.

    This expands the backward pass through
        U = W @ V, s = eigvals(H), Vt = V.T, (W, H) = polar(A).
    """
    A_internal = A
    transposed = A.shape[0] < A.shape[1]
    if transposed:
        A_internal = A.T
        U, Vt = Vt.T, U.T
        grad_U, grad_Vt = grad_Vt.T, grad_U.T

    mask = (s > threshold).astype(s.dtype)
    U = U * mask[None, :]
    Vt = Vt * mask[:, None]
    s = s * mask
    grad_U = grad_U * mask[None, :]
    grad_s = grad_s * mask
    grad_Vt = grad_Vt * mask[:, None]

    V = Vt.T
    W = U @ Vt
    H = V @ jnp.diag(s) @ Vt

    grad_W = grad_U @ Vt
    grad_V_from_U = W.T @ grad_U
    grad_V_from_Vt = grad_Vt.T
    grad_V = grad_V_from_U + grad_V_from_Vt

    grad_H = _eigh_pseudo_bwd(eps, (V, s), (grad_V, grad_s))[0]
    grad_A = _polar_pseudo_bwd(polar_method, eps, (W, V, s, A_internal), (grad_W, grad_H))[0]

    if transposed:
        return grad_A.T
    return grad_A


def _svd_pseudo_bwd(polar_method, *args):
    if len(args) == 6:
        cans_tol, threshold, eps, eps_qr, res, g = args
    elif len(args) == 4:
        cans_tol, eps, res, g = args
        threshold = 1e-8
        eps_qr = 1e-5
    else:
        raise TypeError(
            "_svd_pseudo_bwd expects either "
            "(polar_method, cans_tol, threshold, eps, eps_qr, res, g) or "
            "(polar_method, cans_tol, eps, res, g)"
        )

    U, s, Vt, A = res
    grad_U, grad_s, grad_Vt = g
    grad_A = svd_pseudo_grad(
        A,
        U,
        s,
        Vt,
        grad_U,
        grad_s,
        grad_Vt,
        polar_method=polar_method,
        threshold=threshold,
        eps=eps,
    )
    return (grad_A,)


svd_pseudo.defvjp(_svd_pseudo_fwd, _svd_pseudo_bwd)
