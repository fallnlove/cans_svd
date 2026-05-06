from functools import partial
from typing import List

import jax
import jax.numpy as jnp

def _make_K(lam, eps=0):
    """
    K_ij = 1 / (λ_i - λ_j), i ≠ j
    K_ii = 0
    """

    s_expanded = jnp.expand_dims(lam, axis=-1)  # [..., k, 1]
    s_expanded_t = jnp.expand_dims(lam, axis=-2)  # [..., 1, k]
    K = 1 / (s_expanded - s_expanded_t + eps)
    
    # Make anti-symmetric: take upper triangular and subtract transpose
    K = jnp.triu(K, k=1) - jnp.triu(K, k=1).swapaxes(-2, -1)
    
    return K


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "polar_method",
        "cans_tol",
        "eps_qr",
    ),
)
@partial(
    jax.jit,
    static_argnames=[
        "polar_method",
        "cans_tol",
        "eps_qr",
    ],
)
def svd_townsend(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    eps_qr: float=1e-5,
) -> List[jnp.array]:

    def perform_qr(U, s):
        U, R = jax.lax.linalg.qr(U, full_matrices=False)
        s = jnp.diag(R) * s
        return U, s    

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


def _svd_townsend_fwd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    eps_qr: float=1e-5,
):
    U, s, Vt = svd_townsend(A, polar_method, cans_tol, eps_qr)
    return (U, s, Vt), (U, s, Vt, A)


def _svd_townsend_bwd(polar_method, cans_tol, eps_qr, res, g):
    U, s, Vt, A = res
    grad_U, grad_s, grad_Vt = g
    n_start = U.shape[0]
    if U.shape[0] < Vt.shape[1]:
        U, Vt = Vt.T, U.T
        grad_U, grad_Vt = grad_Vt.T, grad_U.T
    F = _make_K(s**2, 0).mT

    ut_grad_U = U.mT @ grad_U
    u_ut_grad_U = U @ ut_grad_U
    vt_grad_V = Vt @ grad_Vt.mT
    grad_Vt_v_vt= vt_grad_V.mT @ Vt

    grad_A = (U @ (F * (ut_grad_U - ut_grad_U.mT)) * s[None, :] + \
                 (grad_U - u_ut_grad_U) * (1 / s[None, :]) + \
                    U * grad_s[None, :]) @ Vt + \
                    U @ (
                        s[:, None] * (F * (vt_grad_V - vt_grad_V.mT)) @ Vt + \
                        (1 / s[:, None]) * (grad_Vt - grad_Vt_v_vt)
                    )

    if n_start == grad_A.shape[0]:
        return (grad_A,)
    else:
        return (grad_A.T,)

svd_townsend.defvjp(_svd_townsend_fwd, _svd_townsend_bwd)
