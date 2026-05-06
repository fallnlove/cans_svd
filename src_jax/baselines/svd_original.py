from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from .eig_original import ordinary_gradients
from .utils import check_gradient_overflow



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
def svd_original(
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


def _svd_fwd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    eps_qr: float=1e-5,
):
    U, s, Vt = svd_original(A, polar_method, cans_tol, eps_qr)

    return (U, s, Vt), (U, s, Vt)


def _svd_bwd(polar_method, cans_tol, eps_qr, res, g):
    U, s, Vt = res
    grad_U, grad_s, grad_Vt = g

    n_start = U.shape[0]
    if U.shape[0] < Vt.shape[1]:
        U, Vt = Vt.T, U.T
        grad_U, grad_Vt = grad_Vt.T, grad_U.T

    s_squared = s ** 2
    F = ordinary_gradients(s_squared).mT
    F = check_gradient_overflow(F)

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
    
    grad_A = check_gradient_overflow(grad_A)
    
    if n_start == grad_A.shape[0]:
        return (grad_A,)
    else:
        return (grad_A.T,)


svd_original.defvjp(_svd_fwd, _svd_bwd)
