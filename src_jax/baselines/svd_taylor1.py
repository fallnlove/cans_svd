from functools import partial
from typing import List

import jax
import jax.numpy as jnp


@jax.jit
def F_taylor(s):
    s = jnp.square(s)
    s = jnp.where(s<1e-8, 0, s)
    I = jnp.eye(s.shape[-1], dtype=s.dtype)
    p = s[:, None] / s[None, :] - I
    p = jnp.where(jnp.isfinite(p), p, 0)
    p = jnp.where(p < 1., p, 1. / p)
    a1 = jnp.tile(s[:, None], (s.shape[-1], )).mT
    a1_t = a1.mT
    a1 = 1. / jnp.where(a1 >= a1_t, a1, - a1_t)
    a1 *= jnp.ones_like(a1) - I
    a1 = jnp.where(jnp.isfinite(a1), a1, 0)
    p_app = jnp.ones_like(p)
    p_hat = jnp.ones_like(p)
    for _ in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


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
def svd_taylor(
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


def _svd_taylor_fwd(
    A: jnp.array,
    polar_method: str="cans",
    cans_tol: float=1e-5,
    eps_qr: float=1e-5,
):
    U, s, Vt = svd_taylor(A, polar_method, cans_tol, eps_qr)
    return (U, s, Vt), (U, s, Vt, A)


def _svd_taylor_bwd(polar_method, cans_tol, eps_qr, res, g):
    U, s, Vt, A = res
    grad_U, grad_s, grad_Vt = g
    n_start = U.shape[0]
    if U.shape[0] < Vt.shape[1]:
        U, Vt = Vt.T, U.T
        grad_U, grad_Vt = grad_Vt.T, grad_U.T
    s = s + 1e-8
    s_inv = jnp.diag(jnp.where(s > 0, 1 / s, 0))
    utdu = U.mT @ grad_U
    vtdv = Vt @ grad_Vt.T
    ################### the only diff #######################
    F = F_taylor(s)
    Fmat_u = F * (utdu-utdu.mT)
    Fmat_v = F * (vtdv-vtdv.mT)
    #########################################################
    c_u1 = Fmat_u @ jnp.diag(s)
    c_u1 = U @ c_u1
    Im = jnp.eye(U.shape[-2], dtype=U.dtype)
    c_u2 = Im - U @ U.mT
    c_u2 = c_u2 @ grad_U @ s_inv
    c_u = (c_u1 + c_u2) @ Vt
    Ik = jnp.eye(s.shape[-1], dtype=s.dtype)
    c_s = U @ (jnp.diag(grad_s)) @ Vt
    c_v1 = jnp.diag(s) @ Fmat_v @ Vt
    In = jnp.eye(Vt.shape[-1], dtype=Vt.dtype)
    c_v2 = In - Vt.T @ Vt
    c_v2 = s_inv @ grad_Vt @ c_v2
    c_v = U @ (c_v1 + c_v2)
    
    grad_A = c_u + c_s + c_v

    if n_start == grad_A.shape[0]:
        return (grad_A,)
    else:
        return (grad_A.T,)


svd_taylor.defvjp(_svd_taylor_fwd, _svd_taylor_bwd)
