from dataclasses import dataclass
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from src_jax.cans_iteration import cans_iteration



@partial(jax.jit)
def _solve_lyapunov_sym(A, w, Q, eps):
    """
    Solve the symmetric Lyapunov equation

        H X + X H = A

    using the eigenvalue decomposition of H.

    Parameters
    ----------
    A : jnp.array
        Symmetric right-hand side matrix of shape (..., n, n).
    w : jnp.array
        Eigenvalues of H (reuse to avoid recomputation).
    Q : jnp.array
        Eigenvectors of H (reuse to avoid recomputation).
    tol : float, optional
        Threshold to avoid division by small values (w_i + w_j).

    Returns
    -------
    X : torch.Tensor
        Solution of the Lyapunov equation with shape (..., n, n).
    """

    Qt = Q.T
    A_tilde = Qt @ A @ Q

    denom = w[..., None] + w[..., None, :]
    X_tilde = A_tilde * jnp.where(jnp.abs(denom) < eps, 0.0, 1 / denom)

    X = Q @ X_tilde @ Qt
    return X


@partial(
    jax.custom_vjp,
    nondiff_argnames=("impl", "eps"),
)
def polar_decomposition_pseudo(
    A: jnp.ndarray,
    impl: str = "cans",
    eps: float = 1e-8,
):
    """
    Compute the polar decomposition of a matrix using the specified method.

    Args:
        A (jnp.array): The input matrix.
        impl (str): The implementation method to use ("cans" or "qdwh").

    Returns:
        W: jnp.array: The unitary factor.
        H: jnp.array: The Hermitian positive semi-definite factor.
    """
    Q = None
    M = A
    if A.shape[0] > 2 * A.shape[1]:
        Q, m = jnp.linalg.qr(A)
        M = m
        
    if impl == "cans":
        W = cans_iteration(M)
        H = W.T @ A
        H = (H + H.T.conj()) / 2
    elif impl == "qdwh":
        from jax.lax.linalg import qdwh
        W, H, _, _ = qdwh(M)
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    if Q is not None:
        W = Q @ W

    return W, H


def _polar_pseudo_fwd(A, impl="cans", eps=1e-8):
    W, H = polar_decomposition_pseudo(A, impl, eps)
    Q, w = jax.lax.linalg.eigh(H)
    return (W, H), (W, Q, w, A)


def _polar_pseudo_bwd(impl, eps, res, g):
    W, Q, w, A = res
    grad_W, grad_H = g

    diag = jnp.where(jnp.abs(w) < eps, 0.0, 1 / w)

    H_inv = Q @ (diag[..., None] * Q.mT)

    B = grad_H - W.mT @ grad_W @ H_inv
    B = (B + B.mT) / 2
    
    grad_A = jnp.zeros_like(A)
    grad_A += grad_W @ H_inv
    
    X = _solve_lyapunov_sym(B, w, Q, eps)
    
    grad_A += 2 * A @ X
    return (grad_A,)

polar_decomposition_pseudo.defvjp(_polar_pseudo_fwd, _polar_pseudo_bwd)
