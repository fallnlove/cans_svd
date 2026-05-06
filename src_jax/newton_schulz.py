import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(
    jax.jit,
    static_argnames=[
        "num_iters",
        "tol",
    ],
)
def newton_schulz(A, num_iters=50, tol=1e-6):
    """
    Computes the polar factor using the Newton-Schulz iteration.

    Args:
        A: Input matrix of shape (m, n).
        num_iters: Maximum number of iterations for the Newton-Schulz method.
        tol: Tolerance for convergence. The iteration will stop when the Frobenius norm of (X2 - I) is less than tol.
    Returns:
        X: Approximation of the polar factor of A.
        cnt: Number of iterations taken to converge.
    """
    if A.shape[0] < A.shape[1]:
        A = A.T
    one_norm = jnp.linalg.norm(A, ord=1)
    inf_norm = jnp.linalg.norm(A, ord=np.inf)
    alpha_inverse = jax.lax.rsqrt(one_norm) * jax.lax.rsqrt(inf_norm)
    alpha_inverse = jnp.where(one_norm == 0, 1, alpha_inverse)
    A = A * alpha_inverse

    X2 = A.T @ A

    I = jnp.eye(A.shape[1], dtype=A.dtype)

    def body_fun(val):
        A, X2, cnt = val
        A_new = 3 / 2 * A - 1 / 2 * A @ X2
        X2_new = A_new.T @ A_new
        cnt += 1
        return A_new, X2_new, cnt
    
    def cond_fun(val):
        A, X2, cnt = val
        err = jnp.linalg.norm(X2 - I, ord='fro') / jnp.linalg.norm(I, ord='fro')
        return jnp.logical_and(err > tol, cnt < num_iters)

    cnt = 0
    A, X2, cnt = jax.lax.while_loop(
        cond_fun, body_fun, (A, X2, cnt)
    )

    return A, cnt
