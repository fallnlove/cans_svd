from functools import partial
from typing import List

import jax
import jax.numpy as jnp


from src_jax.cans_iteration import cans_iteration


@partial(
    jax.jit,
    static_argnames=[
        "max_iter",
        "degree",
        "preprocess",
        "preprocess_iters",
        "denom_type",
        "eigh_impl",
        "tol",
    ],
)
def cans_svd(
    matrix: jnp.array,
    degree: int = 3,
    preprocess: bool = True,
    preprocess_iters: int = 2,
    delta: float = 0.99,
    a: float = 1e-3,
    max_iter: int = 50,
    denom_type="one_inf",
    eigh_impl=None,
    tol=1e-5,
) -> List[jnp.array]:
    """
    Compute the SVD of a matrix using the CANS method.

    Args:
        matrix (jnp.array): The input matrix.
        degree (int): The degree of the polynomial approximation.
        preprocess (bool): Whether to use preprocessing.
        preprocess_iters (int): Number of preprocessing iterations.
        delta (float): The delta parameter for CANS.
        a (float): The scaling parameter for CANS.
        num_iters (int): Number of iterations for the CANS method.
        denom_type (str): Denominator type: "one_inf" or "frob".
        eigh_impl (EighImplementation): Algorithm for finding eigh in JAX.
        tol (float): Tolerance for CANS algorithm.

    Returns:
        U: jnp.array: Left singular vectors.
        S: jnp.array: Singular values.
        Vt: jnp.array: Right singular vectors (transposed).
    """
    n_start = matrix.shape[0]
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    W = cans_iteration(
        matrix,
        max_iter=max_iter,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
        denom_type=denom_type,
        tol=tol,
    )

    H = W.T @ matrix
    V, S = jax.lax.linalg.eigh(
        H,
        symmetrize_input=True,
        implementation=eigh_impl,
    )
    U = W @ V
    U, R = jax.lax.linalg.qr(
        U,
        full_matrices=False,
    )
    s = jnp.diag(R) * S

    V = jnp.where(s < 0, -V, V)
    s = jnp.abs(s)

    idx = jnp.argsort(s, descending=True)

    U = U.at[:, jnp.arange(s.shape[0])].set(U[:, idx])
    V = V[:, idx]
    s = s[idx]

    if n_start == matrix.shape[0]:
        return U, s, V.T 
    else:
        return V, s, U.T
