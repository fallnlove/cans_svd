from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from jax.lax.linalg import qdwh


@partial(
    jax.jit,
    static_argnames=[
        "eigh_impl",
    ],
)
def qdwh_svd(
    matrix: jnp.array,
    eigh_impl=None,
    
) -> List[jnp.array]:
    """
    Compute the SVD of a matrix using the QDWH method.

    Args:
        matrix (jnp.array): The input matrix.
        eigh_impl (EighImplementation): Algorithm for finding eigh in JAX.

    Returns:
        U: jnp.array: Left singular vectors.
        S: jnp.array: Singular values.
        Vt: jnp.array: Right singular vectors (transposed).
    """
    n_start = matrix.shape[0]
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    W, H, _, _ = qdwh(matrix)

    V, S = jax.lax.linalg.eigh(
        H,
        symmetrize_input=False,
        implementation=eigh_impl,
    )
    U = W @ V
    U, R = jax.lax.linalg.qr(U)
    s = jnp.diag(R) * S

    V = jnp.where(s < 0, -V, V)
    s = jnp.abs(s)

    idx = jnp.argsort(s, descending=True)

    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    if n_start == matrix.shape[0]:
        return U, s, V.T 
    else:
        return V, s, U.T
