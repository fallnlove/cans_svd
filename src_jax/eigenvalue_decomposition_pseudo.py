import jax
import jax.numpy as jnp
from functools import partial

def _eigh_forward(A, symmetrize_input=True):
    """
    Eigen-decomposition for a single symmetric matrix A.
    Eigenvalues are returned in descending order.
    """
    V, w = jax.lax.linalg.eigh(A, symmetrize_input=symmetrize_input, sort_eigenvalues=True)

    idx = jnp.arange(w.shape[0] - 1, -1, -1)
    w = w[..., idx]
    V = V[..., :, idx]

    return V, w


def _make_K(lam, eps):
    """
    K_ij = 1 / (λ_i - λ_j), i ≠ j
    K_ii = 0
    """

    s_expanded = jnp.expand_dims(lam, axis=-1)  # [..., k, 1]
    s_expanded_t = jnp.expand_dims(lam, axis=-2)  # [..., 1, k]
    K = jnp.where(jnp.abs(s_expanded - s_expanded_t) < eps, 0.0, 1 / (s_expanded - s_expanded_t))
    
    # Make anti-symmetric: take upper triangular and subtract transpose
    K = jnp.triu(K, k=1) - jnp.triu(K, k=1).swapaxes(-2, -1)
    
    return K


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "eps",
    ),
)
def symmetric_eigh_pseudo(A, eps=1e-8):
    """
    Symmetric eigendecomposition with analytic backward.

    Args
    ----
    A : (n, n) symmetric matrix
    eps : eigenvalue gap regularization

    Returns
    -------
    w : (n,)   eigenvalues (descending)
    V : (n, n) eigenvectors
    """
    return _eigh_forward(A, symmetrize_input=True)


def _eigh_pseudo_fwd(A, eps):
    V, w = symmetric_eigh_pseudo(A, eps)
    return (V, w), (V, w)


def _eigh_pseudo_bwd(eps, res, g):
    V, w = res
    g_V, g_w = g

    K = _make_K(w, eps)
    grad_A = V @ (K.mT * (1/2 * (V.T @ g_V - (V.T @ g_V).T)) + jnp.diag(g_w)) @ V.mT
    grad_A = 0.5 * (grad_A + grad_A.swapaxes(-2, -1))

    return (grad_A,)

symmetric_eigh_pseudo.defvjp(_eigh_pseudo_fwd, _eigh_pseudo_bwd)
