'''
Differentiable eigendecomposition using Top-N eigenvalues for backward pass (JAX).

Forward Pass: Truncated eigendecomposition (Top-N eigenvalues) returning U, S
Backward Pass: Ordinary eigendecomposition gradients
'''
import jax
import jax.numpy as jnp
from functools import partial
from .utils import check_gradient_overflow

def ordinary_gradients(s):
    """
    Compute ordinary eigendecomposition gradients.
    
    Args:
        s: Eigenvalues [batch_size, k] or [k]
        
    Returns:
        Gradient matrix [batch_size, k, k] or [k, k]
    """
    # Compute 1 / (s_i - s_j) for i != j, 0 for i == j
    s_expanded = jnp.expand_dims(s, axis=-1)  # [..., k, 1]
    s_expanded_t = jnp.expand_dims(s, axis=-2)  # [..., 1, k]
    p = (1 / (s_expanded - s_expanded_t))
    
    # Make anti-symmetric: take upper triangular and subtract transpose
    p = jnp.triu(p, k=1) - jnp.triu(p, k=1).swapaxes(-2, -1)
    
    return p


def _eigh_forward(A, top_n=None):
    """
    Eigen-decomposition for symmetric matrix A.
    Eigenvalues are returned in descending order.
    """
    # Handle both batched and unbatched inputs
    *batch_size, n, m = A.shape
    if n != m:
        raise ValueError("Eigendecomposition requires square matrices. Got shape {}".format(A.shape))
    
    # For symmetric matrices, eigh gives eigendecomposition
    V, w = jax.lax.linalg.eigh(A, symmetrize_input=True, sort_eigenvalues=True)
    
    # Reverse to get descending order
    idx = jnp.arange(w.shape[-1] - 1, -1, -1)
    w = w[..., idx]
    V = V[..., :, idx]
    
    # Keep only top-N eigenvalues if specified
    if top_n is not None and top_n < w.shape[-1]:
        k = w.shape[-1]
        eps = jnp.finfo(w.dtype).eps
        # Set bottom values to eps
        if w.ndim == 1:
            w = w.at[top_n:].set(eps)
        else:
            indices = jnp.arange(top_n, k)
            w = w.at[:, indices].set(eps)
    
    # Ensure eigenvalues are positive and above epsilon
    eps = jnp.finfo(w.dtype).eps
    w = jnp.clip(w, min=eps)
    
    return V, w


@partial(jax.custom_vjp, nondiff_argnames=('top_n',))
def eig_topn(A, top_n=50):
    """
    Differentiable eigendecomposition using Top-N eigenvalues.
    
    Forward: Computes truncated eigendecomposition keeping only top-N eigenvalues, returns U, S
    Backward: Uses ordinary eigendecomposition gradients
    
    Note: Input matrix is assumed to be symmetric.
    
    Args:
        A: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
        top_n: Number of top eigenvalues to keep (default: 50)
        
    Returns:
        U: Eigenvectors [batch_size, n, k] or [n, k]
        S: Eigenvalues [batch_size, k] or [k] (only top_n kept)
    """
    return _eigh_forward(A, top_n=top_n)


def _eig_topn_fwd(A, top_n):
    U, S = _eigh_forward(A, top_n=top_n)
    return (U, S), (U, S, A)


def _eig_topn_bwd(top_n, res, g):
    U, S, A = res
    g_U, g_S = g
    
    # Compute ordinary gradients
    K = ordinary_gradients(S)  # [batch_size, k, k]
    K = check_gradient_overflow(K)
    
    # For eigendecomposition: A = U * diag(S) * U^T
    # Gradient w.r.t. input: U * (K^T ⊙ (U^T * grad_U) + diag(grad_S)) * U^T
    VT_gU = U.swapaxes(-2, -1) @ g_U  # [batch_size, k, k]
    grad_U_component = K.swapaxes(-2, -1) * VT_gU  # [batch_size, k, k]
    
    # Create diagonal matrix from g_S
    grad_S_component = jnp.diag(g_S)

    # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
    grad_intermediate = grad_U_component + grad_S_component
    grad_input = U @ grad_intermediate @ U.swapaxes(-2, -1)
    grad_input = check_gradient_overflow(grad_input)
    
    return (grad_input,)


eig_topn.defvjp(_eig_topn_fwd, _eig_topn_bwd)
