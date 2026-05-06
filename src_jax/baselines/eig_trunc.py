'''
Differentiable eigendecomposition using truncated gradients for backward pass (JAX).

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Eigendecomposition gradients with truncation to prevent overflow
'''
import jax
import jax.numpy as jnp
from functools import partial
from .utils import check_gradient_overflow


def truncated_gradients(s):
    """
    Compute truncated eigendecomposition gradients to prevent overflow.
    
    Args:
        s: Eigenvalues [batch_size, k] or [k]
        
    Returns:
        Truncated gradient matrix [batch_size, k, k] or [k, k]
    """
    # Compute 1 / (s_i - s_j) for i != j, 0 for i == j
    s_expanded = jnp.expand_dims(s, axis=-1)  # [..., k, 1]
    s_expanded_t = jnp.expand_dims(s, axis=-2)  # [..., 1, k]
    p = (1 / (s_expanded - s_expanded_t))
    
    # Make anti-symmetric: take upper triangular and subtract transpose
    p = jnp.triu(p, k=1) - jnp.triu(p, k=1).swapaxes(-2, -1)
    
    # Truncate to prevent overflow
    p = jnp.clip(p, min=-1e10, max=1e10)
    
    return p


def _eigh_forward(A):
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
    
    # Ensure eigenvalues are positive and above epsilon
    eps = jnp.finfo(w.dtype).eps
    w = jnp.clip(w, min=eps)
    
    return V, w


@partial(jax.custom_vjp)
def eig_trunc(A):
    """
    Differentiable eigendecomposition using truncated gradients.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses truncated gradients to prevent overflow
    
    Note: Input matrix is assumed to be symmetric.
    
    Args:
        A: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
        
    Returns:
        U: Eigenvectors [batch_size, n, k] or [n, k]
        S: Eigenvalues [batch_size, k] or [k]
    """
    return _eigh_forward(A)


def _eig_trunc_fwd(A):
    U, S = _eigh_forward(A)
    return (U, S), (U, S, A)


def _eig_trunc_bwd(res, g):
    U, S, A = res
    g_U, g_S = g
    
    # Compute truncated gradients
    K = truncated_gradients(S)  # [batch_size, k, k]
    
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


eig_trunc.defvjp(_eig_trunc_fwd, _eig_trunc_bwd)
