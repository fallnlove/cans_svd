'''
Differentiable eigendecomposition using ordinary gradients for backward pass (JAX).

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Ordinary eigendecomposition gradients with overflow check
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
    p = 1 / (s_expanded - s_expanded_t)
    
    # Make anti-symmetric: take upper triangular and subtract transpose
    p = jnp.triu(p, k=1) - jnp.triu(p, k=1).swapaxes(-2, -1)
    
    return p


def _eigh_forward(A):
    """
    Eigen-decomposition for symmetric matrix A.
    Eigenvalues are returned in descending order.
    """

    # For symmetric matrices, eigh gives eigendecomposition
    # Returns eigenvalues in ascending order, so we reverse

    *batch_size, n, m = A.shape
    if n != m:
        raise ValueError("Eigendecomposition requires square matrices. Got shape {}".format(A.shape))
    
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
def eig_original(A):
    """
    Differentiable eigendecomposition using ordinary gradients.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses ordinary eigendecomposition gradients with overflow checks
    
    Note: Input matrix is assumed to be symmetric.
    
    Args:
        A: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
        
    Returns:
        U: Eigenvectors [batch_size, n, k] or [n, k]
        S: Eigenvalues [batch_size, k] or [k]
    """
    return _eigh_forward(A)


def _eig_original_fwd(A):
    U, S = _eigh_forward(A)
    return (U, S), (U, S, A)


def _eig_original_bwd(res, g):
    U, S, A = res
    g_U, g_S = g
    
    
    # Compute ordinary gradients
    K = ordinary_gradients(S)  # [batch_size, k, k]
    K = check_gradient_overflow(K)
    
    # For eigendecomposition: A = U * diag(S) * U^T
    # Gradient w.r.t. input: U * (K^T ⊙ (U^T * grad_U) + diag(grad_S)) * U^T
    VT_gU = U.swapaxes(-2, -1) @ g_U  # [batch_size, k, k]
    grad_U_component = K.swapaxes(-2, -1) * VT_gU  # [batch_size, k, k]

    # Batched case: [batch_size, k] -> [batch_size, k, k]
    grad_S_component = jnp.diag(g_S)

    # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
    grad_intermediate = grad_U_component + grad_S_component
    grad_input = U @ grad_intermediate @ U.swapaxes(-2, -1)
    grad_input = check_gradient_overflow(grad_input)
    
    return (grad_input,)


eig_original.defvjp(_eig_original_fwd, _eig_original_bwd)
