'''
Differentiable eigendecomposition using Power Iteration (PI) for backward pass (JAX).

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Power Iteration to approximate gradients
'''
import jax
import jax.numpy as jnp
from functools import partial


def _eigh_forward(A):
    """
    Eigen-decomposition for symmetric matrix A.
    Eigenvalues are returned in descending order.
    """
    
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
def eig_pi(A):
    """
    Differentiable eigendecomposition using Power Iteration.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses Power Iteration to approximate gradients
    
    Note: Input matrix is assumed to be symmetric.
    
    Args:
        A: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
        
    Returns:
        U: Eigenvectors [batch_size, n, k] or [n, k]
        S: Eigenvalues [batch_size, k] or [k]
    """
    return _eigh_forward(A)


def _eig_pi_fwd(A):
    U, S = _eigh_forward(A)
    return (U, S), (U, S, A)


def _eig_pi_bwd(res, g):
    U, S, A = res
    g_U, g_S = g
    
    *batch_size, n, k = U.shape
    dtype = A.dtype
    
    # Power Iteration method for gradient computation
    # For eigendecomposition: A = U * diag(S) * U^T
    # Compute gradient using Power Iteration for grad_U component
    grad_U_component = jnp.zeros((*batch_size, k, k), dtype=dtype)
    
    if g_U is not None:
        # Power Iteration approach for grad_U
        I = jnp.eye(n, dtype=dtype)
        I = jnp.broadcast_to(I, (*batch_size, n, n))
        
        # num = I - U @ U^T (approximate zero matrix or orthoprojector)
        num = I - U @ U.swapaxes(-2, -1)  # [batch_size, n, n]
        denom = jnp.linalg.norm(A @ U, axis=(-1, -2), keepdims=True)
        denom = jnp.clip(denom, min=1e-10)
        
        ak = num / denom  # [*batch_size, n, n]
        term1 = ak
        q = A / denom  # [*batch_size, n, n]
        
        # Power Iteration to compute gradient
        for _ in range(20):
            ak = q @ ak
            term1 = term1 + ak
        
        grad_U_proj = term1 @ g_U  # [*batch_size, n, k]
        grad_U_component = U.swapaxes(-2, -1) @ grad_U_proj  # [*batch_size, k, k]
    
    # Create diagonal matrix from g_S
    grad_S_component = jnp.diag(g_S)

    # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
    grad_intermediate = grad_U_component + grad_S_component
    grad_input = U @ grad_intermediate @ U.swapaxes(-2, -1)
    
    return (grad_input,)


eig_pi.defvjp(_eig_pi_fwd, _eig_pi_bwd)
