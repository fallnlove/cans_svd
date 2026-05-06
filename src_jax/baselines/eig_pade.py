'''
Differentiable eigendecomposition using Padé approximants for backward pass (JAX).

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Padé approximants to closely approximate gradients
'''
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from mpmath import *
from .utils import check_gradient_overflow

# Initialize Padé coefficients
mp.dps = 20
one = mpf(1)
mp.pretty = True


def f(x):
    return one / (one - x)


a = taylor(f, 0, 100)
# [50, 50] Padé coefficients of the geometric series
pade_p, pade_q = pade(a, 50, 50)
pade_p = jnp.array(np.array(pade_p).astype(float))
pade_q = jnp.array(np.array(pade_q).astype(float))


def pade_gradients(s):
    """
    Compute Padé approximants for gradient computation.
    
    Args:
        s: Eigenvalues [batch_size, k] or [k]
        
    Returns:
        Approximated gradient matrix [batch_size, k, k] or [k, k]
    """
    # Create matrix of s_i / s_j - 1
    s_expanded = jnp.expand_dims(s, axis=-1)  # [..., k, 1]
    s_expanded_t = jnp.expand_dims(s, axis=-2)  # [..., 1, k]
    I = jnp.eye(s.shape[-1], dtype=s.dtype)
    if s.ndim > 1:
        I = jnp.expand_dims(I, axis=0)
        I = jnp.broadcast_to(I, (*s.shape[:-1], s.shape[-1], s.shape[-1]))
    
    p = s_expanded / s_expanded_t - I
    p = jnp.where(p < 1.0, p, 1.0 / p)
    
    # Compute a1
    a1 = jnp.expand_dims(s, axis=-1)  # [..., k, 1]
    a1 = jnp.broadcast_to(a1, (*s.shape, s.shape[-1]))  # [..., k, k]
    a1_t = a1.swapaxes(-2, -1)
    a1 = 1.0 / jnp.where(a1 >= a1_t, a1, -a1_t)
    a1 = a1 * (1.0 - I)
    
    # Move Padé coefficients to the same device/dtype as input
    device_p = pade_p.astype(s.dtype)
    device_q = pade_q.astype(s.dtype)
    
    # Compute Padé approximants
    p_app = jnp.ones_like(p) * device_p[0]
    q_app = jnp.ones_like(p) * device_q[0]
    p_hat = jnp.ones_like(p)
    for i in range(50):
        p_hat = p_hat * p
        p_app = p_app + device_p[i + 1] * p_hat
        q_app = q_app + device_q[i + 1] * p_hat
    a1 = a1 * p_app / q_app  # rational approximation
    return a1


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
def eig_pade(A):
    """
    Differentiable eigendecomposition using Padé approximants.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses Padé approximants to compute smooth gradients
    
    Note: Input matrix is assumed to be symmetric.
    
    Args:
        A: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
        
    Returns:
        U: Eigenvectors [batch_size, n, k] or [n, k]
        S: Eigenvalues [batch_size, k] or [k]
    """
    return _eigh_forward(A)


def _eig_pade_fwd(A):
    U, S = _eigh_forward(A)
    return (U, S), (U, S, A)


def _eig_pade_bwd(res, g):
    U, S, A = res
    g_U, g_S = g
    
    # Compute Padé approximants for gradient computation
    # K approximates 1/(s_i - s_j) for i != j
    K = pade_gradients(S)  # [batch_size, k, k]
    K = check_gradient_overflow(K)
    
    # For eigendecomposition: A = U * diag(S) * U^T
    # Gradient w.r.t. input: U * (K^T ⊙ (U^T * grad_U) + diag(grad_S)) * U^T
    VT_gU = U.swapaxes(-2, -1) @ g_U  # [batch_size, k, k]
    grad_U_component = K.swapaxes(-2, -1) * VT_gU  # [batch_size, k, k]
    
    grad_S_component = jnp.diag(g_S)

    # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
    grad_intermediate = grad_U_component + grad_S_component
    grad_input = U @ grad_intermediate @ U.swapaxes(-2, -1)
    grad_input = check_gradient_overflow(grad_input)
    
    return (grad_input,)


eig_pade.defvjp(_eig_pade_fwd, _eig_pade_bwd)
