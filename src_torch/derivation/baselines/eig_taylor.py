'''
Differentiable eigendecomposition using Taylor polynomial for backward pass.

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Taylor polynomial to approximate smooth gradients
'''
import torch
from torch.autograd import Function


def check_gradient_overflow(grad_input):
    """Check and fix gradient overflow issues."""
    grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
    grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
    grad_input[grad_input != grad_input] = 0  # Fix NaN values
    return grad_input


def taylor_polynomial(s):
    """
    Taylor polynomial to approximate eigendecomposition gradients.
    
    Args:
        s: Eigenvalues [batch_size, k]
        
    Returns:
        Approximated gradient matrix [batch_size, k, k]
    """
    batch_size, k = s.shape
    dtype = s.dtype
    device = s.device
    
    # Create matrix of s_i / s_j - 1
    s_expanded = s.unsqueeze(-1)  # [batch_size, k, 1]
    s_expanded_t = s.unsqueeze(-2)  # [batch_size, 1, k]
    I = torch.eye(k, device=device, dtype=dtype).view(1, k, k).repeat(batch_size, 1, 1)
    p = s_expanded / s_expanded_t - I
    p = torch.where(p < 1., p, 1. / p)
    
    # Compute a1
    a1 = s.view(batch_size, k, 1).repeat(1, 1, k)
    a1_t = a1.transpose(1, 2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, -a1_t)
    a1 *= (torch.ones(k, k, device=device, dtype=dtype).view(1, k, k).repeat(batch_size, 1, 1) - I)
    
    # Compute Taylor polynomial approximation
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


class EIG_Taylor(Function):
    """
    Differentiable eigendecomposition using Taylor polynomial.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses Taylor polynomial to compute smooth gradients
    
    Note: Input matrix is assumed to be symmetric.
    
    Usage:
        U, S = EIG_Taylor.apply(symmetric_matrix)
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: Eigendecomposition.
        
        Args:
            input: Symmetric input matrix [batch_size, n, n] or [n, n] (assumed to be symmetric)
            
        Returns:
            U: Eigenvectors [batch_size, n, k] or [n, k]
            S: Eigenvalues [batch_size, k] or [k]
        """
        # Handle both batched and unbatched inputs
        is_batched = input.dim() == 3
        if not is_batched:
            input = input.unsqueeze(0)
        
        batch_size, n, m = input.shape
        if n != m:
            raise ValueError("Eigendecomposition requires square matrices. Got shape {}".format(input.shape))
        
        dtype = input.dtype
        device = input.device
        
        # Move to CPU for SVD (used for eigendecomposition of symmetric matrices)
        # Input is assumed to be symmetric
        input_cpu = input.cpu()
        
        # For symmetric matrices, SVD gives eigendecomposition
        _, S, U = torch.svd(input_cpu, some=True, compute_uv=True)
        
        # Move back to original device
        U = U.to(device)
        S = S.to(device)
        
        # Ensure eigenvalues are positive and above epsilon
        eps = torch.finfo(dtype).eps
        S = torch.clamp(S, min=eps)
        
        # Save for backward
        ctx.save_for_backward(U, S, input)
        ctx.is_batched = is_batched
        
        if not is_batched:
            U = U.squeeze(0)
            S = S.squeeze(0)
        
        return U, S

    @staticmethod
    def backward(ctx, grad_U, grad_S):
        """
        Backward pass using Taylor polynomial.
        
        Args:
            grad_U: Gradient w.r.t. eigenvectors [batch_size, n, k] or [n, k]
            grad_S: Gradient w.r.t. eigenvalues [batch_size, k] or [k]
            
        Returns:
            Gradient w.r.t. input matrix
        """
        U, S, input = ctx.saved_tensors
        is_batched = ctx.is_batched
        
        # Handle unbatched case
        if not is_batched:
            U = U.unsqueeze(0)
            S = S.unsqueeze(0)
            grad_U = grad_U.unsqueeze(0) if grad_U is not None else None
            grad_S = grad_S.unsqueeze(0) if grad_S is not None else None
            input = input.unsqueeze(0)
        
        batch_size, n, k = U.shape
        dtype = U.dtype
        
        # Compute Taylor polynomial for gradient computation
        K = taylor_polynomial(S)  # [batch_size, k, k]
        K = check_gradient_overflow(K)
        
        # For eigendecomposition: A = U * diag(S) * U^T
        # Gradient w.r.t. input: U * (K^T ⊙ (U^T * grad_U) + diag(grad_S)) * U^T
        grad_U_component = torch.zeros(batch_size, n, k, device=U.device, dtype=dtype)
        if grad_U is not None:
            grad_U_component = K.transpose(-2, -1) * torch.bmm(U.transpose(-2, -1), grad_U)  # [batch_size, k, k]
        
        grad_S_component = torch.zeros(batch_size, k, k, device=U.device, dtype=dtype)
        if grad_S is not None:
            grad_S_component = torch.diag_embed(grad_S)  # [batch_size, k, k]
        
        # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
        grad_input = torch.bmm(torch.bmm(U, grad_U_component + grad_S_component), U.transpose(-2, -1))
        grad_input = check_gradient_overflow(grad_input)
        
        if not is_batched:
            grad_input = grad_input.squeeze(0)
        
        return grad_input
