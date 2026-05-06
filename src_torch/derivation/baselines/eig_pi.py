'''
Differentiable eigendecomposition using Power Iteration (PI) for backward pass.

Forward Pass: Eigendecomposition returning U, S (eigenvectors and eigenvalues)
Backward Pass: Power Iteration to approximate gradients
'''
import torch
from torch.autograd import Function


class EIG_PI(Function):
    """
    Differentiable eigendecomposition using Power Iteration.
    
    Forward: Computes eigendecomposition of symmetric input matrix, returns U, S
    Backward: Uses Power Iteration to approximate gradients
    
    Note: Input matrix is assumed to be symmetric.
    
    Usage:
        U, S = EIG_PI.apply(symmetric_matrix)
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
        Backward pass using Power Iteration.
        
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
        device = input.device
        dtype = input.dtype
        
        # Power Iteration method for gradient computation
        # For eigendecomposition: A = U * diag(S) * U^T
        # Compute gradient using Power Iteration for grad_U component
        grad_U_component = torch.zeros(batch_size, n, k, device=device, dtype=dtype)
        if grad_U is not None:
            # Power Iteration approach for grad_U
            I = torch.eye(n, device=device, dtype=dtype).view(1, n, n).repeat(batch_size, 1, 1)
            num = I - torch.bmm(U, U.transpose(-2, -1)) # appx zero matrix or orthoprojector in case U is not square or not orthonormal
            denom = torch.norm(torch.bmm(input, U), dim=(1, 2), keepdim=True).clamp(min=1e-10)
            ak = torch.div(num, denom) # smth like a zero matrix?
            term1 = ak.clone()
            q = torch.div(input, denom)
            # Power Iteration to compute gradient
            for _ in range(20):
                ak = torch.bmm(q, ak)
                term1 += ak
            grad_U_proj = torch.bmm(term1, grad_U)  # [batch_size, n, k]
            grad_U_component = torch.bmm(U.transpose(-2, -1), grad_U_proj)  # [batch_size, k, k]
        
        grad_S_component = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
        if grad_S is not None:
            grad_S_component = torch.diag_embed(grad_S)  # [batch_size, k, k]
        
        # Project back to input space: U * (grad_U_component + grad_S_component) * U^T
        grad_input = torch.bmm(torch.bmm(U, grad_U_component + grad_S_component), U.transpose(-2, -1))
        
        if not is_batched:
            grad_input = grad_input.squeeze(0)
        
        return grad_input
