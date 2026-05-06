import torch
import math


def _eigen_decomposition_forward(A):
    """
    Stable eigen-decomposition for symmetric matrices.

    Parameters
    ----------
    A : torch.Tensor
        Symmetric matrix of shape (..., n, n).

    Returns
    -------
    w : torch.Tensor
        Eigenvalues of A with shape (..., n), sorted in descending order.
    Q : torch.Tensor
        Eigenvectors of A with shape (..., n, n), corresponding to eigenvalues in w.
    """
    w, Q = torch.linalg.eigh(A.detach(), )

    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(w, dim=-1, descending=True)
    w = torch.gather(w, -1, idx)
    batch_dims = Q.shape[:-2]
    n = Q.shape[-1]
    idx_expanded = idx.unsqueeze(-2).expand(*batch_dims, n, n)
    Q = torch.gather(Q, -1, idx_expanded)

    return w, Q


def _make_K(lam, eps):
    lam_i = lam.unsqueeze(-1)
    lam_j = lam.unsqueeze(-2)
    diff = lam_i - lam_j + eps

    mask = ~torch.eye(diff.shape[-1], device=diff.device, dtype=torch.bool)
    K = torch.zeros_like(diff)
    K = (1.0 / diff) * mask

    return K

class SymmetricEigendecomposition(torch.autograd.Function):
    """
    Custom autograd function for eigenvalue decomposition of symmetric SPD matrices.
    
    Forward pass: Computes eigenvalues and eigenvectors of A where A = Q Λ Q^T
    Backward pass: Computes gradients w.r.t. input matrix A given gradients
                w.r.t. eigenvalues and eigenvectors.
    
    Args:
        A: Symmetric positive definite matrix of shape (..., n, n)
    
    Returns:
        eigenvalues: Tensor of shape (..., n) containing eigenvalues in descending order
        eigenvectors: Tensor of shape (..., n, n) containing eigenvectors as columns
    """
    
    @staticmethod
    def forward(ctx, A, eps=1e-8):
        """
        Forward pass: Eigendecomposition of symmetric SPD matrix.
        
        Args:
            ctx: Context object to save tensors for backward pass
            A: Symmetric SPD matrix of shape (..., n, n)
            eps: Small epsilon for numerical stability
        
        Returns:
            eigenvalues: Eigenvalues in descending order, shape (..., n)
            eigenvectors: Eigenvectors as columns, shape (..., n, n)        
        """
        A = 0.5 * (A + A.mT)  # Ensure symmetry
        eigenvalues, eigenvectors = _eigen_decomposition_forward(A)
        
        # Save for backward pass
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.eps = eps

        return eigenvalues, eigenvectors
    
    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        """
        Implements analytic backward for symmetric eigenvalue decomposition:
            A = V diag(lambda) V^T
        """
        eigenvalues, eigenvectors = ctx.saved_tensors
        eps = ctx.eps

        V = eigenvectors
        lam = eigenvalues
        dL_dlam = grad_eigenvalues
        dL_dV = grad_eigenvectors

        VT_dL_dV = V.mT @ dL_dV

        K = _make_K(lam, eps)
        K = K.mT
        

        offdiag_term = K * VT_dL_dV

        diag_term = torch.diag_embed(dL_dlam)
        grad_A = V @ (offdiag_term + diag_term) @ V.mT

        grad_A = 0.5 * (grad_A + grad_A.mT)

        return grad_A, None



def eigh_with_grad(A, eps=1e-8):
    """
    Convenience function for eigendecomposition of symmetric SPD matrix.
    
    Args:
        A: Symmetric positive definite matrix of shape (..., n, n)
        eps: Small epsilon for numerical stability (default: 1e-8)
        series_terms: Number of terms in geometric series expansion for close eigenvalues (default: 10).
                      Only used when series_threshold is None.
        series_threshold: Truncation error threshold for dynamic series computation (default: None).
                          If None, uses fixed series_terms. If not None, computes per-element number
                          of terms as ceil(log(series_threshold) / log(abs(r))) where r = lambda_i/lambda_j.
    
    Returns:
        eigenvalues: Tensor of shape (..., n) containing eigenvalues in descending order
        eigenvectors: Tensor of shape (..., n, n) containing eigenvectors as columns
    
    Example:
        >>> A = torch.randn(5, 5)
        >>> A = A @ A.T  # Make it symmetric positive definite
        >>> eigenvalues, eigenvectors = eigendecomposition(A)
        >>> # Verify: A ≈ eigenvectors @ diag(eigenvalues) @ eigenvectors.T
        >>> reconstructed = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.T
        >>> assert torch.allclose(A, reconstructed, atol=1e-5)
    """
    if not A.requires_grad:
        return _eigen_decomposition_forward(A)
    
    return SymmetricEigendecomposition.apply(A, eps)
