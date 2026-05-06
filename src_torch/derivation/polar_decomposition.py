import torch
from src_torch.polar_decomposition.cans import polar_decomposition_cans
from src_torch.polar_decomposition.qdwh import polar_decomposition_qdwh
from dataclasses import dataclass
from typing import Optional


@dataclass
class LazySpectrum:
    """
    Container for eigen-decomposition of a symmetric matrix H.

    Attributes
    ----------
    w : torch.Tensor or None
        Eigenvalues of H.
    Q : torch.Tensor or None
        Eigenvectors of H.
    """
    w: Optional[torch.Tensor] = None
    Q: Optional[torch.Tensor] = None



def _solve_lyapunov_sym(w, Q, A):
    """
    Solve the symmetric Lyapunov equation

        H X + X H = A

    using the eigenvalue decomposition of H.

    Parameters
    ----------
    w : torch.Tensor
        Eigenvalues of H.
    Q : torch.Tensor
        Eigenvectors of H.
        Symmetric right-hand side matrix of shape (..., n, n).
    tol : float, optional
        Threshold to avoid division by small values (w_i + w_j).

    Returns
    -------
    X : torch.Tensor
        Solution of the Lyapunov equation with shape (..., n, n).
    """

    Qt = Q.transpose(-2, -1)
    A_tilde = Qt @ A @ Q

    denom = w.unsqueeze(-1) + w.unsqueeze(-2)

    X_tilde = A_tilde / denom

    X = Q @ X_tilde @ Qt
    return X


def _polar_batched(A, polar_func):
    """
    Apply polar decomposition to a single matrix or a batch of matrices.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (n, m) or batch of matrices (..., n, m).
    polar_func : callable
        Function implementing polar decomposition for a single matrix.

    Returns
    -------
    W : torch.Tensor
        Orthogonal factor(s) with the same leading batch dimensions as `A`.
    H : torch.Tensor
        Symmetric factor(s) with the same leading batch dimensions as `A`.
    """
    if A.ndim == 2:
        return polar_func(A.detach())

    Ws, Hs = zip(*(polar_func(Ab.detach()) for Ab in A))
    return torch.stack(Ws), torch.stack(Hs)


def _polar_decomposition_forward(A, method='cans', cans_tol=1e-6):
    """
    Forward pass for polar decomposition.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix or batch of matrices of shape (..., m, n).
    method : str, optional
        Polar decomposition method. Default is 'cans'.
    eps : float, optional
        Numerical regularization parameter.
    lazy_spec : LazySpectrum, optional
        Optional container holding eigenvalues and eigenvectors of H.

    Returns
    -------
    W : torch.Tensor
        Orthogonal (or semi-orthogonal) factor of the polar decomposition.
    H : torch.Tensor
        Symmetric positive semi-definite factor of the polar decomposition.
    """
    if method == 'cans':
        polar_func = lambda A: polar_decomposition_cans(A, tol=cans_tol)
    elif method == 'qdwh':
        polar_func = polar_decomposition_qdwh
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    W, H = _polar_batched(A, polar_func=polar_func)

    return W, H


class PolarDecomposition(torch.autograd.Function):
    """
    Custom autograd function for polar decomposition A = W H.

    Forward computes the polar factors W (orthogonal) and H (symmetric).
    Backward propagates gradients using a Lyapunov equation.
    """
    def forward(ctx, A, method='cans', eps=1e-8):
        W, H = _polar_decomposition_forward(A, method)

        ctx.save_for_backward(W, H, A.clone())
        ctx.eps = eps

        return W, H
    
    @staticmethod
    def backward(ctx, grad_W, grad_H):
        W, H, A = ctx.saved_tensors

        w, Q = torch.linalg.eigh(H)
        w = w + ctx.eps

        inv_w = 1.0 / w
        H_inv = Q @ (inv_w.unsqueeze(-1) * Q.mT)

        grad_A = torch.zeros_like(A)

        tmp = grad_W @ H_inv
        grad_A += tmp

        grad_H = grad_H - W.mT @ tmp
        grad_H = (grad_H + grad_H.mT) / 2

        X = _solve_lyapunov_sym(w, Q, grad_H)
        grad_A += 2 * A @ X

        return grad_A, None, None, None



def polar_decomposition_with_grad(
    A,
    method='cans',
    eps=1e-8,
):
    """
    Differentiable polar decomposition A = W H.

    Computes the polar decomposition of a matrix (or batch of matrices)
    with a custom backward pass.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix or batch of matrices of shape (..., m, n).
        Gradients are propagated w.r.t. A.
    method : str, optional
        Polar decomposition method. Default is 'cans'.
    eps : float, optional
        Numerical regularization parameter used in backward computations.
    lazy_spec : LazySpectrum, optional
        Optional container holding eigenvalues and eigenvectors of H.
        If provided, these values can be reused in the backward pass
        to avoid recomputing an eigendecomposition of H.

    Returns
    -------
    W : torch.Tensor
        Orthogonal (or semi-orthogonal) factor of the polar decomposition.
    H : torch.Tensor
        Symmetric positive semi-definite factor of the polar decomposition.

    Examples
    --------
    Basic usage without lazy spectrum reuse
    >>> import torch
    >>> torch.manual_seed(0)
    >>> n = 5
    >>> A = torch.randn(n, n, dtype=torch.float64, requires_grad=True)
    >>>
    >>> W, H = polar_decomposition_with_grad(A)
    >>> loss = (W ** 2).sum() + 0.5 * (H ** 2).sum()
    >>> loss.backward()
    >>>
    >>> A.grad.shape
    torch.Size([5, 5])

    Using LazySpectrum to reuse eigendecomposition of H between
    forward and backward
    >>> import torch
    >>> from torch.autograd import gradcheck
    >>>
    >>> torch.manual_seed(2)
    >>> n = 5
    >>> A = torch.randn(n, n, dtype=torch.float64, requires_grad=True)
    >>>
    >>> lazy_spec = LazySpectrum()
    >>>
    >>> def scalar_loss_with_lazy(X):
    ...     W, H = polar_decomposition_with_grad(
    ...         X,
    ...         lazy_spec=lazy_spec,
    ...     )
    ...     # Precompute and store eigendecomposition of H
    ...     with torch.no_grad():
    ...         w, Q = torch.linalg.eigh(H)
    ...         lazy_spec.w = w
    ...         lazy_spec.Q = Q
    ...     return (W ** 2).sum() + 0.5 * (H ** 2).sum()
    >>>
    """
    if not A.requires_grad:
        return _polar_decomposition_forward(A, method)
    
    return PolarDecomposition.apply(A, method, eps)
