import torch
from src_torch.polar_decomposition.cans import polar_decomposition_cans
from src_torch.polar_decomposition.qdwh import polar_decomposition_qdwh

from src_torch.derivation.polar_decomposition import _polar_decomposition_forward
from src_torch.derivation.eigenvalue_decomposition import _eigen_decomposition_forward, _make_K
from dataclasses import dataclass
from typing import Optional


def _svd_forward(A, polar_method='cans', threshold=1e-8):
    """
    Forward pass for SVD via polar decomposition and eigendecomposition.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (..., m, n).
    eps_polar : float, optional
        Numerical regularization parameter for polar decomposition.
    eps_eigh : float, optional
        Numerical regularization parameter for eigendecomposition.
    polar_method : str, optional
        Method for polar decomposition ('cans' or 'qdwh').

    Returns
    -------
    U : torch.Tensor
        Left singular vectors of shape (..., m, r).
    S : torch.Tensor
        Singular values of shape (..., r).
    Vt : torch.Tensor
        Right singular vectors (transposed) of shape (..., r, n).
    """
    is_transposed = False
    if A.shape[-1] > A.shape[-2]:
        A = A.mT
        is_transposed = True
        
    W, H = _polar_decomposition_forward(A, method=polar_method)

    s, V = _eigen_decomposition_forward(H)

    U = W @ V

    if torch.any(torch.abs(torch.linalg.norm(U, dim=-2) - 1) > 1e-5):
        U, R = torch.linalg.qr(U, mode="reduced")
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        s = diag_R * s

    sign = torch.sign(s)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    V = V * sign.unsqueeze(-2)
    s = s.abs()

    idx = torch.argsort(s, dim=-1, descending=True)

    U = torch.gather(
        U, dim=-1,
        index=idx.unsqueeze(-2).expand(*U.shape[:-1], idx.shape[-1])
    )

    V = torch.gather(
        V, dim=-1,
        index=idx.unsqueeze(-2).expand(*V.shape[:-1], idx.shape[-1])
    )

    S = torch.gather(s, dim=-1, index=idx)
    real_rank = torch.sum(S > threshold, dim=-1)
    # print("Real rank:", real_rank)
    
    if is_transposed:
        return V, S, U, real_rank
    
    return U, S, V, real_rank


class SVD(torch.autograd.Function):
    """
    Custom autograd function for polar decomposition A = W H.

    Forward computes the polar factors W (orthogonal) and H (symmetric).
    Backward propagates gradients using a Lyapunov equation.
    """
    def forward(ctx, A, polar_method: str = "cans", threshold: float = 1e-8, eps: float = 1e-8):

        U, S, V, real_rank = _svd_forward(A, polar_method, threshold)
        
        ctx.save_for_backward(U, S, V)
        ctx.threshold = threshold
        ctx.eps = eps
        ctx.real_rank = real_rank
        
        return U, S, V.mT


    def backward(ctx, grad_U, grad_S, grad_Vt):
        U, S, V = ctx.saved_tensors
        eps = ctx.eps
        r = ctx.real_rank

        # --- slice active subspace ---
        U_r = U[..., :, :r]              # (m, r)
        V_r = V[..., :, :r]              # (n, r)
        S_r = S[..., :r]                 # (r,)

        gU = grad_U[..., :, :r]           # (m, r)
        gVt = grad_Vt[..., :r, :]         # (r, n)
        gS = grad_S[..., :r]              # (r,)

        # --- output gradient ---
        grad_A = torch.zeros(
            (*U.shape[:-2], U.shape[-2], V.shape[-2]),
            device=U.device,
            dtype=U.dtype,
        )

        inv_Sr = 1.0 / S_r

        grad_A += (gU * inv_Sr.unsqueeze(-2)) @ V_r.mT

        tmp = torch.diag_embed(gS)

        K = _make_K(S_r, eps).mT

        tmp += K * (V_r.mT @ gVt.mT + U_r.mT @ gU)
        tmp -= (U_r.mT @ gU) * inv_Sr.unsqueeze(-2)
        tmp = 0.5 * (tmp + tmp.mT)

        denom = S_r.unsqueeze(-1) + S_r.unsqueeze(-2) + eps
        tmp = tmp / denom
        
        grad_A += 2.0 * (U_r * S_r.unsqueeze(-2)) @ tmp @ V_r.mT

        return grad_A, None, None, None


def svd_differentiable(
    A,
    polar_method: str = "cans",
    threshold: float = 1e-8,
    eps: float = 1e-8,
):
    if not A.requires_grad:
        U, S, V, _ = _svd_forward(A, polar_method, threshold)
        return U, S, V.mT

    
    return SVD.apply(A, polar_method, threshold, eps)
