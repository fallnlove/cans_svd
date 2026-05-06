import torch

from src_torch.derivation.eigenvalue_decomposition import _eigen_decomposition_forward
from src_torch.derivation.polar_decomposition import _polar_decomposition_forward


def _make_K_tikhonov(lam, eps):
    diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    K = diff / (diff * diff + eps**2)
    K = torch.triu(K, diagonal=1)
    return K - K.mT


def _eigh_tikhonov_bwd(eps, res, grad):
    V, w = res
    grad_V, grad_w = grad

    VT_grad_V = V.mT @ grad_V
    skew = 0.5 * (VT_grad_V - VT_grad_V.mT)
    K = _make_K_tikhonov(w, eps)

    grad_A = V @ (K.mT * skew + torch.diag_embed(grad_w)) @ V.mT
    return 0.5 * (grad_A + grad_A.mT)


def _solve_lyapunov_tikhonov(A, w, Q, eps):
    Qt = Q.mT
    A_tilde = Qt @ A @ Q

    denom = w.unsqueeze(-1) + w.unsqueeze(-2)
    X_tilde = A_tilde * denom / (denom * denom + eps)

    return Q @ X_tilde @ Qt


def _polar_tikhonov_bwd(polar_method, eps, res, grad):
    del polar_method
    W, H, A = res
    grad_W, grad_H = grad

    w, Q = torch.linalg.eigh(H)
    H_inv_diag = w / (w * w + eps**2)
    H_inv = Q @ (H_inv_diag.unsqueeze(-1) * Q.mT)

    B = grad_H - W.mT @ grad_W @ H_inv
    B = 0.5 * (B + B.mT)

    grad_A = grad_W @ H_inv
    X = _solve_lyapunov_tikhonov(B, w, Q, eps)
    grad_A = grad_A + 2.0 * A @ X
    return grad_A


def _svd_tikhonov_forward(A, cans_tol=1e-5, eps_qr=1e-5):
    transposed = A.shape[-1] > A.shape[-2]
    if transposed:
        A = A.mT

    W, H = _polar_decomposition_forward(A, method="cans", cans_tol=cans_tol)
    s, V = _eigen_decomposition_forward(H)
    U = W @ V

    if torch.any(torch.abs(torch.linalg.norm(U, dim=-2) - 1) > eps_qr):
        U, R = torch.linalg.qr(U, mode="reduced")
        s = torch.diagonal(R, dim1=-2, dim2=-1) * s

    sign = torch.sign(s)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    V = V * sign.unsqueeze(-2)
    s = s.abs()

    idx = torch.argsort(s, dim=-1, descending=True)
    U = torch.gather(
        U,
        dim=-1,
        index=idx.unsqueeze(-2).expand(*U.shape[:-1], idx.shape[-1]),
    )
    V = torch.gather(
        V,
        dim=-1,
        index=idx.unsqueeze(-2).expand(*V.shape[:-1], idx.shape[-1]),
    )
    s = torch.gather(s, dim=-1, index=idx)

    if transposed:
        return V, s, U.mT
    return U, s, V.mT


def svd_tikhonov_grad(
    A,
    U,
    s,
    Vt,
    grad_U,
    grad_s,
    grad_Vt,
    cans_tol=1e-5,
):
    del cans_tol
    eps = 1e-8
    threshold = 1e-8
    A_internal = A
    transposed = A.shape[-1] > A.shape[-2]
    if transposed:
        A_internal = A.mT
        U, Vt = Vt.mT, U.mT
        grad_U, grad_Vt = grad_Vt.mT, grad_U.mT

    mask = (s > threshold).to(dtype=s.dtype)
    U = U * mask.unsqueeze(-2)
    Vt = Vt * mask.unsqueeze(-1)
    s = s * mask
    grad_U = grad_U * mask.unsqueeze(-2)
    grad_s = grad_s * mask
    grad_Vt = grad_Vt * mask.unsqueeze(-1)

    V = Vt.mT
    W = U @ Vt
    H = V @ torch.diag_embed(s) @ Vt

    grad_W = grad_U @ Vt
    grad_V = W.mT @ grad_U + grad_Vt.mT

    grad_H = _eigh_tikhonov_bwd(eps, (V, s), (grad_V, grad_s))
    grad_A = _polar_tikhonov_bwd(
        "cans",
        eps,
        (W, H, A_internal),
        (grad_W, grad_H),
    )

    if transposed:
        return grad_A.mT
    return grad_A


class SVDTikhonov(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, cans_tol=1e-5, eps_qr=1e-5):
        U, s, Vt = _svd_tikhonov_forward(A, cans_tol, eps_qr)
        ctx.save_for_backward(A, U, s, Vt)
        ctx.cans_tol = cans_tol
        return U, s, Vt

    @staticmethod
    def backward(ctx, grad_U, grad_s, grad_Vt):
        A, U, s, Vt = ctx.saved_tensors
        grad_A = svd_tikhonov_grad(
            A,
            U,
            s,
            Vt,
            grad_U,
            grad_s,
            grad_Vt,
            cans_tol=ctx.cans_tol,
        )
        return grad_A, None, None


def svd_tikhonov(
    A,
    cans_tol=1e-5,
    eps_qr=1e-5,
):
    if not A.requires_grad:
        return _svd_tikhonov_forward(A, cans_tol, eps_qr)
    return SVDTikhonov.apply(A, cans_tol, eps_qr)


svd_tikhonov_differentiable = svd_tikhonov
