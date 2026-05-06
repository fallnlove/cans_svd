import torch
import pytest


from src_torch.derivation.polar_decomposition import LazySpectrum, polar_decomposition_with_grad


torch.set_default_dtype(torch.float64)


def _finite_diff_grad(f, A, eps=1e-6):
    grad = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            E = torch.zeros_like(A)
            E[i, j] = eps
            grad[i, j] = (f(A + E) - f(A - E)) / (2 * eps)
    return grad


def test_forward_correctness():
    torch.manual_seed(42)
    n = 5

    A = torch.randn(n, n, requires_grad=True)
    W, H = polar_decomposition_with_grad(A)

    recon_error = torch.norm(A - W @ H)
    ortho_error = torch.norm(W.T @ W - torch.eye(n))
    sym_error = torch.norm(H - H.T)

    assert recon_error < 1e-6
    assert ortho_error < 1e-6
    assert sym_error < 1e-10



def test_backward_sanity():
    torch.manual_seed(0)
    n = 5

    A = torch.randn(n, n, requires_grad=True)
    W, H = polar_decomposition_with_grad(A)

    loss = W.sum() + H.sum()
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()
    assert torch.norm(A.grad) > 0


def test_finite_difference_gradient():
    torch.manual_seed(1)
    n = 4

    A = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

    def scalar_loss(X):
        W, H = polar_decomposition_with_grad(X)
        return (W ** 2).sum() + 0.5 * (H ** 2).sum()

    assert torch.autograd.gradcheck(
        scalar_loss,
        (A,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )



def test_batch_support():
    torch.manual_seed(2)
    B, n = 4, 5

    A = torch.randn(B, n, n, requires_grad=True)
    W, H = polar_decomposition_with_grad(A)

    loss = W.sum() + H.sum()
    loss.backward()

    assert A.grad.shape == A.shape
    assert torch.isfinite(A.grad).all()



def test_nearly_singular_case():
    torch.manual_seed(3)
    n = 5

    U, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.tensor([1e-6, 1e-5, 1e-3, 1.0, 3.0])
    A = (U @ torch.diag(s) @ U.T).requires_grad_()

    W, H = polar_decomposition_with_grad(A)
    loss = W.sum()
    loss.backward()

    assert torch.isfinite(A.grad).all()
    assert torch.norm(A.grad) > 0


def _scalar_loss(A, lazy_spec=None):
    W, H = polar_decomposition_with_grad(A, lazy_spec=lazy_spec)
    return (W ** 2).sum() + 0.5 * (H ** 2).sum()



def test_gradcheck_polar():
    torch.manual_seed(0)

    n = 3
    A = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

    def f(X):
        W, H = polar_decomposition_with_grad(X)
        # return (W ** 2).sum() + 0.5 * (H ** 2).sum()
        return W.abs().sum() + 0.5 * (H ** 2).sum()

    torch.autograd.gradcheck(
        f,
        (A,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )
