import torch
import pytest

from src_torch.derivation.eigenvalue_decomposition import eigh_with_grad


torch.set_default_dtype(torch.float64)

def make_spd(A, eps=0.1):
    return A @ A.transpose(-2, -1) + eps * torch.eye(A.shape[-1], device=A.device)


def test_spd_reconstruction():
    torch.manual_seed(42)
    n = 5

    A_base = torch.randn(n, n, requires_grad=False)
    A = make_spd(A_base)
    # A.retain_grad()

    eigvals, eigvecs = eigh_with_grad(A)

    A_rec = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    assert torch.allclose(A, A_rec, atol=1e-6)
    assert eigvals.shape == (n,)
    assert eigvecs.shape == (n, n)


def test_orthogonality():
    torch.manual_seed(0)
    n = 6

    A = make_spd(torch.randn(n, n))
    eigvals, eigvecs = eigh_with_grad(A)

    I = torch.eye(n)
    assert torch.allclose(eigvecs.T @ eigvecs, I, atol=1e-6)



def test_backward_exists_and_symmetric():
    torch.manual_seed(42)
    n = 5

    A_base = torch.randn(n, n, requires_grad=True)
    A = make_spd(A_base)
    A.retain_grad()

    _, eigvecs = eigh_with_grad(A)
    loss = eigvecs.sum()
    loss.backward()

    assert A.grad is not None
    assert torch.allclose(A.grad, A.grad.T, atol=1e-8)
    assert torch.norm(A.grad) > 0



def test_batch_forward_backward():
    torch.manual_seed(0)
    batch, n = 3, 5

    A_base = torch.randn(batch, n, n, requires_grad=True)
    A = make_spd(A_base)
    A.retain_grad()

    eigvals, eigvecs = eigh_with_grad(A)

    assert eigvals.shape == (batch, n)
    assert eigvecs.shape == (batch, n, n)

    loss = eigvals.sum() + eigvecs.sum()
    loss.backward()

    assert A.grad is not None
    assert torch.allclose(A.grad, A.grad.transpose(-2, -1), atol=1e-8)



def test_gradient_identity_matrix():
    A = torch.eye(3, requires_grad=True) * 2.0
    A.retain_grad()

    eigvals, _ = eigh_with_grad(A)
    loss = eigvals.sum()
    loss.backward()

    assert A.grad is not None
    assert torch.allclose(A.grad, torch.eye(3), atol=1e-6)



def test_close_eigenvalues_series_stability():
    A = torch.tensor(
        [[2.0, 0.0002, 0.0],
         [0.0, 2.0001, 0.0001],
         [0.000151, 0.00001, 5.0]],
        requires_grad=True,
        dtype=torch.float64,
    )
    A.retain_grad()

    eigvals, eigvecs = eigh_with_grad(A @ A.T)

    P = eigvecs[:, :2] @ eigvecs[:, :2].T
    loss = ((torch.eye(3) - P) @ A).mean()
    loss.backward()

    assert torch.isfinite(eigvals).all()
    assert A.grad is not None
    assert torch.isfinite(A.grad).all()



def test_against_torch_eigh_gradient_stability():
    A = torch.tensor(
        [[2.0, 0.0002, 0.0],
         [0.0, 2.0001, 0.0001],
         [0.000151, 0.00001, 5.0]],
        requires_grad=True,
        dtype=torch.float64,
    )
    A.retain_grad()

    eigvals, eigvecs = torch.linalg.eigh(A @ A.T)

    loss = ((torch.eye(3) - eigvecs[:, 1:] @ eigvecs[:, 1:].T) @ A).mean()
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


def test_gradcheck_eigh():
    torch.manual_seed(0)

    n = 3
    B = torch.randn(n, n, dtype=torch.float64)
    A = (B @ B.T + 1 * torch.eye(n)).requires_grad_()

    def f(X):
        eigvals, eigvecs = eigh_with_grad(X)
        return eigvals.sum() + eigvecs.sum()

    torch.autograd.gradcheck(
        f,
        (A,),
        eps=1e-8,
        atol=1e-4,
        rtol=1e-4,
    )
