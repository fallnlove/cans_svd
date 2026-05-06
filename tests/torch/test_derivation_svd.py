import torch
import pytest

from src_torch.derivation.svd import svd_differentiable
from src_torch.derivation.svd_pseudo import svd_pseudo
from src_torch.derivation.svd_tikhonov import svd_tikhonov

torch.set_default_dtype(torch.float64)


def random_matrix(m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g)



def test_shapes_tall_matrix():
    A = random_matrix(20, 10)

    U, S, Vt = svd_differentiable(A)

    assert U.shape == (20, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 10)


def test_shapes_wide_matrix():
    A = random_matrix(10, 20)

    U, S, Vt = svd_differentiable(A)

    assert U.shape == (10, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 20)



def test_orthogonality():
    A = random_matrix(30, 15)

    U, S, Vt = svd_differentiable(A)

    Iu = U.T @ U
    Iv = Vt @ Vt.T

    assert torch.allclose(Iu, torch.eye(15), atol=1e-6)
    assert torch.allclose(Iv, torch.eye(15), atol=1e-6)



def test_reconstruction():
    A = random_matrix(25, 12)

    U, S, Vt = svd_differentiable(A)

    A_hat = U @ torch.diag(S) @ Vt
    rel_err = torch.norm(A - A_hat) / torch.norm(A)

    assert rel_err < 1e-6



def test_singular_values_sorted():
    A = random_matrix(40, 10)

    _, S, _ = svd_differentiable(A)

    assert torch.all(S[:-1] >= S[1:])



def test_low_rank_matrix():
    torch.manual_seed(0)

    m, n, r = 30, 20, 5
    U0, _ = torch.linalg.qr(torch.randn(m, r))
    V0, _ = torch.linalg.qr(torch.randn(n, r))
    S0 = torch.linspace(5.0, 1.0, r)

    A = U0 @ torch.diag(S0) @ V0.T

    U, S, Vt = svd_differentiable(A)

    assert torch.allclose(S[r:], torch.zeros_like(S[r:]), atol=1e-6)

    A_hat = U @ torch.diag(S) @ Vt
    assert torch.allclose(A, A_hat, atol=1e-6)



def test_backward_runs():
    A = random_matrix(15, 10).requires_grad_()

    U, S, Vt = svd_differentiable(A)
    loss = S.sum() + (U ** 2).sum() * 1e-3
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()



@pytest.mark.parametrize(
    "shape",
    [
        (3, 6, 4),
        (5, 4, 6),
    ],
)
def test_svd_forward_batch_nograd(shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64)

    U, S, Vt = svd_differentiable(A)

    batch = A.shape[:-2]
    m, n = A.shape[-2:]
    k = min(m, n)

    assert U.shape == (*batch, m, k)
    assert S.shape == (*batch, k)
    assert Vt.shape == (*batch, k, n)

    A_rec = U @ torch.diag_embed(S) @ Vt
    err = torch.norm(A - A_rec) / torch.norm(A)

    assert err < 1e-8



@pytest.mark.parametrize(
    "shape",
    [
        (3, 6, 4),
        (5, 4, 6),
    ],
)
def test_svd_forward_batch(shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    U, S, Vt = svd_differentiable(A)

    batch = A.shape[:-2]
    m, n = A.shape[-2:]
    k = min(m, n)

    assert U.shape == (*batch, m, k)
    assert S.shape == (*batch, k)
    assert Vt.shape == (*batch, k, n)

    A_rec = U @ torch.diag_embed(S) @ Vt
    err = torch.norm(A - A_rec) / torch.norm(A)

    assert err < 1e-8


@pytest.mark.parametrize(
    "shape",
    [
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck(shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    def f(A):
        _, S, _ = svd_differentiable(A)
        return S

    torch.autograd.gradcheck(
        f,
        (A,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck2(shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    def f(A):
        U, _, _ = svd_differentiable(A)
        return torch.sum(U.abs())

    torch.autograd.gradcheck(
        f,
        (A,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )



def test_against_torch_svd():
    A = random_matrix(20, 10)

    U, S, Vt = svd_differentiable(A)
    U0, S0, Vt0 = torch.linalg.svd(A, full_matrices=False)

    assert torch.allclose(S, S0, atol=1e-6)


@pytest.mark.parametrize("svd_fn", [svd_tikhonov, svd_pseudo])
@pytest.mark.parametrize(
    "shape",
    [
        (6, 4),
        (4, 6),
    ],
)
def test_regularized_svd_forward_and_backward(svd_fn, shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    U, S, Vt = svd_fn(A, cans_tol=1e-6, eps_qr=1e-5)
    m, n = shape
    k = min(m, n)

    assert U.shape == (m, k)
    assert S.shape == (k,)
    assert Vt.shape == (k, n)

    A_rec = U @ torch.diag(S) @ Vt
    err = torch.norm(A - A_rec) / torch.norm(A)
    assert err < 1e-6

    loss = S.sum() + 1e-3 * (U.square().sum() + Vt.square().sum())
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


@pytest.mark.parametrize("svd_fn", [svd_tikhonov, svd_pseudo])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 2),
        (2, 3),
        (6, 4),
        (4, 6),
    ],
)
def test_regularized_svd_gradcheck(svd_fn, shape):
    torch.manual_seed(0)

    A = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    def f(A):
        U, S, Vt = svd_fn(A, cans_tol=1e-8, eps_qr=1e-5)
        return U.abs().sum() + S.abs().sum() + Vt.abs().sum()

    torch.autograd.gradcheck(
        f,
        (A,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )
