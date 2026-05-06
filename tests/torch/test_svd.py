import torch
import pytest

from src_torch.svd.cans_svd import cans_svd


torch.set_default_dtype(torch.float64)


def random_matrix(m, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g)


def test_shapes():
    A = random_matrix(20, 10)

    U, S, Vt = cans_svd(A, n=30)

    assert U.shape == (20, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 10)


def test_orthogonality():
    A = random_matrix(30, 15)

    U, S, Vt = cans_svd(A, n=40)

    Iu = U.T @ U
    Iv = Vt @ Vt.T

    assert torch.allclose(Iu, torch.eye(15), atol=1e-6)
    assert torch.allclose(Iv, torch.eye(15), atol=1e-6)


def test_reconstruction():
    A = random_matrix(25, 12)

    U, S, Vt = cans_svd(A, n=50)

    A_rec = U @ torch.diag(S) @ Vt
    rel_err = torch.linalg.norm(A - A_rec) / torch.linalg.norm(A)

    assert rel_err < 1e-5


def test_singular_values_nonnegative():
    A = random_matrix(20, 20)

    _, S, _ = cans_svd(A)

    assert torch.all(S >= -1e-10)


def test_compare_with_torch_svd():
    A = random_matrix(40, 15)

    Uc, Sc, Vtc = cans_svd(A, n=60)
    Ut, St, Vtt = torch.linalg.svd(A, full_matrices=False)

    Sc_sorted = torch.sort(Sc, descending=True).values
    St_sorted = torch.sort(St, descending=True).values

    rel_err = torch.linalg.norm(Sc_sorted - St_sorted) / torch.linalg.norm(St_sorted)
    assert rel_err < 1e-4


def test_preprocess_flag_consistency():
    A = random_matrix(30, 10)

    U1, S1, Vt1 = cans_svd(A, preprocess=True)
    U2, S2, Vt2 = cans_svd(A, preprocess=False)

    S1s = torch.sort(S1).values
    S2s = torch.sort(S2).values

    assert torch.allclose(S1s, S2s, atol=1e-4)


@pytest.mark.parametrize("degree", [3, 5])
def test_degree(degree):
    A = random_matrix(20, 8)

    U, S, Vt = cans_svd(A, degree=degree, n=40)

    A_rec = U @ torch.diag(S) @ Vt
    rel_err = torch.linalg.norm(A - A_rec) / torch.linalg.norm(A)

    assert rel_err < 1e-4


def test_low_rank_matrix():
    torch.manual_seed(0)

    m, n, r = 40, 30, 5

    U0, _ = torch.linalg.qr(torch.randn(m, r))
    V0, _ = torch.linalg.qr(torch.randn(n, r))
    S0 = torch.linspace(10.0, 1.0, r)

    A = U0 @ torch.diag(S0) @ V0.T

    U, S, Vt = cans_svd(A)

    Iu = U.T @ U
    Iv = Vt @ Vt.T
    
    assert torch.allclose(Iv, torch.eye(n), atol=1e-6)
    assert torch.allclose(Iu, torch.eye(n), atol=1e-6)

    A_rec = U @ torch.diag(S) @ Vt
    rel_err = torch.linalg.norm(A - A_rec) / torch.linalg.norm(A)
    assert rel_err < 1e-6

    S_sorted = torch.sort(S, descending=True).values

    assert torch.allclose(S_sorted[:r], S0, rtol=1e-4, atol=1e-6)
    assert torch.all(S_sorted[r:] < 1e-6)

