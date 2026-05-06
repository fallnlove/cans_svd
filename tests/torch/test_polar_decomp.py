import torch
import pytest

from src_torch.polar_decomposition.cans import polar_decomposition_cans
from src_torch.polar_decomposition.qdwh import polar_decomposition_qdwh


def random_matrix(m, n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=dtype)


def test_output_shapes_tall_matrix():
    A = random_matrix(30, 10)

    Q, H = polar_decomposition_cans(A, n_iters=40)

    assert Q.shape == (30, 10)
    assert H.shape == (10, 10)


def test_output_shapes_wide_matrix():
    A = random_matrix(30, 10)

    Q, H = polar_decomposition_cans(A, n_iters=40)

    assert Q.shape == (30, 10)
    assert H.shape[0] == H.shape[1]


def test_small_matrix():
    A = random_matrix(5, 3)

    Q, H = polar_decomposition_cans(A, n_iters=30)

    assert Q.shape == (5, 3)
    assert H.shape == (3, 3)



def test_Q_orthogonality():
    A = random_matrix(40, 15)

    Q, _ = polar_decomposition_cans(A, n_iters=50)

    I = torch.eye(Q.shape[1], dtype=A.dtype)
    assert torch.allclose(Q.T @ Q, I, atol=1e-6)


def test_H_symmetric():
    A = random_matrix(25, 10)

    _, H = polar_decomposition_cans(A, n_iters=40)

    assert torch.allclose(H, H.T, atol=1e-6)


def test_H_positive_semidefinite():
    A = random_matrix(30, 12)

    _, H = polar_decomposition_cans(A, n_iters=50)

    eigvals = torch.linalg.eigvalsh(H)
    assert torch.all(eigvals >= -1e-8)


def test_reconstruction_accuracy():
    A = random_matrix(35, 14)

    Q, H = polar_decomposition_cans(A, n_iters=60)

    A_hat = Q @ H
    rel_err = torch.norm(A - A_hat) / torch.norm(A)

    assert rel_err < 1e-5



def test_with_preprocess():
    A = random_matrix(40, 12)

    Q, H = polar_decomposition_cans(
        A,
        n_iters=40,
        preprocess=True,
        preprocess_iters=3,
        degree=3,
    )

    I = torch.eye(Q.shape[1], dtype=A.dtype)
    assert torch.allclose(Q.T @ Q, I, atol=1e-6)

    A_hat = Q @ H
    assert torch.norm(A - A_hat) / torch.norm(A) < 1e-5



def test_rank_deficient_matrix():
    m, n, r = 30, 12, 5
    U = torch.randn(m, r)
    V = torch.randn(r, n)
    A = U @ V

    Q, H = polar_decomposition_cans(A, n_iters=70)

    assert torch.isfinite(Q).all()
    assert torch.isfinite(H).all()

    # I = torch.eye(Q.shape[1])
    # assert torch.allclose(Q.T @ Q, I, atol=1e-5) ## not necessarily orthogonal

    eigvals = torch.linalg.eigvalsh(H)
    print(eigvals)
    assert torch.min(eigvals) >= -1e-5
    assert torch.sum(eigvals < 1e-5) >= (n - r)
    
    # check resconstruction
    A_hat = Q @ H
    rel_err = torch.norm(A - A_hat) / torch.norm(A)
    assert rel_err < 1e-5
        

def test_qdwh_basic_properties():
    A = random_matrix(40, 15, dtype=torch.float32)

    Q, H = polar_decomposition_qdwh(A)

    assert Q.shape == (40, 15)
    assert H.shape == (15, 15)

    I = torch.eye(Q.shape[1], dtype=torch.float32)
    assert torch.allclose(Q.T @ Q, I, atol=1e-4)

    assert torch.allclose(H, H.T, atol=1e-7)

    eigvals = torch.linalg.eigvalsh(H)
    assert torch.all(eigvals >= -1e-8)

    A_hat = Q @ H
    rel_err = torch.norm(A - A_hat) / torch.norm(A)
    assert rel_err < 1e-6


def test_qdwh_rank_deficient_matrix():
    m, n, r = 30, 12, 5
    U = torch.randn(m, r, dtype=torch.float32)
    V = torch.randn(r, n, dtype=torch.float32)
    A = U @ V 

    Q, H = polar_decomposition_qdwh(A)

    assert torch.isfinite(Q).all()
    assert torch.isfinite(H).all()

    eigvals = torch.linalg.eigvalsh(H)
    assert torch.min(eigvals) >= -1e-4
    assert torch.sum(eigvals < 1e-4) >= (n - r)

    A_hat = Q @ H
    rel_err = torch.norm(A - A_hat) / torch.norm(A)
    assert rel_err < 1e-4
