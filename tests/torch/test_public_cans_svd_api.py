import pytest
import torch

from cans_svd import svd


torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("differentiation", ["tikhonov", "pseudo"])
@pytest.mark.parametrize("matmul_precision", ["fp32", "tf32"])
def test_public_svd_torch_backend(differentiation, matmul_precision):
    torch.manual_seed(0)
    A = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

    U, S, Vt = svd(
        A,
        backend="torch",
        differentiation=differentiation,
        matmul_precision=matmul_precision,
    )

    assert U.shape == (5, 3)
    assert S.shape == (3,)
    assert Vt.shape == (3, 3)

    A_rec = U @ torch.diag(S) @ Vt
    assert torch.linalg.norm(A - A_rec) / torch.linalg.norm(A) < 1e-4

    loss = S.sum() + 1e-3 * (U.square().sum() + Vt.square().sum())
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


def test_public_svd_rejects_unknown_options():
    A = torch.eye(2, dtype=torch.float64)

    with pytest.raises(ValueError):
        svd(A, backend="numpy")

    with pytest.raises(ValueError):
        svd(A, differentiation="unknown")

    with pytest.raises(ValueError):
        svd(A, matmul_precision="fp16")
