import torch

from src_torch.svd.cans import cans_iteration

def cans_svd(matrix: torch.Tensor, degree: int = 3, preprocess: bool = True,
               preprocess_iters: int = 2, delta: float = 0.99, a: float = 1e-3,
               n: int = 50, eps_qr: float = 1e-5) -> torch.Tensor:
    """
    Compute the SVD of a matrix using the CANS method.

    Args:
        matrix (torch.Tensor): The input matrix.
        degree (int): The degree of the polynomial approximation.
        preprocess (bool): Whether to use preprocessing.
        preprocess_iters (int): Number of preprocessing iterations.
        delta (float): The delta parameter for CANS.
        a (float): The scaling parameter for CANS.
        n (int): Number of iterations for the CANS method.
        eps_qr (float): Tolerance for performing QR decomposition to ensure orthogonality.

    Returns:
        U: torch.Tensor: Left singular vectors.
        S: torch.Tensor: Singular values.
        Vt: torch.Tensor: Right singular vectors (transposed).
    """
    device = matrix.device
    dtype = matrix.dtype

    n_start = matrix.shape[0]
    transpose = False
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transpose = True

    W = cans_iteration(
        matrix,
        n=n,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
    )

    H = W.T @ matrix
    H = 0.5 * (H + H.T)

    s, V = torch.linalg.eigh(H)   # ascending order
    U = W @ V
    
    if torch.any(torch.abs(torch.linalg.norm(U, dim=0) - 1) > eps_qr):
        U, R = torch.linalg.qr(U, mode="reduced")
        s = torch.diagonal(R) * s

    sign = torch.sign(s)
    sign[sign == 0] = 1.0

    V = V * sign.unsqueeze(0)
    s = s.abs()

    idx = torch.argsort(s, descending=True)

    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    if not transpose:
        return U, s, V.T
    else:
        return V, s, U.T
