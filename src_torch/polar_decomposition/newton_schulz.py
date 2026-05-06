import torch
import numpy as np


def newton_schulz(A, num_iters, **kwargs):
    if A.shape[0] < A.shape[1]:
        A = A.T
    X2 = A.T @ A
    X3 = A @ X2

    denom = torch.norm(X3, p='fro') ** (1/3) + 1e-7
    A = A / denom
    X2 = X2 / (denom ** 2)
    X3 = X3 / (denom ** 3)

    I = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    errors = [torch.norm(X2 - I, p='fro') / torch.norm(I, p='fro')]
    mult = [0]

    for i in range(num_iters):
        A = 3 / 2 * A - 1 / 2 * X3
        X2 = A.T @ A
        X3 = A @ X2
        errors.append(torch.norm(X2 - I, p='fro') / torch.norm(I, p='fro'))
        mult.append(mult[-1] + 2)

    return A, errors, mult
